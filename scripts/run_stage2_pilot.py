#!/usr/bin/env python3
"""Stage 2 PILOT: Dynamic K prediction — quick feasibility check.

Scope:
  - Topologies: Germany50, GEANT, Abilene (if cheap)
  - W ∈ {0.5, 1.0} only
  - Reduced training: 20 epochs supervised, 5 REINFORCE
  - Reduced samples: 25 train / 15 val per topo
  - Eval on 50 test steps max

Goal: verify K_pred is no longer collapsed and varies meaningfully.
Decision rule: proceed to full sweep only if K std > 2 and no MLU collapse.

Output: results/gnn_plus/stage2_pilot/   (does NOT touch stage2_dynamic_k/)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ---------- constants ----------
CONFIG_PATH = "configs/phase1_reactive_full.yaml"
SEED = 42
LT = 15          # shorter LP timeout for pilot
DEVICE = "cpu"

GNN_ORIG_CKPT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
OUTPUT_ROOT = Path("results/gnn_plus/stage2_pilot")

K_CANDIDATES = [15, 20, 25, 30, 35, 40, 45, 50]
K_MIN = 1
K_MAX = 50

W_VALUES = [0.5, 1.0]  # pilot only

# Reduced training for pilot
SUP_LR = 5e-4
SUP_WD = 1e-5
SUP_EPOCHS = 20       # reduced from 30
SUP_PATIENCE = 6      # reduced from 8
SUP_MARGIN = 0.1
MAX_TRAIN_PER_TOPO = 25   # reduced from 40
MAX_VAL_PER_TOPO = 15     # reduced from 20

RL_LR = 1e-4
RL_EPOCHS = 5          # reduced from 10
RL_PATIENCE = 3        # reduced from 4
RL_EMA = 0.9

MAX_TEST_STEPS = 50    # reduced from 75

PILOT_TOPOS = {"germany50_real", "geant_core", "abilene_backbone"}


# ---------------------------------------------------------------------------
#  Dynamic K Head (same as full script)
# ---------------------------------------------------------------------------

class DynamicKHead(nn.Module):
    """K prediction head: graph_embed + traffic_stats -> K ∈ [K_MIN, K_MAX]."""

    def __init__(self, hidden_dim: int, k_min: int = 1, k_max: int = 50, dropout: float = 0.1):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 4, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, graph_embed, traffic_stats):
        x = torch.cat([graph_embed, traffic_stats], dim=-1)
        raw = self.head(x).squeeze(-1)
        k_frac = torch.sigmoid(raw)
        return self.k_min + k_frac * (self.k_max - self.k_min)


class GNNWithDynamicK(nn.Module):
    """Wraps GNNFlowSelector + DynamicKHead."""

    def __init__(self, base_model, hidden_dim=64, k_min=1, k_max=50, dropout=0.1):
        super().__init__()
        self.base = base_model
        self.k_head = DynamicKHead(hidden_dim, k_min, k_max, dropout)
        self._hidden_dim = hidden_dim

    def forward(self, graph_data, od_data, traffic_stats):
        scores, _, info = self.base(graph_data, od_data)
        # Recompute graph embedding
        node_feat = graph_data["node_features"]
        edge_index = graph_data["edge_index"]
        edge_feat = graph_data["edge_features"]
        h = F.relu(self.base.node_proj(node_feat))
        e = F.relu(self.base.edge_proj(edge_feat))
        for layer in self.base.gnn_layers:
            h = layer(h, edge_index, e)
        graph_embed = h.mean(dim=0)
        k_continuous = self.k_head(graph_embed, traffic_stats)
        k_int = int(torch.clamp(torch.round(k_continuous), K_MIN, K_MAX).item())
        info["k_pred"] = k_int
        info["k_continuous"] = float(k_continuous.item())
        return scores, k_continuous, k_int, info

    def select_critical_flows(self, graph_data, od_data, traffic_stats, active_mask):
        with torch.no_grad():
            scores, k_continuous, k_int, info = self.forward(graph_data, od_data, traffic_stats)
        scores_np = scores.detach().cpu().numpy().astype(np.float32)
        active = np.asarray(active_mask, dtype=bool)
        active_indices = np.where(active)[0]
        if active_indices.size == 0 or k_int <= 0:
            info["k_used"] = 0
            return [], info
        take = min(k_int, active_indices.size)
        active_scores = scores_np[active_indices]
        top_local = np.argsort(-active_scores, kind="mergesort")[:take]
        selected = [int(active_indices[i]) for i in top_local]
        info["k_used"] = take
        return selected, info


# ---------------------------------------------------------------------------
#  Setup
# ---------------------------------------------------------------------------

def setup():
    from te.baselines import (ecmp_splits, select_bottleneck_critical,
                               select_sensitivity_critical, select_topk_by_demand)
    from te.lp_solver import solve_selected_path_lp
    from te.simulator import apply_routing
    from phase1_reactive.eval.common import load_bundle, load_named_dataset, collect_specs
    from phase1_reactive.eval.core import split_indices
    from phase1_reactive.drl.gnn_selector import (
        GNNSelectorConfig, GNNFlowSelector,
        load_gnn_selector, save_gnn_selector,
        build_graph_tensors, build_od_features)
    from phase1_reactive.drl.state_builder import compute_reactive_telemetry
    from phase1_reactive.drl.gnn_training import _ranking_loss
    return {
        "ecmp_splits": ecmp_splits,
        "select_bottleneck_critical": select_bottleneck_critical,
        "select_sensitivity_critical": select_sensitivity_critical,
        "select_topk_by_demand": select_topk_by_demand,
        "solve_selected_path_lp": solve_selected_path_lp,
        "apply_routing": apply_routing,
        "load_bundle": load_bundle,
        "load_named_dataset": load_named_dataset,
        "collect_specs": collect_specs,
        "split_indices": split_indices,
        "GNNSelectorConfig": GNNSelectorConfig,
        "GNNFlowSelector": GNNFlowSelector,
        "load_gnn_selector": load_gnn_selector,
        "save_gnn_selector": save_gnn_selector,
        "build_graph_tensors": build_graph_tensors,
        "build_od_features": build_od_features,
        "compute_reactive_telemetry": compute_reactive_telemetry,
        "_ranking_loss": _ranking_loss,
    }


def load_pilot_topologies(M):
    bundle = M["load_bundle"](CONFIG_PATH)
    eval_specs = M["collect_specs"](bundle, "eval_topologies")
    gen_specs = M["collect_specs"](bundle, "generalization_topologies")
    datasets = {}
    for spec in eval_specs + gen_specs:
        if spec.key not in PILOT_TOPOS:
            continue
        try:
            ds, pl = M["load_named_dataset"](bundle, spec, 500)
        except Exception as e:
            print(f"  Skip {spec.key}: {e}"); continue
        ds.path_library = pl
        datasets[ds.key] = ds
        n_tms = len(ds.tm) if hasattr(ds, "tm") else 0
        print(f"  {ds.key}: {len(ds.nodes)}N, {len(ds.edges)}E, {n_tms} TMs")
    return datasets


# ---------------------------------------------------------------------------
#  Oracle K + traffic stats
# ---------------------------------------------------------------------------

def compute_oracle_k_target(M, tm, ecmp_base, path_library, capacities):
    best_k = 40; best_mlu = float("inf"); best_sel = []; best_method = "topk"
    for k in K_CANDIDATES:
        for name, fn in [
            ("topk", lambda k=k: M["select_topk_by_demand"](tm, k)),
            ("bottleneck", lambda k=k: M["select_bottleneck_critical"](
                tm, ecmp_base, path_library, capacities, k)),
            ("sensitivity", lambda k=k: M["select_sensitivity_critical"](
                tm, ecmp_base, path_library, capacities, k)),
        ]:
            try:
                sel = fn()
                lp = M["solve_selected_path_lp"](
                    tm_vector=tm, selected_ods=sel, base_splits=ecmp_base,
                    path_library=path_library, capacities=capacities,
                    time_limit_sec=LT)
                mlu = float(lp.routing.mlu)
                if np.isfinite(mlu) and mlu < best_mlu:
                    best_mlu = mlu; best_k = k; best_sel = sel; best_method = name
            except Exception:
                continue
    return best_k, best_sel, best_mlu, best_method


def compute_traffic_stats(telemetry, tm_vector, num_edges):
    util = np.asarray(telemetry.utilization, dtype=np.float64)[:num_edges]
    tm = np.asarray(tm_vector, dtype=np.float64)
    mean_util = float(np.mean(util))
    max_util = float(np.max(util))
    frac_congested = float(np.mean(util > 0.9))
    tm_active = tm[tm > 1e-12]
    demand_cv = float(np.std(tm_active) / (np.mean(tm_active) + 1e-12)) if len(tm_active) > 1 else 0.0
    demand_cv = min(demand_cv, 5.0) / 5.0
    return torch.tensor([mean_util, max_util, frac_congested, demand_cv],
                        dtype=torch.float32, device=DEVICE)


# ---------------------------------------------------------------------------
#  Sample collection
# ---------------------------------------------------------------------------

def collect_samples(M, datasets, split, max_per):
    rng = np.random.default_rng(SEED)
    samples = []; k_targets = []
    for key, ds in datasets.items():
        pl = ds.path_library; ecmp_base = M["ecmp_splits"](pl)
        caps = np.asarray(ds.capacities, dtype=float)
        indices = M["split_indices"](ds, split)
        if len(indices) > max_per:
            indices = sorted(rng.choice(indices, size=max_per, replace=False).tolist())
        count = 0; prev_tm = None
        for t_idx in indices:
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12: prev_tm = tm; continue
            routing = M["apply_routing"](tm, ecmp_base, pl, caps)
            telem = M["compute_reactive_telemetry"](
                tm, ecmp_base, pl, routing, np.asarray(ds.weights, dtype=float))
            oracle_k, oracle_sel, oracle_mlu, oracle_method = \
                compute_oracle_k_target(M, tm, ecmp_base, pl, caps)
            if not oracle_sel: prev_tm = tm; continue
            samples.append({"dataset": ds, "path_library": pl,
                "tm_vector": tm, "prev_tm": prev_tm, "telemetry": telem,
                "oracle_selected": oracle_sel, "oracle_mlu": oracle_mlu,
                "oracle_k": oracle_k, "oracle_method": oracle_method,
                "k_crit": oracle_k, "capacities": caps})
            k_targets.append(oracle_k)
            count += 1; prev_tm = tm
        print(f"    {key}: {count} {split}")
    if k_targets:
        kt = np.array(k_targets)
        print(f"  Oracle K: mean={kt.mean():.1f} std={kt.std():.1f} "
              f"min={kt.min()} max={kt.max()} unique={len(np.unique(kt))}")
    return samples


# ---------------------------------------------------------------------------
#  Save / Load
# ---------------------------------------------------------------------------

def save_model(wrapper, path, extra_meta=None):
    payload = {"base_state_dict": wrapper.base.state_dict(),
               "k_head_state_dict": wrapper.k_head.state_dict(),
               "hidden_dim": wrapper._hidden_dim,
               "k_min": wrapper.k_head.k_min, "k_max": wrapper.k_head.k_max,
               "model_type": "dynamic_k_pilot"}
    if extra_meta: payload.update(extra_meta)
    torch.save(payload, str(path))


def load_model(M, path, device="cpu"):
    payload = torch.load(str(path), map_location=torch.device(device), weights_only=False)
    cfg = M["GNNSelectorConfig"](); cfg.learn_k_crit = False; cfg.device = device
    base = M["GNNFlowSelector"](cfg).to(device)
    base.load_state_dict(payload["base_state_dict"])
    wrapper = GNNWithDynamicK(base, hidden_dim=payload["hidden_dim"],
                               k_min=payload["k_min"], k_max=payload["k_max"])
    wrapper.k_head.load_state_dict(payload["k_head_state_dict"])
    wrapper.eval(); return wrapper


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train_one(M, w_k, train_samples, val_samples):
    tag = f"w{str(w_k).replace('.', '')}"
    out_dir = OUTPUT_ROOT / f"training_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original GNN weights (disable its k_head)
    base_model, _ = M["load_gnn_selector"](GNN_ORIG_CKPT, device=DEVICE)
    cfg = M["GNNSelectorConfig"](); cfg.learn_k_crit = False; cfg.dropout = 0.1
    base = M["GNNFlowSelector"](cfg).to(DEVICE)
    orig_sd = base_model.state_dict(); new_sd = base.state_dict()
    for k, v in orig_sd.items():
        if k in new_sd and new_sd[k].shape == v.shape: new_sd[k] = v
    base.load_state_dict(new_sd)

    wrapper = GNNWithDynamicK(base, hidden_dim=64, k_min=K_MIN, k_max=K_MAX, dropout=0.1)
    wrapper.to(DEVICE)

    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=SUP_LR, weight_decay=SUP_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUP_EPOCHS)
    rng = np.random.default_rng(SEED)
    logs = []; best_vl = float("inf"); best_ep = 0; stale = 0
    t0 = time.perf_counter()

    print(f"\n{'='*50}", flush=True)
    print(f"[Supervised W={w_k}]", flush=True)
    for epoch in range(1, SUP_EPOCHS + 1):
        ep0 = time.perf_counter()
        wrapper.train()
        ep_losses = []; ep_kl = []; ep_kp = []
        for si in rng.permutation(len(train_samples)):
            s = train_samples[si]
            gd = M["build_graph_tensors"](s["dataset"], telemetry=s["telemetry"], device=DEVICE)
            od = M["build_od_features"](s["dataset"], s["tm_vector"],
                s["path_library"], telemetry=s["telemetry"], device=DEVICE)
            ts = compute_traffic_stats(s["telemetry"], s["tm_vector"], gd["num_edges"])
            scores, k_cont, k_int, info = wrapper(gd, od, ts)
            om = torch.zeros(scores.size(0), device=DEVICE)
            for oid in s["oracle_selected"]:
                if oid < scores.size(0): om[oid] = 1.0
            flow_loss = M["_ranking_loss"](scores, om, margin=SUP_MARGIN)
            k_target = torch.tensor(float(s["oracle_k"]), device=DEVICE)
            k_loss = F.mse_loss(k_cont, k_target)
            loss = flow_loss + w_k * k_loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(float(loss.item()))
            ep_kl.append(float(k_loss.item()))
            ep_kp.append(float(k_cont.item()))
        scheduler.step()
        tl = float(np.mean(ep_losses)); kl = float(np.mean(ep_kl))
        kp_mean = float(np.mean(ep_kp)); kp_std = float(np.std(ep_kp))

        wrapper.eval()
        vls = []; vkp = []
        with torch.no_grad():
            for s in val_samples:
                gd = M["build_graph_tensors"](s["dataset"], telemetry=s["telemetry"], device=DEVICE)
                od = M["build_od_features"](s["dataset"], s["tm_vector"],
                    s["path_library"], telemetry=s["telemetry"], device=DEVICE)
                ts = compute_traffic_stats(s["telemetry"], s["tm_vector"], gd["num_edges"])
                scores, k_cont, k_int, info = wrapper(gd, od, ts)
                om = torch.zeros(scores.size(0), device=DEVICE)
                for oid in s["oracle_selected"]:
                    if oid < scores.size(0): om[oid] = 1.0
                vl = M["_ranking_loss"](scores, om, margin=SUP_MARGIN)
                k_target = torch.tensor(float(s["oracle_k"]), device=DEVICE)
                vk = F.mse_loss(k_cont, k_target)
                vls.append(float((vl + w_k * vk).item()))
                vkp.append(float(k_cont.item()))
        val_loss = float(np.mean(vls))
        val_k_mean = float(np.mean(vkp)); val_k_std = float(np.std(vkp))
        et = time.perf_counter() - ep0

        logs.append({"stage": "supervised", "epoch": epoch, "train_loss": tl,
            "val_loss": val_loss, "k_loss": kl,
            "k_pred_mean": kp_mean, "k_pred_std": kp_std,
            "val_k_mean": val_k_mean, "val_k_std": val_k_std,
            "alpha": float(wrapper.base.alpha.item()), "epoch_time_sec": et})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}: loss={tl:.4f} vl={val_loss:.4f} "
                  f"kl={kl:.1f} K={kp_mean:.1f}±{kp_std:.1f} "
                  f"valK={val_k_mean:.1f}±{val_k_std:.1f} [{et:.1f}s]", flush=True)
        if val_loss + 1e-6 < best_vl:
            best_vl = val_loss; best_ep = epoch; stale = 0
            save_model(wrapper, out_dir / "supervised.pt",
                       extra_meta={"best_epoch": best_ep, "w_k": w_k})
        else: stale += 1
        if stale >= SUP_PATIENCE: print(f"  Early stop at {epoch}"); break

    sup_time = time.perf_counter() - t0
    print(f"  Supervised done: {sup_time:.0f}s, best_ep={best_ep}", flush=True)
    pd.DataFrame(logs).to_csv(out_dir / "supervised_log.csv", index=False)

    wrapper = load_model(M, out_dir / "supervised.pt", device=DEVICE)

    # REINFORCE
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=RL_LR)
    rng = np.random.default_rng(SEED)
    baseline_reward = None; best_vm = float("inf"); best_re = 0; stale = 0
    rl_logs = []; t0 = time.perf_counter()
    print(f"\n[REINFORCE W={w_k}]", flush=True)

    for epoch in range(1, RL_EPOCHS + 1):
        ep0 = time.perf_counter()
        wrapper.train()
        ep_mlu = []; ep_kp = []
        for si in rng.permutation(len(train_samples)):
            s = train_samples[si]
            gd = M["build_graph_tensors"](s["dataset"], telemetry=s["telemetry"], device=DEVICE)
            od = M["build_od_features"](s["dataset"], s["tm_vector"],
                s["path_library"], telemetry=s["telemetry"], device=DEVICE)
            ts = compute_traffic_stats(s["telemetry"], s["tm_vector"], gd["num_edges"])
            scores, k_cont, k_int, info = wrapper(gd, od, ts)
            active = s["tm_vector"] > 0; ai = np.where(active)[0]
            if ai.size == 0: continue
            take = min(k_int, ai.size)
            as_ = scores[torch.tensor(ai, dtype=torch.long, device=DEVICE)]
            lp_ = F.log_softmax(as_, dim=0)
            _, tl_ = torch.topk(as_, take)
            sel = [int(ai[i]) for i in tl_.cpu().numpy()]
            slp = lp_[tl_].sum()
            ecmp = M["ecmp_splits"](s["path_library"])
            try:
                lp = M["solve_selected_path_lp"](tm_vector=s["tm_vector"], selected_ods=sel,
                    base_splits=ecmp, path_library=s["path_library"],
                    capacities=s["capacities"], time_limit_sec=10)
                mlu = float(lp.routing.mlu)
            except Exception: continue
            if not np.isfinite(mlu): continue
            ep_mlu.append(mlu); ep_kp.append(k_int)
            reward = -mlu
            if baseline_reward is None: baseline_reward = reward
            else: baseline_reward = RL_EMA * baseline_reward + (1 - RL_EMA) * reward
            adv = reward - baseline_reward
            flow_rl = -adv * slp
            om = torch.zeros(scores.size(0), device=DEVICE)
            for oid in s["oracle_selected"]:
                if oid < scores.size(0): om[oid] = 1.0
            rl = M["_ranking_loss"](scores, om, margin=0.05)
            k_target = torch.tensor(float(s["oracle_k"]), device=DEVICE)
            k_loss = F.mse_loss(k_cont, k_target)
            total = flow_rl + 0.3 * rl + w_k * k_loss
            optimizer.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            optimizer.step()

        mm = float(np.mean(ep_mlu)) if ep_mlu else float("inf")
        mk = float(np.mean(ep_kp)) if ep_kp else 0; sk = float(np.std(ep_kp)) if ep_kp else 0

        wrapper.eval(); vms = []; vks = []
        with torch.no_grad():
            for s in val_samples:
                gd = M["build_graph_tensors"](s["dataset"], telemetry=s["telemetry"], device=DEVICE)
                od = M["build_od_features"](s["dataset"], s["tm_vector"],
                    s["path_library"], telemetry=s["telemetry"], device=DEVICE)
                ts = compute_traffic_stats(s["telemetry"], s["tm_vector"], gd["num_edges"])
                scores, k_cont, k_int, info = wrapper(gd, od, ts)
                active = s["tm_vector"] > 0; ai = np.where(active)[0]
                if ai.size == 0: continue
                take = min(k_int, ai.size)
                as_ = scores[torch.tensor(ai, dtype=torch.long, device=DEVICE)]
                _, tl_ = torch.topk(as_, take)
                sel = [int(ai[i]) for i in tl_.cpu().numpy()]
                ecmp = M["ecmp_splits"](s["path_library"])
                try:
                    lp = M["solve_selected_path_lp"](tm_vector=s["tm_vector"], selected_ods=sel,
                        base_splits=ecmp, path_library=s["path_library"],
                        capacities=s["capacities"], time_limit_sec=10)
                    vms.append(float(lp.routing.mlu))
                except Exception: pass
                vks.append(k_int)
        vm = float(np.mean(vms)) if vms else float("inf")
        vmk = float(np.mean(vks)) if vks else 0; vsk = float(np.std(vks)) if vks else 0
        et = time.perf_counter() - ep0
        rl_logs.append({"stage": "reinforce", "epoch": epoch,
            "train_mlu": mm, "val_mlu": vm,
            "train_k_mean": mk, "train_k_std": sk,
            "val_k_mean": vmk, "val_k_std": vsk,
            "alpha": float(wrapper.base.alpha.item()), "epoch_time_sec": et})
        print(f"  RL Ep {epoch}: mlu={mm:.2f} val={vm:.2f} "
              f"K={mk:.1f}±{sk:.1f} [{et:.0f}s]", flush=True)
        if vm + 1e-6 < best_vm:
            best_vm = vm; best_re = epoch; stale = 0
            save_model(wrapper, out_dir / "final.pt",
                       extra_meta={"best_epoch": best_re, "w_k": w_k})
        else: stale += 1
        if stale >= RL_PATIENCE: print(f"  RL early stop at {epoch}"); break

    rl_time = time.perf_counter() - t0
    pd.DataFrame(rl_logs).to_csv(out_dir / "reinforce_log.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "w_k": w_k, "sup_time": sup_time, "sup_best_epoch": best_ep,
        "rl_time": rl_time, "rl_best_epoch": best_re, "rl_best_val_mlu": best_vm,
        "total_time": sup_time + rl_time,
    }, indent=2) + "\n", encoding="utf-8")

    ckpt = out_dir / "final.pt"
    if not ckpt.exists(): ckpt = out_dir / "supervised.pt"
    wrapper = load_model(M, ckpt, device=DEVICE)
    return wrapper, sup_time + rl_time


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def _dist(cur, prev, k):
    if prev is None: return 0.0
    return len(set(cur) ^ set(prev)) / max(k, 1)


def evaluate_all(M, datasets, models_dict):
    eval_dir = OUTPUT_ROOT / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    gnn_orig, _ = M["load_gnn_selector"](GNN_ORIG_CKPT, device=DEVICE)
    gnn_orig.eval()
    for m in models_dict.values(): m.eval()

    rows = []
    tags = list(models_dict.keys())

    for topo_key in sorted(datasets.keys()):
        ds = datasets[topo_key]
        pl = ds.path_library; ecmp_base = M["ecmp_splits"](pl)
        caps = np.asarray(ds.capacities, dtype=float)
        test_idx = M["split_indices"](ds, "test")[:MAX_TEST_STEPS]
        print(f"\n[Eval] {topo_key}: {len(test_idx)} steps", flush=True)

        prev_sels = {"orig": None}
        for tag in tags: prev_sels[tag] = None
        prev_tm = None

        for step_i, t_idx in enumerate(test_idx):
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12: prev_tm = tm; continue
            routing = M["apply_routing"](tm, ecmp_base, pl, caps)
            telem = M["compute_reactive_telemetry"](
                tm, ecmp_base, pl, routing, np.asarray(ds.weights, dtype=float))
            amask = (np.asarray(tm, dtype=np.float64) > 1e-12).astype(np.float32)

            o_k, o_sel, o_mlu, _ = compute_oracle_k_target(M, tm, ecmp_base, pl, caps)
            row = {"topology": topo_key, "step": step_i, "tm_idx": t_idx,
                   "oracle_mlu": o_mlu, "oracle_k": o_k}

            # Original GNN (fixed K=40)
            t0 = time.perf_counter()
            gd = M["build_graph_tensors"](ds, telemetry=telem, device=DEVICE)
            od = M["build_od_features"](ds, tm, pl, telemetry=telem, device=DEVICE)
            with torch.no_grad():
                sel, info = gnn_orig.select_critical_flows(
                    gd, od, active_mask=amask, k_crit_default=40, force_default_k=True)
            t_orig = (time.perf_counter() - t0) * 1000
            try:
                lp = M["solve_selected_path_lp"](tm_vector=tm, selected_ods=sel,
                    base_splits=ecmp_base, path_library=pl, capacities=caps, time_limit_sec=LT)
                mlu = float(lp.routing.mlu)
            except Exception: mlu = float("inf")
            row["mlu_orig"] = mlu; row["k_orig"] = 40; row["time_ms_orig"] = t_orig
            row["dist_orig"] = _dist(sel, prev_sels["orig"], 40)
            row["pr_orig"] = (mlu - o_mlu) / (o_mlu + 1e-12)
            prev_sels["orig"] = sel

            # Dynamic K variants
            for tag, model in models_dict.items():
                t0 = time.perf_counter()
                gd = M["build_graph_tensors"](ds, telemetry=telem, device=DEVICE)
                od = M["build_od_features"](ds, tm, pl, telemetry=telem, device=DEVICE)
                ts = compute_traffic_stats(telem, tm, gd["num_edges"])
                with torch.no_grad():
                    sel, info = model.select_critical_flows(gd, od, ts, active_mask=amask)
                t_m = (time.perf_counter() - t0) * 1000
                k_pred = info.get("k_pred", 40); k_used = info.get("k_used", 40)
                try:
                    lp = M["solve_selected_path_lp"](tm_vector=tm, selected_ods=sel,
                        base_splits=ecmp_base, path_library=pl, capacities=caps, time_limit_sec=LT)
                    mlu_m = float(lp.routing.mlu)
                except Exception: mlu_m = float("inf")
                row[f"mlu_{tag}"] = mlu_m
                row[f"k_pred_{tag}"] = k_pred; row[f"k_used_{tag}"] = k_used
                row[f"time_ms_{tag}"] = t_m
                row[f"dist_{tag}"] = _dist(sel, prev_sels[tag], k_used)
                row[f"pr_{tag}"] = (mlu_m - o_mlu) / (o_mlu + 1e-12)
                prev_sels[tag] = sel

            rows.append(row)
            prev_tm = tm
            if (step_i + 1) % 25 == 0:
                parts = [f"orig={row['mlu_orig']:.4f}(K=40)"]
                for tag in tags:
                    parts.append(f"{tag}={row[f'mlu_{tag}']:.4f}(K={row[f'k_pred_{tag}']})")
                print(f"    step {step_i+1}: {' '.join(parts)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(eval_dir / "pilot_per_timestep.csv", index=False)

    # Summary per topology
    summary_rows = []
    for topo_key in sorted(datasets.keys()):
        sub = df[df["topology"] == topo_key]
        if sub.empty: continue
        row = {"topology": topo_key, "num_steps": len(sub),
               "mean_mlu_orig": sub["mlu_orig"].mean(),
               "mean_dist_orig": sub["dist_orig"].mean(),
               "mean_pr_orig": sub["pr_orig"].mean(),
               "oracle_k_mean": sub["oracle_k"].mean(),
               "oracle_k_std": sub["oracle_k"].std(),
               "oracle_k_min": sub["oracle_k"].min(),
               "oracle_k_max": sub["oracle_k"].max()}
        for tag in tags:
            row[f"mean_mlu_{tag}"] = sub[f"mlu_{tag}"].mean()
            row[f"mean_dist_{tag}"] = sub[f"dist_{tag}"].mean()
            row[f"mean_pr_{tag}"] = sub[f"pr_{tag}"].mean()
            row[f"mean_k_pred_{tag}"] = sub[f"k_pred_{tag}"].mean()
            row[f"std_k_pred_{tag}"] = sub[f"k_pred_{tag}"].std()
            row[f"min_k_pred_{tag}"] = sub[f"k_pred_{tag}"].min()
            row[f"max_k_pred_{tag}"] = sub[f"k_pred_{tag}"].max()
            row[f"mlu_improve_{tag}_pct"] = (
                (sub["mlu_orig"].mean() - sub[f"mlu_{tag}"].mean())
                / (sub["mlu_orig"].mean() + 1e-12) * 100)
            row[f"dist_improve_{tag}_pct"] = (
                (sub["dist_orig"].mean() - sub[f"dist_{tag}"].mean())
                / (sub["dist_orig"].mean() + 1e-12) * 100)
            row[f"wins_{tag}"] = int((sub[f"mlu_{tag}"] < sub["mlu_orig"]).sum())
            row[f"ties_{tag}"] = int((abs(sub[f"mlu_{tag}"] - sub["mlu_orig"]) < 1e-6).sum())
        summary_rows.append(row)

    # Total aggregate
    row = {"topology": "TOTAL", "num_steps": len(df),
           "mean_mlu_orig": df["mlu_orig"].mean(),
           "mean_dist_orig": df["dist_orig"].mean(),
           "mean_pr_orig": df["pr_orig"].mean(),
           "oracle_k_mean": df["oracle_k"].mean(),
           "oracle_k_std": df["oracle_k"].std(),
           "oracle_k_min": df["oracle_k"].min(),
           "oracle_k_max": df["oracle_k"].max()}
    for tag in tags:
        row[f"mean_mlu_{tag}"] = df[f"mlu_{tag}"].mean()
        row[f"mean_dist_{tag}"] = df[f"dist_{tag}"].mean()
        row[f"mean_pr_{tag}"] = df[f"pr_{tag}"].mean()
        row[f"mean_k_pred_{tag}"] = df[f"k_pred_{tag}"].mean()
        row[f"std_k_pred_{tag}"] = df[f"k_pred_{tag}"].std()
        row[f"min_k_pred_{tag}"] = df[f"k_pred_{tag}"].min()
        row[f"max_k_pred_{tag}"] = df[f"k_pred_{tag}"].max()
        row[f"mlu_improve_{tag}_pct"] = (
            (df["mlu_orig"].mean() - df[f"mlu_{tag}"].mean())
            / (df["mlu_orig"].mean() + 1e-12) * 100)
        row[f"dist_improve_{tag}_pct"] = (
            (df["dist_orig"].mean() - df[f"dist_{tag}"].mean())
            / (df["dist_orig"].mean() + 1e-12) * 100)
        row[f"wins_{tag}"] = int((df[f"mlu_{tag}"] < df["mlu_orig"]).sum())
        row[f"ties_{tag}"] = int((abs(df[f"mlu_{tag}"] - df["mlu_orig"]) < 1e-6).sum())
    summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(eval_dir / "pilot_summary.csv", index=False)
    print(f"\n[Eval] Saved {len(df)} rows + summary")
    return df, summary_df


# ---------------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------------

def generate_plots(df, summary_df, tags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = OUTPUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    colors = {"orig": "tab:blue", "w05": "tab:orange", "w10": "tab:red"}
    labels_map = {"orig": "Original (K=40)", "w05": "DynK W=0.5", "w10": "DynK W=1.0"}
    topos = sorted(df["topology"].unique())

    # 1. K histogram per topology per W
    fig, axes = plt.subplots(len(tags), len(topos),
                              figsize=(4.5 * len(topos), 3.5 * len(tags)), squeeze=False)
    for j, tag in enumerate(tags):
        for i, topo in enumerate(topos):
            ax = axes[j, i]; sub = df[df["topology"] == topo]
            k_vals = sub[f"k_pred_{tag}"].values
            ax.hist(k_vals, bins=range(0, 55, 2), alpha=0.7,
                   color=colors.get(tag, "gray"), edgecolor="black")
            ax.axvline(40, color="tab:blue", ls="--", lw=2, label="Fixed K=40")
            ax.axvline(sub["oracle_k"].mean(), color="black", ls=":",
                      lw=2, label=f"Oracle μ={sub['oracle_k'].mean():.0f}")
            ax.axvline(np.mean(k_vals), color="red", ls="-.",
                      lw=2, label=f"Pred μ={np.mean(k_vals):.1f}")
            ax.set_title(f"{topo}\n{labels_map.get(tag, tag)}", fontsize=9, fontweight="bold")
            ax.set_xlabel("K"); ax.set_ylabel("Count"); ax.legend(fontsize=6)
            ax.set_xlim(0, 55)
    fig.suptitle("K Prediction Distribution — Pilot", fontweight="bold", fontsize=14)
    fig.tight_layout(); fig.savefig(plot_dir / "k_histogram.png", dpi=150); plt.close(fig)

    # 2. K over time per topology (one plot per topo, all W overlaid)
    for topo in topos:
        sub = df[df["topology"] == topo].sort_values("step")
        if sub.empty: continue
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        steps = sub["step"].values

        # MLU
        ax1.plot(steps, sub["mlu_orig"].values, label="Original (K=40)",
                color="tab:blue", lw=1.5, alpha=0.8)
        for tag in tags:
            ax1.plot(steps, sub[f"mlu_{tag}"].values, label=labels_map.get(tag, tag),
                    color=colors.get(tag, "gray"), lw=1.5, alpha=0.8)
        ax1.set_ylabel("MLU"); ax1.set_title(f"{topo} — MLU over Time", fontweight="bold")
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        # K
        ax2.axhline(40, color="tab:blue", ls="--", lw=1.5, label="Fixed K=40", alpha=0.6)
        ax2.plot(steps, sub["oracle_k"].values, color="black", ls=":",
                lw=2, label="Oracle K", alpha=0.7)
        for tag in tags:
            ax2.plot(steps, sub[f"k_pred_{tag}"].values, label=labels_map.get(tag, tag),
                    color=colors.get(tag, "gray"), lw=1.5, alpha=0.8)
        ax2.set_ylabel("K"); ax2.set_xlabel("Timestep")
        ax2.set_title("K Prediction over Time", fontweight="bold")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"k_over_time_{topo}.png", dpi=150)
        plt.close(fig)

    # 3. MLU CDF per topology
    n = len(topos)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    for i, topo in enumerate(topos):
        ax = axes[0, i]; sub = df[df["topology"] == topo]
        for key in ["orig"] + tags:
            col = f"mlu_{key}" if key != "orig" else "mlu_orig"
            vals = np.sort(sub[col].values)
            cdf = np.arange(1, len(vals)+1) / len(vals)
            ax.plot(vals, cdf, label=labels_map.get(key, key),
                   color=colors.get(key, "gray"), lw=2)
        ax.set_title(topo, fontweight="bold"); ax.set_xlabel("MLU"); ax.set_ylabel("CDF")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle("MLU CDF — Pilot Topologies", fontweight="bold")
    fig.tight_layout(); fig.savefig(plot_dir / "mlu_cdf.png", dpi=150); plt.close(fig)

    # 4. Disturbance CDF
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    for i, topo in enumerate(topos):
        ax = axes[0, i]; sub = df[df["topology"] == topo]
        for key in ["orig"] + tags:
            col = f"dist_{key}" if key != "orig" else "dist_orig"
            vals = np.sort(sub[col].values)
            cdf = np.arange(1, len(vals)+1) / len(vals)
            ax.plot(vals, cdf, label=labels_map.get(key, key),
                   color=colors.get(key, "gray"), lw=2)
        ax.set_title(topo, fontweight="bold"); ax.set_xlabel("Disturbance"); ax.set_ylabel("CDF")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle("Disturbance CDF — Pilot Topologies", fontweight="bold")
    fig.tight_layout(); fig.savefig(plot_dir / "disturbance_cdf.png", dpi=150); plt.close(fig)

    print(f"[Plots] Saved to {plot_dir}")


# ---------------------------------------------------------------------------
#  Pilot verdict
# ---------------------------------------------------------------------------

def print_verdict(summary_df, tags):
    """Print go/no-go decision based on pilot results."""
    print("\n" + "=" * 70)
    print("PILOT VERDICT")
    print("=" * 70)

    total = summary_df[summary_df["topology"] == "TOTAL"]
    if total.empty:
        print("  ERROR: No total row found")
        return

    go = True
    reasons = []

    for tag in tags:
        k_std = total[f"std_k_pred_{tag}"].values[0]
        k_mean = total[f"mean_k_pred_{tag}"].values[0]
        k_min = total[f"min_k_pred_{tag}"].values[0]
        k_max = total[f"max_k_pred_{tag}"].values[0]
        mlu_impr = total[f"mlu_improve_{tag}_pct"].values[0]
        dist_impr = total[f"dist_improve_{tag}_pct"].values[0]

        print(f"\n  {tag}:")
        print(f"    K: mean={k_mean:.1f}, std={k_std:.1f}, range=[{k_min}, {k_max}]")
        print(f"    MLU improvement: {mlu_impr:+.2f}%")
        print(f"    Disturbance improvement: {dist_impr:+.2f}%")

        # Check criteria
        if k_std < 2.0:
            reasons.append(f"{tag}: K std={k_std:.1f} < 2.0 (collapsed)")
            go = False
        if mlu_impr < -5.0:
            reasons.append(f"{tag}: MLU collapsed ({mlu_impr:+.2f}%)")
            go = False

    print(f"\n  Decision: {'GO — proceed to full sweep' if go else 'NO-GO — dynamic K not useful'}")
    if reasons:
        print("  Reasons for NO-GO:")
        for r in reasons:
            print(f"    - {r}")

    return go


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 2 PILOT: DYNAMIC K (Germany50, GEANT, Abilene)")
    print(f"W values: {W_VALUES}")
    print("=" * 70)
    total_start = time.perf_counter()

    M = setup()

    print("\n[1] Loading pilot topologies...", flush=True)
    datasets = load_pilot_topologies(M)
    print(f"  Loaded: {list(datasets.keys())}")

    print("\n[2] Collecting samples with oracle K sweep...", flush=True)
    print("  (Sweeping 8 K values × 3 selectors per timestep — slower than Stage 1)")
    train_samples = collect_samples(M, datasets, "train", MAX_TRAIN_PER_TOPO)
    val_samples = collect_samples(M, datasets, "val", MAX_VAL_PER_TOPO)
    print(f"  Total: {len(train_samples)} train, {len(val_samples)} val")

    # Save oracle K distribution
    oracle_dir = OUTPUT_ROOT / "oracle_k"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    k_data = [{"topology": s["dataset"].key, "oracle_k": s["oracle_k"],
               "oracle_method": s["oracle_method"], "oracle_mlu": s["oracle_mlu"],
               "split": "train"} for s in train_samples]
    k_data += [{"topology": s["dataset"].key, "oracle_k": s["oracle_k"],
                "oracle_method": s["oracle_method"], "oracle_mlu": s["oracle_mlu"],
                "split": "val"} for s in val_samples]
    pd.DataFrame(k_data).to_csv(oracle_dir / "oracle_k_distribution.csv", index=False)

    models = {}
    for w_k in W_VALUES:
        tag = f"w{str(w_k).replace('.', '')}"
        print(f"\n[3] Training W={w_k} ({tag})...", flush=True)
        model, t_time = train_one(M, w_k, train_samples, val_samples)
        models[tag] = model
        print(f"  Training time: {t_time:.0f}s")

    print("\n[4] Evaluating...", flush=True)
    df, summary_df = evaluate_all(M, datasets, models)

    print("\n[5] Generating plots...", flush=True)
    tags = list(models.keys())
    generate_plots(df, summary_df, tags)

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print("STAGE 2 PILOT RESULTS")
    print("=" * 70)
    display_cols = ["topology", "num_steps", "mean_mlu_orig",
                    "oracle_k_mean", "oracle_k_std"]
    for tag in tags:
        display_cols += [f"mean_mlu_{tag}", f"mlu_improve_{tag}_pct",
                        f"mean_k_pred_{tag}", f"std_k_pred_{tag}",
                        f"min_k_pred_{tag}", f"max_k_pred_{tag}",
                        f"mean_dist_{tag}", f"dist_improve_{tag}_pct",
                        f"wins_{tag}"]
    available = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available].to_string(index=False))
    print(f"\nTotal wall time: {total_time:.0f}s")

    go = print_verdict(summary_df, tags)
    print("=" * 70)
