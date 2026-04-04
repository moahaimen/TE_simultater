#!/usr/bin/env python3
"""Stage 2 LOCK: Fix learning signal and prove K is actually learned.

Corrections from pilot:
  - K target normalized to [0,1] with sigmoid output
  - Strong K-loss weights W ∈ {5, 10}
  - Proof metrics: corr(K_pred, K_target), MAE, per-topology + overall
  - Isolated in results/gnn_plus/stage2_lock/
  - Folders: training_w5/, training_w10/ (not w05)

Goal: Verify dynamic K is actually learned before moving to advanced Stage 2.
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
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ---------- constants ----------
CONFIG_PATH = "configs/phase1_reactive_full.yaml"
SEED = 42
LT = 15
DEVICE = "cpu"

GNN_ORIG_CKPT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
OUTPUT_ROOT = Path("results/gnn_plus/stage2_lock")

K_CANDIDATES = [15, 20, 25, 30, 35, 40, 45, 50]
K_MIN = 1
K_MAX = 50

W_VALUES = [5.0, 10.0]  # Strong K-loss only

SUP_LR = 5e-4
SUP_WD = 1e-5
SUP_EPOCHS = 25
SUP_PATIENCE = 6
SUP_MARGIN = 0.1
MAX_TRAIN_PER_TOPO = 25
MAX_VAL_PER_TOPO = 15

RL_LR = 1e-4
RL_EPOCHS = 8
RL_PATIENCE = 4
RL_EMA = 0.9

MAX_TEST_STEPS = 50

PILOT_TOPOS = {"germany50_real", "geant_core", "abilene_backbone"}


# ---------------------------------------------------------------------------
#  Dynamic K Head (with NORMALIZED output via sigmoid)
# ---------------------------------------------------------------------------

class DynamicKHead(nn.Module):
    """K prediction head: graph_embed + traffic_stats -> K_norm ∈ [0,1]."""

    def __init__(self, hidden_dim: int, k_min: int = 1, k_max: int = 50, dropout: float = 0.1):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.k_range = k_max - k_min
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 4, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, graph_embed, traffic_stats):
        """Returns normalized K in [0, 1] via sigmoid."""
        x = torch.cat([graph_embed, traffic_stats], dim=-1)
        raw = self.head(x).squeeze(-1)
        k_norm = torch.sigmoid(raw)  # [0, 1]
        return k_norm

    def denormalize(self, k_norm):
        """Convert normalized K back to actual K in [k_min, k_max]."""
        return self.k_min + k_norm * self.k_range


class GNNWithDynamicK(nn.Module):
    """Wraps GNNFlowSelector + DynamicKHead with normalized K."""

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
        
        # Predict NORMALIZED K
        k_norm = self.k_head(graph_embed, traffic_stats)  # [0, 1]
        k_continuous = self.k_head.denormalize(k_norm)  # [k_min, k_max]
        k_int = int(torch.clamp(torch.round(k_continuous), K_MIN, K_MAX).item())
        
        info["k_pred"] = k_int
        info["k_continuous"] = float(k_continuous.item())
        info["k_norm"] = float(k_norm.item())
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
            
            # Store normalized oracle K
            oracle_k_norm = oracle_k / K_MAX
            
            samples.append({"dataset": ds, "path_library": pl,
                "tm_vector": tm, "prev_tm": prev_tm, "telemetry": telem,
                "oracle_selected": oracle_sel, "oracle_mlu": oracle_mlu,
                "oracle_k": oracle_k, "oracle_k_norm": oracle_k_norm,
                "oracle_method": oracle_method,
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
               "model_type": "dynamic_k_lock"}
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
#  Training with NORMALIZED K loss
# ---------------------------------------------------------------------------

def train_one(M, w_k, train_samples, val_samples):
    """Train with normalized K target and strong K-loss."""
    tag = f"w{int(w_k)}"  # w5, w10 (not w05)
    out_dir = OUTPUT_ROOT / f"training_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original GNN weights
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
    print(f"[Supervised W={w_k} (normalized K)]", flush=True)
    for epoch in range(1, SUP_EPOCHS + 1):
        ep0 = time.perf_counter()
        wrapper.train()
        ep_losses = []; ep_kl = []; ep_kp = []; ep_kn = []  # k_norm
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
            
            # NORMALIZED K loss
            k_target_norm = torch.tensor(float(s["oracle_k_norm"]), device=DEVICE)
            k_pred_norm = torch.tensor(float(info["k_norm"]), device=DEVICE)
            k_loss = F.mse_loss(k_pred_norm, k_target_norm)  # Loss on [0,1]
            
            loss = flow_loss + w_k * k_loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            optimizer.step()
            
            ep_losses.append(float(loss.item()))
            ep_kl.append(float(k_loss.item()))
            ep_kp.append(float(k_cont.item()))
            ep_kn.append(float(info["k_norm"]))
        scheduler.step()
        tl = float(np.mean(ep_losses)); kl = float(np.mean(ep_kl))
        kp_mean = float(np.mean(ep_kp)); kp_std = float(np.std(ep_kp))
        kn_mean = float(np.mean(ep_kn))

        wrapper.eval()
        vls = []; vkp = []; vkn = []
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
                k_target_norm = torch.tensor(float(s["oracle_k_norm"]), device=DEVICE)
                k_pred_norm = torch.tensor(float(info["k_norm"]), device=DEVICE)
                vk = F.mse_loss(k_pred_norm, k_target_norm)
                vls.append(float((vl + w_k * vk).item()))
                vkp.append(float(k_cont.item()))
                vkn.append(float(info["k_norm"]))
        val_loss = float(np.mean(vls))
        val_k_mean = float(np.mean(vkp)); val_k_std = float(np.std(vkp))
        val_kn_mean = float(np.mean(vkn))
        et = time.perf_counter() - ep0

        logs.append({"stage": "supervised", "epoch": epoch, "train_loss": tl,
            "val_loss": val_loss, "k_loss": kl,
            "k_pred_mean": kp_mean, "k_pred_std": kp_std, "k_norm_mean": kn_mean,
            "val_k_mean": val_k_mean, "val_k_std": val_k_std, "val_k_norm_mean": val_kn_mean,
            "alpha": float(wrapper.base.alpha.item()), "epoch_time_sec": et})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}: loss={tl:.4f} vl={val_loss:.4f} "
                  f"kl={kl:.4f} K={kp_mean:.1f}±{kp_std:.1f} K_norm={kn_mean:.3f} "
                  f"[{et:.1f}s]", flush=True)
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
        ep_mlu = []; ep_kp = []; ep_kn = []
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
            ep_mlu.append(mlu); ep_kp.append(k_int); ep_kn.append(info["k_norm"])
            reward = -mlu
            if baseline_reward is None: baseline_reward = reward
            else: baseline_reward = RL_EMA * baseline_reward + (1 - RL_EMA) * reward
            adv = reward - baseline_reward
            flow_rl = -adv * slp
            om = torch.zeros(scores.size(0), device=DEVICE)
            for oid in s["oracle_selected"]:
                if oid < scores.size(0): om[oid] = 1.0
            rl = M["_ranking_loss"](scores, om, margin=0.05)
            k_target_norm = torch.tensor(float(s["oracle_k_norm"]), device=DEVICE)
            k_pred_norm = torch.tensor(float(info["k_norm"]), device=DEVICE)
            k_loss = F.mse_loss(k_pred_norm, k_target_norm)
            total = flow_rl + 0.3 * rl + w_k * k_loss
            optimizer.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            optimizer.step()

        mm = float(np.mean(ep_mlu)) if ep_mlu else float("inf")
        mk = float(np.mean(ep_kp)) if ep_kp else 0; sk = float(np.std(ep_kp)) if ep_kp else 0
        mkn = float(np.mean(ep_kn)) if ep_kn else 0

        wrapper.eval(); vms = []; vks = []; vkns = []
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
                vks.append(k_int); vkns.append(info["k_norm"])
        vm = float(np.mean(vms)) if vms else float("inf")
        vmk = float(np.mean(vks)) if vks else 0; vsk = float(np.std(vks)) if vks else 0
        vkn = float(np.mean(vkns)) if vkns else 0
        et = time.perf_counter() - ep0
        rl_logs.append({"stage": "reinforce", "epoch": epoch,
            "train_mlu": mm, "val_mlu": vm,
            "train_k_mean": mk, "train_k_std": sk, "train_k_norm_mean": mkn,
            "val_k_mean": vmk, "val_k_std": vsk, "val_k_norm_mean": vkn,
            "alpha": float(wrapper.base.alpha.item()), "epoch_time_sec": et})
        print(f"  RL Ep {epoch}: mlu={mm:.2f} val={vm:.2f} "
              f"K={mk:.1f}±{sk:.1f} K_norm={mkn:.3f} [{et:.0f}s]", flush=True)
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
#  Evaluation with PROOF metrics
# ---------------------------------------------------------------------------

def evaluate_with_proof(M, datasets, model, w_name):
    """Evaluate and collect proof metrics per topology + overall."""
    print(f"\n[Evaluating {w_name} with proof metrics]")
    
    eval_dir = OUTPUT_ROOT / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Per-topology results
    k_preds_by_topo = {}
    k_targets_by_topo = {}
    timesteps_by_topo = {}
    
    rows = []
    
    for topo_key in sorted(datasets.keys()):
        ds = datasets[topo_key]
        pl = ds.path_library; ecmp_base = M["ecmp_splits"](pl)
        caps = np.asarray(ds.capacities, dtype=float)
        test_idx = M["split_indices"](ds, "test")[:MAX_TEST_STEPS]
        print(f"\n  {topo_key}: {len(test_idx)} steps", flush=True)
        
        k_preds = []
        k_targets = []
        timesteps = []
        
        for step_i, t_idx in enumerate(test_idx):
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12: continue
            
            routing = M["apply_routing"](tm, ecmp_base, pl, caps)
            telem = M["compute_reactive_telemetry"](
                tm, ecmp_base, pl, routing, np.asarray(ds.weights, dtype=float))
            
            # Get oracle K
            o_k, o_sel, o_mlu, _ = compute_oracle_k_target(M, tm, ecmp_base, pl, caps)
            
            # Get prediction
            gd = M["build_graph_tensors"](ds, telemetry=telem, device=DEVICE)
            od = M["build_od_features"](ds, tm, pl, telemetry=telem, device=DEVICE)
            ts = compute_traffic_stats(telem, tm, gd["num_edges"])
            
            with torch.no_grad():
                scores, k_cont, k_int, info = model(gd, od, ts)
                k_pred = k_int
            
            k_preds.append(k_pred)
            k_targets.append(o_k)
            timesteps.append(step_i)
            
            row = {
                "topology": topo_key,
                "step": step_i,
                "k_pred": k_pred,
                "k_target": o_k,
                "k_norm_pred": info["k_norm"],
                "k_norm_target": o_k / K_MAX,
            }
            rows.append(row)
            
            if (step_i + 1) % 10 == 0:
                print(f" {step_i+1}(K={k_pred},tgt={o_k})", end="", flush=True)
        
        print()
        
        # SANITY CHECK: Report unique K values and first 20 predictions
        if k_preds:
            unique_k = len(set(k_preds))
            print(f"  [SANITY] {topo_key}: {unique_k} unique K values out of {len(k_preds)} predictions")
            print(f"  [SANITY] First 20 K_pred: {k_preds[:20]}")
            if unique_k <= 2:
                print(f"  ⚠️ WARNING: K_pred nearly constant! Only {unique_k} unique values.")
        
        k_preds_by_topo[topo_key] = k_preds
        k_targets_by_topo[topo_key] = k_targets
        timesteps_by_topo[topo_key] = timesteps
    
    # Save per-timestep CSV
    df = pd.DataFrame(rows)
    df.to_csv(eval_dir / f"per_timestep_{w_name}.csv", index=False)
    
    return k_preds_by_topo, k_targets_by_topo, timesteps_by_topo


def compute_learning_proof(k_preds_by_topo, k_targets_by_topo):
    """Compute correlation and MAE per topology + overall."""
    results = {}
    
    # Overall
    all_preds = []
    all_targets = []
    for topo in k_preds_by_topo:
        all_preds.extend(k_preds_by_topo[topo])
        all_targets.extend(k_targets_by_topo[topo])
    
    if len(all_preds) > 1 and len(set(all_targets)) > 1:
        corr, pval = pearsonr(all_preds, all_targets)
    else:
        corr, pval = 0.0, 1.0
    
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    
    results['TOTAL'] = {
        'pearson_corr': corr,
        'p_value': pval,
        'mae': mae,
        'k_pred_mean': np.mean(all_preds),
        'k_pred_std': np.std(all_preds),
        'k_pred_min': np.min(all_preds),
        'k_pred_max': np.max(all_preds),
        'k_target_mean': np.mean(all_targets),
        'k_target_std': np.std(all_targets),
        'n_samples': len(all_preds),
    }
    
    # Per topology
    for topo in sorted(k_preds_by_topo.keys()):
        k_p = k_preds_by_topo[topo]
        k_t = k_targets_by_topo[topo]
        
        if len(k_p) > 1 and len(set(k_t)) > 1:
            corr, pval = pearsonr(k_p, k_t)
        else:
            corr, pval = 0.0, 1.0
        
        mae = np.mean(np.abs(np.array(k_p) - np.array(k_t)))
        
        results[topo] = {
            'pearson_corr': corr,
            'p_value': pval,
            'mae': mae,
            'k_pred_mean': np.mean(k_p),
            'k_pred_std': np.std(k_p),
            'k_pred_min': np.min(k_p),
            'k_pred_max': np.max(k_p),
            'k_target_mean': np.mean(k_t),
            'k_target_std': np.std(k_t),
            'n_samples': len(k_p),
        }
    
    return results


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def generate_plots(k_preds_by_topo, k_targets_by_topo, timesteps_by_topo, w_name):
    """Generate K histograms, over-time plots, and scatter plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    plots_dir = OUTPUT_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for topo in k_preds_by_topo:
        k_preds = np.array(k_preds_by_topo[topo])
        k_targets = np.array(k_targets_by_topo[topo])
        timesteps = np.array(timesteps_by_topo[topo])
        
        # 1. Histogram
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        axes[0].hist(k_preds, bins=20, alpha=0.6, label='K_pred', color='blue', edgecolor='black')
        axes[0].hist(k_targets, bins=20, alpha=0.6, label='K_target', color='red', edgecolor='black')
        axes[0].set_xlabel('K')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{topo}: K Distribution ({w_name})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. K over time
        axes[1].plot(timesteps, k_preds, 'o-', label='K_pred', color='blue', markersize=3, alpha=0.7)
        axes[1].plot(timesteps, k_targets, 's-', label='K_target', color='red', markersize=3, alpha=0.7)
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('K')
        axes[1].set_title(f'{topo}: K over Time ({w_name})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{topo}_{w_name}_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Scatter plot: K_pred vs K_target
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(k_targets, k_preds, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Diagonal line
        min_val = min(k_targets.min(), k_preds.min())
        max_val = max(k_targets.max(), k_preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        
        # Correlation annotation
        if len(k_preds) > 1:
            corr, _ = pearsonr(k_preds, k_targets)
            corr_status = "✓ Strong" if corr > 0.6 else ("~ Partial" if corr > 0.3 else "✗ Weak")
            ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}\n{corr_status}', 
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   verticalalignment='top')
        
        ax.set_xlabel('K_target (Oracle)')
        ax.set_ylabel('K_pred (Model)')
        ax.set_title(f'{topo}: K_pred vs K_target ({w_name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{topo}_{w_name}_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    print("="*70)
    print("STAGE 2 LOCK: Verify K is actually learned")
    print(f"W values: {W_VALUES} (strong K-loss)")
    print(f"K target: NORMALIZED [0,1] with sigmoid output")
    print(f"Folders: training_w5/, training_w10/ (not w05)")
    print(f"Output: {OUTPUT_ROOT}")
    print("="*70)
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    M = setup()
    
    # Load topologies
    print("\n[1] Loading topologies...")
    datasets = load_pilot_topologies(M)
    print(f"\nLoaded {len(datasets)} topologies: {list(datasets.keys())}")
    
    if len(datasets) == 0:
        print("\n" + "="*70)
        print("ERROR: Zero topologies loaded!")
        print("STOPPING - Cannot proceed without data.")
        print("="*70)
        return
    
    # Count total TMs
    total_tms = sum(len(ds.tm) if hasattr(ds, 'tm') else 0 for ds in datasets.values())
    print(f"Total traffic matrices: {total_tms}")
    
    # Collect samples
    print("\n[2] Collecting samples with oracle K sweep...")
    train_samples = collect_samples(M, datasets, 'train', MAX_TRAIN_PER_TOPO)
    val_samples = collect_samples(M, datasets, 'val', MAX_VAL_PER_TOPO)
    
    print(f"\nTotal samples: {len(train_samples)} train, {len(val_samples)} val")
    
    if len(train_samples) == 0:
        print("\n" + "="*70)
        print("ERROR: No training samples collected!")
        print("STOPPING - Cannot train without samples.")
        print("="*70)
        return
    
    # Train both W values
    all_results = {}
    
    for w_val in W_VALUES:
        w_name = f"w{int(w_val)}"  # w5, w10
        print(f"\n{'='*70}")
        print(f"[3] Training {w_name} (W={w_val}, normalized K)")
        print(f"{'='*70}")
        
        model, train_time = train_one(M, w_val, train_samples, val_samples)
        
        # Evaluate with proof metrics
        k_preds_by_topo, k_targets_by_topo, timesteps_by_topo = \
            evaluate_with_proof(M, datasets, model, w_name)
        
        # Compute learning proof
        proof_results = compute_learning_proof(k_preds_by_topo, k_targets_by_topo)
        all_results[w_name] = proof_results
        
        # Generate plots
        generate_plots(k_preds_by_topo, k_targets_by_topo, timesteps_by_topo, w_name)
        
        # Print proof metrics
        print(f"\n[Proof Metrics for {w_name}]")
        print(f"  {'Topology':20s}  {'Corr':>8s}  {'Status':>10s}  {'MAE':>6s}  {'Std':>6s}  {'Range':>15s}")
        print(f"  {'-'*20}  {'-'*8}  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*15}")
        
        for topo in ['abilene_backbone', 'geant_core', 'germany50_real', 'TOTAL']:
            if topo in proof_results:
                r = proof_results[topo]
                corr = r['pearson_corr']
                if corr > 0.6:
                    status = "✓ Strong"
                elif corr > 0.3:
                    status = "~ Partial"
                else:
                    status = "✗ Weak"
                
                k_range = f"[{r['k_pred_min']:.0f}, {r['k_pred_max']:.0f}]"
                print(f"  {topo:20s}  {corr:>+8.3f}  {status:>10s}  {r['mae']:>6.1f}  "
                      f"{r['k_pred_std']:>6.1f}  {k_range:>15s}")
    
    # Save summary
    print("\n[4] Saving results...")
    
    summary_rows = []
    for w_name in all_results:
        for topo, metrics in all_results[w_name].items():
            row = {'w_name': w_name, 'topology': topo}
            row.update(metrics)
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_ROOT / "learning_proof_summary.csv", index=False)
    print(f"  Saved: {OUTPUT_ROOT / 'learning_proof_summary.csv'}")
    
    # Verdict
    print("\n" + "="*70)
    print("STAGE 2 LOCK VERDICT")
    print("="*70)
    
    for w_name in all_results:
        total_corr = all_results[w_name]['TOTAL']['pearson_corr']
        
        if total_corr > 0.6:
            verdict = "PASS - Convincing learning"
            rec = "Stage 2 is LOCKED. Proceed to full sweep."
        elif total_corr > 0.3:
            verdict = "PARTIAL - Weak learning"
            rec = "Needs architecture fix before proceeding."
        else:
            verdict = "FAIL - No learning detected"
            rec = "Major redesign required."
        
        print(f"\n{w_name}:")
        print(f"  Overall corr: {total_corr:.3f}")
        print(f"  Verdict: {verdict}")
        print(f"  Recommendation: {rec}")
    
    print(f"\nResults saved to: {OUTPUT_ROOT}")
    print("="*70)


if __name__ == "__main__":
    main()
