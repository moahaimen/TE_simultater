#!/usr/bin/env python3
"""Stage 2: Dynamic K prediction with original features.

Isolates the dynamic K mechanism from feature enrichment.
Uses the original GNN features (34 effective dims) with a redesigned K_pred head.

Key changes from screening (where K collapsed to 39):
  1. K_target is per-timestep oracle: sweep K ∈ {15,20,25,30,35,40,45,50},
     pick the K that yields lowest post-LP MLU
  2. K_pred head gets traffic statistics as extra input (not just topology)
  3. K-loss weight W is swept: {0.1, 0.5, 1.0, 5.0}
  4. Stability: K_target variance forces the model to learn varying K

Output: results/gnn_plus/stage2_dynamic_k/
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
LT = 20
DEVICE = "cpu"

GNN_ORIG_CKPT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
OUTPUT_ROOT = Path("results/gnn_plus/stage2_dynamic_k")

# K sweep for oracle K_target
K_CANDIDATES = [15, 20, 25, 30, 35, 40, 45, 50]
K_MIN = 1
K_MAX = 50

# W sweep for K-loss weight
W_VALUES = [0.1, 0.5, 1.0, 5.0]

# Training hyperparameters (match Stage 1)
SUP_LR = 5e-4
SUP_WD = 1e-5
SUP_EPOCHS = 30
SUP_PATIENCE = 8
SUP_MARGIN = 0.1
MAX_TRAIN_PER_TOPO = 40
MAX_VAL_PER_TOPO = 20

RL_LR = 1e-4
RL_EPOCHS = 10
RL_PATIENCE = 4
RL_EMA = 0.9

MAX_TEST_STEPS = 75

KNOWN_TOPOS = {"abilene", "geant", "cernet", "rocketfuel_ebone",
               "rocketfuel_sprintlink", "rocketfuel_tiscali"}
UNSEEN_TOPOS = {"germany50", "topologyzoo_vtlwavenet2011"}


# ---------------------------------------------------------------------------
#  Dynamic K GNN Model
# ---------------------------------------------------------------------------

class DynamicKHead(nn.Module):
    """K prediction head that uses graph embedding + traffic statistics.

    Traffic statistics provide per-timestep signal (not just topology):
      - mean utilization
      - max utilization
      - fraction of edges congested (util > 0.9)
      - demand coefficient of variation (std/mean)

    Architecture: [hidden_dim + 4, 32, 16, 1] -> sigmoid -> [K_MIN, K_MAX]
    """

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
        """
        graph_embed: [hidden_dim] — mean-pooled node embeddings
        traffic_stats: [4] — (mean_util, max_util, frac_congested, demand_cv)

        Returns:
            k_continuous: scalar in [k_min, k_max] (differentiable)
        """
        x = torch.cat([graph_embed, traffic_stats], dim=-1)
        raw = self.head(x).squeeze(-1)  # scalar
        k_frac = torch.sigmoid(raw)  # [0, 1]
        k_continuous = self.k_min + k_frac * (self.k_max - self.k_min)
        return k_continuous


# ---------------------------------------------------------------------------
#  Imports
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


def load_all_topologies(M):
    bundle = M["load_bundle"](CONFIG_PATH)
    eval_specs = M["collect_specs"](bundle, "eval_topologies")
    gen_specs = M["collect_specs"](bundle, "generalization_topologies")
    datasets = {}
    for spec in eval_specs + gen_specs:
        try:
            ds, pl = M["load_named_dataset"](bundle, spec, 500)
        except Exception as e:
            print(f"  Skip {spec.key}: {e}"); continue
        ds.path_library = pl
        datasets[ds.key] = ds
        label = "known" if ds.key in KNOWN_TOPOS else "unseen"
        n_tms = len(ds.tm) if hasattr(ds, "tm") else 0
        print(f"  [{label}] {ds.key}: {len(ds.nodes)}N, {len(ds.edges)}E, {n_tms} TMs")
    return datasets


# ---------------------------------------------------------------------------
#  Oracle K_target computation
# ---------------------------------------------------------------------------

def compute_oracle_k_target(M, tm, ecmp_base, path_library, capacities):
    """Sweep K ∈ K_CANDIDATES, return the K that gives lowest post-LP MLU.

    Also returns the oracle selection at the best K and the MLU achieved.
    """
    best_k = 40
    best_mlu = float("inf")
    best_sel = []
    best_method = "topk"

    for k in K_CANDIDATES:
        for name, fn in [
            ("topk", lambda: M["select_topk_by_demand"](tm, k)),
            ("bottleneck", lambda: M["select_bottleneck_critical"](
                tm, ecmp_base, path_library, capacities, k)),
            ("sensitivity", lambda: M["select_sensitivity_critical"](
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
                    best_mlu = mlu
                    best_k = k
                    best_sel = sel
                    best_method = name
            except Exception:
                continue

    return best_k, best_sel, best_mlu, best_method


def compute_traffic_stats(telemetry, tm_vector, num_edges):
    """Compute 4 traffic statistics for the K_pred head.

    Returns tensor [4]: (mean_util, max_util, frac_congested, demand_cv)
    """
    util = np.asarray(telemetry.utilization, dtype=np.float64)[:num_edges]
    tm = np.asarray(tm_vector, dtype=np.float64)

    mean_util = float(np.mean(util))
    max_util = float(np.max(util))
    frac_congested = float(np.mean(util > 0.9))

    tm_active = tm[tm > 1e-12]
    if len(tm_active) > 1:
        demand_cv = float(np.std(tm_active) / (np.mean(tm_active) + 1e-12))
    else:
        demand_cv = 0.0

    # Normalize to [0, 1] range
    demand_cv = min(demand_cv, 5.0) / 5.0  # clip CV to [0, 5] then normalize

    return torch.tensor([mean_util, max_util, frac_congested, demand_cv],
                        dtype=torch.float32, device=DEVICE)


# ---------------------------------------------------------------------------
#  Sample collection
# ---------------------------------------------------------------------------

def collect_samples_with_oracle_k(M, datasets, split, max_per, topo_filter):
    """Collect samples with per-timestep oracle K_target."""
    rng = np.random.default_rng(SEED)
    samples = []
    k_targets = []

    for key, ds in datasets.items():
        if key not in topo_filter:
            continue
        pl = ds.path_library
        ecmp_base = M["ecmp_splits"](pl)
        caps = np.asarray(ds.capacities, dtype=float)
        indices = M["split_indices"](ds, split)
        if len(indices) > max_per:
            indices = sorted(rng.choice(indices, size=max_per, replace=False).tolist())
        count = 0
        prev_tm = None
        for t_idx in indices:
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12:
                prev_tm = tm; continue
            routing = M["apply_routing"](tm, ecmp_base, pl, caps)
            telem = M["compute_reactive_telemetry"](
                tm, ecmp_base, pl, routing, np.asarray(ds.weights, dtype=float))

            # Oracle K sweep
            oracle_k, oracle_sel, oracle_mlu, oracle_method = \
                compute_oracle_k_target(M, tm, ecmp_base, pl, caps)

            if not oracle_sel:
                prev_tm = tm; continue

            samples.append({
                "dataset": ds, "path_library": pl,
                "tm_vector": tm, "prev_tm": prev_tm, "telemetry": telem,
                "oracle_selected": oracle_sel, "oracle_mlu": oracle_mlu,
                "oracle_k": oracle_k, "oracle_method": oracle_method,
                "k_crit": oracle_k,  # use oracle K for selection training
                "capacities": caps,
            })
            k_targets.append(oracle_k)
            count += 1
            prev_tm = tm
        print(f"    {key}: {count} {split} samples")

    if k_targets:
        kt = np.array(k_targets)
        print(f"  Oracle K distribution: mean={kt.mean():.1f}, std={kt.std():.1f}, "
              f"min={kt.min()}, max={kt.max()}, unique={len(np.unique(kt))}")

    return samples


# ---------------------------------------------------------------------------
#  Model wrapper with Dynamic K head
# ---------------------------------------------------------------------------

class GNNWithDynamicK(nn.Module):
    """Wraps a GNNFlowSelector + adds a separate DynamicKHead.

    The base GNN model's own k_head is disabled (learn_k_crit=False).
    The DynamicKHead is a separate module trained jointly.
    """

    def __init__(self, base_model, hidden_dim=64, k_min=1, k_max=50, dropout=0.1):
        super().__init__()
        self.base = base_model
        self.k_head = DynamicKHead(hidden_dim, k_min, k_max, dropout)
        self._hidden_dim = hidden_dim

    def forward(self, graph_data, od_data, traffic_stats):
        """Run base model + K prediction.

        Returns:
            scores: [num_od] flow selection scores
            k_continuous: scalar in [k_min, k_max]
            info: dict
        """
        scores, _, info = self.base(graph_data, od_data)

        # Get graph embedding from base model (recompute)
        node_feat = graph_data["node_features"]
        edge_index = graph_data["edge_index"]
        edge_feat = graph_data["edge_features"]
        h = F.relu(self.base.node_proj(node_feat))
        e = F.relu(self.base.edge_proj(edge_feat))
        for layer in self.base.gnn_layers:
            h = layer(h, edge_index, e)
        graph_embed = h.mean(dim=0)  # [hidden_dim]

        k_continuous = self.k_head(graph_embed, traffic_stats)
        k_int = int(torch.clamp(torch.round(k_continuous), K_MIN, K_MAX).item())

        info["k_pred"] = k_int
        info["k_continuous"] = float(k_continuous.item())

        return scores, k_continuous, k_int, info

    def select_critical_flows(self, graph_data, od_data, traffic_stats,
                              active_mask, k_crit_default=40):
        """Inference: use predicted K for selection."""
        with torch.no_grad():
            scores, k_continuous, k_int, info = self.forward(
                graph_data, od_data, traffic_stats)

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


def save_dynamic_k_model(wrapper, path, extra_meta=None):
    """Save the full wrapper (base + k_head)."""
    payload = {
        "base_state_dict": wrapper.base.state_dict(),
        "k_head_state_dict": wrapper.k_head.state_dict(),
        "hidden_dim": wrapper._hidden_dim,
        "k_min": wrapper.k_head.k_min,
        "k_max": wrapper.k_head.k_max,
        "model_type": "dynamic_k",
    }
    if extra_meta:
        payload.update(extra_meta)
    torch.save(payload, str(path))


def load_dynamic_k_model(M, path, device="cpu"):
    """Load the full wrapper."""
    payload = torch.load(str(path), map_location=torch.device(device), weights_only=False)
    # Recreate base model
    cfg = M["GNNSelectorConfig"]()
    cfg.learn_k_crit = False
    cfg.device = device
    base = M["GNNFlowSelector"](cfg).to(device)
    base.load_state_dict(payload["base_state_dict"])

    wrapper = GNNWithDynamicK(
        base, hidden_dim=payload["hidden_dim"],
        k_min=payload["k_min"], k_max=payload["k_max"])
    wrapper.k_head.load_state_dict(payload["k_head_state_dict"])
    wrapper.eval()
    return wrapper


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train_one_config(M, w_k, train_samples, val_samples):
    """Train GNN + DynamicK head with K-loss weight W."""
    tag = f"w{str(w_k).replace('.', '')}"
    out_dir = OUTPUT_ROOT / f"training_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start from original GNN checkpoint (disable its k_head)
    base_model, base_cfg = M["load_gnn_selector"](GNN_ORIG_CKPT, device=DEVICE)
    base_cfg.learn_k_crit = False
    # Recreate model without k_head
    cfg = M["GNNSelectorConfig"]()
    cfg.learn_k_crit = False
    cfg.dropout = 0.1  # original dropout
    base = M["GNNFlowSelector"](cfg).to(DEVICE)

    # Copy weights from original checkpoint (skip k_head params)
    orig_sd = base_model.state_dict()
    new_sd = base.state_dict()
    for k, v in orig_sd.items():
        if k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k] = v
    base.load_state_dict(new_sd)

    # Create wrapper with DynamicK head
    wrapper = GNNWithDynamicK(base, hidden_dim=64, k_min=K_MIN, k_max=K_MAX, dropout=0.1)
    wrapper.to(DEVICE)

    # Optimizer: train both base and k_head jointly
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=SUP_LR, weight_decay=SUP_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUP_EPOCHS)
    rng = np.random.default_rng(SEED)
    logs = []
    best_vl = float("inf"); best_ep = 0; stale = 0
    t0 = time.perf_counter()

    print(f"\n[Supervised W={w_k}]", flush=True)
    for epoch in range(1, SUP_EPOCHS + 1):
        ep0 = time.perf_counter()
        wrapper.train()
        ep_losses = []
        ep_k_losses = []
        ep_k_preds = []

        for si in rng.permutation(len(train_samples)):
            s = train_samples[si]
            gd = M["build_graph_tensors"](s["dataset"], telemetry=s["telemetry"], device=DEVICE)
            od = M["build_od_features"](s["dataset"], s["tm_vector"],
                s["path_library"], telemetry=s["telemetry"], device=DEVICE)
            ts = compute_traffic_stats(s["telemetry"], s["tm_vector"],
                                       gd["num_edges"])

            scores, k_continuous, k_int, info = wrapper(gd, od, ts)

            # Flow ranking loss
            om = torch.zeros(scores.size(0), device=DEVICE)
            for oid in s["oracle_selected"]:
                if oid < scores.size(0): om[oid] = 1.0
            flow_loss = M["_ranking_loss"](scores, om, margin=SUP_MARGIN)

            # K loss: MSE between predicted K and oracle K
            k_target = torch.tensor(float(s["oracle_k"]), device=DEVICE)
            k_loss = F.mse_loss(k_continuous, k_target)

            # Total loss
            loss = flow_loss + w_k * k_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            optimizer.step()

            ep_losses.append(float(loss.item()))
            ep_k_losses.append(float(k_loss.item()))
            ep_k_preds.append(float(k_continuous.item()))

        scheduler.step()
        tl = float(np.mean(ep_losses)) if ep_losses else 0.0
        kl = float(np.mean(ep_k_losses)) if ep_k_losses else 0.0
        kp_mean = float(np.mean(ep_k_preds)) if ep_k_preds else 0.0
        kp_std = float(np.std(ep_k_preds)) if ep_k_preds else 0.0

        # Validation
        wrapper.eval()
        vls = []
        vks = []
        with torch.no_grad():
            for s in val_samples:
                gd = M["build_graph_tensors"](s["dataset"], telemetry=s["telemetry"], device=DEVICE)
                od = M["build_od_features"](s["dataset"], s["tm_vector"],
                    s["path_library"], telemetry=s["telemetry"], device=DEVICE)
                ts = compute_traffic_stats(s["telemetry"], s["tm_vector"],
                                           gd["num_edges"])
                scores, k_continuous, k_int, info = wrapper(gd, od, ts)
                om = torch.zeros(scores.size(0), device=DEVICE)
                for oid in s["oracle_selected"]:
                    if oid < scores.size(0): om[oid] = 1.0
                vl = M["_ranking_loss"](scores, om, margin=SUP_MARGIN)
                k_target = torch.tensor(float(s["oracle_k"]), device=DEVICE)
                vk = F.mse_loss(k_continuous, k_target)
                vls.append(float((vl + w_k * vk).item()))
                vks.append(float(k_continuous.item()))

        val_loss = float(np.mean(vls)) if vls else 0.0
        val_k_mean = float(np.mean(vks)) if vks else 0.0
        val_k_std = float(np.std(vks)) if vks else 0.0
        et = time.perf_counter() - ep0

        logs.append({"stage": "supervised", "epoch": epoch, "train_loss": tl,
            "val_loss": val_loss, "k_loss": kl,
            "k_pred_mean": kp_mean, "k_pred_std": kp_std,
            "val_k_mean": val_k_mean, "val_k_std": val_k_std,
            "alpha": float(wrapper.base.alpha.item()), "epoch_time_sec": et})

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}: tl={tl:.4f} vl={val_loss:.4f} "
                  f"kl={kl:.2f} K_pred={kp_mean:.1f}±{kp_std:.1f} "
                  f"val_K={val_k_mean:.1f}±{val_k_std:.1f} [{et:.1f}s]", flush=True)

        if val_loss + 1e-6 < best_vl:
            best_vl = val_loss; best_ep = epoch; stale = 0
            save_dynamic_k_model(wrapper, out_dir / "supervised.pt",
                extra_meta={"best_epoch": best_ep, "stage": "supervised", "w_k": w_k})
        else:
            stale += 1
        if stale >= SUP_PATIENCE:
            print(f"  Early stop at {epoch}"); break

    sup_time = time.perf_counter() - t0
    print(f"  Supervised done: {sup_time:.0f}s, best_ep={best_ep}", flush=True)
    pd.DataFrame(logs).to_csv(out_dir / "supervised_log.csv", index=False)

    # Reload best supervised
    wrapper = load_dynamic_k_model(M, out_dir / "supervised.pt", device=DEVICE)

    # ---- REINFORCE with dynamic K ----
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=RL_LR)
    rng = np.random.default_rng(SEED)
    baseline_reward = None; best_vm = float("inf"); best_re = 0; stale = 0
    rl_logs = []; t0 = time.perf_counter()
    print(f"\n[REINFORCE W={w_k}]", flush=True)

    for epoch in range(1, RL_EPOCHS + 1):
        ep0 = time.perf_counter()
        wrapper.train()
        ep_mlu = []
        ep_k_preds = []

        for si in rng.permutation(len(train_samples)):
            s = train_samples[si]
            gd = M["build_graph_tensors"](s["dataset"], telemetry=s["telemetry"], device=DEVICE)
            od = M["build_od_features"](s["dataset"], s["tm_vector"],
                s["path_library"], telemetry=s["telemetry"], device=DEVICE)
            ts = compute_traffic_stats(s["telemetry"], s["tm_vector"],
                                       gd["num_edges"])

            scores, k_continuous, k_int, info = wrapper(gd, od, ts)
            active = s["tm_vector"] > 0
            ai = np.where(active)[0]
            if ai.size == 0: continue

            # Use predicted K for selection
            take = min(k_int, ai.size)
            as_ = scores[torch.tensor(ai, dtype=torch.long, device=DEVICE)]
            lp_ = F.log_softmax(as_, dim=0)
            _, tl_ = torch.topk(as_, take)
            sel = [int(ai[i]) for i in tl_.cpu().numpy()]
            slp = lp_[tl_].sum()

            ecmp = M["ecmp_splits"](s["path_library"])
            try:
                lp = M["solve_selected_path_lp"](
                    tm_vector=s["tm_vector"], selected_ods=sel,
                    base_splits=ecmp, path_library=s["path_library"],
                    capacities=s["capacities"], time_limit_sec=10)
                mlu = float(lp.routing.mlu)
            except Exception: continue
            if not np.isfinite(mlu): continue

            ep_mlu.append(mlu)
            ep_k_preds.append(k_int)
            reward = -mlu
            if baseline_reward is None: baseline_reward = reward
            else: baseline_reward = RL_EMA * baseline_reward + (1 - RL_EMA) * reward
            adv = reward - baseline_reward

            # REINFORCE loss for flow selection
            flow_rl_loss = -adv * slp

            # Ranking alignment
            om = torch.zeros(scores.size(0), device=DEVICE)
            for oid in s["oracle_selected"]:
                if oid < scores.size(0): om[oid] = 1.0
            rank_loss = M["_ranking_loss"](scores, om, margin=0.05)

            # K loss (still enforce oracle K alignment during RL)
            k_target = torch.tensor(float(s["oracle_k"]), device=DEVICE)
            k_loss = F.mse_loss(k_continuous, k_target)

            total = flow_rl_loss + 0.3 * rank_loss + w_k * k_loss
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            optimizer.step()

        mm = float(np.mean(ep_mlu)) if ep_mlu else float("inf")
        mk = float(np.mean(ep_k_preds)) if ep_k_preds else 0
        sk = float(np.std(ep_k_preds)) if ep_k_preds else 0

        # Validation
        wrapper.eval()
        vms = []
        vks = []
        with torch.no_grad():
            for s in val_samples:
                gd = M["build_graph_tensors"](s["dataset"], telemetry=s["telemetry"], device=DEVICE)
                od = M["build_od_features"](s["dataset"], s["tm_vector"],
                    s["path_library"], telemetry=s["telemetry"], device=DEVICE)
                ts = compute_traffic_stats(s["telemetry"], s["tm_vector"],
                                           gd["num_edges"])
                scores, k_continuous, k_int, info = wrapper(gd, od, ts)
                active = s["tm_vector"] > 0
                ai = np.where(active)[0]
                if ai.size == 0: continue
                take = min(k_int, ai.size)
                as_ = scores[torch.tensor(ai, dtype=torch.long, device=DEVICE)]
                _, tl_ = torch.topk(as_, take)
                sel = [int(ai[i]) for i in tl_.cpu().numpy()]
                ecmp = M["ecmp_splits"](s["path_library"])
                try:
                    lp = M["solve_selected_path_lp"](
                        tm_vector=s["tm_vector"], selected_ods=sel,
                        base_splits=ecmp, path_library=s["path_library"],
                        capacities=s["capacities"], time_limit_sec=10)
                    vms.append(float(lp.routing.mlu))
                except Exception: pass
                vks.append(k_int)

        vm = float(np.mean(vms)) if vms else float("inf")
        vmk = float(np.mean(vks)) if vks else 0
        vsk = float(np.std(vks)) if vks else 0
        et = time.perf_counter() - ep0

        rl_logs.append({"stage": "reinforce", "epoch": epoch,
            "train_mlu": mm, "val_mlu": vm,
            "train_k_mean": mk, "train_k_std": sk,
            "val_k_mean": vmk, "val_k_std": vsk,
            "alpha": float(wrapper.base.alpha.item()), "epoch_time_sec": et})
        print(f"  RL Ep {epoch}: mlu={mm:.2f} val={vm:.2f} "
              f"K={mk:.1f}±{sk:.1f} valK={vmk:.1f}±{vsk:.1f} [{et:.0f}s]", flush=True)

        if vm + 1e-6 < best_vm:
            best_vm = vm; best_re = epoch; stale = 0
            save_dynamic_k_model(wrapper, out_dir / "final.pt",
                extra_meta={"best_epoch": best_re, "stage": "reinforce", "w_k": w_k})
        else:
            stale += 1
        if stale >= RL_PATIENCE:
            print(f"  RL early stop at {epoch}"); break

    rl_time = time.perf_counter() - t0
    pd.DataFrame(rl_logs).to_csv(out_dir / "reinforce_log.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "w_k": w_k, "sup_time": sup_time, "sup_best_epoch": best_ep,
        "rl_time": rl_time, "rl_best_epoch": best_re, "rl_best_val_mlu": best_vm,
        "total_time": sup_time + rl_time,
    }, indent=2) + "\n", encoding="utf-8")

    # Reload best
    ckpt = out_dir / "final.pt"
    if not ckpt.exists(): ckpt = out_dir / "supervised.pt"
    wrapper = load_dynamic_k_model(M, ckpt, device=DEVICE)
    return wrapper, sup_time + rl_time


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def _dist(cur, prev, k):
    if prev is None: return 0.0
    return len(set(cur) ^ set(prev)) / max(k, 1)


def evaluate_all(M, datasets, models_dict):
    """Evaluate original GNN (fixed K=40) vs all dynamic K variants."""
    eval_dir = OUTPUT_ROOT / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    gnn_orig, _ = M["load_gnn_selector"](GNN_ORIG_CKPT, device=DEVICE)
    gnn_orig.eval()
    for m in models_dict.values():
        m.eval()

    rows = []
    eval_order = sorted(KNOWN_TOPOS & set(datasets.keys())) + \
                 sorted(UNSEEN_TOPOS & set(datasets.keys()))

    for topo_key in eval_order:
        if topo_key not in datasets: continue
        ds = datasets[topo_key]
        pl = ds.path_library; ecmp_base = M["ecmp_splits"](pl)
        caps = np.asarray(ds.capacities, dtype=float)
        is_unseen = topo_key in UNSEEN_TOPOS

        test_idx = M["split_indices"](ds, "test")[:MAX_TEST_STEPS]
        label = "UNSEEN" if is_unseen else "known"
        print(f"\n[Eval] {topo_key} ({label}): {len(test_idx)} steps", flush=True)

        prev_sels = {"orig": None}
        for tag in models_dict: prev_sels[tag] = None
        prev_tm = None

        for step_i, t_idx in enumerate(test_idx):
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12: prev_tm = tm; continue
            routing = M["apply_routing"](tm, ecmp_base, pl, caps)
            telem = M["compute_reactive_telemetry"](
                tm, ecmp_base, pl, routing, np.asarray(ds.weights, dtype=float))
            amask = (np.asarray(tm, dtype=np.float64) > 1e-12).astype(np.float32)

            # Oracle (sweep K for reference)
            o_k, o_sel, o_mlu, _ = compute_oracle_k_target(M, tm, ecmp_base, pl, caps)

            row = {"topology": topo_key, "is_unseen": is_unseen,
                   "step": step_i, "tm_idx": t_idx,
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
            row["mlu_orig"] = mlu
            row["k_orig"] = 40
            row["time_ms_orig"] = t_orig
            row["dist_orig"] = _dist(sel, prev_sels["orig"], 40)
            row["pr_orig"] = (mlu - o_mlu) / (o_mlu + 1e-12)
            prev_sels["orig"] = sel

            # Each dynamic K variant
            for tag, model in models_dict.items():
                t0 = time.perf_counter()
                gd = M["build_graph_tensors"](ds, telemetry=telem, device=DEVICE)
                od = M["build_od_features"](ds, tm, pl, telemetry=telem, device=DEVICE)
                ts = compute_traffic_stats(telem, tm, gd["num_edges"])
                with torch.no_grad():
                    sel, info = model.select_critical_flows(
                        gd, od, ts, active_mask=amask, k_crit_default=40)
                t_m = (time.perf_counter() - t0) * 1000
                k_used = info.get("k_used", 40)
                k_pred = info.get("k_pred", 40)
                try:
                    lp = M["solve_selected_path_lp"](tm_vector=tm, selected_ods=sel,
                        base_splits=ecmp_base, path_library=pl, capacities=caps, time_limit_sec=LT)
                    mlu_m = float(lp.routing.mlu)
                except Exception: mlu_m = float("inf")
                row[f"mlu_{tag}"] = mlu_m
                row[f"k_pred_{tag}"] = k_pred
                row[f"k_used_{tag}"] = k_used
                row[f"time_ms_{tag}"] = t_m
                row[f"dist_{tag}"] = _dist(sel, prev_sels[tag], k_used)
                row[f"pr_{tag}"] = (mlu_m - o_mlu) / (o_mlu + 1e-12)
                prev_sels[tag] = sel

            rows.append(row)
            prev_tm = tm
            if (step_i + 1) % 25 == 0:
                parts = [f"orig={row['mlu_orig']:.4f}(K=40)"]
                for tag in models_dict:
                    parts.append(f"{tag}={row[f'mlu_{tag}']:.4f}(K={row[f'k_pred_{tag}']})")
                print(f"    step {step_i+1}: {' '.join(parts)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(eval_dir / "dynamic_k_per_timestep.csv", index=False)

    # Summary
    tags = list(models_dict.keys())
    summary_rows = []
    for topo_key in eval_order:
        sub = df[df["topology"] == topo_key]
        if sub.empty: continue
        summary_rows.append(_summary(sub, topo_key, topo_key in UNSEEN_TOPOS, tags))

    for label, mask in [("KNOWN_AGG", ~df["is_unseen"]), ("UNSEEN_AGG", df["is_unseen"]),
                         ("TOTAL_AGG", pd.Series(True, index=df.index))]:
        sub = df[mask]
        if not sub.empty:
            summary_rows.append(_summary(sub, label, "UNSEEN" in label, tags))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(eval_dir / "dynamic_k_summary.csv", index=False)
    print(f"\n[Eval] Saved {len(df)} rows + summary")
    return df, summary_df


def _summary(sub, label, is_unseen, tags):
    row = {"topology": label, "is_unseen": is_unseen, "num_steps": len(sub),
           "mean_mlu_orig": sub["mlu_orig"].mean(),
           "mean_dist_orig": sub["dist_orig"].mean(),
           "oracle_k_mean": sub["oracle_k"].mean() if "oracle_k" in sub.columns else 40,
           "oracle_k_std": sub["oracle_k"].std() if "oracle_k" in sub.columns else 0}
    for tag in tags:
        row[f"mean_mlu_{tag}"] = sub[f"mlu_{tag}"].mean()
        row[f"mean_dist_{tag}"] = sub[f"dist_{tag}"].mean()
        row[f"mean_k_pred_{tag}"] = sub[f"k_pred_{tag}"].mean()
        row[f"std_k_pred_{tag}"] = sub[f"k_pred_{tag}"].std()
        row[f"mlu_improve_{tag}_pct"] = (
            (sub["mlu_orig"].mean() - sub[f"mlu_{tag}"].mean())
            / (sub["mlu_orig"].mean() + 1e-12) * 100)
        row[f"dist_improve_{tag}_pct"] = (
            (sub["dist_orig"].mean() - sub[f"dist_{tag}"].mean())
            / (sub["dist_orig"].mean() + 1e-12) * 100)
        row[f"wins_{tag}"] = int((sub[f"mlu_{tag}"] < sub["mlu_orig"]).sum())
    return row


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def generate_plots(df, summary_df, tags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = OUTPUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    colors = {"orig": "tab:blue", "w01": "tab:orange", "w05": "tab:green",
              "w10": "tab:red", "w50": "tab:purple"}
    labels_map = {"orig": "Original (K=40)", "w01": "DynK W=0.1",
                  "w05": "DynK W=0.5", "w10": "DynK W=1.0", "w50": "DynK W=5.0"}

    topos = sorted(df["topology"].unique())
    known = [t for t in topos if t in KNOWN_TOPOS]
    unseen = [t for t in topos if t in UNSEEN_TOPOS]

    # Plot 1: K distribution per topology (histograms)
    all_topos = known + unseen
    if all_topos:
        n = len(all_topos)
        fig, axes = plt.subplots(len(tags), n, figsize=(4 * n, 3.5 * len(tags)), squeeze=False)
        for j, tag in enumerate(tags):
            for i, topo in enumerate(all_topos):
                ax = axes[j, i]
                sub = df[df["topology"] == topo]
                k_col = f"k_pred_{tag}"
                if k_col in sub.columns:
                    ax.hist(sub[k_col].values, bins=range(0, 55, 5), alpha=0.7,
                           color=colors.get(tag, "gray"), edgecolor="black")
                    oracle_col = "oracle_k"
                    if oracle_col in sub.columns:
                        ax.axvline(sub[oracle_col].mean(), color="black", ls="--",
                                  lw=2, label=f"Oracle mean={sub[oracle_col].mean():.0f}")
                    ax.axvline(sub[k_col].mean(), color="red", ls=":",
                              lw=2, label=f"Pred mean={sub[k_col].mean():.0f}")
                    ax.legend(fontsize=6)
                ax.set_title(f"{topo}\n{labels_map.get(tag, tag)}", fontsize=8, fontweight="bold")
                ax.set_xlabel("K_pred"); ax.set_ylabel("Count")
        fig.suptitle("K Prediction Distribution per Topology", fontweight="bold", fontsize=14)
        fig.tight_layout()
        fig.savefig(plot_dir / "k_distribution.png", dpi=150)
        plt.close(fig)

    # Plot 2: MLU CDF per topology
    for group_name, group_topos in [("known", known), ("unseen", unseen)]:
        if not group_topos: continue
        n = len(group_topos)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
        for i, topo in enumerate(group_topos):
            ax = axes[0, i]; sub = df[df["topology"] == topo]
            for key in ["orig"] + tags:
                col = f"mlu_{key}" if key != "orig" else "mlu_orig"
                if col not in sub.columns: continue
                vals = np.sort(sub[col].values)
                cdf = np.arange(1, len(vals)+1) / len(vals)
                ax.plot(vals, cdf, label=labels_map.get(key, key),
                       color=colors.get(key, "gray"), lw=2)
            ax.set_title(topo, fontweight="bold"); ax.set_xlabel("MLU"); ax.set_ylabel("CDF")
            ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        fig.suptitle(f"MLU CDF — {group_name.capitalize()} Topologies (Dynamic K)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(plot_dir / f"mlu_cdf_{group_name}.png", dpi=150)
        plt.close(fig)

    # Plot 3: K over time for one topology (most interesting = Sprintlink)
    for topo in ["rocketfuel_sprintlink"] + all_topos[:2]:
        sub = df[df["topology"] == topo]
        if sub.empty: continue
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        steps = sub["step"].values
        ax1.plot(steps, sub["mlu_orig"].values, label="Original (K=40)",
                color="tab:blue", lw=1.5, alpha=0.8)
        for tag in tags:
            col = f"mlu_{tag}"
            if col in sub.columns:
                ax1.plot(steps, sub[col].values, label=labels_map.get(tag, tag),
                        color=colors.get(tag, "gray"), lw=1.5, alpha=0.8)
        ax1.set_ylabel("MLU"); ax1.set_title(f"{topo} — MLU over Time", fontweight="bold")
        ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

        ax2.axhline(40, color="tab:blue", ls="--", lw=1, label="Fixed K=40")
        if "oracle_k" in sub.columns:
            ax2.plot(steps, sub["oracle_k"].values, color="black", ls=":",
                    lw=2, label="Oracle K", alpha=0.7)
        for tag in tags:
            col = f"k_pred_{tag}"
            if col in sub.columns:
                ax2.plot(steps, sub[col].values, label=labels_map.get(tag, tag),
                        color=colors.get(tag, "gray"), lw=1.5, alpha=0.8)
        ax2.set_ylabel("K"); ax2.set_xlabel("Timestep")
        ax2.set_title("K Prediction over Time", fontweight="bold")
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"k_over_time_{topo}.png", dpi=150)
        plt.close(fig)
        break  # only first found

    # Plot 4: Aggregate bar chart (MLU + disturbance)
    agg_rows = summary_df[summary_df["topology"].isin(["KNOWN_AGG", "UNSEEN_AGG", "TOTAL_AGG"])]
    if not agg_rows.empty:
        agg_labels = agg_rows["topology"].tolist()
        x = np.arange(len(agg_labels))
        n_bars = 1 + len(tags)
        w = 0.8 / n_bars

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for j, key in enumerate(["orig"] + tags):
            col = "mean_mlu_orig" if key == "orig" else f"mean_mlu_{key}"
            if col not in agg_rows.columns: continue
            vals = agg_rows[col].tolist()
            bars = ax1.bar(x + (j - n_bars/2 + 0.5) * w, vals, w,
                          label=labels_map.get(key, key), color=colors.get(key, "gray"), alpha=0.8)
            for b in bars:
                ax1.text(b.get_x()+b.get_width()/2, b.get_height(),
                        f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=6)
        ax1.set_xticks(x); ax1.set_xticklabels(agg_labels)
        ax1.set_ylabel("Mean MLU"); ax1.set_title("Aggregate MLU", fontweight="bold")
        ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3, axis="y")

        for j, key in enumerate(["orig"] + tags):
            col = "mean_dist_orig" if key == "orig" else f"mean_dist_{key}"
            if col not in agg_rows.columns: continue
            vals = agg_rows[col].tolist()
            ax2.bar(x + (j - n_bars/2 + 0.5) * w, vals, w,
                   label=labels_map.get(key, key), color=colors.get(key, "gray"), alpha=0.8)
        ax2.set_xticks(x); ax2.set_xticklabels(agg_labels)
        ax2.set_ylabel("Mean Disturbance"); ax2.set_title("Aggregate Disturbance", fontweight="bold")
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(plot_dir / "aggregate_comparison.png", dpi=150)
        plt.close(fig)

    print(f"[Plots] Saved to {plot_dir}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 2: DYNAMIC K PREDICTION (ORIGINAL FEATURES)")
    print("=" * 70)
    total_start = time.perf_counter()

    M = setup()

    print("\n[1] Loading topologies...", flush=True)
    datasets = load_all_topologies(M)

    print("\n[2] Collecting samples with oracle K sweep...", flush=True)
    print("  (This is slower than Stage 1 — sweeping 8 K values × 3 selectors per timestep)")
    train_samples = collect_samples_with_oracle_k(M, datasets, "train",
                                                   MAX_TRAIN_PER_TOPO, KNOWN_TOPOS)
    val_samples = collect_samples_with_oracle_k(M, datasets, "val",
                                                 MAX_VAL_PER_TOPO, KNOWN_TOPOS)
    print(f"  Total: {len(train_samples)} train, {len(val_samples)} val")

    # Save oracle K distribution
    oracle_dir = OUTPUT_ROOT / "oracle_k"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    k_data = []
    for s in train_samples + val_samples:
        k_data.append({"topology": s["dataset"].key, "oracle_k": s["oracle_k"],
                       "oracle_method": s["oracle_method"], "oracle_mlu": s["oracle_mlu"]})
    pd.DataFrame(k_data).to_csv(oracle_dir / "oracle_k_distribution.csv", index=False)
    print(f"  Saved oracle K distribution to {oracle_dir}")

    # Train all W configs
    models = {}
    for w_k in W_VALUES:
        tag = f"w{str(w_k).replace('.', '')}"
        print(f"\n[3] Training W={w_k} ({tag})...", flush=True)
        model, t_time = train_one_config(M, w_k, train_samples, val_samples)
        models[tag] = model
        print(f"  Total training time: {t_time:.0f}s")

    print("\n[4] Evaluating all configs...", flush=True)
    df, summary_df = evaluate_all(M, datasets, models)

    print("\n[5] Generating plots...", flush=True)
    generate_plots(df, summary_df, list(models.keys()))

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print("STAGE 2 DYNAMIC K RESULTS")
    print("=" * 70)
    display_cols = ["topology", "num_steps", "mean_mlu_orig", "oracle_k_mean"]
    for tag in models:
        display_cols += [f"mean_mlu_{tag}", f"mlu_improve_{tag}_pct",
                        f"mean_k_pred_{tag}", f"std_k_pred_{tag}", f"wins_{tag}"]
    available = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available].to_string(index=False))
    print(f"\nTotal wall time: {total_time:.0f}s")
    print("=" * 70)
