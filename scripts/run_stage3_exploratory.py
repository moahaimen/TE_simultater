#!/usr/bin/env python3
"""Stage 3 Exploratory Screening: Combined prototype assessment.

Compares 3 variants:
  1. Original GNN (baseline, reuse existing)
  2. Stage 1 winner (enriched features, dropout=0.2, fixed K=40, reuse existing)
  3. Stage 3 prototype (Stage 1 winner + Stage 2 pilot dynamic-K with W=1.0)

Constraints:
  - Reuse existing baselines where possible
  - Train ONLY the Stage 3 prototype (W=1.0 only)
  - Scope: Germany50 + GEANT (Abilene if cheap)
  - Outputs isolated in results/gnn_plus/stage3_exploratory/

IMPORTANT: This is EXPLORATORY only. Stage 2 dynamic-K was NOT validated.
This experiment answers: does the combined idea show promise?
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
LT = 15
DEVICE = "cpu"

GNN_ORIG_CKPT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
STAGE1_WINNER_CKPT = Path("results/gnn_plus/stage1_final/gnn_stage1_enriched_dropout.pt")
OUTPUT_ROOT = Path("results/gnn_plus/stage3_exploratory")

K_CANDIDATES = [15, 20, 25, 30, 35, 40, 45, 50]
K_MIN = 1
K_MAX = 50

# Stage 3 prototype: W=1.0 only (from Stage 2 pilot, NOT the failed lock)
W_PROTOTYPE = 1.0

SUP_LR = 5e-4
SUP_WD = 1e-5
SUP_EPOCHS = 20
SUP_PATIENCE = 6
SUP_MARGIN = 0.1
MAX_TRAIN_PER_TOPO = 25
MAX_VAL_PER_TOPO = 15

RL_LR = 1e-4
RL_EPOCHS = 5
RL_PATIENCE = 4
RL_EMA = 0.9

MAX_TEST_STEPS = 50

PILOT_TOPOS = {"germany50_real", "geant_core", "abilene_backbone"}


# ---------------------------------------------------------------------------
#  Dynamic K Head (from Stage 2 pilot - RAW K, no normalization)
# ---------------------------------------------------------------------------

class DynamicKHead(nn.Module):
    """K prediction head: graph_embed + traffic_stats -> continuous K."""

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
        """Returns continuous K in [k_min, k_max]."""
        x = torch.cat([graph_embed, traffic_stats], dim=-1)
        raw = self.head(x).squeeze(-1)
        k_continuous = torch.clamp(raw, self.k_min, self.k_max)
        return k_continuous


class GNNWithDynamicK(nn.Module):
    """Wraps GNNFlowSelector + DynamicKHead (Stage 2 pilot style)."""

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
        
        # Predict K (raw, not normalized)
        k_continuous = self.k_head(graph_embed, traffic_stats)
        k_int = int(torch.clamp(torch.round(k_continuous), K_MIN, K_MAX).item())
        
        info["k_pred"] = k_int
        info["k_continuous"] = float(k_continuous.item())
        return scores, k_continuous, k_int, info

    def select_critical_flows(self, graph_data, od_data, traffic_stats, active_mask):
        with torch.no_grad():
            scores, k_cont, k_int, info = self.forward(graph_data, od_data, traffic_stats)
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
#  Sample collection (for Stage 3 prototype training)
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
               "model_type": "stage3_prototype"}
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
#  Train Stage 3 Prototype only
# ---------------------------------------------------------------------------

def train_stage3_prototype(M, w_k, train_samples, val_samples):
    """Train Stage 3 prototype (Stage 1 winner + Stage 2 pilot dynamic-K)."""
    out_dir = OUTPUT_ROOT / "stage3_prototype"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Training Stage 3 Prototype: W={w_k}]")
    
    # Load Stage 1 winner as base (enriched features, dropout=0.2)
    if STAGE1_WINNER_CKPT.exists():
        print(f"  Loading Stage 1 winner from {STAGE1_WINNER_CKPT}")
        base_model, _ = M["load_gnn_selector"](STAGE1_WINNER_CKPT, device=DEVICE)
    else:
        print(f"  Stage 1 winner not found, using original GNN")
        base_model, _ = M["load_gnn_selector"](GNN_ORIG_CKPT, device=DEVICE)
    
    cfg = M["GNNSelectorConfig"](); cfg.learn_k_crit = False; cfg.dropout = 0.2
    base = M["GNNFlowSelector"](cfg).to(DEVICE)
    orig_sd = base_model.state_dict(); new_sd = base.state_dict()
    for k, v in orig_sd.items():
        if k in new_sd and new_sd[k].shape == v.shape: new_sd[k] = v
    base.load_state_dict(new_sd)

    # Add dynamic-K head (Stage 2 pilot style - no normalization)
    wrapper = GNNWithDynamicK(base, hidden_dim=64, k_min=K_MIN, k_max=K_MAX, dropout=0.2)
    wrapper.to(DEVICE)

    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=SUP_LR, weight_decay=SUP_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUP_EPOCHS)
    rng = np.random.default_rng(SEED)
    logs = []; best_vl = float("inf"); best_ep = 0; stale = 0
    t0 = time.perf_counter()

    print(f"\n[Supervised W={w_k}]", flush=True)
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
            
            # RAW K loss (Stage 2 pilot style, NOT normalized)
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

        logs.append({"epoch": epoch, "train_loss": tl, "val_loss": val_loss,
            "k_loss": kl, "k_pred_mean": kp_mean, "k_pred_std": kp_std,
            "val_k_mean": val_k_mean, "val_k_std": val_k_std})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}: loss={tl:.4f} vl={val_loss:.4f} "
                  f"kl={kl:.1f} K={kp_mean:.1f}±{kp_std:.1f} [{et:.1f}s]", flush=True)
        if val_loss + 1e-6 < best_vl:
            best_vl = val_loss; best_ep = epoch; stale = 0
            save_model(wrapper, out_dir / "supervised.pt",
                       extra_meta={"best_epoch": best_ep, "w_k": w_k})
        else: stale += 1
        if stale >= SUP_PATIENCE: print(f"  Early stop at {epoch}"); break

    sup_time = time.perf_counter() - t0
    print(f"  Supervised done: {sup_time:.0f}s, best_ep={best_ep}", flush=True)
    pd.DataFrame(logs).to_csv(out_dir / "training_log.csv", index=False)

    wrapper = load_model(M, out_dir / "supervised.pt", device=DEVICE)
    return wrapper, sup_time


# ---------------------------------------------------------------------------
#  Evaluate all 3 variants
# ---------------------------------------------------------------------------

def _dist(cur, prev, k):
    if prev is None: return 0.0
    return len(set(cur) ^ set(prev)) / max(k, 1)


def evaluate_all(M, datasets, stage3_model):
    """Evaluate: Original GNN, Stage 1 winner, Stage 3 prototype."""
    eval_dir = OUTPUT_ROOT / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load baselines (reuse existing)
    gnn_orig, _ = M["load_gnn_selector"](GNN_ORIG_CKPT, device=DEVICE)
    gnn_orig.eval()
    
    if STAGE1_WINNER_CKPT.exists():
        stage1_model, _ = M["load_gnn_selector"](STAGE1_WINNER_CKPT, device=DEVICE)
        stage1_model.eval()
        has_stage1 = True
    else:
        print("WARNING: Stage 1 winner not found, skipping")
        has_stage1 = False
    
    stage3_model.eval()

    rows = []
    variants = [("orig", gnn_orig, False)]
    if has_stage1:
        variants.append(("stage1", stage1_model, False))
    variants.append(("stage3", stage3_model, True))  # True = dynamic K

    for topo_key in sorted(datasets.keys()):
        ds = datasets[topo_key]
        pl = ds.path_library; ecmp_base = M["ecmp_splits"](pl)
        caps = np.asarray(ds.capacities, dtype=float)
        test_idx = M["split_indices"](ds, "test")[:MAX_TEST_STEPS]
        print(f"\n[Eval] {topo_key}: {len(test_idx)} steps", flush=True)

        prev_sels = {tag: None for tag, _, _ in variants}

        for step_i, t_idx in enumerate(test_idx):
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12: continue
            
            routing = M["apply_routing"](tm, ecmp_base, pl, caps)
            telem = M["compute_reactive_telemetry"](
                tm, ecmp_base, pl, routing, np.asarray(ds.weights, dtype=float))
            amask = (np.asarray(tm, dtype=np.float64) > 1e-12).astype(np.float32)

            o_k, o_sel, o_mlu, _ = compute_oracle_k_target(M, tm, ecmp_base, pl, caps)
            row = {"topology": topo_key, "step": step_i, "tm_idx": t_idx,
                   "oracle_mlu": o_mlu, "oracle_k": o_k}

            for tag, model, is_dynamic in variants:
                t0 = time.perf_counter()
                gd = M["build_graph_tensors"](ds, telemetry=telem, device=DEVICE)
                od = M["build_od_features"](ds, tm, pl, telemetry=telem, device=DEVICE)
                
                if is_dynamic:
                    ts = compute_traffic_stats(telem, tm, gd["num_edges"])
                    sel, info = model.select_critical_flows(gd, od, ts, active_mask=amask)
                    k_pred = info.get("k_pred", 40); k_used = info.get("k_used", 40)
                else:
                    with torch.no_grad():
                        sel, info = model.select_critical_flows(
                            gd, od, active_mask=amask, k_crit_default=40, force_default_k=True)
                    k_pred = 40; k_used = 40
                
                t_ms = (time.perf_counter() - t0) * 1000
                
                try:
                    lp = M["solve_selected_path_lp"](tm_vector=tm, selected_ods=sel,
                        base_splits=ecmp_base, path_library=pl, capacities=caps, time_limit_sec=LT)
                    mlu = float(lp.routing.mlu)
                except Exception: 
                    mlu = float("inf")
                
                row[f"mlu_{tag}"] = mlu
                row[f"k_pred_{tag}"] = k_pred
                row[f"k_used_{tag}"] = k_used
                row[f"time_ms_{tag}"] = t_ms
                row[f"dist_{tag}"] = _dist(sel, prev_sels[tag], k_used)
                row[f"pr_{tag}"] = (mlu - o_mlu) / (o_mlu + 1e-12)
                prev_sels[tag] = sel

            rows.append(row)
            
            if (step_i + 1) % 25 == 0:
                parts = []
                for tag, _, _ in variants:
                    parts.append(f"{tag}={row[f'mlu_{tag}']:.4f}(K={row[f'k_pred_{tag}']})")
                print(f"    step {step_i+1}: {' '.join(parts)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(eval_dir / "per_timestep.csv", index=False)

    # Summary
    summary_rows = []
    for topo_key in sorted(datasets.keys()) + ["TOTAL"]:
        if topo_key == "TOTAL":
            sub = df
        else:
            sub = df[df["topology"] == topo_key]
        if sub.empty: continue
        
        row = {"topology": topo_key, "num_steps": len(sub),
               "oracle_k_mean": sub["oracle_k"].mean(),
               "oracle_k_std": sub["oracle_k"].std()}
        
        for tag, _, is_dynamic in variants:
            row[f"mean_mlu_{tag}"] = sub[f"mlu_{tag}"].mean()
            row[f"mean_dist_{tag}"] = sub[f"dist_{tag}"].mean()
            row[f"mean_pr_{tag}"] = sub[f"pr_{tag}"].mean()
            row[f"mean_time_ms_{tag}"] = sub[f"time_ms_{tag}"].mean()
            row[f"mean_k_pred_{tag}"] = sub[f"k_pred_{tag}"].mean()
            row[f"std_k_pred_{tag}"] = sub[f"k_pred_{tag}"].std()
            row[f"min_k_pred_{tag}"] = sub[f"k_pred_{tag}"].min()
            row[f"max_k_pred_{tag}"] = sub[f"k_pred_{tag}"].max()
            
            # Win rate vs original
            wins = int((sub[f"mlu_{tag}"] < sub["mlu_orig"]).sum())
            ties = int((abs(sub[f"mlu_{tag}"] - sub["mlu_orig"]) < 1e-6).sum())
            row[f"wins_vs_orig_{tag}"] = wins
            row[f"ties_vs_orig_{tag}"] = ties
            row[f"losses_vs_orig_{tag}"] = len(sub) - wins - ties
        
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(eval_dir / "summary.csv", index=False)
    print(f"\n[Eval] Saved {len(df)} rows + summary")
    return df, summary_df, variants


# ---------------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------------

def generate_plots(df, summary_df, variants):
    """Generate comparison plots for 3 variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = OUTPUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    tags = [tag for tag, _, _ in variants]
    colors = {"orig": "tab:blue", "stage1": "tab:orange", "stage3": "tab:green"}
    labels = {"orig": "Original GNN (K=40)", "stage1": "Stage 1 Winner (K=40)", 
              "stage3": "Stage 3 Prototype (DynK)"}
    topos = sorted(df["topology"].unique())

    # 1. MLU CDF
    fig, axes = plt.subplots(1, len(topos), figsize=(4.5 * len(topos), 4), squeeze=False)
    for i, topo in enumerate(topos):
        ax = axes[0, i]; sub = df[df["topology"] == topo]
        for tag in tags:
            col = f"mlu_{tag}"
            vals = np.sort(sub[col].values)
            cdf = np.arange(1, len(vals)+1) / len(vals)
            ax.plot(vals, cdf, label=labels.get(tag, tag),
                   color=colors.get(tag, "gray"), lw=2)
        ax.set_title(topo, fontweight="bold")
        ax.set_xlabel("MLU")
        ax.set_ylabel("CDF")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("MLU CDF Comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_dir / "mlu_cdf.png", dpi=150)
    plt.close(fig)

    # 2. K over time (for Stage 3 dynamic K)
    if "stage3" in tags:
        for topo in topos:
            sub = df[df["topology"] == topo].sort_values("step")
            if sub.empty: continue
            fig, ax = plt.subplots(figsize=(10, 4))
            steps = sub["step"].values
            
            ax.axhline(40, color="tab:blue", ls="--", lw=1.5, label="Fixed K=40", alpha=0.6)
            ax.plot(steps, sub["oracle_k"].values, color="black", ls=":",
                    lw=2, label="Oracle K", alpha=0.7)
            ax.plot(steps, sub["k_pred_stage3"].values, label="Stage 3 K_pred",
                    color="tab:green", lw=1.5, alpha=0.8)
            
            ax.set_ylabel("K")
            ax.set_xlabel("Timestep")
            ax.set_title(f"{topo}: K Prediction (Stage 3 Prototype)", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / f"k_over_time_{topo}.png", dpi=150)
            plt.close(fig)

    # 3. Win rate bar chart
    if len(topos) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(topos))
        width = 0.35
        
        for i, tag in enumerate(tags):
            if tag == "orig": continue
            wins = [summary_df[summary_df["topology"] == t][f"wins_vs_orig_{tag}"].values[0] 
                    for t in topos]
            ax.bar(x + i*width, wins, width, label=labels.get(tag, tag), 
                   color=colors.get(tag, "gray"))
        
        ax.set_ylabel("Wins vs Original GNN")
        ax.set_xlabel("Topology")
        ax.set_title("Win Rate Comparison", fontweight="bold")
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(topos)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(plot_dir / "win_rate.png", dpi=150)
        plt.close(fig)

    print(f"[Plots] Saved to {plot_dir}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    print("="*70)
    print("STAGE 3 EXPLORATORY SCREENING")
    print("Compare: Original GNN | Stage 1 Winner | Stage 3 Prototype")
    print("IMPORTANT: Exploratory only. Stage 2 was NOT validated.")
    print(f"Output: {OUTPUT_ROOT}")
    print("="*70)
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    M = setup()
    
    # Load topologies
    print("\n[1] Loading topologies...")
    datasets = load_pilot_topologies(M)
    print(f"\nLoaded {len(datasets)} topologies: {list(datasets.keys())}")
    
    if len(datasets) == 0:
        print("\nERROR: Zero topologies loaded! STOPPING.")
        return
    
    # Collect samples for Stage 3 prototype training
    print("\n[2] Collecting samples for Stage 3 prototype training...")
    train_samples = collect_samples(M, datasets, 'train', MAX_TRAIN_PER_TOPO)
    val_samples = collect_samples(M, datasets, 'val', MAX_VAL_PER_TOPO)
    print(f"\nTotal samples: {len(train_samples)} train, {len(val_samples)} val")
    
    if len(train_samples) == 0:
        print("\nERROR: No training samples! STOPPING.")
        return
    
    # Train Stage 3 prototype only (W=1.0 from Stage 2 pilot)
    print(f"\n[3] Training Stage 3 Prototype (W={W_PROTOTYPE})...")
    stage3_model, train_time = train_stage3_prototype(M, W_PROTOTYPE, train_samples, val_samples)
    
    # Evaluate all 3 variants
    print(f"\n[4] Evaluating all variants...")
    df, summary_df, variants = evaluate_all(M, datasets, stage3_model)
    
    # Generate plots
    print(f"\n[5] Generating plots...")
    generate_plots(df, summary_df, variants)
    
    # Print results
    print("\n" + "="*70)
    print("STAGE 3 EXPLORATORY RESULTS")
    print("="*70)
    
    tags = [tag for tag, _, _ in variants]
    
    for _, row in summary_df.iterrows():
        topo = row['topology']
        print(f"\n{topo}:")
        print(f"  Steps: {row['num_steps']}")
        print(f"  Oracle K: {row['oracle_k_mean']:.1f} ± {row['oracle_k_std']:.1f}")
        
        for tag in tags:
            mlu = row[f'mean_mlu_{tag}']
            dist = row[f'mean_dist_{tag}']
            pr = row[f'mean_pr_{tag}']
            time_ms = row[f'mean_time_ms_{tag}']
            k_mean = row[f'mean_k_pred_{tag}']
            k_std = row[f'std_k_pred_{tag}']
            
            print(f"  {tag:8s}: MLU={mlu:.4f} PR={pr:+.3f} dist={dist:.3f} time={time_ms:.1f}ms", end="")
            if tag == "stage3":
                print(f" K={k_mean:.1f}±{k_std:.1f} [{row[f'min_k_pred_{tag}']:.0f},{row[f'max_k_pred_{tag}']:.0f}]")
            else:
                print(f" K={k_mean:.0f}")
            
            if tag != "orig":
                wins = row[f'wins_vs_orig_{tag}']
                ties = row[f'ties_vs_orig_{tag}']
                losses = row[f'losses_vs_orig_{tag}']
                print(f"           vs orig: {wins}W/{ties}T/{losses}L")
    
    # Blunt conclusion
    print("\n" + "="*70)
    print("BLUNT CONCLUSION")
    print("="*70)
    
    total_row = summary_df[summary_df["topology"] == "TOTAL"]
    if len(total_row) > 0:
        stage3_wins = total_row["wins_vs_orig_stage3"].values[0]
        stage3_losses = total_row["losses_vs_orig_stage3"].values[0]
        stage3_k_std = total_row["std_k_pred_stage3"].values[0]
        
        if "stage1" in tags:
            stage1_wins = total_row["wins_vs_orig_stage1"].values[0]
            print(f"Stage 1 winner vs orig: {stage1_wins} wins, {stage1_losses} losses")
        
        print(f"Stage 3 prototype vs orig: {stage3_wins} wins, {stage3_losses} losses")
        print(f"Stage 3 K std: {stage3_k_std:.2f} (higher = more dynamic)")
        
        # Decision rule
        if stage3_wins > stage3_losses and stage3_k_std > 2.0:
            verdict = "PROMISING for future work"
            reason = "Wins vs baseline AND shows K variation"
        elif stage3_wins > stage3_losses:
            verdict = "MIXED - better MLU but static K"
            reason = "Wins but K still collapsed"
        else:
            verdict = "NOT worth continuing"
            reason = "No improvement over fixed K"
        
        print(f"\nVerdict: {verdict}")
        print(f"Reason: {reason}")
    
    print(f"\nResults saved to: {OUTPUT_ROOT}")
    print("="*70)


if __name__ == "__main__":
    main()
