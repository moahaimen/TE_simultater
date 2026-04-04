#!/usr/bin/env python3
"""GNN+ Screening Experiment: fast evaluation of enriched features + dynamic K.

Compares:
  1. Original GNN (fixed K=40, original features) — loaded from existing checkpoint
  2. GNN+ (enriched features, dynamic K ∈ [15, 40]) — newly trained here

Topologies: Germany50, GEANT, Abilene (3 topologies)

Output: results/gnn_plus/
  - training/  (GNN+ model, training log)
  - eval/      (per-timestep CSV, summary CSV)
  - plots/     (2-4 comparison plots)

IMPORTANT: This is a SCREENING experiment only.
  - Tests combined upgrade (features + dynamic K)
  - Does NOT isolate the effect of features vs dynamic K
  - Does NOT replace the existing GNN baseline
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
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ---------- constants ----------
CONFIG_PATH = "configs/phase1_reactive_full.yaml"
SEED = 42
K_CRIT = 40
LT = 20          # LP time limit (sec)
DEVICE = "cpu"
MAX_TEST_STEPS = 75   # test timesteps per topology

GNN_ORIG_CKPT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
OUTPUT_ROOT = Path("results/gnn_plus")
TRAIN_DIR = OUTPUT_ROOT / "training"
EVAL_DIR = OUTPUT_ROOT / "eval"
PLOT_DIR = OUTPUT_ROOT / "plots"

TOPOLOGIES = ["abilene", "geant", "germany50"]

# ---------- setup ----------

def setup():
    """Import all pipeline modules."""
    from te.baselines import (
        ecmp_splits, select_bottleneck_critical,
        select_sensitivity_critical, select_topk_by_demand,
    )
    from te.lp_solver import solve_selected_path_lp
    from te.simulator import apply_routing
    from phase1_reactive.eval.common import (
        load_bundle, load_named_dataset, collect_specs, max_steps_from_args,
    )
    from phase1_reactive.eval.core import split_indices
    from phase1_reactive.drl.gnn_selector import (
        load_gnn_selector, build_graph_tensors, build_od_features,
        GNNSelectorConfig, GNNFlowSelector,
    )
    from phase1_reactive.drl.gnn_plus_selector import (
        GNNPlusConfig, GNNPlusFlowSelector,
        build_graph_tensors_plus, build_od_features_plus,
        save_gnn_plus, load_gnn_plus,
    )
    from phase1_reactive.drl.state_builder import compute_reactive_telemetry
    from phase1_reactive.drl.gnn_training import _collect_oracle_labels, _ranking_loss

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
        "load_gnn_selector": load_gnn_selector,
        "build_graph_tensors": build_graph_tensors,
        "build_od_features": build_od_features,
        "GNNSelectorConfig": GNNSelectorConfig,
        "GNNFlowSelector": GNNFlowSelector,
        "GNNPlusConfig": GNNPlusConfig,
        "GNNPlusFlowSelector": GNNPlusFlowSelector,
        "build_graph_tensors_plus": build_graph_tensors_plus,
        "build_od_features_plus": build_od_features_plus,
        "save_gnn_plus": save_gnn_plus,
        "load_gnn_plus": load_gnn_plus,
        "compute_reactive_telemetry": compute_reactive_telemetry,
        "_collect_oracle_labels": _collect_oracle_labels,
        "_ranking_loss": _ranking_loss,
    }


def load_topologies(M):
    """Load datasets for the screening topologies."""
    bundle = M["load_bundle"](CONFIG_PATH)
    eval_specs = M["collect_specs"](bundle, "eval_topologies")
    gen_specs = M["collect_specs"](bundle, "generalization_topologies")
    all_specs = eval_specs + gen_specs

    datasets = {}
    for spec in all_specs:
        try:
            ds, pl = M["load_named_dataset"](bundle, spec, 500)
        except Exception as e:
            print(f"  Skip {spec.key}: {e}")
            continue
        if ds.key in TOPOLOGIES:
            ds.path_library = pl
            datasets[ds.key] = ds
            n_tms = len(ds.tm) if hasattr(ds, 'tm') else 0
            print(f"  Loaded {ds.key}: {len(ds.nodes)} nodes, {len(ds.edges)} edges, "
                  f"{len(ds.od_pairs)} ODs, {n_tms} TMs")
    return datasets


# ---------- GNN+ Training ----------

def train_gnn_plus(M, datasets):
    """Train GNN+ model on the screening topologies."""
    from phase1_reactive.drl.gnn_selector import save_gnn_selector
    from te.simulator import apply_routing as apply_rt

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    cfg = M["GNNPlusConfig"]()
    model = M["GNNPlusFlowSelector"](cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Collect training samples
    print("\n[GNN+ Training] Collecting oracle labels...", flush=True)
    train_samples = []
    val_samples = []
    rng_sub = np.random.default_rng(SEED)

    for topo_key, ds in datasets.items():
        path_library = ds.path_library
        ecmp_base = M["ecmp_splits"](path_library)
        capacities = np.asarray(ds.capacities, dtype=float)

        train_idx = M["split_indices"](ds, "train")
        val_idx = M["split_indices"](ds, "val")

        # Subsample train: max 40 per topo
        if len(train_idx) > 40:
            train_idx = sorted(rng_sub.choice(train_idx, size=40, replace=False).tolist())
        if len(val_idx) > 15:
            val_idx = sorted(rng_sub.choice(val_idx, size=15, replace=False).tolist())

        print(f"  {topo_key}: {len(train_idx)} train, {len(val_idx)} val", flush=True)

        prev_tm = None
        for t_idx in train_idx:
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12:
                prev_tm = tm
                continue
            routing = apply_rt(tm, ecmp_base, path_library, capacities)
            telemetry = M["compute_reactive_telemetry"](
                tm, ecmp_base, path_library, routing,
                np.asarray(ds.weights, dtype=float),
            )
            oracle_sel, oracle_mlu, _, _ = M["_collect_oracle_labels"](
                ds, path_library, tm, ecmp_base, capacities, K_CRIT, lp_time_limit_sec=LT,
            )
            if oracle_sel:
                train_samples.append({
                    "dataset": ds, "path_library": path_library,
                    "tm_vector": tm, "prev_tm": prev_tm,
                    "telemetry": telemetry,
                    "oracle_selected": oracle_sel, "oracle_mlu": oracle_mlu,
                    "k_crit": K_CRIT, "capacities": capacities,
                    "prev_util": None,  # first pass: no previous util
                })
            prev_tm = tm

        prev_tm = None
        for t_idx in val_idx:
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12:
                prev_tm = tm
                continue
            routing = apply_rt(tm, ecmp_base, path_library, capacities)
            telemetry = M["compute_reactive_telemetry"](
                tm, ecmp_base, path_library, routing,
                np.asarray(ds.weights, dtype=float),
            )
            oracle_sel, oracle_mlu, _, _ = M["_collect_oracle_labels"](
                ds, path_library, tm, ecmp_base, capacities, K_CRIT, lp_time_limit_sec=LT,
            )
            if oracle_sel:
                val_samples.append({
                    "dataset": ds, "path_library": path_library,
                    "tm_vector": tm, "prev_tm": prev_tm,
                    "telemetry": telemetry,
                    "oracle_selected": oracle_sel, "oracle_mlu": oracle_mlu,
                    "k_crit": K_CRIT, "capacities": capacities,
                    "prev_util": None,
                })
            prev_tm = tm

    print(f"[GNN+ Training] {len(train_samples)} train, {len(val_samples)} val samples", flush=True)

    # Training loop
    rng = np.random.default_rng(SEED)
    logs = []
    best_val_loss = float("inf")
    best_epoch = 0
    stale = 0
    max_epochs = 30
    patience = 8
    t_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_samples))
        ep_losses = []

        for si in order:
            s = train_samples[si]
            # Use GNN+ feature builders
            graph_data = M["build_graph_tensors_plus"](
                s["dataset"], tm_vector=s["tm_vector"],
                path_library=s["path_library"],
                telemetry=s["telemetry"], prev_util=s["prev_util"],
                device=DEVICE,
            )
            od_data = M["build_od_features_plus"](
                s["dataset"], s["tm_vector"], s["path_library"],
                telemetry=s["telemetry"], prev_tm=s["prev_tm"],
                device=DEVICE,
            )

            scores, k_pred, info = model(graph_data, od_data)

            num_od = scores.size(0)
            oracle_mask = torch.zeros(num_od, device=DEVICE)
            for oid in s["oracle_selected"]:
                if oid < num_od:
                    oracle_mask[oid] = 1.0

            loss = M["_ranking_loss"](scores, oracle_mask, margin=0.1)

            # k_crit loss: target is the actual k_crit used
            if k_pred is not None:
                k_target_val = float(s["k_crit"])
                k_frac_target = (k_target_val - 15.0) / (40.0 - 15.0)  # normalize to [0,1]
                # Get raw k_frac from model for loss
                graph_embed = model.gnn_layers[-1](
                    model.gnn_layers[-2](
                        model.gnn_layers[0](
                            F.relu(model.node_proj(graph_data["node_features"])),
                            graph_data["edge_index"],
                            F.relu(model.edge_proj(graph_data["edge_features"])),
                        ),
                        graph_data["edge_index"],
                        F.relu(model.edge_proj(graph_data["edge_features"])),
                    ),
                    graph_data["edge_index"],
                    F.relu(model.edge_proj(graph_data["edge_features"])),
                ).mean(dim=0)
                k_frac_pred = model.k_head(graph_embed).squeeze()
                k_loss = F.mse_loss(k_frac_pred, torch.tensor(k_frac_target, device=DEVICE))
                loss = loss + 0.01 * k_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(float(loss.item()))

        scheduler.step()
        train_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        # Validation
        model.eval()
        val_losses = []
        val_overlaps = []
        with torch.no_grad():
            for s in val_samples:
                graph_data = M["build_graph_tensors_plus"](
                    s["dataset"], tm_vector=s["tm_vector"],
                    path_library=s["path_library"],
                    telemetry=s["telemetry"], prev_util=s["prev_util"],
                    device=DEVICE,
                )
                od_data = M["build_od_features_plus"](
                    s["dataset"], s["tm_vector"], s["path_library"],
                    telemetry=s["telemetry"], prev_tm=s["prev_tm"],
                    device=DEVICE,
                )
                scores, k_pred, info = model(graph_data, od_data)

                num_od = scores.size(0)
                oracle_mask = torch.zeros(num_od, device=DEVICE)
                for oid in s["oracle_selected"]:
                    if oid < num_od:
                        oracle_mask[oid] = 1.0

                vloss = M["_ranking_loss"](scores, oracle_mask, margin=0.1)
                val_losses.append(float(vloss.item()))

                # Selection overlap
                k = s["k_crit"]
                scores_np = scores.cpu().numpy()
                active = s["tm_vector"] > 0
                active_idx = np.where(active)[0]
                if active_idx.size > 0:
                    take = min(k, active_idx.size)
                    active_scores = scores_np[active_idx]
                    top_local = np.argsort(-active_scores)[:take]
                    pred_set = set(active_idx[top_local].tolist())
                    oracle_set = set(s["oracle_selected"])
                    overlap = len(pred_set & oracle_set) / max(len(pred_set | oracle_set), 1)
                    val_overlaps.append(overlap)

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_overlap = float(np.mean(val_overlaps)) if val_overlaps else 0.0
        ep_time = time.perf_counter() - ep_start

        logs.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_overlap": val_overlap,
            "alpha": float(model.alpha.item()),
            "k_pred": k_pred,
            "lr": float(scheduler.get_last_lr()[0]),
            "epoch_time_sec": ep_time,
        })
        print(f"  Epoch {epoch:3d}: train={train_loss:.4f}  val={val_loss:.4f}  "
              f"overlap={val_overlap:.3f}  alpha={model.alpha.item():.3f}  "
              f"k_pred={k_pred}  [{ep_time:.1f}s]", flush=True)

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale = 0
            M["save_gnn_plus"](model, TRAIN_DIR / "gnn_plus_model.pt", extra_meta={
                "best_epoch": best_epoch, "best_val_loss": best_val_loss,
            })
        else:
            stale += 1
        if stale >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    total_time = time.perf_counter() - t_start
    print(f"[GNN+ Training] Done in {total_time:.1f}s, best epoch={best_epoch}", flush=True)

    # Save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(TRAIN_DIR / "gnn_plus_train_log.csv", index=False)
    summary = {
        "training_time_sec": total_time,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_train_samples": len(train_samples),
        "total_val_samples": len(val_samples),
        "final_alpha": float(model.alpha.item()),
    }
    (TRAIN_DIR / "gnn_plus_train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    return model, total_time, best_epoch


# ---------- Evaluation ----------

def evaluate_both(M, datasets, gnn_plus_model):
    """Evaluate original GNN and GNN+ side-by-side on test splits."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load original GNN
    print("\n[Eval] Loading original GNN from checkpoint...", flush=True)
    gnn_orig, _ = M["load_gnn_selector"](GNN_ORIG_CKPT, device=DEVICE)
    gnn_orig.eval()
    gnn_plus_model.eval()

    rows = []

    for topo_key in TOPOLOGIES:
        if topo_key not in datasets:
            print(f"  SKIP {topo_key} (not loaded)")
            continue

        ds = datasets[topo_key]
        path_library = ds.path_library
        ecmp_base = M["ecmp_splits"](path_library)
        capacities = np.asarray(ds.capacities, dtype=float)

        test_idx = M["split_indices"](ds, "test")
        if len(test_idx) > MAX_TEST_STEPS:
            test_idx = test_idx[:MAX_TEST_STEPS]

        print(f"\n[Eval] {topo_key}: {len(test_idx)} test steps", flush=True)

        prev_tm = None
        prev_util = None
        prev_sel_orig = None
        prev_sel_plus = None

        for step_i, t_idx in enumerate(test_idx):
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12:
                prev_tm = tm
                continue

            routing_ecmp = M["apply_routing"](tm, ecmp_base, path_library, capacities)
            telemetry = M["compute_reactive_telemetry"](
                tm, ecmp_base, path_library, routing_ecmp,
                np.asarray(ds.weights, dtype=float),
            )

            active_mask = (np.asarray(tm, dtype=np.float64) > 1e-12).astype(np.float32)

            # --- Oracle (for PR computation) ---
            oracle_sel, oracle_mlu, oracle_method, _ = M["_collect_oracle_labels"](
                ds, path_library, tm, ecmp_base, capacities, K_CRIT, lp_time_limit_sec=LT,
            )

            # --- Original GNN (fixed K=40) ---
            t0 = time.perf_counter()
            graph_data_orig = M["build_graph_tensors"](ds, telemetry=telemetry, device=DEVICE)
            od_data_orig = M["build_od_features"](
                ds, tm, path_library, telemetry=telemetry, device=DEVICE,
            )
            with torch.no_grad():
                sel_orig, info_orig = gnn_orig.select_critical_flows(
                    graph_data_orig, od_data_orig,
                    active_mask=active_mask, k_crit_default=K_CRIT,
                    force_default_k=True,
                )
            time_orig = (time.perf_counter() - t0) * 1000

            # LP for original GNN
            try:
                lp_orig = M["solve_selected_path_lp"](
                    tm_vector=tm, selected_ods=sel_orig, base_splits=ecmp_base,
                    path_library=path_library, capacities=capacities,
                    time_limit_sec=LT,
                )
                mlu_orig = float(lp_orig.routing.mlu)
            except Exception:
                mlu_orig = float("inf")

            # --- GNN+ (dynamic K) ---
            t0 = time.perf_counter()
            graph_data_plus = M["build_graph_tensors_plus"](
                ds, tm_vector=tm, path_library=path_library,
                telemetry=telemetry, prev_util=prev_util, device=DEVICE,
            )
            od_data_plus = M["build_od_features_plus"](
                ds, tm, path_library,
                telemetry=telemetry, prev_tm=prev_tm, device=DEVICE,
            )
            with torch.no_grad():
                sel_plus, info_plus = gnn_plus_model.select_critical_flows(
                    graph_data_plus, od_data_plus,
                    active_mask=active_mask, k_crit_default=K_CRIT,
                    force_default_k=False,  # use dynamic K
                )
            time_plus = (time.perf_counter() - t0) * 1000

            # LP for GNN+
            try:
                lp_plus = M["solve_selected_path_lp"](
                    tm_vector=tm, selected_ods=sel_plus, base_splits=ecmp_base,
                    path_library=path_library, capacities=capacities,
                    time_limit_sec=LT,
                )
                mlu_plus = float(lp_plus.routing.mlu)
            except Exception:
                mlu_plus = float("inf")

            # Disturbance
            dist_orig = _compute_disturbance(sel_orig, prev_sel_orig, K_CRIT)
            dist_plus = _compute_disturbance(sel_plus, prev_sel_plus, K_CRIT)

            # PR = (method_MLU - oracle_MLU) / oracle_MLU
            pr_orig = (mlu_orig - oracle_mlu) / (oracle_mlu + 1e-12) if oracle_mlu > 0 else 0.0
            pr_plus = (mlu_plus - oracle_mlu) / (oracle_mlu + 1e-12) if oracle_mlu > 0 else 0.0

            k_used_plus = info_plus.get("k_used", K_CRIT)
            k_dynamic_plus = info_plus.get("k_dynamic", None)

            rows.append({
                "topology": topo_key,
                "step": step_i,
                "tm_idx": t_idx,
                # Original GNN
                "mlu_orig": mlu_orig,
                "pr_orig": pr_orig,
                "time_ms_orig": time_orig,
                "disturbance_orig": dist_orig,
                "k_used_orig": K_CRIT,
                "alpha_orig": info_orig.get("alpha", 0),
                "confidence_orig": info_orig.get("confidence", 0),
                "w_bn_orig": info_orig.get("w_bottleneck", 0),
                "w_sens_orig": info_orig.get("w_sensitivity", 0),
                # GNN+
                "mlu_plus": mlu_plus,
                "pr_plus": pr_plus,
                "time_ms_plus": time_plus,
                "disturbance_plus": dist_plus,
                "k_used_plus": k_used_plus,
                "k_dynamic_plus": k_dynamic_plus if k_dynamic_plus is not None else K_CRIT,
                "alpha_plus": info_plus.get("alpha", 0),
                "confidence_plus": info_plus.get("confidence", 0),
                "w_bn_plus": info_plus.get("w_bottleneck", 0),
                "w_sens_plus": info_plus.get("w_sensitivity", 0),
                # Oracle reference
                "oracle_mlu": oracle_mlu,
                "oracle_method": oracle_method,
            })

            prev_sel_orig = sel_orig
            prev_sel_plus = sel_plus
            prev_tm = tm
            prev_util = np.asarray(telemetry.utilization, dtype=np.float64) if telemetry else None

            if (step_i + 1) % 25 == 0:
                print(f"    step {step_i+1}/{len(test_idx)}: "
                      f"orig_MLU={mlu_orig:.4f} plus_MLU={mlu_plus:.4f} "
                      f"K+={k_used_plus}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "gnn_plus_per_timestep.csv", index=False)
    print(f"\n[Eval] Saved {len(df)} rows to {EVAL_DIR / 'gnn_plus_per_timestep.csv'}")

    # Summary
    summary_rows = []
    for topo_key in TOPOLOGIES:
        sub = df[df["topology"] == topo_key]
        if sub.empty:
            continue
        summary_rows.append({
            "topology": topo_key,
            "num_steps": len(sub),
            "mean_mlu_orig": sub["mlu_orig"].mean(),
            "mean_mlu_plus": sub["mlu_plus"].mean(),
            "mean_pr_orig": sub["pr_orig"].mean(),
            "mean_pr_plus": sub["pr_plus"].mean(),
            "mean_time_ms_orig": sub["time_ms_orig"].mean(),
            "mean_time_ms_plus": sub["time_ms_plus"].mean(),
            "mean_disturbance_orig": sub["disturbance_orig"].mean(),
            "mean_disturbance_plus": sub["disturbance_plus"].mean(),
            "k_plus_mean": sub["k_used_plus"].mean(),
            "k_plus_min": sub["k_used_plus"].min(),
            "k_plus_max": sub["k_used_plus"].max(),
            "k_plus_std": sub["k_used_plus"].std(),
            "mean_oracle_mlu": sub["oracle_mlu"].mean(),
            "mlu_improvement_pct": ((sub["mlu_orig"].mean() - sub["mlu_plus"].mean())
                                     / (sub["mlu_orig"].mean() + 1e-12) * 100),
            "plus_wins": int((sub["mlu_plus"] < sub["mlu_orig"]).sum()),
            "orig_wins": int((sub["mlu_orig"] < sub["mlu_plus"]).sum()),
            "ties": int((sub["mlu_orig"] == sub["mlu_plus"]).sum()),
        })

    # Add aggregate row
    if not df.empty:
        summary_rows.append({
            "topology": "AGGREGATE",
            "num_steps": len(df),
            "mean_mlu_orig": df["mlu_orig"].mean(),
            "mean_mlu_plus": df["mlu_plus"].mean(),
            "mean_pr_orig": df["pr_orig"].mean(),
            "mean_pr_plus": df["pr_plus"].mean(),
            "mean_time_ms_orig": df["time_ms_orig"].mean(),
            "mean_time_ms_plus": df["time_ms_plus"].mean(),
            "mean_disturbance_orig": df["disturbance_orig"].mean(),
            "mean_disturbance_plus": df["disturbance_plus"].mean(),
            "k_plus_mean": df["k_used_plus"].mean(),
            "k_plus_min": df["k_used_plus"].min(),
            "k_plus_max": df["k_used_plus"].max(),
            "k_plus_std": df["k_used_plus"].std(),
            "mean_oracle_mlu": df["oracle_mlu"].mean(),
            "mlu_improvement_pct": ((df["mlu_orig"].mean() - df["mlu_plus"].mean())
                                     / (df["mlu_orig"].mean() + 1e-12) * 100),
            "plus_wins": int((df["mlu_plus"] < df["mlu_orig"]).sum()),
            "orig_wins": int((df["mlu_orig"] < df["mlu_plus"]).sum()),
            "ties": int((df["mlu_orig"] == df["mlu_plus"]).sum()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(EVAL_DIR / "gnn_plus_summary.csv", index=False)
    print(f"[Eval] Saved summary to {EVAL_DIR / 'gnn_plus_summary.csv'}")

    return df, summary_df


def _compute_disturbance(current_ods, prev_ods, k_crit):
    if prev_ods is None:
        return 0.0
    cur_set = set(current_ods)
    prev_set = set(prev_ods)
    sym_diff = len(cur_set ^ prev_set)
    return sym_diff / max(k_crit, 1)


# ---------- Plotting ----------

def generate_plots(df, summary_df):
    """Generate 4 comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: MLU CDF comparison per topology
    fig, axes = plt.subplots(1, len(TOPOLOGIES), figsize=(5 * len(TOPOLOGIES), 4), squeeze=False)
    for i, topo in enumerate(TOPOLOGIES):
        ax = axes[0, i]
        sub = df[df["topology"] == topo]
        if sub.empty:
            continue
        for col, label, color in [("mlu_orig", "Original GNN (K=40)", "tab:blue"),
                                   ("mlu_plus", "GNN+ (dynamic K)", "tab:orange")]:
            vals = np.sort(sub[col].values)
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, label=label, color=color, linewidth=2)
        ax.set_title(topo, fontsize=13, fontweight="bold")
        ax.set_xlabel("MLU")
        ax.set_ylabel("CDF")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("MLU CDF: Original GNN vs GNN+", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mlu_cdf_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'mlu_cdf_comparison.png'}")

    # Plot 2: K distribution for GNN+
    fig, axes = plt.subplots(1, len(TOPOLOGIES), figsize=(5 * len(TOPOLOGIES), 4), squeeze=False)
    for i, topo in enumerate(TOPOLOGIES):
        ax = axes[0, i]
        sub = df[df["topology"] == topo]
        if sub.empty:
            continue
        k_vals = sub["k_used_plus"].values
        ax.hist(k_vals, bins=range(int(min(k_vals)) - 1, int(max(k_vals)) + 2),
                color="tab:orange", alpha=0.7, edgecolor="black")
        ax.axvline(40, color="tab:blue", linestyle="--", linewidth=2, label=f"Fixed K=40")
        ax.axvline(np.mean(k_vals), color="tab:red", linestyle="-", linewidth=2,
                   label=f"Mean K={np.mean(k_vals):.1f}")
        ax.set_title(topo, fontsize=13, fontweight="bold")
        ax.set_xlabel("K used")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("GNN+ Dynamic K Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "k_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'k_distribution.png'}")

    # Plot 3: Bar chart — mean MLU comparison
    topos_present = [t for t in TOPOLOGIES if t in df["topology"].values]
    x = np.arange(len(topos_present))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    mlu_orig = [summary_df[summary_df["topology"] == t]["mean_mlu_orig"].values[0] for t in topos_present]
    mlu_plus = [summary_df[summary_df["topology"] == t]["mean_mlu_plus"].values[0] for t in topos_present]
    bars1 = ax.bar(x - width/2, mlu_orig, width, label="Original GNN (K=40)", color="tab:blue", alpha=0.8)
    bars2 = ax.bar(x + width/2, mlu_plus, width, label="GNN+ (dynamic K)", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(topos_present, fontsize=11)
    ax.set_ylabel("Mean MLU", fontsize=12)
    ax.set_title("Mean MLU: Original GNN vs GNN+", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mean_mlu_bar_chart.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'mean_mlu_bar_chart.png'}")

    # Plot 4: Disturbance comparison
    fig, axes = plt.subplots(1, len(TOPOLOGIES), figsize=(5 * len(TOPOLOGIES), 4), squeeze=False)
    for i, topo in enumerate(TOPOLOGIES):
        ax = axes[0, i]
        sub = df[df["topology"] == topo]
        if sub.empty:
            continue
        for col, label, color in [("disturbance_orig", "Original GNN", "tab:blue"),
                                   ("disturbance_plus", "GNN+", "tab:orange")]:
            vals = np.sort(sub[col].values)
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, label=label, color=color, linewidth=2)
        ax.set_title(topo, fontsize=13, fontweight="bold")
        ax.set_xlabel("Disturbance")
        ax.set_ylabel("CDF")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Disturbance CDF: Original GNN vs GNN+", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "disturbance_cdf_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'disturbance_cdf_comparison.png'}")


# ---------- Main ----------

if __name__ == "__main__":
    print("=" * 70)
    print("GNN+ SCREENING EXPERIMENT")
    print("=" * 70)

    print("\n[1/4] Setting up modules...", flush=True)
    M = setup()

    print("\n[2/4] Loading topologies...", flush=True)
    datasets = load_topologies(M)

    print("\n[3/4] Training GNN+...", flush=True)
    gnn_plus_model, train_time, best_epoch = train_gnn_plus(M, datasets)

    # Reload best checkpoint
    gnn_plus_model, _ = M["load_gnn_plus"](TRAIN_DIR / "gnn_plus_model.pt", device=DEVICE)

    print("\n[4/4] Evaluating Original GNN vs GNN+...", flush=True)
    df, summary_df = evaluate_both(M, datasets, gnn_plus_model)

    print("\n[5/5] Generating plots...", flush=True)
    generate_plots(df, summary_df)

    print("\n" + "=" * 70)
    print("SCREENING RESULTS SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nTraining time: {train_time:.1f}s, Best epoch: {best_epoch}")
    print(f"\nAll outputs in: {OUTPUT_ROOT}")
    print("=" * 70)
