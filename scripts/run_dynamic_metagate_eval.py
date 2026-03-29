#!/usr/bin/env python3
"""Evaluate the 4-expert MLP MetaGate with unseen-topology integrity.

Architecture:
  - MLP Meta-Gate selects among {Bottleneck, TopK, Sensitivity, GNN}
  - ONE unified gate trained on pooled data from 6 known topologies
  - Germany50 and VtlWavenet2011 are unseen at gate-training time
  - Per-component end-to-end timing: BN+TopK+Sens+GNN+features+MLP+LP

Pipeline per timestep:
  1. Run all 4 experts (BN, TopK, Sens, GNN) → 4 OD selections
  2. Extract features from TM + expert outputs + GNN diagnostics
  3. MLP classifier predicts best expert
  4. Run predicted expert's selection through LP → MLU

Oracle labels:
  For each training TM, all 4 experts are evaluated with LP.
  Oracle = argmin(BN_MLU, TopK_MLU, Sens_MLU, GNN_MLU).

Output:
  results/dynamic_metagate/metagate_results.csv
  results/dynamic_metagate/metagate_summary.csv
  results/dynamic_metagate/metagate_decisions.csv
  results/dynamic_metagate/metagate_timing.csv
  results/dynamic_metagate/models/metagate_unified.pt
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

CONFIG_PATH = "configs/phase1_reactive_full.yaml"
MAX_STEPS = 500
SEED = 42
K_CRIT = 40
LT = 20  # LP time limit
DEVICE = "cpu"
OUTPUT_DIR = Path("results/dynamic_metagate")
GNN_CHECKPOINT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")

SELECTOR_NAMES = ["bottleneck", "topk", "sensitivity", "gnn"]
KNOWN_TOPOLOGIES = {"abilene", "geant", "cernet", "rocketfuel_ebone",
                    "rocketfuel_sprintlink", "rocketfuel_tiscali"}
UNSEEN_TOPOLOGIES = {"germany50", "topologyzoo_vtlwavenet2011"}


def setup():
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
    from phase1_reactive.drl.dynamic_meta_gate import (
        DynamicMetaGate, MetaGateConfig, extract_features,
    )
    from phase1_reactive.drl.gnn_selector import (
        load_gnn_selector, build_graph_tensors, build_od_features,
    )
    from phase1_reactive.drl.state_builder import compute_reactive_telemetry
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
        "max_steps_from_args": max_steps_from_args,
        "split_indices": split_indices,
        "DynamicMetaGate": DynamicMetaGate,
        "MetaGateConfig": MetaGateConfig,
        "extract_features": extract_features,
        "load_gnn_selector": load_gnn_selector,
        "build_graph_tensors": build_graph_tensors,
        "build_od_features": build_od_features,
        "compute_reactive_telemetry": compute_reactive_telemetry,
    }


def run_heuristic_selector(M, method, tm, ecmp_base, path_library, capacities, k_crit):
    """Run one of the 3 heuristic selectors."""
    if method == "topk":
        return M["select_topk_by_demand"](tm, k_crit)
    elif method == "bottleneck":
        return M["select_bottleneck_critical"](tm, ecmp_base, path_library, capacities, k_crit)
    elif method == "sensitivity":
        return M["select_sensitivity_critical"](tm, ecmp_base, path_library, capacities, k_crit)
    else:
        raise ValueError(f"Unknown heuristic: {method}")


def run_gnn_selector(M, tm, dataset, path_library, gnn_model, k_crit,
                     telemetry=None):
    """Run GNN expert, returns (selected_ods, info_dict)."""
    graph_data = M["build_graph_tensors"](dataset, telemetry=telemetry, device=DEVICE)
    od_data = M["build_od_features"](dataset, tm, path_library,
                                      telemetry=telemetry, device=DEVICE)
    active_mask = (np.asarray(tm, dtype=np.float64) > 1e-12).astype(np.float32)
    with torch.no_grad():
        selected, info = gnn_model.select_critical_flows(
            graph_data, od_data, active_mask=active_mask, k_crit_default=k_crit,
        )
    return selected, info


def run_all_experts(M, tm, ecmp_base, path_library, capacities, dataset,
                    gnn_model, k_crit, telemetry=None):
    """Run all 4 experts, returning selections and per-expert timing.

    Returns:
        selector_results: dict {name: list[int]} of OD selections
        gnn_info: dict with GNN diagnostics (alpha, confidence, etc.)
        timing: dict {name: float} per-expert time in ms
    """
    timing = {}

    # BN
    t0 = time.perf_counter()
    bn_sel = run_heuristic_selector(M, "bottleneck", tm, ecmp_base, path_library, capacities, k_crit)
    timing["bottleneck"] = (time.perf_counter() - t0) * 1000

    # TopK
    t0 = time.perf_counter()
    topk_sel = run_heuristic_selector(M, "topk", tm, ecmp_base, path_library, capacities, k_crit)
    timing["topk"] = (time.perf_counter() - t0) * 1000

    # Sensitivity
    t0 = time.perf_counter()
    sens_sel = run_heuristic_selector(M, "sensitivity", tm, ecmp_base, path_library, capacities, k_crit)
    timing["sensitivity"] = (time.perf_counter() - t0) * 1000

    # GNN
    t0 = time.perf_counter()
    gnn_sel, gnn_info = run_gnn_selector(M, tm, dataset, path_library, gnn_model, k_crit,
                                          telemetry=telemetry)
    timing["gnn"] = (time.perf_counter() - t0) * 1000

    selector_results = {
        "bottleneck": bn_sel,
        "topk": topk_sel,
        "sensitivity": sens_sel,
        "gnn": gnn_sel,
    }
    return selector_results, gnn_info, timing


def compute_features_and_oracle_for_topology(M, dataset, path_library, timesteps,
                                              gnn_model, k_crit):
    """Compute features + 4-class oracle labels for one topology's timesteps.

    Runs all 4 experts + LP per timestep to determine oracle label.
    """
    ecmp_base = M["ecmp_splits"](path_library)
    capacities = np.asarray(dataset.capacities, dtype=float)
    num_nodes = len(dataset.nodes)
    num_edges = len(dataset.edges)
    weights = np.asarray(dataset.weights, dtype=float)

    features_list = []
    labels_list = []
    valid_timesteps = []
    mlu_records = []

    for t_idx in timesteps:
        tm = np.asarray(dataset.tm[t_idx], dtype=float)
        if np.max(tm) < 1e-12:
            continue

        # Compute telemetry and ECMP baseline for GNN and features
        routing_ecmp = M["apply_routing"](tm, ecmp_base, path_library, capacities)
        telemetry = M["compute_reactive_telemetry"](tm, ecmp_base, path_library, routing_ecmp, weights)
        ecmp_link_utils = np.array(routing_ecmp.utilization, dtype=np.float32) if hasattr(routing_ecmp, 'utilization') else None

        # Run all 4 experts
        selector_results, gnn_info, _ = run_all_experts(
            M, tm, ecmp_base, path_library, capacities, dataset,
            gnn_model, k_crit, telemetry=telemetry,
        )

        # Extract features (with ECMP baseline and GNN diagnostics)
        feats = M["extract_features"](tm, selector_results, num_nodes, num_edges, k_crit,
                                       gnn_info=gnn_info, ecmp_link_utils=ecmp_link_utils)

        # Run LP for each expert to get oracle label
        mlus = {}
        for name in SELECTOR_NAMES:
            try:
                lp = M["solve_selected_path_lp"](
                    tm, selector_results[name], ecmp_base, path_library, capacities,
                    time_limit_sec=LT,
                )
                routing = M["apply_routing"](tm, lp.splits, path_library, capacities)
                mlus[name] = float(routing.mlu)
            except Exception:
                mlus[name] = float("inf")

        best_name = min(mlus, key=mlus.get)
        best_label = SELECTOR_NAMES.index(best_name)

        features_list.append(feats)
        labels_list.append(best_label)
        valid_timesteps.append(t_idx)
        mlu_records.append(mlus)

    if features_list:
        return (np.stack(features_list), np.array(labels_list, dtype=np.int64),
                valid_timesteps, mlu_records)
    return np.zeros((0, 49)), np.array([], dtype=np.int64), [], []


def evaluate_on_topology(M, dataset, path_library, gate, gnn_model, k_crit,
                         test_oracle_df):
    """Evaluate the trained unified gate on one topology's test split.

    Returns per-timestep results and decisions with full timing breakdown.
    """
    ds_key = dataset.key
    test_indices = M["split_indices"](dataset, "test")
    ecmp_base = M["ecmp_splits"](path_library)
    capacities = np.asarray(dataset.capacities, dtype=float)
    num_nodes = len(dataset.nodes)
    num_edges = len(dataset.edges)
    weights = np.asarray(dataset.weights, dtype=float)

    # Pre-computed test oracle (from all_results.csv)
    topo_oracle = test_oracle_df[test_oracle_df["dataset"] == ds_key].copy()
    oracle_by_ts = topo_oracle.set_index("timestep") if not topo_oracle.empty else pd.DataFrame()

    results = []
    decisions = []

    for t_idx in test_indices:
        tm = np.asarray(dataset.tm[t_idx], dtype=float)
        if np.max(tm) < 1e-12:
            continue

        # --- End-to-end timing starts ---
        t_total_start = time.perf_counter()

        # Telemetry and ECMP baseline for GNN and features
        routing_ecmp = M["apply_routing"](tm, ecmp_base, path_library, capacities)
        telemetry = M["compute_reactive_telemetry"](tm, ecmp_base, path_library, routing_ecmp, weights)
        ecmp_link_utils = np.array(routing_ecmp.utilization, dtype=np.float32) if hasattr(routing_ecmp, 'utilization') else None

        # Run all 4 experts with per-component timing
        selector_results, gnn_info, expert_timing = run_all_experts(
            M, tm, ecmp_base, path_library, capacities, dataset,
            gnn_model, k_crit, telemetry=telemetry,
        )

        # Feature extraction timing
        t_feat_start = time.perf_counter()
        feats = M["extract_features"](tm, selector_results, num_nodes, num_edges, k_crit,
                                       gnn_info=gnn_info, ecmp_link_utils=ecmp_link_utils)
        t_feat_ms = (time.perf_counter() - t_feat_start) * 1000

        # MLP inference timing
        t_mlp_start = time.perf_counter()
        pred_class, probs = gate.predict(feats)
        t_mlp_ms = (time.perf_counter() - t_mlp_start) * 1000
        pred_name = SELECTOR_NAMES[pred_class]
        confidence = float(probs[pred_class])

        # LP solve on predicted expert's selection
        t_lp_start = time.perf_counter()
        selected_ods = selector_results[pred_name]
        lp = M["solve_selected_path_lp"](tm, selected_ods, ecmp_base, path_library,
                                          capacities, time_limit_sec=LT)
        routing = M["apply_routing"](tm, lp.splits, path_library, capacities)
        t_lp_ms = (time.perf_counter() - t_lp_start) * 1000

        t_total_ms = (time.perf_counter() - t_total_start) * 1000
        metagate_mlu = float(routing.mlu)

        # Decision time = everything except LP
        t_decision_ms = sum(expert_timing.values()) + t_feat_ms + t_mlp_ms

        # Oracle from pre-computed CSV (includes GNN now)
        oracle_selector = "unknown"
        oracle_mlu = metagate_mlu
        bn_mlu = topk_mlu = sens_mlu = gnn_mlu = np.nan

        if not oracle_by_ts.empty and t_idx in oracle_by_ts.index:
            row = oracle_by_ts.loc[t_idx]
            oracle_selector = str(row["oracle_selector"])
            oracle_mlu = float(row["oracle_mlu"])
            bn_mlu = float(row.get("bottleneck", np.nan))
            topk_mlu = float(row.get("topk", np.nan))
            sens_mlu = float(row.get("sensitivity", np.nan))
            gnn_mlu = float(row.get("gnn", np.nan))

        results.append({
            "dataset": ds_key,
            "topology_type": "unseen" if ds_key in UNSEEN_TOPOLOGIES else "known",
            "timestep": int(t_idx),
            "metagate_selector": pred_name,
            "metagate_confidence": confidence,
            "metagate_mlu": metagate_mlu,
            "oracle_selector": oracle_selector,
            "oracle_mlu": oracle_mlu,
            "bn_mlu": bn_mlu,
            "topk_mlu": topk_mlu,
            "sens_mlu": sens_mlu,
            "gnn_mlu": gnn_mlu,
            "correct": 1 if pred_name == oracle_selector else 0,
            # Timing breakdown
            "t_bn_ms": expert_timing["bottleneck"],
            "t_topk_ms": expert_timing["topk"],
            "t_sens_ms": expert_timing["sensitivity"],
            "t_gnn_ms": expert_timing["gnn"],
            "t_features_ms": t_feat_ms,
            "t_mlp_ms": t_mlp_ms,
            "t_lp_ms": t_lp_ms,
            "t_decision_ms": t_decision_ms,
            "t_total_ms": t_total_ms,
        })

        decisions.append({
            "dataset": ds_key,
            "topology_type": "unseen" if ds_key in UNSEEN_TOPOLOGIES else "known",
            "timestep": int(t_idx),
            "predicted": pred_name,
            "oracle": oracle_selector,
            "confidence": confidence,
            "prob_bn": float(probs[0]),
            "prob_topk": float(probs[1]),
            "prob_sens": float(probs[2]),
            "prob_gnn": float(probs[3]),
        })

    return results, decisions


def load_test_oracle_4class(oracle_csv: Path) -> pd.DataFrame:
    """Load pre-computed test oracle labels including GNN as 4th option."""
    df = pd.read_csv(oracle_csv)
    sel_df = df[df["method"].isin(SELECTOR_NAMES)].copy()

    # For each (dataset, timestep), find the expert with lowest MLU
    idx = sel_df.groupby(["dataset", "timestep"])["mlu"].idxmin()
    oracle = sel_df.loc[idx, ["dataset", "timestep", "method", "mlu"]].copy()
    oracle = oracle.rename(columns={"method": "oracle_selector", "mlu": "oracle_mlu"})

    # Pivot to get all 4 MLUs per timestep
    pivot = sel_df.pivot_table(index=["dataset", "timestep"], columns="method",
                                values="mlu").reset_index()
    pivot.columns.name = None
    for s in SELECTOR_NAMES:
        if s not in pivot.columns:
            pivot[s] = np.nan
    oracle = oracle.merge(pivot[["dataset", "timestep"] + SELECTOR_NAMES],
                          on=["dataset", "timestep"])

    label_map = {name: i for i, name in enumerate(SELECTOR_NAMES)}
    oracle["oracle_label"] = oracle["oracle_selector"].map(label_map)
    return oracle


def main():
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  4-EXPERT MLP METAGATE EVALUATION")
    print("  Experts: {Bottleneck, TopK, Sensitivity, GNN}")
    print("  Unified gate trained on 6 known topologies")
    print("  Unseen test: Germany50, VtlWavenet2011")
    print("=" * 70)

    M = setup()

    # Load GNN model
    print(f"\nLoading GNN model from {GNN_CHECKPOINT}...")
    if not GNN_CHECKPOINT.exists():
        print(f"  ERROR: GNN checkpoint not found at {GNN_CHECKPOINT}")
        sys.exit(1)
    gnn_model, gnn_cfg = M["load_gnn_selector"](str(GNN_CHECKPOINT), device=DEVICE)
    print(f"  GNN model loaded successfully")

    # Load test oracle (4-class, from existing all_results.csv)
    oracle_csv = Path("results/requirements_compliant_eval/all_results.csv")
    print(f"\nLoading test oracle labels from {oracle_csv}...")
    test_oracle_df = load_test_oracle_4class(oracle_csv)
    print(f"  {len(test_oracle_df)} test oracle labels")
    for ds in sorted(test_oracle_df["dataset"].unique()):
        sub = test_oracle_df[test_oracle_df["dataset"] == ds]
        dist = sub["oracle_selector"].value_counts().to_dict()
        print(f"    {ds}: {dist}")

    # Load all datasets
    bundle = M["load_bundle"](CONFIG_PATH)
    eval_specs = M["collect_specs"](bundle, "eval_topologies")
    gen_specs = M["collect_specs"](bundle, "generalization_topologies")
    max_steps = M["max_steps_from_args"](bundle, MAX_STEPS)

    known_datasets = []
    unseen_datasets = []
    for spec in eval_specs:
        try:
            dataset, pl = M["load_named_dataset"](bundle, spec, max_steps)
            known_datasets.append((dataset, pl))
        except Exception as e:
            print(f"  Skip {spec.key}: {e}")
    for spec in gen_specs:
        try:
            dataset, pl = M["load_named_dataset"](bundle, spec, max_steps)
            unseen_datasets.append((dataset, pl))
        except Exception as e:
            print(f"  Skip {spec.key}: {e}")

    print(f"  Known topologies: {len(known_datasets)}, Unseen: {len(unseen_datasets)}")

    # ================================================================
    # PHASE 1: Compute oracle labels + features for KNOWN topologies
    # ================================================================
    print("\n" + "=" * 70)
    print("  PHASE 1: Computing 4-class oracle on KNOWN topologies (train+val)")
    print("=" * 70)

    all_train_X = []
    all_train_y = []
    all_val_X = []
    all_val_y = []

    for dataset, pl in known_datasets:
        ds_key = dataset.key
        train_indices = M["split_indices"](dataset, "train")
        val_indices = M["split_indices"](dataset, "val")

        print(f"\n  {ds_key}: computing oracle for {len(train_indices)} train TMs...")
        t0 = time.time()
        train_X, train_y, train_ts, train_mlus = compute_features_and_oracle_for_topology(
            M, dataset, pl, train_indices, gnn_model, K_CRIT,
        )
        elapsed = time.time() - t0
        print(f"    Train: {len(train_X)} samples in {elapsed:.1f}s")

        if len(train_X) > 0:
            # Show oracle distribution
            for i, name in enumerate(SELECTOR_NAMES):
                c = int(np.sum(train_y == i))
                print(f"      {name}: {c}/{len(train_y)} ({100*c/len(train_y):.0f}%)")
            all_train_X.append(train_X)
            all_train_y.append(train_y)

        print(f"    Computing val oracle for {len(val_indices)} TMs...")
        val_X, val_y, val_ts, val_mlus = compute_features_and_oracle_for_topology(
            M, dataset, pl, val_indices, gnn_model, K_CRIT,
        )
        print(f"    Val: {len(val_X)} samples")
        if len(val_X) > 0:
            all_val_X.append(val_X)
            all_val_y.append(val_y)

    # Pool all known topology data
    pooled_train_X = np.concatenate(all_train_X, axis=0)
    pooled_train_y = np.concatenate(all_train_y, axis=0)
    pooled_val_X = np.concatenate(all_val_X, axis=0) if all_val_X else None
    pooled_val_y = np.concatenate(all_val_y, axis=0) if all_val_y else None

    print(f"\n  Pooled training set: {len(pooled_train_X)} samples from {len(known_datasets)} topologies")
    print(f"  Pooled validation set: {len(pooled_val_X) if pooled_val_X is not None else 0} samples")
    print(f"  Overall oracle distribution:")
    for i, name in enumerate(SELECTOR_NAMES):
        c = int(np.sum(pooled_train_y == i))
        print(f"    {name}: {c}/{len(pooled_train_y)} ({100*c/len(pooled_train_y):.0f}%)")

    # ================================================================
    # PHASE 2: Train ONE unified MLP gate
    # ================================================================
    print("\n" + "=" * 70)
    print("  PHASE 2: Training UNIFIED MLP MetaGate")
    print("=" * 70)

    config = M["MetaGateConfig"](hidden_dim=128, dropout=0.3, learning_rate=5e-4,
                                  num_epochs=300, batch_size=64)
    gate = M["DynamicMetaGate"](config)
    train_acc, val_acc = gate.train(pooled_train_X, pooled_train_y,
                                    pooled_val_X, pooled_val_y)
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Val accuracy:   {val_acc:.3f}")

    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    gate.save(model_dir / "metagate_unified.pt")
    print(f"  Model saved to {model_dir / 'metagate_unified.pt'}")

    # ================================================================
    # PHASE 3: Evaluate on ALL topologies (known + unseen)
    # ================================================================
    print("\n" + "=" * 70)
    print("  PHASE 3: Evaluating on ALL 8 topologies")
    print("=" * 70)

    all_results = []
    all_decisions = []

    for dataset, pl in known_datasets + unseen_datasets:
        ds_key = dataset.key
        topo_type = "UNSEEN" if ds_key in UNSEEN_TOPOLOGIES else "known"
        print(f"\n  [{topo_type}] {ds_key}...")

        try:
            results, decisions = evaluate_on_topology(
                M, dataset, pl, gate, gnn_model, K_CRIT, test_oracle_df,
            )
            all_results.extend(results)
            all_decisions.extend(decisions)

            if results:
                df = pd.DataFrame(results)
                acc = df["correct"].mean()
                mg_mlu = df["metagate_mlu"].mean()
                sel_counts = df["metagate_selector"].value_counts()

                print(f"    Accuracy: {acc:.1%} ({df['correct'].sum()}/{len(df)})")
                print(f"    MetaGate MLU: {mg_mlu:.6f}")
                print(f"    Selector distribution:")
                for name in SELECTOR_NAMES:
                    c = sel_counts.get(name, 0)
                    print(f"      {name}: {c}/{len(df)} ({100*c/len(df):.0f}%)")

                # Per-timestep inference log: switch count
                changes = 0
                prev = None
                for _, row in df.iterrows():
                    if prev is not None and row["metagate_selector"] != prev:
                        changes += 1
                    prev = row["metagate_selector"]
                print(f"    Selector switches: {changes}/{len(df)-1}")

                # Timing summary
                print(f"    Timing (mean): BN={df['t_bn_ms'].mean():.1f}ms "
                      f"TopK={df['t_topk_ms'].mean():.1f}ms "
                      f"Sens={df['t_sens_ms'].mean():.1f}ms "
                      f"GNN={df['t_gnn_ms'].mean():.1f}ms "
                      f"Feat={df['t_features_ms'].mean():.2f}ms "
                      f"MLP={df['t_mlp_ms'].mean():.3f}ms "
                      f"LP={df['t_lp_ms'].mean():.1f}ms "
                      f"Total={df['t_total_ms'].mean():.1f}ms")

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ================================================================
    # PHASE 4: Save results
    # ================================================================
    if not all_results:
        print("\nNo results generated!")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "metagate_results.csv", index=False)

    decisions_df = pd.DataFrame(all_decisions)
    decisions_df.to_csv(OUTPUT_DIR / "metagate_decisions.csv", index=False)

    # Summary table
    summary = results_df.groupby("dataset").agg(
        topology_type=("topology_type", "first"),
        accuracy=("correct", "mean"),
        metagate_mlu=("metagate_mlu", "mean"),
        oracle_mlu=("oracle_mlu", "mean"),
        bn_mlu=("bn_mlu", "mean"),
        topk_mlu=("topk_mlu", "mean"),
        sens_mlu=("sens_mlu", "mean"),
        gnn_mlu=("gnn_mlu", "mean"),
        t_decision_ms=("t_decision_ms", "mean"),
        t_lp_ms=("t_lp_ms", "mean"),
        t_total_ms=("t_total_ms", "mean"),
        n_timesteps=("timestep", "count"),
    ).reset_index()

    summary["best_forced_mlu"] = summary[["bn_mlu", "topk_mlu", "sens_mlu", "gnn_mlu"]].min(axis=1)
    summary["metagate_vs_oracle_gap_pct"] = (
        (summary["metagate_mlu"] - summary["oracle_mlu"]) / summary["oracle_mlu"] * 100
    )
    summary.to_csv(OUTPUT_DIR / "metagate_summary.csv", index=False)

    # Timing summary
    timing_summary = results_df.groupby("dataset").agg(
        t_bn_ms=("t_bn_ms", "mean"),
        t_topk_ms=("t_topk_ms", "mean"),
        t_sens_ms=("t_sens_ms", "mean"),
        t_gnn_ms=("t_gnn_ms", "mean"),
        t_features_ms=("t_features_ms", "mean"),
        t_mlp_ms=("t_mlp_ms", "mean"),
        t_lp_ms=("t_lp_ms", "mean"),
        t_decision_ms=("t_decision_ms", "mean"),
        t_total_ms=("t_total_ms", "mean"),
    ).reset_index()
    timing_summary.to_csv(OUTPUT_DIR / "metagate_timing.csv", index=False)

    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY: 4-EXPERT MLP METAGATE")
    print("=" * 70)
    print(f"\n{'Topology':<28} {'Type':<7} {'Acc':>6} {'MG_MLU':>12} {'Oracle':>12} "
          f"{'BestForced':>12} {'vs_Oracle':>10} {'Decision':>8} {'Total':>8}")
    print("-" * 110)
    for _, row in summary.iterrows():
        print(f"{row['dataset']:<28} {row['topology_type']:<7} {row['accuracy']:>5.1%} "
              f"{row['metagate_mlu']:>12.6f} {row['oracle_mlu']:>12.6f} "
              f"{row['best_forced_mlu']:>12.6f} {row['metagate_vs_oracle_gap_pct']:>+9.2f}% "
              f"{row['t_decision_ms']:>7.1f}ms {row['t_total_ms']:>7.1f}ms")

    # Overall
    total_acc = results_df["correct"].mean()
    known_acc = results_df[results_df["topology_type"] == "known"]["correct"].mean()
    unseen_results = results_df[results_df["topology_type"] == "unseen"]
    unseen_acc = unseen_results["correct"].mean() if len(unseen_results) > 0 else 0.0

    print(f"\n  Overall accuracy:     {total_acc:.1%}")
    print(f"  Known topo accuracy:  {known_acc:.1%}")
    print(f"  Unseen topo accuracy: {unseen_acc:.1%}")
    print(f"\n  Mean decision time:   {results_df['t_decision_ms'].mean():.1f}ms")
    print(f"  Mean LP time:         {results_df['t_lp_ms'].mean():.1f}ms")
    print(f"  Mean total time:      {results_df['t_total_ms'].mean():.1f}ms")
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
