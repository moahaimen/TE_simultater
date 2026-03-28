#!/usr/bin/env python3
"""Evaluate the Dynamic MetaGate: train on oracle labels, test per-timestep prediction.

This script proves the meta-gate TRULY selects dynamically per timestep:
  1. For each topology: run all 3 selectors + LP on train TMs to get oracle labels
  2. Compute features, train MetaGate classifier on train split
  3. Evaluate on test split: MetaGate predicts selector -> runs that selector + LP -> MLU
  4. Compares against forced-BN, forced-TopK, forced-Sens, and oracle (always best)

Output:
  results/dynamic_metagate/metagate_results.csv       - per-timestep results
  results/dynamic_metagate/metagate_summary.csv       - per-topology summary
  results/dynamic_metagate/metagate_decisions.csv      - which selector was chosen per TM
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

CONFIG_PATH = "configs/phase1_reactive_full.yaml"
MAX_STEPS = 500
SEED = 42
K_CRIT = 40
LT = 20  # LP time limit
OUTPUT_DIR = Path("results/dynamic_metagate")
ORACLE_CSV = Path("results/requirements_compliant_eval/all_results.csv")

SELECTOR_NAMES = ["bottleneck", "topk", "sensitivity"]


def setup():
    from te.baselines import (
        ecmp_splits, select_bottleneck_critical,
        select_sensitivity_critical, select_topk_by_demand,
    )
    from te.disturbance import compute_disturbance
    from te.lp_solver import solve_selected_path_lp
    from te.simulator import apply_routing
    from phase1_reactive.eval.common import (
        load_bundle, load_named_dataset, collect_specs, max_steps_from_args,
    )
    from phase1_reactive.eval.core import split_indices
    from phase1_reactive.drl.dynamic_meta_gate import (
        DynamicMetaGate, MetaGateConfig, extract_features,
    )
    return {
        "ecmp_splits": ecmp_splits,
        "select_bottleneck_critical": select_bottleneck_critical,
        "select_sensitivity_critical": select_sensitivity_critical,
        "select_topk_by_demand": select_topk_by_demand,
        "compute_disturbance": compute_disturbance,
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
    }


def run_selector(M, method, tm, ecmp_base, path_library, capacities, k_crit):
    if method == "topk":
        return M["select_topk_by_demand"](tm, k_crit)
    elif method == "bottleneck":
        return M["select_bottleneck_critical"](tm, ecmp_base, path_library, capacities, k_crit)
    elif method == "sensitivity":
        return M["select_sensitivity_critical"](tm, ecmp_base, path_library, capacities, k_crit)
    else:
        raise ValueError(f"Unknown selector: {method}")


def load_test_oracle(oracle_csv: Path) -> pd.DataFrame:
    """Load pre-computed oracle labels for test timesteps only."""
    df = pd.read_csv(oracle_csv)
    sel_df = df[df["method"].isin(SELECTOR_NAMES)].copy()
    idx = sel_df.groupby(["dataset", "timestep"])["mlu"].idxmin()
    oracle = sel_df.loc[idx, ["dataset", "timestep", "method", "mlu"]].copy()
    oracle = oracle.rename(columns={"method": "oracle_selector", "mlu": "oracle_mlu"})
    pivot = sel_df.pivot_table(index=["dataset", "timestep"], columns="method", values="mlu").reset_index()
    pivot.columns.name = None
    for s in SELECTOR_NAMES:
        if s not in pivot.columns:
            pivot[s] = np.nan
    oracle = oracle.merge(pivot[["dataset", "timestep"] + SELECTOR_NAMES], on=["dataset", "timestep"])
    label_map = {name: i for i, name in enumerate(SELECTOR_NAMES)}
    oracle["oracle_label"] = oracle["oracle_selector"].map(label_map)
    return oracle


def compute_features_and_oracle(M, dataset, path_library, timesteps, k_crit):
    """Compute features AND oracle labels by running all 3 selectors + LP."""
    ecmp_base = M["ecmp_splits"](path_library)
    capacities = np.asarray(dataset.capacities, dtype=float)
    num_nodes = len(dataset.nodes)
    num_edges = len(dataset.edges)

    features_list = []
    labels_list = []
    valid_timesteps = []
    mlu_records = []  # for debugging

    for t_idx in timesteps:
        tm = np.asarray(dataset.tm[t_idx], dtype=float)
        if np.max(tm) < 1e-12:
            continue

        # Run all 3 selectors (fast)
        selector_results = {}
        for name in SELECTOR_NAMES:
            selected = run_selector(M, name, tm, ecmp_base, path_library, capacities, k_crit)
            selector_results[name] = selected

        # Extract features
        feats = M["extract_features"](tm, selector_results, num_nodes, num_edges, k_crit)

        # Run LP for each selector to get oracle label
        mlus = {}
        for name in SELECTOR_NAMES:
            try:
                lp = M["solve_selected_path_lp"](
                    tm, selector_results[name], ecmp_base, path_library, capacities, time_limit_sec=LT
                )
                routing = M["apply_routing"](tm, lp.splits, path_library, capacities)
                mlus[name] = float(routing.mlu)
            except Exception:
                mlus[name] = float("inf")

        # Oracle = selector with lowest MLU
        best_name = min(mlus, key=mlus.get)
        best_label = SELECTOR_NAMES.index(best_name)

        features_list.append(feats)
        labels_list.append(best_label)
        valid_timesteps.append(t_idx)
        mlu_records.append(mlus)

    if features_list:
        return np.stack(features_list), np.array(labels_list, dtype=np.int64), valid_timesteps, mlu_records
    return np.zeros((0, 23)), np.array([], dtype=np.int64), [], []


def compute_features_only(M, dataset, path_library, timesteps, k_crit):
    """Compute features only (no LP), for test-time inference."""
    ecmp_base = M["ecmp_splits"](path_library)
    capacities = np.asarray(dataset.capacities, dtype=float)
    num_nodes = len(dataset.nodes)
    num_edges = len(dataset.edges)

    features_list = []
    selector_cache = {}  # t_idx -> selector_results
    valid_timesteps = []

    for t_idx in timesteps:
        tm = np.asarray(dataset.tm[t_idx], dtype=float)
        if np.max(tm) < 1e-12:
            continue

        selector_results = {}
        for name in SELECTOR_NAMES:
            selected = run_selector(M, name, tm, ecmp_base, path_library, capacities, k_crit)
            selector_results[name] = selected

        feats = M["extract_features"](tm, selector_results, num_nodes, num_edges, k_crit)
        features_list.append(feats)
        selector_cache[t_idx] = selector_results
        valid_timesteps.append(t_idx)

    if features_list:
        return np.stack(features_list), valid_timesteps, selector_cache
    return np.zeros((0, 23)), [], {}


def evaluate_metagate_on_topology(M, dataset, path_library, test_oracle_df, k_crit):
    """Train and evaluate the dynamic meta-gate on one topology."""
    ds_key = dataset.key
    print(f"\n{'='*60}")
    print(f"  Topology: {ds_key}")
    print(f"{'='*60}")

    train_indices = M["split_indices"](dataset, "train")
    val_indices = M["split_indices"](dataset, "val")
    test_indices = M["split_indices"](dataset, "test")

    # Step 1: Compute features + oracle labels for train+val (runs LP for each selector)
    print(f"  Step 1: Computing train oracle labels ({len(train_indices)} TMs x 3 selectors + LP)...")
    t0 = time.time()
    train_X, train_y, train_ts, train_mlus = compute_features_and_oracle(
        M, dataset, path_library, train_indices, k_crit
    )
    train_time = time.time() - t0
    print(f"    Done in {train_time:.1f}s, {len(train_X)} valid samples")

    if len(train_X) < 10:
        print(f"  Only {len(train_X)} training samples, skipping")
        return [], []

    print(f"  Computing val oracle labels ({len(val_indices)} TMs)...")
    val_X, val_y, val_ts, val_mlus = compute_features_and_oracle(
        M, dataset, path_library, val_indices, k_crit
    )
    print(f"    {len(val_X)} valid val samples")

    # Show oracle distribution
    for i, name in enumerate(SELECTOR_NAMES):
        count = int(np.sum(train_y == i))
        pct = 100 * count / len(train_y) if len(train_y) > 0 else 0
        print(f"    Oracle train dist: {name} = {count}/{len(train_y)} ({pct:.0f}%)")

    # Step 2: Train meta-gate
    print(f"  Step 2: Training MetaGate on {len(train_X)} samples...")
    config = M["MetaGateConfig"](hidden_dim=64, dropout=0.2, learning_rate=1e-3, num_epochs=150, batch_size=32)
    gate = M["DynamicMetaGate"](config)
    train_acc, val_acc = gate.train(
        train_X, train_y,
        val_X if len(val_X) > 0 else None,
        val_y if len(val_y) > 0 else None,
    )
    print(f"    Train accuracy: {train_acc:.3f}, Val accuracy: {val_acc:.3f}")

    # Save model
    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    gate.save(model_dir / f"metagate_{ds_key}.pt")

    # Step 3: Evaluate on test split
    print(f"  Step 3: Evaluating on {len(test_indices)} test TMs...")
    ecmp_base = M["ecmp_splits"](path_library)
    capacities = np.asarray(dataset.capacities, dtype=float)
    num_nodes = len(dataset.nodes)
    num_edges = len(dataset.edges)

    # Get pre-computed test oracle
    topo_oracle = test_oracle_df[test_oracle_df["dataset"] == ds_key].copy()
    oracle_by_ts = topo_oracle.set_index("timestep") if not topo_oracle.empty else pd.DataFrame()

    results = []
    decisions = []

    for t_idx in test_indices:
        tm = np.asarray(dataset.tm[t_idx], dtype=float)
        if np.max(tm) < 1e-12:
            continue

        # Run all 3 selectors (fast)
        selector_results = {}
        for name in SELECTOR_NAMES:
            selected = run_selector(M, name, tm, ecmp_base, path_library, capacities, k_crit)
            selector_results[name] = selected

        # Extract features and predict
        feats = M["extract_features"](tm, selector_results, num_nodes, num_edges, k_crit)
        pred_class, probs = gate.predict(feats)
        pred_name = SELECTOR_NAMES[pred_class]
        confidence = float(probs[pred_class])

        # Run predicted selector + LP
        t_start = time.perf_counter()
        selected_ods = selector_results[pred_name]
        lp = M["solve_selected_path_lp"](tm, selected_ods, ecmp_base, path_library, capacities, time_limit_sec=LT)
        routing = M["apply_routing"](tm, lp.splits, path_library, capacities)
        metagate_time = (time.perf_counter() - t_start) * 1000
        metagate_mlu = float(routing.mlu)

        # Get oracle info from pre-computed CSV
        oracle_selector = "unknown"
        oracle_mlu = metagate_mlu
        bn_mlu = np.nan
        topk_mlu = np.nan
        sens_mlu = np.nan

        if not oracle_by_ts.empty and t_idx in oracle_by_ts.index:
            row = oracle_by_ts.loc[t_idx]
            oracle_selector = str(row["oracle_selector"])
            oracle_mlu = float(row["oracle_mlu"])
            bn_mlu = float(row["bottleneck"])
            topk_mlu = float(row["topk"])
            sens_mlu = float(row["sensitivity"])

        results.append({
            "dataset": ds_key,
            "timestep": int(t_idx),
            "metagate_selector": pred_name,
            "metagate_confidence": confidence,
            "metagate_mlu": metagate_mlu,
            "oracle_selector": oracle_selector,
            "oracle_mlu": oracle_mlu,
            "bn_mlu": bn_mlu,
            "topk_mlu": topk_mlu,
            "sens_mlu": sens_mlu,
            "exec_time_ms": metagate_time,
            "correct": 1 if pred_name == oracle_selector else 0,
        })

        decisions.append({
            "dataset": ds_key,
            "timestep": int(t_idx),
            "predicted": pred_name,
            "oracle": oracle_selector,
            "confidence": confidence,
            "prob_bn": float(probs[0]),
            "prob_topk": float(probs[1]),
            "prob_sens": float(probs[2]),
        })

    if results:
        df = pd.DataFrame(results)
        accuracy = df["correct"].mean()
        mean_mg_mlu = df["metagate_mlu"].mean()
        mean_oracle_mlu = df["oracle_mlu"].mean()
        mean_bn_mlu = df["bn_mlu"].mean()
        mean_topk_mlu = df["topk_mlu"].mean()
        mean_sens_mlu = df["sens_mlu"].mean()

        sel_counts = df["metagate_selector"].value_counts()

        print(f"\n  --- Results for {ds_key} ---")
        print(f"  MetaGate accuracy:    {accuracy:.1%} ({df['correct'].sum()}/{len(df)})")
        print(f"  MetaGate mean MLU:    {mean_mg_mlu:.6f}")
        print(f"  Oracle mean MLU:      {mean_oracle_mlu:.6f}")
        print(f"  Forced BN mean MLU:   {mean_bn_mlu:.6f}")
        print(f"  Forced TopK mean MLU: {mean_topk_mlu:.6f}")
        print(f"  Forced Sens mean MLU: {mean_sens_mlu:.6f}")
        print(f"  Selector distribution:")
        for name in SELECTOR_NAMES:
            c = sel_counts.get(name, 0)
            print(f"    {name}: {c}/{len(df)} ({100*c/len(df):.0f}%)")

        # Show the meta-gate REALLY changes per timestep
        changes = 0
        prev = None
        for _, row in df.iterrows():
            if prev is not None and row["metagate_selector"] != prev:
                changes += 1
            prev = row["metagate_selector"]
        print(f"  Selector switches:    {changes}/{len(df)-1} transitions ({100*changes/max(len(df)-1,1):.0f}% dynamic)")

    return results, decisions


def main():
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DYNAMIC METAGATE EVALUATION")
    print("  Per-timestep selector among {Bottleneck, TopK, Sensitivity}")
    print("=" * 70)

    M = setup()

    # Load pre-computed test oracle labels
    print(f"\nLoading test oracle labels from {ORACLE_CSV}...")
    test_oracle_df = load_test_oracle(ORACLE_CSV)
    print(f"  {len(test_oracle_df)} test oracle labels across {test_oracle_df['dataset'].nunique()} topologies")

    # Load all datasets
    bundle = M["load_bundle"](CONFIG_PATH)
    eval_specs = M["collect_specs"](bundle, "eval_topologies")
    gen_specs = M["collect_specs"](bundle, "generalization_topologies")
    max_steps = M["max_steps_from_args"](bundle, MAX_STEPS)

    all_datasets = []
    for spec in eval_specs + gen_specs:
        try:
            dataset, pl = M["load_named_dataset"](bundle, spec, max_steps)
            all_datasets.append((dataset, pl))
        except Exception as e:
            print(f"  Skip {spec.key}: {e}")

    print(f"  Loaded {len(all_datasets)} topologies")

    # Evaluate on each topology
    all_results = []
    all_decisions = []

    for dataset, pl in all_datasets:
        try:
            results, decisions = evaluate_metagate_on_topology(
                M, dataset, pl, test_oracle_df, K_CRIT
            )
            all_results.extend(results)
            all_decisions.extend(decisions)
        except Exception as e:
            print(f"\n  FAILED on {dataset.key}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_DIR / "metagate_results.csv", index=False)

        decisions_df = pd.DataFrame(all_decisions)
        decisions_df.to_csv(OUTPUT_DIR / "metagate_decisions.csv", index=False)

        # Summary table
        summary = results_df.groupby("dataset").agg(
            accuracy=("correct", "mean"),
            metagate_mlu=("metagate_mlu", "mean"),
            oracle_mlu=("oracle_mlu", "mean"),
            bn_mlu=("bn_mlu", "mean"),
            topk_mlu=("topk_mlu", "mean"),
            sens_mlu=("sens_mlu", "mean"),
            mean_exec_ms=("exec_time_ms", "mean"),
            n_timesteps=("timestep", "count"),
        ).reset_index()

        # Best forced = min(forced_bn, forced_topk, forced_sens) mean
        summary["best_forced_mlu"] = summary[["bn_mlu", "topk_mlu", "sens_mlu"]].min(axis=1)
        summary["metagate_vs_best_forced_pct"] = (
            (summary["best_forced_mlu"] - summary["metagate_mlu"]) / summary["best_forced_mlu"] * 100
        )
        summary["metagate_vs_oracle_gap_pct"] = (
            (summary["metagate_mlu"] - summary["oracle_mlu"]) / summary["oracle_mlu"] * 100
        )

        summary.to_csv(OUTPUT_DIR / "metagate_summary.csv", index=False)

        print("\n" + "=" * 70)
        print("  FINAL SUMMARY: DYNAMIC METAGATE")
        print("=" * 70)
        print(f"\n{'Topology':<25} {'Acc':>6} {'MG_MLU':>10} {'Oracle':>10} {'BestForced':>10} {'vs_Oracle':>10}")
        print("-" * 75)
        for _, row in summary.iterrows():
            print(f"{row['dataset']:<25} {row['accuracy']:>5.1%} {row['metagate_mlu']:>10.6f} "
                  f"{row['oracle_mlu']:>10.6f} {row['best_forced_mlu']:>10.6f} "
                  f"{row['metagate_vs_oracle_gap_pct']:>+9.2f}%")

        # Overall stats
        total_acc = results_df["correct"].mean()
        total_mg = results_df["metagate_mlu"].mean()
        total_oracle = results_df["oracle_mlu"].mean()
        print(f"\n  Overall accuracy: {total_acc:.1%}")
        print(f"  Overall MetaGate MLU: {total_mg:.6f}")
        print(f"  Overall Oracle MLU:   {total_oracle:.6f}")
        print(f"\nResults saved to {OUTPUT_DIR}/")

    else:
        print("\nNo results generated!")


if __name__ == "__main__":
    main()
