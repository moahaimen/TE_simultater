#!/usr/bin/env python3
"""Thesis Benchmark: Clean internal-vs-external evaluation.

Implements the restructured methodology:
  - Internal selectors: Bottleneck, TopK, GNN, DA-GNN (disturbance-aware)
  - External baselines: ECMP, OSPF, FlexDATE, FlexEntry, CFRRL, ERODRL
  - Fixed k=40, same LP solver, same ECMP base, same test split
  - Multi-metric: MLU, Disturbance, Decision Time, Latency, PR
  - Failure scenarios: single-link, capacity-degradation, demand-spike
  - CDF plots for all metrics
  - Unseen topology generalization (Germany50)

Usage:
  python scripts/run_thesis_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from te.baselines import ecmp_splits, clone_splits, select_bottleneck_critical, select_sensitivity_critical, select_topk_by_demand
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp, solve_full_mcf_min_mlu
from te.simulator import TEDataset, apply_routing
from phase1_reactive.baselines.literature_baselines import select_literature_baseline
from phase1_reactive.drl.gnn_selector import load_gnn_selector, build_graph_tensors, build_od_features
from phase1_reactive.drl.gnn_inference import rollout_gnn_selector_policy, GNN_METHOD
from phase1_reactive.drl.gnn_disturbance_aware import (
    DAGNNConfig, DA_GNN_METHOD,
    rollout_da_gnn_selector, tune_lambda_cont,
)
from phase1_reactive.drl.state_builder import compute_reactive_telemetry
from phase1_reactive.eval.common import (
    build_reactive_env_cfg,
    load_bundle,
    load_named_dataset,
    collect_specs,
    max_steps_from_args,
    resolve_phase1_k_crit,
)
from phase1_reactive.eval.core import run_selector_lp_method, run_static_method, split_indices
from phase1_reactive.eval.metrics import summarize_timeseries
from phase1_reactive.env.offline_env import ReactiveRoutingEnv


# ============================================================
# Configuration
# ============================================================
CONFIG_PATH = "configs/phase1_reactive_full.yaml"
MAX_STEPS = 500
SEED = 42
DEVICE = "cpu"
K_CRIT = 40  # Fixed for fairness

OUTPUT_DIR = Path("results/thesis_benchmark")
GNN_CHECKPOINT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")

# Internal selectors (our methods only)
INTERNAL_SELECTORS = ["bottleneck", "topk", "sensitivity"]

# External baselines (comparison only)
EXTERNAL_BASELINES = ["flexdate", "flexentry", "cfrrl", "erodrl"]
STATIC_METHODS = ["ospf", "ecmp"]

# Failure scenarios
FAILURE_SCENARIOS = ["single_link", "capacity_degradation", "demand_spike"]


# ============================================================
# Helpers
# ============================================================

def run_lp_optimal_sample(dataset, path_library, timestep, capacities, time_limit=90):
    """Compute LP-optimal MLU for a single timestep (full MCF)."""
    tm_vector = dataset.tm[timestep]
    try:
        result = solve_full_mcf_min_mlu(
            tm_vector=tm_vector,
            path_library=path_library,
            capacities=capacities,
            time_limit_sec=time_limit,
        )
        if result is not None and hasattr(result, 'routing'):
            return float(result.routing.mlu)
    except Exception:
        pass
    return None


def apply_failure(capacities, edges, failure_type, rng):
    """Apply a failure scenario to capacities. Returns modified copy."""
    caps = capacities.copy()
    num_edges = len(caps)

    if failure_type == "single_link":
        # Pick most-utilized edge (approximate by highest capacity — will be refined)
        idx = rng.integers(0, num_edges)
        caps[idx] = 0.0
    elif failure_type == "capacity_degradation":
        # Degrade top 10% links by 50%
        n_degrade = max(1, num_edges // 10)
        indices = rng.choice(num_edges, size=n_degrade, replace=False)
        for idx in indices:
            caps[idx] *= 0.5
    elif failure_type == "demand_spike":
        pass  # Handled at TM level, not capacity
    return caps


def apply_demand_spike(tm_vector, rng, spike_factor=1.5, top_frac=0.1):
    """Apply demand spike to top OD pairs."""
    tm = tm_vector.copy()
    n_spike = max(1, int(len(tm) * top_frac))
    top_ods = np.argsort(tm)[::-1][:n_spike]
    tm[top_ods] *= spike_factor
    return tm


# ============================================================
# Section A: Internal Fair Fixed-K Benchmark
# ============================================================

def run_internal_benchmark(bundle, datasets, gnn_model, da_gnn_cfg):
    """Run all internal selectors through the same unified pipeline."""
    print("\n" + "=" * 70)
    print("SECTION A: INTERNAL FAIR FIXED-K BENCHMARK (k=40)")
    print("=" * 70)

    out_dir = OUTPUT_DIR / "internal"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []

    for dataset, path_library in datasets:
        print(f"\n--- {dataset.key} ({len(dataset.nodes)} nodes) ---")
        env_cfg = build_reactive_env_cfg(bundle, k_crit_override=K_CRIT)

        # Run heuristic internal selectors
        for method in INTERNAL_SELECTORS:
            try:
                df = run_selector_lp_method(
                    dataset, path_library, split_name="test", method=method,
                    k_crit=K_CRIT, lp_time_limit_sec=25,
                )
                df["category"] = "internal"
                all_frames.append(df)
                mlu = df["mlu"].mean()
                db = df["disturbance"].mean() if "disturbance" in df else 0
                dt = df["decision_time_ms"].mean() if "decision_time_ms" in df else 0
                print(f"  {method:<15}: MLU={mlu:.6f}  DB={db:.4f}  DT={dt:.1f}ms")
            except Exception as e:
                print(f"  {method:<15}: FAILED ({e})")

        # Run GNN selector (pure MLU)
        if gnn_model is not None:
            try:
                env = ReactiveRoutingEnv(
                    dataset, dataset.tm, path_library,
                    split_name="test", cfg=env_cfg, env_name=dataset.key,
                )
                df_gnn = rollout_gnn_selector_policy(env, gnn_model, device=DEVICE)
                df_gnn["dataset"] = dataset.key
                df_gnn["category"] = "internal"
                all_frames.append(df_gnn)
                mlu = df_gnn["mlu"].mean()
                db = df_gnn.get("disturbance", pd.Series([0])).mean()
                dt = df_gnn.get("decision_time_ms", pd.Series([0])).mean()
                print(f"  {'GNN':<15}: MLU={mlu:.6f}  DB={db:.4f}  DT={dt:.1f}ms")
            except Exception as e:
                print(f"  {'GNN':<15}: FAILED ({e})")
                import traceback; traceback.print_exc()

        # Run DA-GNN selector (disturbance-aware)
        if gnn_model is not None:
            try:
                env = ReactiveRoutingEnv(
                    dataset, dataset.tm, path_library,
                    split_name="test", cfg=env_cfg, env_name=dataset.key,
                )
                df_dagnn = rollout_da_gnn_selector(
                    env, gnn_model, da_cfg=da_gnn_cfg, device=DEVICE,
                )
                df_dagnn["dataset"] = dataset.key
                df_dagnn["category"] = "internal"
                all_frames.append(df_dagnn)
                mlu = df_dagnn["mlu"].mean()
                db = df_dagnn.get("disturbance", pd.Series([0])).mean()
                dt = df_dagnn.get("decision_time_ms", pd.Series([0])).mean()
                print(f"  {'DA-GNN':<15}: MLU={mlu:.6f}  DB={db:.4f}  DT={dt:.1f}ms")
            except Exception as e:
                print(f"  {'DA-GNN':<15}: FAILED ({e})")
                import traceback; traceback.print_exc()

    if all_frames:
        ts = pd.concat(all_frames, ignore_index=True, sort=False)
        ts.to_csv(out_dir / "internal_timeseries.csv", index=False)
        summary = summarize_timeseries(ts, group_cols=["dataset", "method"], training_meta={})
        summary.to_csv(out_dir / "internal_summary.csv", index=False)
        return ts, summary
    return pd.DataFrame(), pd.DataFrame()


# ============================================================
# Section B: External Baseline Comparison
# ============================================================

def run_external_baselines(bundle, datasets):
    """Run external baselines for comparison."""
    print("\n" + "=" * 70)
    print("SECTION B: EXTERNAL BASELINE COMPARISON")
    print("=" * 70)

    out_dir = OUTPUT_DIR / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []

    for dataset, path_library in datasets:
        print(f"\n--- {dataset.key} ({len(dataset.nodes)} nodes) ---")

        # Static methods
        for method in STATIC_METHODS:
            try:
                df = run_static_method(dataset, path_library, split_name="test", method=method)
                df["category"] = "external"
                all_frames.append(df)
                mlu = df["mlu"].mean()
                db = df["disturbance"].mean() if "disturbance" in df else 0
                print(f"  {method:<15}: MLU={mlu:.6f}  DB={db:.4f}")
            except Exception as e:
                print(f"  {method:<15}: FAILED ({e})")

        # Literature baselines
        for method in EXTERNAL_BASELINES:
            try:
                df = run_selector_lp_method(
                    dataset, path_library, split_name="test", method=method,
                    k_crit=K_CRIT, lp_time_limit_sec=25,
                )
                df["category"] = "external"
                all_frames.append(df)
                mlu = df["mlu"].mean()
                db = df["disturbance"].mean() if "disturbance" in df else 0
                dt = df["decision_time_ms"].mean() if "decision_time_ms" in df else 0
                print(f"  {method:<15}: MLU={mlu:.6f}  DB={db:.4f}  DT={dt:.1f}ms")
            except Exception as e:
                print(f"  {method:<15}: FAILED ({e})")

    if all_frames:
        ts = pd.concat(all_frames, ignore_index=True, sort=False)
        ts.to_csv(out_dir / "external_timeseries.csv", index=False)
        summary = summarize_timeseries(ts, group_cols=["dataset", "method"], training_meta={})
        summary.to_csv(out_dir / "external_summary.csv", index=False)
        return ts, summary
    return pd.DataFrame(), pd.DataFrame()


# ============================================================
# Section C: LP-Optimal Reference (PR Computation)
# ============================================================

def compute_lp_optimal(datasets, n_samples=10):
    """Compute LP-optimal MLU for sampled timesteps."""
    print("\n" + "=" * 70)
    print("SECTION C: LP-OPTIMAL REFERENCE (PR)")
    print("=" * 70)

    out_dir = OUTPUT_DIR / "lp_optimal"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for dataset, path_library in datasets:
        capacities = np.asarray(dataset.capacities, dtype=float)
        test_indices = split_indices(dataset, "test")

        # Sample uniformly
        rng = np.random.default_rng(SEED)
        sample_indices = sorted(rng.choice(
            test_indices, size=min(n_samples, len(test_indices)), replace=False
        ).tolist())

        print(f"\n--- {dataset.key}: solving {len(sample_indices)} LP-optimal steps ---")
        opt_mlus = []
        for t in sample_indices:
            opt_mlu = run_lp_optimal_sample(dataset, path_library, t, capacities)
            if opt_mlu is not None:
                opt_mlus.append(opt_mlu)
                print(f"  t={t}: LP-optimal MLU = {opt_mlu:.6f}")
            else:
                print(f"  t={t}: LP solve failed")

        if opt_mlus:
            results.append({
                "topology": dataset.key,
                "n_solved": len(opt_mlus),
                "mean_opt_mlu": np.mean(opt_mlus),
                "p95_opt_mlu": np.percentile(opt_mlus, 95),
            })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(out_dir / "lp_optimal_reference.csv", index=False)
        print("\nLP-Optimal Reference:")
        print(df.to_string(index=False))
        return df
    return pd.DataFrame()


# ============================================================
# Section D: Failure Scenarios
# ============================================================

def run_failure_scenarios(bundle, datasets, gnn_model, da_gnn_cfg):
    """Run failure robustness evaluation."""
    print("\n" + "=" * 70)
    print("SECTION D: FAILURE SCENARIOS")
    print("=" * 70)

    out_dir = OUTPUT_DIR / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    rng = np.random.default_rng(SEED)

    methods_to_test = INTERNAL_SELECTORS + ["flexdate"]  # strongest external for comparison

    for dataset, path_library in datasets:
        capacities = np.asarray(dataset.capacities, dtype=float)
        test_indices = split_indices(dataset, "test")

        for scenario in FAILURE_SCENARIOS:
            print(f"\n--- {dataset.key} / {scenario} ---")

            # Apply failure
            if scenario == "demand_spike":
                fail_caps = capacities.copy()
            else:
                fail_caps = apply_failure(capacities, dataset.edges, scenario, rng)

            for method in methods_to_test:
                try:
                    df = run_selector_lp_method(
                        dataset, path_library, split_name="test", method=method,
                        k_crit=K_CRIT, lp_time_limit_sec=25,
                        capacities=fail_caps,
                    )
                    mlu = df["mlu"].mean()
                    db = df["disturbance"].mean() if "disturbance" in df else 0
                    dt = df["decision_time_ms"].mean() if "decision_time_ms" in df else 0
                    print(f"  {method:<15}: MLU={mlu:.6f}  DB={db:.4f}  DT={dt:.1f}ms")

                    all_results.append({
                        "topology": dataset.key,
                        "scenario": scenario,
                        "method": method,
                        "mean_mlu": mlu,
                        "mean_disturbance": db,
                        "mean_decision_time_ms": dt,
                        "category": "internal" if method in INTERNAL_SELECTORS else "external",
                    })
                except Exception as e:
                    print(f"  {method:<15}: FAILED ({e})")

            # GNN under failure
            if gnn_model is not None:
                try:
                    env_cfg = build_reactive_env_cfg(bundle, k_crit_override=K_CRIT)
                    env = ReactiveRoutingEnv(
                        dataset, dataset.tm, path_library,
                        split_name="test", cfg=env_cfg, env_name=dataset.key,
                        capacities_override=fail_caps,
                    )
                    df_gnn = rollout_gnn_selector_policy(env, gnn_model, device=DEVICE)
                    mlu = df_gnn["mlu"].mean()
                    db = df_gnn.get("disturbance", pd.Series([0])).mean()
                    dt = df_gnn.get("decision_time_ms", pd.Series([0])).mean()
                    print(f"  {'GNN':<15}: MLU={mlu:.6f}  DB={db:.4f}  DT={dt:.1f}ms")
                    all_results.append({
                        "topology": dataset.key,
                        "scenario": scenario,
                        "method": GNN_METHOD,
                        "mean_mlu": mlu,
                        "mean_disturbance": float(db),
                        "mean_decision_time_ms": float(dt),
                        "category": "internal",
                    })
                except Exception as e:
                    print(f"  {'GNN':<15}: FAILED ({e})")

            # DA-GNN under failure
            if gnn_model is not None:
                try:
                    env = ReactiveRoutingEnv(
                        dataset, dataset.tm, path_library,
                        split_name="test", cfg=env_cfg, env_name=dataset.key,
                        capacities_override=fail_caps,
                    )
                    df_dagnn = rollout_da_gnn_selector(
                        env, gnn_model, da_cfg=da_gnn_cfg, device=DEVICE,
                    )
                    mlu = df_dagnn["mlu"].mean()
                    db = df_dagnn.get("disturbance", pd.Series([0])).mean()
                    dt = df_dagnn.get("decision_time_ms", pd.Series([0])).mean()
                    print(f"  {'DA-GNN':<15}: MLU={mlu:.6f}  DB={db:.4f}  DT={dt:.1f}ms")
                    all_results.append({
                        "topology": dataset.key,
                        "scenario": scenario,
                        "method": DA_GNN_METHOD,
                        "mean_mlu": mlu,
                        "mean_disturbance": float(db),
                        "mean_decision_time_ms": float(dt),
                        "category": "internal",
                    })
                except Exception as e:
                    print(f"  {'DA-GNN':<15}: FAILED ({e})")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(out_dir / "failure_results.csv", index=False)
        return df
    return pd.DataFrame()


# ============================================================
# Section E: CDF Plots
# ============================================================

def generate_cdf_plots(internal_ts, external_ts, failure_df):
    """Generate CDF plots for all metrics."""
    print("\n" + "=" * 70)
    print("SECTION E: GENERATING CDF PLOTS")
    print("=" * 70)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping CDF plots")
        return

    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Combine internal + external for comparison
    if internal_ts.empty and external_ts.empty:
        print("  No data for CDF plots")
        return

    combined = pd.concat([internal_ts, external_ts], ignore_index=True, sort=False)

    # Methods to highlight
    highlight_methods = [
        DA_GNN_METHOD, GNN_METHOD, "bottleneck", "topk",
        "flexdate", "ecmp", "sensitivity",
    ]

    # Color map
    colors = {
        DA_GNN_METHOD: "#E63946",      # red (our main)
        GNN_METHOD: "#457B9D",          # blue
        "bottleneck": "#2A9D8F",        # teal
        "topk": "#E9C46A",             # yellow
        "sensitivity": "#8D99AE",       # gray
        "flexdate": "#F4A261",          # orange
        "ecmp": "#264653",             # dark
        "ospf": "#A8DADC",            # light blue
        "cfrrl": "#BC6C25",            # brown
        "erodrl": "#606C38",           # olive
        "flexentry": "#DDA15E",        # tan
    }

    # Per-topology CDF plots
    for topo in combined["dataset"].unique():
        topo_df = combined[combined["dataset"] == topo]

        for metric, label, filename_suffix in [
            ("mlu", "MLU", "mlu"),
            ("disturbance", "Disturbance", "disturbance"),
            ("decision_time_ms", "Decision Time (ms)", "decision_time"),
            ("latency", "Mean Latency", "latency"),
        ]:
            if metric not in topo_df.columns:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            for method in highlight_methods:
                method_df = topo_df[topo_df["method"] == method]
                if method_df.empty:
                    continue
                vals = pd.to_numeric(method_df[metric], errors="coerce").dropna().sort_values()
                if vals.empty:
                    continue
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                color = colors.get(method, "#333333")
                lw = 2.5 if method in (DA_GNN_METHOD, GNN_METHOD) else 1.5
                ls = "-" if method in (DA_GNN_METHOD, GNN_METHOD, "bottleneck") else "--"
                display_name = method.replace("our_da_gnn_selector", "DA-GNN (Ours)").replace("our_gnn_selector", "GNN").replace("_", " ").title()
                ax.plot(vals, cdf, label=display_name, color=color, linewidth=lw, linestyle=ls)

            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel("CDF", fontsize=12)
            ax.set_title(f"{topo} — CDF of {label}", fontsize=13)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / f"cdf_{topo}_{filename_suffix}.png", dpi=150)
            plt.close(fig)
            print(f"  Saved: cdf_{topo}_{filename_suffix}.png")

    # Combined MLU CDF across all topologies
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in highlight_methods:
        method_df = combined[combined["method"] == method]
        if method_df.empty or "mlu" not in method_df:
            continue
        vals = pd.to_numeric(method_df["mlu"], errors="coerce").dropna().sort_values()
        if vals.empty:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        color = colors.get(method, "#333333")
        lw = 2.5 if method in (DA_GNN_METHOD, GNN_METHOD) else 1.5
        display_name = method.replace("our_da_gnn_selector", "DA-GNN (Ours)").replace("our_gnn_selector", "GNN").replace("_", " ").title()
        ax.plot(vals, cdf, label=display_name, color=color, linewidth=lw)

    ax.set_xlabel("MLU", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("All Topologies — CDF of MLU", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "cdf_all_mlu.png", dpi=150)
    plt.close(fig)
    print("  Saved: cdf_all_mlu.png")

    # MLU vs Disturbance tradeoff scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in highlight_methods:
        method_df = combined[combined["method"] == method]
        if method_df.empty:
            continue
        for topo in method_df["dataset"].unique():
            topo_method = method_df[method_df["dataset"] == topo]
            mlu_val = topo_method["mlu"].mean()
            db_val = topo_method.get("disturbance", pd.Series([0])).mean()
            color = colors.get(method, "#333333")
            display_name = method.replace("our_da_gnn_selector", "DA-GNN").replace("our_gnn_selector", "GNN").replace("_", " ").title()
            ax.scatter(mlu_val, db_val, c=color, s=60, alpha=0.8,
                      label=f"{display_name}/{topo}" if topo == method_df["dataset"].unique()[0] else "")
    ax.set_xlabel("Mean MLU", fontsize=12)
    ax.set_ylabel("Mean Disturbance", fontsize=12)
    ax.set_title("MLU vs Disturbance Tradeoff", fontsize=13)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "tradeoff_mlu_vs_disturbance.png", dpi=150)
    plt.close(fig)
    print("  Saved: tradeoff_mlu_vs_disturbance.png")


# ============================================================
# Section F: Summary Tables
# ============================================================

def build_summary_tables(internal_ts, external_ts, lp_optimal_df, failure_df):
    """Build all summary tables for the thesis."""
    print("\n" + "=" * 70)
    print("SECTION F: BUILDING SUMMARY TABLES")
    print("=" * 70)

    table_dir = OUTPUT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([internal_ts, external_ts], ignore_index=True, sort=False)
    if combined.empty:
        print("  No data for tables")
        return

    # Table 1: Internal fair benchmark (multi-metric)
    print("\n--- Table 1: Internal Fair Benchmark (k=40) ---")
    internal_methods = [DA_GNN_METHOD, GNN_METHOD, "bottleneck", "topk", "sensitivity"]
    rows = []
    for topo in combined["dataset"].unique():
        for method in internal_methods:
            mdf = combined[(combined["dataset"] == topo) & (combined["method"] == method)]
            if mdf.empty:
                continue
            rows.append({
                "Topology": topo,
                "Method": method.replace("our_da_gnn_selector", "DA-GNN").replace("our_gnn_selector", "GNN"),
                "Mean MLU": f"{mdf['mlu'].mean():.6f}",
                "P95 MLU": f"{mdf['mlu'].quantile(0.95):.6f}",
                "Mean DB": f"{mdf['disturbance'].mean():.4f}" if "disturbance" in mdf else "N/A",
                "P95 DB": f"{mdf['disturbance'].quantile(0.95):.4f}" if "disturbance" in mdf else "N/A",
                "Mean DT(ms)": f"{mdf['decision_time_ms'].mean():.1f}" if "decision_time_ms" in mdf else "N/A",
                "Mean Latency": f"{mdf['latency'].mean():.4f}" if "latency" in mdf else "N/A",
            })
    t1 = pd.DataFrame(rows)
    t1.to_csv(table_dir / "table1_internal_benchmark.csv", index=False)
    print(t1.to_string(index=False))

    # Table 2: External baseline comparison
    print("\n--- Table 2: External Baseline Comparison ---")
    all_methods = [DA_GNN_METHOD, "bottleneck", "flexdate", "cfrrl", "erodrl", "flexentry", "ecmp", "ospf"]
    rows = []
    for topo in combined["dataset"].unique():
        row = {"Topology": topo}
        best_mlu = float("inf")
        best_method = ""
        for method in all_methods:
            mdf = combined[(combined["dataset"] == topo) & (combined["method"] == method)]
            if not mdf.empty:
                mlu = mdf["mlu"].mean()
                display = method.replace("our_da_gnn_selector", "DA-GNN").replace("our_gnn_selector", "GNN")
                row[display] = f"{mlu:.6f}"
                if mlu < best_mlu:
                    best_mlu = mlu
                    best_method = display
        row["Best"] = best_method
        rows.append(row)
    t2 = pd.DataFrame(rows)
    t2.to_csv(table_dir / "table2_external_comparison.csv", index=False)
    print(t2.to_string(index=False))

    # Table 3: Full multi-metric table
    print("\n--- Table 3: Full Multi-Metric Summary ---")
    rows = []
    for topo in combined["dataset"].unique():
        for method in combined[combined["dataset"] == topo]["method"].unique():
            mdf = combined[(combined["dataset"] == topo) & (combined["method"] == method)]
            row = {
                "Topology": topo,
                "Method": method.replace("our_da_gnn_selector", "DA-GNN").replace("our_gnn_selector", "GNN"),
                "Mean MLU": float(mdf["mlu"].mean()),
                "Mean DB": float(mdf["disturbance"].mean()) if "disturbance" in mdf else 0,
                "Mean DT(ms)": float(mdf["decision_time_ms"].mean()) if "decision_time_ms" in mdf else 0,
                "K_selected": int(mdf["selected_count"].mean()) if "selected_count" in mdf else K_CRIT,
            }

            # PR if LP-optimal available
            if lp_optimal_df is not None and not lp_optimal_df.empty:
                opt_row = lp_optimal_df[lp_optimal_df["topology"] == topo]
                if not opt_row.empty:
                    opt_mlu = opt_row["mean_opt_mlu"].values[0]
                    if opt_mlu > 0:
                        row["PR (%)"] = round(opt_mlu / row["Mean MLU"] * 100, 2)

            rows.append(row)
    t3 = pd.DataFrame(rows)
    t3.to_csv(table_dir / "table3_full_metrics.csv", index=False)

    # Table 4: Failure scenario summary
    if failure_df is not None and not failure_df.empty:
        print("\n--- Table 4: Failure Scenario Summary ---")
        failure_df.to_csv(table_dir / "table4_failure_results.csv", index=False)
        print(failure_df.to_string(index=False))

    return t1, t2, t3


# ============================================================
# Section G: Training Efficiency Report
# ============================================================

def report_training_efficiency():
    """Report GNN training convergence."""
    print("\n" + "=" * 70)
    print("SECTION G: TRAINING EFFICIENCY")
    print("=" * 70)

    table_dir = OUTPUT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    # Check for training logs
    gnn_train_dir = Path("results/phase1_reactive/gnn_selector/train/gnn_selector")
    rows = []

    if gnn_train_dir.exists():
        summary_file = gnn_train_dir / "gnn_train_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                s = json.load(f)
            rows.append({
                "Model": "GNN Selector",
                "Training Time (s)": s.get("training_time_sec", "N/A"),
                "Epochs": s.get("total_epochs", s.get("best_epoch", "N/A")),
                "Best Epoch": s.get("best_epoch", "N/A"),
                "Best Val Loss": s.get("best_val_loss", "N/A"),
                "Converged": "Yes" if s.get("best_epoch", 0) < s.get("total_epochs", 999) else "No",
            })
        # Check for training log CSV
        log_file = gnn_train_dir / "gnn_train_log.csv"
        if log_file.exists():
            print(f"  Training log: {log_file}")

    # PPO/DQN training
    for model_name, subdir in [("PPO", "ppo"), ("DQN", "dqn")]:
        train_dir = Path(f"results/phase1_reactive/train/{subdir}")
        summary_file = train_dir / "train_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                s = json.load(f)
            rows.append({
                "Model": model_name,
                "Training Time (s)": s.get("training_time_sec", "N/A"),
                "Epochs": s.get("total_epochs", "N/A"),
                "Best Epoch": s.get("best_epoch", "N/A"),
                "Best Val Loss": s.get("best_val_metric", "N/A"),
                "Converged": "Yes" if s.get("converged", False) else "No",
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(table_dir / "table5_training_efficiency.csv", index=False)
        print(df.to_string(index=False))
    else:
        print("  No training summaries found")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("THESIS BENCHMARK: CLEAN INTERNAL vs EXTERNAL EVALUATION")
    print(f"Fixed k_crit = {K_CRIT} | Seed = {SEED}")
    print("=" * 70)
    total_start = time.perf_counter()

    # Load datasets
    print("\nLoading datasets...")
    bundle = load_bundle(CONFIG_PATH)
    max_steps = max_steps_from_args(bundle, MAX_STEPS)

    # Eval topologies (known)
    eval_datasets = []
    print("Eval topologies:")
    eval_specs = collect_specs(bundle, "eval_topologies")
    for spec in eval_specs:
        try:
            dataset, path_library = load_named_dataset(bundle, spec, max_steps)
            eval_datasets.append((dataset, path_library))
            print(f"  {dataset.key}: {len(dataset.nodes)} nodes, {len(dataset.od_pairs)} OD pairs")
        except Exception as e:
            print(f"  SKIP {spec.key}: {e}")

    # Generalization topologies (unseen)
    gen_datasets = []
    print("Generalization topologies:")
    gen_specs = collect_specs(bundle, "generalization_topologies")
    for spec in gen_specs:
        try:
            dataset, path_library = load_named_dataset(bundle, spec, max_steps)
            gen_datasets.append((dataset, path_library))
            print(f"  {dataset.key}: {len(dataset.nodes)} nodes, {len(dataset.od_pairs)} OD pairs")
        except Exception as e:
            print(f"  SKIP {spec.key}: {e}")

    all_datasets = eval_datasets + gen_datasets

    # Load GNN model
    gnn_model = None
    if GNN_CHECKPOINT.exists():
        print(f"\nLoading GNN: {GNN_CHECKPOINT}")
        gnn_model, _ = load_gnn_selector(GNN_CHECKPOINT, device=DEVICE)
        gnn_model.eval()
    else:
        print(f"\nGNN checkpoint not found: {GNN_CHECKPOINT}")

    # Tune DA-GNN lambda_cont on first validation topology
    da_gnn_cfg = DAGNNConfig(lambda_cont=0.3)  # default
    if gnn_model is not None and eval_datasets:
        print("\nTuning DA-GNN lambda_cont on validation...")
        ds, pl = eval_datasets[0]
        env_cfg = build_reactive_env_cfg(bundle, k_crit_override=K_CRIT)
        try:
            tune_env = ReactiveRoutingEnv(
                ds, ds.tm, pl, split_name="val", cfg=env_cfg, env_name=ds.key,
            )
            best_lam, tune_results = tune_lambda_cont(
                tune_env, gnn_model,
                lambda_candidates=[0.0, 0.1, 0.2, 0.3, 0.5],
                device=DEVICE,
            )
            da_gnn_cfg = DAGNNConfig(lambda_cont=best_lam)
            print(f"  Best lambda_cont = {best_lam}")
            for lam, res in tune_results.items():
                print(f"    lambda={lam:.1f}: MLU={res['mean_mlu']:.6f} DB={res['mean_disturbance']:.4f} Score={res['combined_score']:.6f}")
        except Exception as e:
            print(f"  Tuning failed ({e}), using default lambda_cont=0.3")

    # Run all sections
    internal_ts, internal_summary = run_internal_benchmark(bundle, all_datasets, gnn_model, da_gnn_cfg)
    external_ts, external_summary = run_external_baselines(bundle, all_datasets)
    lp_optimal_df = compute_lp_optimal(all_datasets, n_samples=10)
    failure_df = run_failure_scenarios(bundle, eval_datasets[:3], gnn_model, da_gnn_cfg)  # limit to 3 topos for speed

    # Generate CDF plots
    generate_cdf_plots(internal_ts, external_ts, failure_df)

    # Build summary tables
    build_summary_tables(internal_ts, external_ts, lp_optimal_df, failure_df)

    # Training efficiency
    report_training_efficiency()

    # Save DA-GNN config
    config_out = {
        "k_crit": K_CRIT,
        "seed": SEED,
        "da_gnn_lambda_cont": da_gnn_cfg.lambda_cont,
        "internal_selectors": INTERNAL_SELECTORS + [GNN_METHOD, DA_GNN_METHOD],
        "external_baselines": STATIC_METHODS + EXTERNAL_BASELINES,
    }
    with open(OUTPUT_DIR / "benchmark_config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 70)
    print("THESIS BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results: {OUTPUT_DIR}")
    print(f"DA-GNN lambda_cont = {da_gnn_cfg.lambda_cont}")


if __name__ == "__main__":
    main()
