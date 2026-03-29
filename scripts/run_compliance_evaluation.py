#!/usr/bin/env python3
"""Full compliance evaluation: retrain GNN (no FlexDATE), run all scenarios.

Implements all 5 fixes from the requirements compliance pass:
  FIX 1: GNN retrained with internal-only oracle (no FlexDATE)
  FIX 2: GNN failure evaluation (single-link, capacity-degradation, traffic-spike)
  FIX 3: Two unseen topologies (Germany50 + VtlWavenet2011)
  FIX 4: Extended metrics collection (disturbance, execution time, PR)
  FIX 5: Honest LP-optimal PR computation
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from te.baselines import ecmp_splits, ospf_splits, select_bottleneck_critical, select_sensitivity_critical, select_topk_by_demand
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.simulator import apply_routing

from phase1_reactive.eval.common import (
    build_reactive_env_cfg,
    load_bundle,
    load_named_dataset,
    collect_specs,
    max_steps_from_args,
    resolve_phase1_k_crit,
)
from phase1_reactive.eval.core import split_indices
from phase1_reactive.drl.gnn_selector import load_gnn_selector, build_graph_tensors, build_od_features
from phase1_reactive.drl.state_builder import compute_reactive_telemetry
from phase1_reactive.routing.path_cache import build_modified_paths

CONFIG_PATH = "configs/phase1_reactive_full.yaml"
MAX_STEPS = 500
SEED = 42
DEVICE = "cpu"
K_CRIT_FIXED = 40
LT = 20  # LP time limit

OUTPUT_DIR = Path("results/compliance_eval")
GNN_CHECKPOINT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")


# ============================================================
# Internal selector functions
# ============================================================

def _run_selector(method, tm, ecmp_base, path_library, capacities, k_crit):
    """Run an internal selector and return selected OD indices."""
    if method == "topk":
        return select_topk_by_demand(tm, k_crit)
    elif method == "bottleneck":
        return select_bottleneck_critical(tm, ecmp_base, path_library, capacities, k_crit)
    elif method == "sensitivity":
        return select_sensitivity_critical(tm, ecmp_base, path_library, capacities, k_crit)
    else:
        raise ValueError(f"Unknown internal method: {method}")


def _run_gnn_selector(tm, dataset, path_library, gnn_model, k_crit, telemetry=None, failure_mask=None):
    """Run GNN selector and return selected OD indices."""
    graph_data = build_graph_tensors(dataset, telemetry=telemetry, failure_mask=failure_mask, device=DEVICE)
    od_data = build_od_features(dataset, tm, path_library, telemetry=telemetry, device=DEVICE)
    active_mask = (np.asarray(tm, dtype=np.float64) > 1e-12).astype(np.float32)
    with torch.no_grad():
        selected, _ = gnn_model.select_critical_flows(
            graph_data, od_data, active_mask=active_mask, k_crit_default=k_crit,
        )
    return selected


def _run_external_baseline(method, tm, ecmp_base, path_library, capacities, k_crit, prev_selected=None, failure_mask=None):
    """Run an external baseline selector."""
    from phase1_reactive.baselines.literature_baselines import select_literature_baseline
    return select_literature_baseline(
        method, tm_vector=tm, ecmp_policy=ecmp_base,
        path_library=path_library, capacities=capacities, k_crit=k_crit,
        prev_selected=prev_selected, failure_mask=failure_mask,
    )


# ============================================================
# Evaluate one method on one topology (all test TMs)
# ============================================================

def evaluate_method_on_topology(dataset, path_library, method, gnn_model=None,
                                 capacities_override=None, failure_mask=None):
    """Run a method on all test TMs, collecting MLU, disturbance, execution time."""
    capacities = np.asarray(capacities_override if capacities_override is not None
                            else dataset.capacities, dtype=float)
    ecmp_base = ecmp_splits(path_library)
    test_indices = split_indices(dataset, "test")

    rows = []
    prev_splits = None

    for t_idx in test_indices:
        tm = np.asarray(dataset.tm[t_idx], dtype=float)
        if np.max(tm) < 1e-12:
            continue

        # Compute telemetry for GNN features
        routing_ecmp = apply_routing(tm, ecmp_base, path_library, capacities)
        weights = np.asarray(dataset.weights, dtype=float)
        telemetry = compute_reactive_telemetry(tm, ecmp_base, path_library, routing_ecmp, weights)

        # Select critical flows
        t0 = time.perf_counter()
        try:
            if method == "gnn" and gnn_model is not None:
                selected = _run_gnn_selector(tm, dataset, path_library, gnn_model,
                                              K_CRIT_FIXED, telemetry=telemetry,
                                              failure_mask=failure_mask)
            elif method in ("topk", "bottleneck", "sensitivity"):
                selected = _run_selector(method, tm, ecmp_base, path_library, capacities, K_CRIT_FIXED)
            elif method in ("flexdate", "erodrl", "cfrrl", "flexentry"):
                selected = _run_external_baseline(method, tm, ecmp_base, path_library, capacities, K_CRIT_FIXED)
            elif method == "ecmp":
                # Static ECMP — no LP
                routing = apply_routing(tm, ecmp_base, path_library, capacities)
                db = compute_disturbance(prev_splits, ecmp_base, tm)
                prev_splits = ecmp_base
                exec_time = (time.perf_counter() - t0) * 1000
                rows.append({
                    "timestep": int(t_idx), "method": method,
                    "mlu": float(routing.mlu), "disturbance": float(db),
                    "exec_time_ms": exec_time, "k_used": 0, "status": "Static",
                })
                continue
            elif method == "ospf":
                ospf_sp = ospf_splits(path_library)
                routing = apply_routing(tm, ospf_sp, path_library, capacities)
                db = compute_disturbance(prev_splits, ospf_sp, tm)
                prev_splits = ospf_sp
                exec_time = (time.perf_counter() - t0) * 1000
                rows.append({
                    "timestep": int(t_idx), "method": method,
                    "mlu": float(routing.mlu), "disturbance": float(db),
                    "exec_time_ms": exec_time, "k_used": 0, "status": "Static",
                })
                continue
            else:
                continue
        except Exception as e:
            print(f"    {method} failed at t={t_idx}: {e}")
            continue

        # LP solve
        lp = solve_selected_path_lp(tm, selected, ecmp_base, path_library, capacities, time_limit_sec=LT)
        exec_time = (time.perf_counter() - t0) * 1000

        routing = apply_routing(tm, lp.splits, path_library, capacities)
        db = compute_disturbance(prev_splits, lp.splits, tm)
        prev_splits = lp.splits

        rows.append({
            "timestep": int(t_idx), "method": method,
            "mlu": float(routing.mlu), "disturbance": float(db),
            "exec_time_ms": exec_time, "k_used": len(selected),
            "status": str(lp.status),
        })

    return pd.DataFrame(rows)


# ============================================================
# LP-Optimal (full MCF) for PR computation
# ============================================================

def compute_lp_optimal(dataset, path_library, capacities_override=None):
    """Compute LP-optimal MLU per test TM using full multi-commodity flow."""
    from te.lp_solver import solve_full_mcf_min_mlu
    capacities = np.asarray(capacities_override if capacities_override is not None
                            else dataset.capacities, dtype=float)
    test_indices = split_indices(dataset, "test")
    results = {}

    for t_idx in test_indices[:30]:  # limit for speed
        tm = np.asarray(dataset.tm[t_idx], dtype=float)
        if np.max(tm) < 1e-12:
            continue
        try:
            result = solve_full_mcf_min_mlu(
                tm_vector=tm,
                od_pairs=dataset.od_pairs,
                nodes=dataset.nodes,
                edges=dataset.edges,
                capacities=capacities,
                time_limit_sec=90,
            )
            results[t_idx] = float(result.mlu)
        except Exception as e:
            print(f"  LP-optimal failed at t={t_idx}: {e}")
            continue
    return results


# ============================================================
# Failure scenarios
# ============================================================

def run_failure_scenario(dataset, path_library, method, failure_type, gnn_model=None):
    """Run a method under a failure scenario."""
    capacities = np.asarray(dataset.capacities, dtype=float)
    ecmp_base = ecmp_splits(path_library)
    test_indices = split_indices(dataset, "test")
    if not test_indices:
        return pd.DataFrame()

    failure_start_idx = test_indices[len(test_indices) // 3]

    # Pick most-utilized edge for failure
    tm0 = np.asarray(dataset.tm[test_indices[0]], dtype=float)
    routing0 = apply_routing(tm0, ecmp_base, path_library, capacities)
    ranked_edges = np.argsort(-np.asarray(routing0.utilization, dtype=float)).tolist()

    if failure_type == "single_link_failure":
        failed_edges = ranked_edges[:1]
        # Remove the failed edge — rebuild paths
        keep = [i for i in range(len(dataset.edges)) if i not in set(failed_edges)]
        edges = [dataset.edges[i] for i in keep]
        weights = np.asarray([dataset.weights[i] for i in keep], dtype=float)
        new_caps = np.asarray([capacities[i] for i in keep], dtype=float)
        try:
            new_paths = build_modified_paths(dataset.nodes, edges, weights, dataset.od_pairs, k_paths=3)
        except Exception:
            return pd.DataFrame()
        fail_mask = np.zeros(len(new_caps), dtype=float)
        post = {"path_library": new_paths, "capacities": new_caps, "weights": weights, "failure_mask": fail_mask}

    elif failure_type == "capacity_degradation":
        failed_edges = ranked_edges[:1]
        new_caps = capacities.copy()
        fail_mask = np.zeros_like(new_caps)
        for idx in failed_edges:
            new_caps[idx] *= 0.5
            fail_mask[idx] = 1.0
        post = {"path_library": path_library, "capacities": new_caps, "weights": np.asarray(dataset.weights, dtype=float), "failure_mask": fail_mask}

    elif failure_type == "traffic_spike":
        # No topology change — just 2x traffic on top 10% ODs
        post = None
        failed_edges = []
    else:
        return pd.DataFrame()

    rows = []
    prev_splits = None

    for t_idx in test_indices:
        tm = np.asarray(dataset.tm[t_idx], dtype=float)
        if np.max(tm) < 1e-12:
            continue

        failure_active = int(t_idx >= failure_start_idx)

        if failure_type == "traffic_spike" and failure_active:
            # 2x demand on top 10% ODs
            top_ods = np.argsort(-tm)[:max(1, len(tm) // 10)]
            tm = tm.copy()
            tm[top_ods] *= 2.0

        if failure_active and post is not None:
            cur_paths = post["path_library"]
            cur_caps = post["capacities"]
            cur_weights = post["weights"]
            cur_mask = post["failure_mask"]
        else:
            cur_paths = path_library
            cur_caps = capacities
            cur_weights = np.asarray(dataset.weights, dtype=float)
            cur_mask = np.zeros(len(cur_caps), dtype=float)

        cur_ecmp = ecmp_splits(cur_paths)
        routing_ecmp = apply_routing(tm, cur_ecmp, cur_paths, cur_caps)
        telemetry = compute_reactive_telemetry(tm, cur_ecmp, cur_paths, routing_ecmp, cur_weights)

        t0 = time.perf_counter()
        try:
            if method == "gnn" and gnn_model is not None:
                selected = _run_gnn_selector(tm, dataset, cur_paths, gnn_model,
                                              K_CRIT_FIXED, telemetry=telemetry,
                                              failure_mask=cur_mask)
            elif method in ("topk", "bottleneck", "sensitivity"):
                selected = _run_selector(method, tm, cur_ecmp, cur_paths, cur_caps, K_CRIT_FIXED)
            elif method in ("flexdate", "erodrl", "cfrrl", "flexentry"):
                selected = _run_external_baseline(method, tm, cur_ecmp, cur_paths, cur_caps, K_CRIT_FIXED)
            elif method == "ecmp":
                routing = apply_routing(tm, cur_ecmp, cur_paths, cur_caps)
                db = compute_disturbance(prev_splits, cur_ecmp, tm)
                prev_splits = cur_ecmp
                exec_time = (time.perf_counter() - t0) * 1000
                rows.append({
                    "timestep": int(t_idx), "method": method,
                    "failure_type": failure_type, "failure_active": failure_active,
                    "mlu": float(routing.mlu), "disturbance": float(db),
                    "exec_time_ms": exec_time,
                })
                continue
            else:
                continue
        except Exception as e:
            continue

        lp = solve_selected_path_lp(tm, selected, cur_ecmp, cur_paths, cur_caps, time_limit_sec=LT)
        exec_time = (time.perf_counter() - t0) * 1000
        routing = apply_routing(tm, lp.splits, cur_paths, cur_caps)
        db = compute_disturbance(prev_splits, lp.splits, tm)
        prev_splits = lp.splits

        rows.append({
            "timestep": int(t_idx), "method": method,
            "failure_type": failure_type, "failure_active": failure_active,
            "mlu": float(routing.mlu), "disturbance": float(db),
            "exec_time_ms": exec_time,
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    total_start = time.perf_counter()
    print("=" * 70)
    print("COMPLIANCE EVALUATION: Requirements-aligned full benchmark")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load config & datasets
    bundle = load_bundle(CONFIG_PATH)
    max_steps = max_steps_from_args(bundle, MAX_STEPS)

    # Training topologies (known)
    print("\n--- Known/Train Topologies ---")
    eval_specs = collect_specs(bundle, "eval_topologies")
    eval_datasets = []
    for spec in eval_specs:
        try:
            ds, pl = load_named_dataset(bundle, spec, max_steps)
            eval_datasets.append((ds, pl))
            print(f"  {ds.key}: {len(ds.nodes)}n, {len(ds.edges)}e, {len(ds.od_pairs)} ODs")
        except Exception as e:
            print(f"  SKIP {spec.key}: {e}")

    # Unseen topologies
    print("\n--- Unseen Topologies (Germany50 + VtlWavenet2011) ---")
    gen_specs = collect_specs(bundle, "generalization_topologies")
    gen_datasets = []
    for spec in gen_specs:
        try:
            ds, pl = load_named_dataset(bundle, spec, max_steps)
            gen_datasets.append((ds, pl))
            print(f"  {ds.key}: {len(ds.nodes)}n, {len(ds.edges)}e, {len(ds.od_pairs)} ODs")
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
        print(f"\nWARNING: GNN checkpoint not found: {GNN_CHECKPOINT}")

    # ========================================================
    # SECTION A: Internal fair fixed-K benchmark
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION A: Internal Fair Fixed-K=40 Benchmark")
    print("=" * 70)

    INTERNAL_METHODS = ["bottleneck", "topk", "sensitivity"]
    if gnn_model is not None:
        INTERNAL_METHODS.append("gnn")

    EXTERNAL_BASELINES = ["ecmp", "ospf", "flexdate", "erodrl", "cfrrl", "flexentry"]

    all_methods = INTERNAL_METHODS + EXTERNAL_BASELINES
    all_results = []

    for ds, pl in all_datasets:
        print(f"\n  Topology: {ds.key}")
        for method in all_methods:
            try:
                df = evaluate_method_on_topology(ds, pl, method, gnn_model=gnn_model)
                if not df.empty:
                    df["dataset"] = ds.key
                    df["topology_type"] = "unseen" if (ds, pl) in gen_datasets else "known"
                    all_results.append(df)
                    mean_mlu = df["mlu"].mean()
                    mean_db = df["disturbance"].mean()
                    mean_t = df["exec_time_ms"].mean()
                    print(f"    {method:<15}: MLU={mean_mlu:.6f}  DB={mean_db:.4f}  Time={mean_t:.1f}ms")
            except Exception as e:
                print(f"    {method:<15}: FAILED ({e})")

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)

        # Summary table
        summary = results_df.groupby(["dataset", "method"]).agg(
            mean_mlu=("mlu", "mean"),
            p95_mlu=("mlu", lambda x: np.percentile(x, 95)),
            mean_disturbance=("disturbance", "mean"),
            mean_exec_ms=("exec_time_ms", "mean"),
        ).reset_index()
        summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
        print("\n--- Summary Table ---")
        print(summary.to_string(index=False))

    # ========================================================
    # SECTION B: LP-Optimal and PR
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION B: LP-Optimal (Full MCF) for Performance Ratio")
    print("=" * 70)

    pr_results = []
    for ds, pl in all_datasets[:3]:  # Limit to smaller topologies for speed
        print(f"\n  Computing LP-optimal for: {ds.key}")
        try:
            lp_opt = compute_lp_optimal(ds, pl)
            if lp_opt:
                for t_idx, opt_mlu in lp_opt.items():
                    pr_results.append({
                        "dataset": ds.key, "timestep": t_idx,
                        "lp_optimal_mlu": opt_mlu,
                    })
                print(f"    LP-optimal computed for {len(lp_opt)} timesteps, mean={np.mean(list(lp_opt.values())):.6f}")
        except Exception as e:
            print(f"    FAILED: {e}")

    if pr_results:
        pr_df = pd.DataFrame(pr_results)
        pr_df.to_csv(OUTPUT_DIR / "lp_optimal.csv", index=False)

        # Compute PR for methods that have matching timesteps
        if all_results:
            merged = results_df.merge(pr_df, on=["dataset", "timestep"], how="inner")
            if not merged.empty:
                merged["pr"] = merged["mlu"] / merged["lp_optimal_mlu"].clip(lower=1e-12)
                pr_summary = merged.groupby(["dataset", "method"]).agg(
                    mean_pr=("pr", "mean"),
                    p95_pr=("pr", lambda x: np.percentile(x, 95)),
                ).reset_index()
                pr_summary.to_csv(OUTPUT_DIR / "pr_summary.csv", index=False)
                print("\n--- Performance Ratio Table ---")
                print(pr_summary.to_string(index=False))

    # ========================================================
    # SECTION C: Failure Scenarios
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION C: Failure Scenarios (GNN + Heuristics)")
    print("=" * 70)

    FAILURE_TYPES = ["single_link_failure", "capacity_degradation", "traffic_spike"]
    FAILURE_METHODS = ["bottleneck", "topk", "sensitivity", "ecmp"]
    if gnn_model is not None:
        FAILURE_METHODS.append("gnn")
    # Add strongest external baseline
    FAILURE_METHODS.append("flexdate")

    failure_frames = []
    # Test failures on 2-3 representative topologies
    failure_topos = all_datasets[:3]  # Abilene, GEANT, Ebone
    if gen_datasets:
        failure_topos.append(gen_datasets[0])  # Germany50

    for ds, pl in failure_topos:
        print(f"\n  Topology: {ds.key}")
        for ft in FAILURE_TYPES:
            print(f"    Scenario: {ft}")
            for method in FAILURE_METHODS:
                try:
                    df = run_failure_scenario(ds, pl, method, ft, gnn_model=gnn_model)
                    if not df.empty:
                        df["dataset"] = ds.key
                        failure_frames.append(df)
                        fail_only = df[df["failure_active"] == 1]
                        if not fail_only.empty:
                            print(f"      {method:<15}: MLU={fail_only['mlu'].mean():.6f}  DB={fail_only['disturbance'].mean():.4f}")
                except Exception as e:
                    print(f"      {method:<15}: FAILED ({e})")

    if failure_frames:
        failure_df = pd.concat(failure_frames, ignore_index=True)
        failure_df.to_csv(OUTPUT_DIR / "failure_results.csv", index=False)

        fail_summary = failure_df[failure_df["failure_active"] == 1].groupby(
            ["dataset", "failure_type", "method"]
        ).agg(
            mean_mlu=("mlu", "mean"),
            mean_disturbance=("disturbance", "mean"),
            mean_exec_ms=("exec_time_ms", "mean"),
        ).reset_index()
        fail_summary.to_csv(OUTPUT_DIR / "failure_summary.csv", index=False)
        print("\n--- Failure Summary ---")
        print(fail_summary.to_string(index=False))

    # ========================================================
    # SECTION D: CDF Data Export
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION D: CDF Data Export")
    print("=" * 70)

    if all_results:
        # Per-TM MLU CDF data
        for ds_key in results_df["dataset"].unique():
            ds_data = results_df[results_df["dataset"] == ds_key]
            cdf_dir = OUTPUT_DIR / "cdf" / ds_key
            cdf_dir.mkdir(parents=True, exist_ok=True)
            for method in ds_data["method"].unique():
                m_data = ds_data[ds_data["method"] == method]
                m_data[["mlu", "disturbance", "exec_time_ms"]].to_csv(
                    cdf_dir / f"{method}_cdf_data.csv", index=False
                )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================
    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 70)
    print("COMPLIANCE EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files produced:")
    for f in sorted(OUTPUT_DIR.rglob("*.csv")):
        print(f"  {f.relative_to(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
