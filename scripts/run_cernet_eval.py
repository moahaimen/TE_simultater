#!/usr/bin/env python3
"""CERNET-only evaluation to fill R56 gap.

The main eval skipped CERNET due to a config issue (now fixed).
This script runs CERNET through the same pipeline and appends results.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

K_CRIT_FIXED = 40
LT = 20
DEVICE = "cpu"
OUTPUT_DIR = Path("results/requirements_compliant_eval")
GNN_CHECKPOINT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")

from te.baselines import ecmp_splits, ospf_splits, select_bottleneck_critical, select_sensitivity_critical, select_topk_by_demand
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp, solve_full_mcf_min_mlu
from te.simulator import apply_routing, TEDataset
from te.paths import build_k_shortest_paths
from phase1_reactive.drl.gnn_selector import load_gnn_selector, build_graph_tensors, build_od_features
from phase1_reactive.drl.state_builder import compute_reactive_telemetry
from phase1_reactive.baselines.literature_baselines import select_literature_baseline
from phase1_reactive.routing.path_cache import build_modified_paths


def load_cernet():
    """Load CERNET from processed NPZ."""
    d = np.load("data/processed/cernet.npz", allow_pickle=True)
    nodes = list(d["nodes"])
    edges = list(zip(d["edge_src"], d["edge_dst"]))
    capacities = d["capacities"].astype(float)
    weights = d["weights"].astype(float)
    od_pairs = list(zip(d["od_src"], d["od_dst"]))
    tm = d["tm"].astype(float)

    class CERNETDataset:
        def __init__(self):
            self.key = "cernet"
            self.nodes = nodes
            self.edges = edges
            self.capacities = capacities
            self.weights = weights
            self.od_pairs = od_pairs
            self.tm = tm
            self.num_timesteps = tm.shape[0]
            # Split indices: 70/15/15
            n = tm.shape[0]
            self.train_end = int(n * 0.70)
            self.val_end = int(n * 0.85)

    return CERNETDataset()


def split_indices(ds, split):
    n = ds.num_timesteps
    if split == "train":
        return list(range(0, ds.train_end))
    elif split == "val":
        return list(range(ds.train_end, ds.val_end))
    elif split == "test":
        return list(range(ds.val_end, n))
    return []


def main():
    total_start = time.perf_counter()
    print("=" * 70)
    print("CERNET EVALUATION [R56]")
    print("=" * 70)

    ds = load_cernet()
    print(f"  CERNET: {len(ds.nodes)}n, {len(ds.edges)}e, {len(ds.od_pairs)} ODs, {ds.num_timesteps} TMs")

    # Build path library
    print("  Building k-shortest paths...")
    pl = build_modified_paths(ds.nodes, ds.edges, ds.weights, ds.od_pairs, k_paths=3)
    print(f"  Path library built: {len(pl.od_pairs)} OD pairs")

    # Load GNN
    gnn_model = None
    if GNN_CHECKPOINT.exists():
        gnn_model, _ = load_gnn_selector(GNN_CHECKPOINT, device=DEVICE)
        gnn_model.eval()
        print("  GNN loaded")

    capacities = ds.capacities
    ecmp_base = ecmp_splits(pl)
    test_indices = split_indices(ds, "test")
    print(f"  Test timesteps: {len(test_indices)}")

    # ---- Section A: All methods ----
    INTERNAL = ["bottleneck", "topk", "sensitivity"]
    if gnn_model:
        INTERNAL.append("gnn")
    EXTERNAL = ["ecmp", "ospf", "flexdate", "erodrl", "cfrrl", "flexentry"]
    all_methods = INTERNAL + EXTERNAL

    all_rows = []
    for method in all_methods:
        prev_splits = None
        for t_idx in test_indices:
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12:
                continue
            routing_ecmp = apply_routing(tm, ecmp_base, pl, capacities)
            telemetry = compute_reactive_telemetry(tm, ecmp_base, pl, routing_ecmp, ds.weights)

            t0 = time.perf_counter()
            try:
                if method == "gnn":
                    graph_data = build_graph_tensors(ds, telemetry=telemetry, device=DEVICE)
                    od_data = build_od_features(ds, tm, pl, telemetry=telemetry, device=DEVICE)
                    active_mask = (tm > 1e-12).astype(np.float32)
                    with torch.no_grad():
                        selected, _ = gnn_model.select_critical_flows(
                            graph_data, od_data, active_mask=active_mask, k_crit_default=K_CRIT_FIXED)
                elif method in ("topk", "bottleneck", "sensitivity"):
                    if method == "topk":
                        selected = select_topk_by_demand(tm, K_CRIT_FIXED)
                    elif method == "bottleneck":
                        selected = select_bottleneck_critical(tm, ecmp_base, pl, capacities, K_CRIT_FIXED)
                    else:
                        selected = select_sensitivity_critical(tm, ecmp_base, pl, capacities, K_CRIT_FIXED)
                elif method in ("flexdate", "erodrl", "cfrrl", "flexentry"):
                    selected = select_literature_baseline(
                        method, tm_vector=tm, ecmp_policy=ecmp_base,
                        path_library=pl, capacities=capacities, k_crit=K_CRIT_FIXED)
                elif method == "ecmp":
                    routing = apply_routing(tm, ecmp_base, pl, capacities)
                    db = compute_disturbance(prev_splits, ecmp_base, tm)
                    prev_splits = ecmp_base
                    exec_time = (time.perf_counter() - t0) * 1000
                    all_rows.append({"timestep": t_idx, "method": method,
                                     "mlu": float(routing.mlu), "disturbance": float(db),
                                     "exec_time_ms": exec_time, "k_used": 0, "status": "Static",
                                     "dataset": "cernet", "topology_type": "known"})
                    continue
                elif method == "ospf":
                    ospf_sp = ospf_splits(pl)
                    routing = apply_routing(tm, ospf_sp, pl, capacities)
                    db = compute_disturbance(prev_splits, ospf_sp, tm)
                    prev_splits = ospf_sp
                    exec_time = (time.perf_counter() - t0) * 1000
                    all_rows.append({"timestep": t_idx, "method": method,
                                     "mlu": float(routing.mlu), "disturbance": float(db),
                                     "exec_time_ms": exec_time, "k_used": 0, "status": "Static",
                                     "dataset": "cernet", "topology_type": "known"})
                    continue
                else:
                    continue

                lp = solve_selected_path_lp(tm, selected, ecmp_base, pl, capacities, time_limit_sec=LT)
                exec_time = (time.perf_counter() - t0) * 1000
                routing = apply_routing(tm, lp.splits, pl, capacities)
                db = compute_disturbance(prev_splits, lp.splits, tm)
                prev_splits = lp.splits
                all_rows.append({"timestep": t_idx, "method": method,
                                 "mlu": float(routing.mlu), "disturbance": float(db),
                                 "exec_time_ms": exec_time, "k_used": len(selected),
                                 "status": str(lp.status),
                                 "dataset": "cernet", "topology_type": "known"})
            except Exception as e:
                print(f"    {method} failed at t={t_idx}: {e}")
                continue

        if all_rows:
            m_rows = [r for r in all_rows if r["method"] == method]
            if m_rows:
                mean_mlu = np.mean([r["mlu"] for r in m_rows])
                mean_db = np.mean([r["disturbance"] for r in m_rows])
                mean_t = np.mean([r["exec_time_ms"] for r in m_rows])
                print(f"    {method:<15}: MLU={mean_mlu:.6f}  DB={mean_db:.4f}  Time={mean_t:.1f}ms")

    cernet_df = pd.DataFrame(all_rows)
    cernet_df.to_csv(OUTPUT_DIR / "cernet_results.csv", index=False)

    # ---- LP-Optimal for CERNET ----
    print("\n  Computing LP-optimal for CERNET...")
    lp_rows = []
    for t_idx in test_indices[:30]:
        tm = ds.tm[t_idx]
        if np.max(tm) < 1e-12:
            continue
        try:
            result = solve_full_mcf_min_mlu(
                tm_vector=tm, od_pairs=ds.od_pairs, nodes=ds.nodes,
                edges=ds.edges, capacities=capacities, time_limit_sec=90)
            lp_rows.append({"dataset": "cernet", "timestep": t_idx, "lp_optimal_mlu": float(result.mlu)})
        except Exception as e:
            print(f"    LP failed at t={t_idx}: {e}")
    if lp_rows:
        lp_df = pd.DataFrame(lp_rows)
        lp_df.to_csv(OUTPUT_DIR / "cernet_lp_optimal.csv", index=False)
        print(f"    Computed {len(lp_rows)} LP-optimal solutions, mean={np.mean([r['lp_optimal_mlu'] for r in lp_rows]):.6f}")

    # ---- Failure scenarios for CERNET ----
    print("\n  Running failure scenarios...")
    FAILURE_TYPES = ["single_link_failure", "capacity_degradation", "traffic_spike"]
    failure_methods = ["bottleneck", "topk", "sensitivity", "ecmp", "flexdate"]
    if gnn_model:
        failure_methods.append("gnn")

    failure_start_idx = test_indices[len(test_indices) // 3]
    tm0 = ds.tm[test_indices[0]]
    routing0 = apply_routing(tm0, ecmp_base, pl, capacities)
    ranked_edges = np.argsort(-np.asarray(routing0.utilization, dtype=float)).tolist()

    fail_rows = []
    for ft in FAILURE_TYPES:
        print(f"    {ft}:")
        if ft == "single_link_failure":
            failed_edge_idx = ranked_edges[0]
            fail_caps = capacities.copy()
            fail_caps[failed_edge_idx] = 1e-10
            fail_mask = np.zeros(len(capacities), dtype=float)
            fail_mask[failed_edge_idx] = 1.0
            keep = [i for i in range(len(ds.edges)) if i != failed_edge_idx]
            edges_new = [ds.edges[i] for i in keep]
            weights_new = np.asarray([ds.weights[i] for i in keep], dtype=float)
            caps_new = np.asarray([capacities[i] for i in keep], dtype=float)
            try:
                new_paths = build_modified_paths(ds.nodes, edges_new, weights_new, ds.od_pairs, k_paths=3)
            except:
                print(f"      Cannot rebuild paths - skipping")
                continue
        elif ft == "capacity_degradation":
            failed_edge_idx = ranked_edges[0]
            fail_caps = capacities.copy()
            fail_caps[failed_edge_idx] *= 0.5
            fail_mask = np.zeros(len(capacities), dtype=float)
            fail_mask[failed_edge_idx] = 1.0
            new_paths = None
        else:
            fail_caps = capacities
            fail_mask = np.zeros(len(capacities), dtype=float)
            new_paths = None

        for method in failure_methods:
            prev_splits = None
            for t_idx in test_indices:
                tm = ds.tm[t_idx].copy()
                if np.max(tm) < 1e-12:
                    continue
                failure_active = int(t_idx >= failure_start_idx)

                if ft == "traffic_spike" and failure_active:
                    top_ods = np.argsort(-tm)[:max(1, len(tm) // 10)]
                    tm[top_ods] *= 2.0

                if failure_active and ft == "single_link_failure":
                    cur_paths, cur_caps = new_paths, caps_new
                elif failure_active and ft == "capacity_degradation":
                    cur_paths, cur_caps = pl, fail_caps
                else:
                    cur_paths, cur_caps = pl, capacities

                cur_ecmp = ecmp_splits(cur_paths)
                t0 = time.perf_counter()
                try:
                    if method == "gnn" and gnn_model:
                        routing_e = apply_routing(tm, ecmp_splits(pl), pl, fail_caps if failure_active else capacities)
                        tel = compute_reactive_telemetry(tm, ecmp_splits(pl), pl, routing_e, ds.weights)
                        graph_data = build_graph_tensors(ds, telemetry=tel,
                                                         failure_mask=fail_mask if failure_active else None, device=DEVICE)
                        od_data = build_od_features(ds, tm, pl, telemetry=tel, device=DEVICE)
                        active_mask = (tm > 1e-12).astype(np.float32)
                        with torch.no_grad():
                            selected, _ = gnn_model.select_critical_flows(
                                graph_data, od_data, active_mask=active_mask, k_crit_default=K_CRIT_FIXED)
                    elif method in ("topk", "bottleneck", "sensitivity"):
                        if method == "topk":
                            selected = select_topk_by_demand(tm, K_CRIT_FIXED)
                        elif method == "bottleneck":
                            selected = select_bottleneck_critical(tm, cur_ecmp, cur_paths, cur_caps, K_CRIT_FIXED)
                        else:
                            selected = select_sensitivity_critical(tm, cur_ecmp, cur_paths, cur_caps, K_CRIT_FIXED)
                    elif method in ("flexdate",):
                        selected = select_literature_baseline(
                            method, tm_vector=tm, ecmp_policy=cur_ecmp,
                            path_library=cur_paths, capacities=cur_caps, k_crit=K_CRIT_FIXED)
                    elif method == "ecmp":
                        routing = apply_routing(tm, cur_ecmp, cur_paths, cur_caps)
                        db = compute_disturbance(prev_splits, cur_ecmp, tm)
                        prev_splits = cur_ecmp
                        exec_time = (time.perf_counter() - t0) * 1000
                        fail_rows.append({"timestep": t_idx, "method": method,
                                          "failure_type": ft, "failure_active": failure_active,
                                          "mlu": float(routing.mlu), "disturbance": float(db),
                                          "exec_time_ms": exec_time, "dataset": "cernet"})
                        continue
                    else:
                        continue

                    lp = solve_selected_path_lp(tm, selected, cur_ecmp, cur_paths, cur_caps, time_limit_sec=LT)
                    exec_time = (time.perf_counter() - t0) * 1000
                    routing = apply_routing(tm, lp.splits, cur_paths, cur_caps)
                    db = compute_disturbance(prev_splits, lp.splits, tm)
                    prev_splits = lp.splits
                    fail_rows.append({"timestep": t_idx, "method": method,
                                      "failure_type": ft, "failure_active": failure_active,
                                      "mlu": float(routing.mlu), "disturbance": float(db),
                                      "exec_time_ms": exec_time, "dataset": "cernet"})
                except Exception as e:
                    continue

            f_data = [r for r in fail_rows if r["method"] == method and r["failure_type"] == ft and r["failure_active"] == 1]
            if f_data:
                print(f"      {method:<15}: MLU={np.mean([r['mlu'] for r in f_data]):.6f}  "
                      f"DB={np.mean([r['disturbance'] for r in f_data]):.4f}")

    if fail_rows:
        fail_df = pd.DataFrame(fail_rows)
        fail_df.to_csv(OUTPUT_DIR / "cernet_failure_results.csv", index=False)

    # ---- CDF data ----
    if not cernet_df.empty:
        cdf_dir = OUTPUT_DIR / "cdf" / "cernet"
        cdf_dir.mkdir(parents=True, exist_ok=True)
        for method in cernet_df["method"].unique():
            m_data = cernet_df[cernet_df["method"] == method]
            m_data[["mlu", "disturbance", "exec_time_ms"]].to_csv(
                cdf_dir / f"{method}_cdf_data.csv", index=False)

    # ---- CDF plots for CERNET ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = OUTPUT_DIR / "plots"
        def plot_cdf(data_dict, title, xlabel, filename):
            fig, ax = plt.subplots(figsize=(10, 6))
            for label, values in sorted(data_dict.items()):
                sorted_vals = np.sort(values)
                cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                ax.plot(sorted_vals, cdf, label=label, linewidth=1.5)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("CDF", fontsize=12)
            ax.set_title(title, fontsize=13)
            ax.legend(fontsize=9, loc="lower right")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            plt.tight_layout()
            plt.savefig(plot_dir / filename, dpi=150)
            plt.close()

        cdf_methods = ["bottleneck", "topk", "sensitivity", "gnn", "flexdate", "ecmp"]
        cdf_data = cernet_df[cernet_df["method"].isin(cdf_methods)]
        for metric, xlabel in [("mlu", "MLU"), ("disturbance", "Disturbance"), ("exec_time_ms", "Execution Time (ms)")]:
            data_dict = {}
            for method in cdf_data["method"].unique():
                vals = cdf_data[cdf_data["method"] == method][metric].values
                if metric == "disturbance":
                    vals = vals[vals > 0]
                if len(vals) > 0:
                    data_dict[method] = vals
            if data_dict:
                suffix = metric.replace("exec_time_ms", "exec_time")
                plot_cdf(data_dict, f"CDF of {xlabel} - CERNET [R56]",
                         xlabel, f"cdf_{suffix}_cernet.png")
                print(f"  Saved: cdf_{suffix}_cernet.png")
    except ImportError:
        pass

    total_time = time.perf_counter() - total_start
    print(f"\nCERNET evaluation complete in {total_time:.1f}s")
    print(f"Results: {OUTPUT_DIR}/cernet_*.csv")


if __name__ == "__main__":
    main()
