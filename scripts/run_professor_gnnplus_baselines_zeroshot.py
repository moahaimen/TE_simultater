#!/usr/bin/env python3
"""Run a clean zero-shot GNN+ vs baseline packet-SDN study.

Scope:
  - GNN+ (proposed method)
  - ECMP
  - OSPF
  - Bottleneck
  - TopK
  - Sensitivity

This branch-local runner does not use:
  - MetaGate
  - Stable MetaGate
  - Bayesian calibration
  - per-topology adaptation
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_RUNNER = PROJECT_ROOT / "scripts" / "run_gnnplus_packet_sdn_full.py"
OUT_DIR = PROJECT_ROOT / "results" / "professor_gnnplus_baselines_zeroshot"
METHODS = ["gnnplus", "bottleneck", "topk", "sensitivity", "ecmp", "ospf"]


def load_base_runner():
    spec = importlib.util.spec_from_file_location("run_gnnplus_packet_sdn_full", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base runner from {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_failure_scenario_all_methods(
    base,
    scenario: str,
    tm_vector: np.ndarray,
    method: str,
    dataset,
    path_library,
    ecmp_base: list,
    capacities: np.ndarray,
    weights: np.ndarray,
    topo_mapping,
    gnnplus_model=None,
):
    """Run one failure scenario and return model-based SDN metrics."""
    from te.baselines import ospf_splits
    from phase3.state_builder import compute_telemetry

    normal_routing = base.apply_routing(tm_vector, ecmp_base, path_library, capacities)
    pre_failure_mlu = float(normal_routing.mlu)

    failure_mask = np.ones(len(capacities), dtype=float)
    if scenario == "single_link_failure":
        util = np.asarray(normal_routing.utilization)
        failure_mask[int(np.argmax(util))] = 0.0
    elif scenario == "random_link_failure_1":
        fail_idx = base.random.randint(0, len(capacities) - 1)
        failure_mask[fail_idx] = 0.0
    elif scenario in {"multiple_link_failure", "random_link_failure_2"}:
        for idx in base.random.sample(range(len(capacities)), min(2, len(capacities))):
            failure_mask[idx] = 0.0
    elif scenario == "three_link_failure":
        for idx in base.random.sample(range(len(capacities)), min(3, len(capacities))):
            failure_mask[idx] = 0.0
    elif scenario == "capacity_degradation_50":
        util = np.asarray(normal_routing.utilization)
        for idx in np.where(util > 0.5)[0]:
            failure_mask[idx] = 0.5

    failed_caps = capacities * failure_mask
    current_splits = [s.copy() for s in ecmp_base]
    current_groups, _ = base.build_ecmp_baseline_rules(path_library, topo_mapping, dataset.edges)
    prev_latency = None

    if scenario == "traffic_spike_2x":
        effective_tm = tm_vector.copy()
        top_demands = np.argsort(tm_vector)[-base.K_CRIT:]
        effective_tm[top_demands] *= 2.0
        effective_caps = capacities
    else:
        effective_tm = tm_vector
        effective_caps = failed_caps

    t_total_start = time.perf_counter()

    if method == "ecmp":
        new_splits = [s.copy() for s in ecmp_base]
        selected_ods = []
    elif method == "ospf":
        new_splits = ospf_splits(path_library)
        selected_ods = []
    elif method == "bottleneck":
        selected_ods = base.select_bottleneck_critical(effective_tm, ecmp_base, path_library, effective_caps, base.K_CRIT)
        lp_result = base.solve_selected_path_lp(
            tm_vector=effective_tm,
            selected_ods=selected_ods,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=effective_caps,
            time_limit_sec=20,
        )
        new_splits = [s.copy() for s in lp_result.splits]
    elif method == "topk":
        selected_ods = base.select_topk_by_demand(effective_tm, base.K_CRIT)
        lp_result = base.solve_selected_path_lp(
            tm_vector=effective_tm,
            selected_ods=selected_ods,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=effective_caps,
            time_limit_sec=20,
        )
        new_splits = [s.copy() for s in lp_result.splits]
    elif method == "sensitivity":
        selected_ods = base.select_sensitivity_critical(effective_tm, ecmp_base, path_library, effective_caps, base.K_CRIT)
        lp_result = base.solve_selected_path_lp(
            tm_vector=effective_tm,
            selected_ods=selected_ods,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=effective_caps,
            time_limit_sec=20,
        )
        new_splits = [s.copy() for s in lp_result.splits]
    elif method == "gnnplus" and gnnplus_model is not None:
        selected_ods = base.gnnplus_select_critical(gnnplus_model, dataset, path_library, effective_tm, base.K_CRIT)
        lp_result = base.solve_selected_path_lp(
            tm_vector=effective_tm,
            selected_ods=selected_ods,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=effective_caps,
            time_limit_sec=20,
        )
        new_splits = [s.copy() for s in lp_result.splits]
    else:
        new_splits = [s.copy() for s in ecmp_base]
        selected_ods = []

    decision_time_ms = (time.perf_counter() - t_total_start) * 1000.0

    t_rule_start = time.perf_counter()
    if method == "ecmp":
        new_groups, _ = base.build_ecmp_baseline_rules(path_library, topo_mapping, dataset.edges)
    else:
        new_groups, _ = base.splits_to_openflow_rules(
            new_splits, selected_ods, path_library, topo_mapping, dataset.edges
        )
    changed_groups = base.compute_rule_diff(current_groups, new_groups)
    flow_table_updates = int(len(changed_groups))
    rule_install_delay_ms = (time.perf_counter() - t_rule_start) * 1000.0

    post_routing = base.apply_routing(effective_tm, new_splits, path_library, effective_caps)
    post_recovery_mlu = float(post_routing.mlu)
    disturbance = float(base.compute_disturbance(current_splits, new_splits, effective_tm))
    telemetry = compute_telemetry(
        tm_vector=effective_tm,
        splits=new_splits,
        path_library=path_library,
        routing=post_routing,
        weights=weights,
        prev_latency_by_od=prev_latency,
    )
    failure_recovery_ms = decision_time_ms + rule_install_delay_ms

    return {
        "pre_failure_mlu": pre_failure_mlu,
        "mean_mlu": post_recovery_mlu,
        "failure_recovery_ms": failure_recovery_ms,
        "throughput": float(telemetry.throughput),
        "mean_latency_au": float(telemetry.mean_latency),
        "p95_latency_au": float(telemetry.p95_latency),
        "packet_loss": float(telemetry.packet_loss),
        "jitter_au": float(telemetry.jitter),
        "mean_disturbance": disturbance,
        "decision_time_ms": decision_time_ms,
        "flow_table_updates": float(flow_table_updates),
        "rule_install_delay_ms": rule_install_delay_ms,
    }


def benchmark_topology_failures_all_methods(base, topo_key: str, methods: list[str], gnnplus_cache: dict) -> list[dict]:
    base.logger.info("=== Loading %s (failures) ===", topo_key)
    dataset, path_library = base.load_dataset(topo_key)
    capacities = np.asarray(dataset.capacities, dtype=float)
    weights = np.asarray(dataset.weights, dtype=float)
    ecmp_base = base.ecmp_splits(path_library)
    topo_mapping = base.SDNTopologyMapping.from_mininet(dataset.nodes, dataset.edges, dataset.od_pairs)

    sp = dataset.split
    test_start = int(sp["test_start"])
    tm_data = dataset.tm
    test_indices = list(range(test_start, tm_data.shape[0]))
    sample_indices = test_indices[:: max(1, len(test_indices) // 10)]

    gnnplus_model = None
    if "gnnplus" in methods:
        if topo_key not in gnnplus_cache:
            gnnplus_cache[topo_key] = base.load_gnnplus_model(dataset, path_library)
        gnnplus_model = gnnplus_cache[topo_key]

    all_rows = []
    for scenario in base.FAILURE_SCENARIOS:
        if scenario == "normal":
            continue
        for method in methods:
            base.logger.info("  Running %s on %s (%s)...", method, topo_key, scenario)
            run_results = defaultdict(list)
            for t_idx in sample_indices:
                tm_vec = tm_data[t_idx]
                metrics = run_failure_scenario_all_methods(
                    base=base,
                    scenario=scenario,
                    tm_vector=tm_vec,
                    method=method,
                    dataset=dataset,
                    path_library=path_library,
                    ecmp_base=ecmp_base,
                    capacities=capacities,
                    weights=weights,
                    topo_mapping=topo_mapping,
                    gnnplus_model=gnnplus_model,
                )
                for key, value in metrics.items():
                    run_results[key].append(value)

            row = {
                "topology": topo_key,
                "status": base.TOPOLOGIES[topo_key]["status"],
                "method": method,
                "scenario": scenario,
                "nodes": len(dataset.nodes),
                "edges": len(dataset.edges),
            }
            for key, values in run_results.items():
                row[key] = float(np.mean(values))
            all_rows.append(row)
            base.logger.info(
                "    %s (%s): MLU=%.4f, recovery=%.1fms",
                method,
                scenario,
                row["mean_mlu"],
                row["failure_recovery_ms"],
            )
    return all_rows


def build_sdn_metrics_csv(summary_df: pd.DataFrame, failure_df: pd.DataFrame) -> pd.DataFrame:
    recovery_lookup = (
        failure_df.groupby(["topology", "method"], as_index=False)["failure_recovery_ms"]
        .mean()
        .rename(columns={"failure_recovery_ms": "avg_failure_recovery_ms"})
    )
    metrics = summary_df.merge(recovery_lookup, on=["topology", "method"], how="left")
    metrics.to_csv(OUT_DIR / "packet_sdn_sdn_metrics.csv", index=False)
    return metrics


def main() -> int:
    os.chdir(PROJECT_ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = load_base_runner()
    base.OUT_DIR = OUT_DIR

    base.logger.info("=" * 72)
    base.logger.info("PROFESSOR GNN+ VS BASELINES ZERO-SHOT RERUN")
    base.logger.info("Scope: GNN+, ECMP, OSPF, Bottleneck, TopK, Sensitivity")
    base.logger.info("No MetaGate, no Stable MetaGate, no calibration, no adaptation")
    base.logger.info("Results directory: %s", OUT_DIR)
    base.logger.info("=" * 72)

    gnn_cache = {}
    gnnplus_cache = {}

    normal_results = []
    for topo in list(base.TOPOLOGIES.keys()):
        rows = base.benchmark_topology_normal(topo, METHODS.copy(), gnn_cache, gnnplus_cache)
        rows = [row for row in rows if row["method"] in METHODS]
        normal_results.extend(rows)

    summary_df = pd.DataFrame(normal_results)
    summary_df.to_csv(OUT_DIR / "packet_sdn_summary.csv", index=False)
    base.logger.info("Normal results saved: %s", OUT_DIR / "packet_sdn_summary.csv")

    failure_results = []
    for topo in list(base.TOPOLOGIES.keys()):
        failure_results.extend(benchmark_topology_failures_all_methods(base, topo, METHODS.copy(), gnnplus_cache))

    failure_df = pd.DataFrame(failure_results)
    failure_df.to_csv(OUT_DIR / "packet_sdn_failure.csv", index=False)
    base.logger.info("Failure results saved: %s", OUT_DIR / "packet_sdn_failure.csv")

    build_sdn_metrics_csv(summary_df, failure_df)
    base.logger.info("SDN metrics saved: %s", OUT_DIR / "packet_sdn_sdn_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
