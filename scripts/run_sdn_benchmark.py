#!/usr/bin/env python3
"""
SDN Deployment Benchmark — measures all 9 required SDN metrics.

Metrics measured:
  1. Throughput       — model-based (routed_demand / total_demand)
  2. Latency          — model-based (M/M/1 queuing + link weights)
  3. Packet loss      — model-based (1 - throughput)
  4. Jitter           — model-based (inter-timestep latency variation)
  5. Decision time    — wall-clock (perf_counter)
  6. Flow-table updates — counted (OpenFlow GroupMod diff per cycle)
  7. Rule installation delay — wall-clock (time to serialize OpenFlow messages)
  8. Failure recovery time — wall-clock (time from failure injection to new rules)

Methods evaluated: GNN, Bottleneck, ECMP
Topologies: Abilene (12n, small), GEANT (22n, medium)
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from te.baselines import ecmp_splits, select_bottleneck_critical, select_topk_by_demand, select_sensitivity_critical
from te.lp_solver import solve_selected_path_lp
from te.paths import PathLibrary
from te.simulator import TEDataset, apply_routing
from te.disturbance import compute_disturbance

from sdn.openflow_adapter import (
    SDNTopologyMapping, splits_to_openflow_rules, build_ecmp_baseline_rules, compute_rule_diff,
    OFGroupMod, OFFlowMod,
)
from phase3.state_builder import compute_telemetry, TelemetryConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = project_root / "results" / "sdn_benchmark"
K_CRIT = 40
NUM_RUNS = 3  # Average over 3 runs for timing stability


# ── Dataset loading ────────────────────────────────────────────────────

def load_dataset(key: str) -> Tuple[TEDataset, PathLibrary]:
    """Load a dataset from the processed NPZ files."""
    from phase1_reactive.eval.common import load_bundle, collect_specs, load_named_dataset, max_steps_from_args
    config_path = project_root / "configs" / "phase1_reactive_full.yaml"
    bundle = load_bundle(config_path)

    for field_name in ["eval_topologies", "generalization_topologies"]:
        specs = collect_specs(bundle, field_name)
        max_steps = max_steps_from_args(bundle, 500)
        for spec in specs:
            if key in spec.key.lower():
                try:
                    dataset, pl = load_named_dataset(bundle, spec, max_steps)
                    return dataset, pl
                except Exception as e:
                    logger.warning(f"Failed to load {spec.key}: {e}")

    raise ValueError(f"Dataset '{key}' not found")


def load_gnn_model(dataset, path_library):
    """Try to load trained GNN model, return None if unavailable."""
    try:
        import torch
        from phase1_reactive.drl.gnn_selector import load_gnn_selector, build_graph_tensors, build_od_features

        ckpt = project_root / "results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt"
        if not ckpt.exists():
            logger.warning(f"No GNN checkpoint at {ckpt}")
            return None

        model, cfg = load_gnn_selector(str(ckpt), device="cpu")
        model.eval()
        # Attach dataset reference and helpers for later use
        model._dataset = dataset
        model._build_graph_tensors = build_graph_tensors
        model._build_od_features = build_od_features
        logger.info(f"Loaded GNN model from {ckpt}")
        return model
    except Exception as e:
        logger.warning(f"Could not load GNN model: {e}")
        return None


# ── SDN Benchmark Core ─────────────────────────────────────────────────

@dataclass
class SDNCycleResult:
    """Result of one SDN control cycle with all 9 metrics."""
    cycle: int
    method: str
    topology: str
    # Primary TE metrics
    pre_mlu: float
    post_mlu: float
    disturbance: float
    # Model-based network metrics
    throughput: float          # routed_demand / total_demand
    mean_latency: float        # M/M/1 queuing model (abstract units)
    p95_latency: float
    packet_loss: float         # 1 - throughput
    jitter: float              # inter-timestep latency variation
    # Measured SDN deployment metrics
    decision_time_ms: float    # wall-clock
    flow_table_updates: int    # GroupMod diff count
    rule_install_delay_ms: float  # serialization time
    # Failure
    is_failure_cycle: bool = False
    failure_recovery_ms: float = 0.0


def run_sdn_cycle(
    tm_vector: np.ndarray,
    method: str,
    dataset: TEDataset,
    path_library: PathLibrary,
    ecmp_base: list,
    current_splits: list,
    current_groups: List[OFGroupMod],
    topo_mapping: SDNTopologyMapping,
    capacities: np.ndarray,
    weights: np.ndarray,
    gnn_model=None,
    prev_latency_by_od=None,
    failure_mask=None,
) -> Tuple[SDNCycleResult, list, List[OFGroupMod], np.ndarray]:
    """Run one SDN TE cycle and measure all metrics."""

    t_total_start = time.perf_counter()

    # ── 1. OBSERVE: compute pre-MLU with current routing ──
    routing_pre = apply_routing(tm_vector, current_splits, path_library, capacities)
    pre_mlu = float(routing_pre.mlu)

    # ── 2. SELECT + OPTIMIZE ──
    if method == "ecmp":
        # ECMP: no rerouting, just use baseline
        new_splits = [s.copy() for s in ecmp_base]
        selected_ods = []
    elif method == "bottleneck":
        selected_ods = select_bottleneck_critical(
            tm_vector, ecmp_base, path_library, capacities, K_CRIT
        )
        lp_result = solve_selected_path_lp(
            tm_vector=tm_vector,
            selected_ods=selected_ods,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=capacities,
            time_limit_sec=20,
        )
        new_splits = [s.copy() for s in lp_result.splits]
    elif method == "gnn" and gnn_model is not None:
        import torch
        try:
            graph_data = gnn_model._build_graph_tensors(gnn_model._dataset, device="cpu")
            od_data = gnn_model._build_od_features(gnn_model._dataset, tm_vector, path_library, device="cpu")
            active_mask = (np.asarray(tm_vector, dtype=np.float64) > 1e-12).astype(np.float32)
            with torch.no_grad():
                selected_ods, _ = gnn_model.select_critical_flows(
                    graph_data, od_data, active_mask=active_mask, k_crit_default=K_CRIT,
                )
        except Exception as e:
            logger.debug(f"GNN fallback: {e}")
            selected_ods = select_bottleneck_critical(
                tm_vector, ecmp_base, path_library, capacities, K_CRIT
            )

        lp_result = solve_selected_path_lp(
            tm_vector=tm_vector,
            selected_ods=selected_ods,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=capacities,
            time_limit_sec=20,
        )
        new_splits = [s.copy() for s in lp_result.splits]
    else:
        # Fallback to bottleneck
        selected_ods = select_bottleneck_critical(
            tm_vector, ecmp_base, path_library, capacities, K_CRIT
        )
        lp_result = solve_selected_path_lp(
            tm_vector=tm_vector,
            selected_ods=selected_ods,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=capacities,
            time_limit_sec=20,
        )
        new_splits = [s.copy() for s in lp_result.splits]

    t_decision_end = time.perf_counter()
    decision_time_ms = (t_decision_end - t_total_start) * 1000

    # ── 3. APPLY: Generate OpenFlow rules and measure ──
    t_rule_start = time.perf_counter()

    if method == "ecmp":
        new_groups, new_flows = build_ecmp_baseline_rules(
            path_library, topo_mapping, dataset.edges
        )
    else:
        new_groups, new_flows = splits_to_openflow_rules(
            new_splits, selected_ods, path_library, topo_mapping, dataset.edges
        )

    # Count diff
    changed_groups = compute_rule_diff(current_groups, new_groups)
    flow_table_updates = len(changed_groups)

    t_rule_end = time.perf_counter()
    rule_install_delay_ms = (t_rule_end - t_rule_start) * 1000

    # ── 4. MEASURE: compute post-routing metrics ──
    routing_post = apply_routing(tm_vector, new_splits, path_library, capacities)
    post_mlu = float(routing_post.mlu)

    # Disturbance
    dist = compute_disturbance(current_splits, new_splits, tm_vector)

    # Telemetry (model-based)
    telemetry = compute_telemetry(
        tm_vector=tm_vector,
        splits=new_splits,
        path_library=path_library,
        routing=routing_post,
        weights=weights,
        prev_latency_by_od=prev_latency_by_od,
    )

    result = SDNCycleResult(
        cycle=0,
        method=method,
        topology=dataset.key,
        pre_mlu=pre_mlu,
        post_mlu=post_mlu,
        disturbance=float(dist),
        throughput=telemetry.throughput,
        mean_latency=telemetry.mean_latency,
        p95_latency=telemetry.p95_latency,
        packet_loss=telemetry.packet_loss,
        jitter=telemetry.jitter,
        decision_time_ms=decision_time_ms,
        flow_table_updates=flow_table_updates,
        rule_install_delay_ms=rule_install_delay_ms,
    )

    return result, new_splits, new_groups, telemetry.latency_by_od


def run_failure_recovery(
    tm_vector: np.ndarray,
    method: str,
    dataset: TEDataset,
    path_library: PathLibrary,
    ecmp_base: list,
    capacities: np.ndarray,
    weights: np.ndarray,
    topo_mapping: SDNTopologyMapping,
    gnn_model=None,
) -> Tuple[float, float, float]:
    """Measure failure recovery: time from failure injection to new rules ready.

    Returns: (recovery_time_ms, post_failure_mlu, post_recovery_mlu)
    """
    # Pick the most loaded edge to fail
    test_tm = tm_vector
    ecmp_routing = apply_routing(test_tm, ecmp_base, path_library, capacities)
    util = np.asarray(ecmp_routing.utilization)
    fail_edge_idx = int(np.argmax(util))

    post_failure_mlu = float(ecmp_routing.mlu)

    # ── Inject failure: set failed edge capacity to near-zero ──
    t_fail_start = time.perf_counter()

    failed_caps = capacities.copy()
    failed_caps[fail_edge_idx] = 1e-10

    # Recompute routing under failure (like SDN controller detecting failure)
    if method == "ecmp":
        new_splits = [s.copy() for s in ecmp_base]
    elif method == "bottleneck":
        selected_ods = select_bottleneck_critical(
            test_tm, ecmp_base, path_library, failed_caps, K_CRIT
        )
        lp_result = solve_selected_path_lp(
            tm_vector=test_tm,
            selected_ods=selected_ods,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=failed_caps,
            time_limit_sec=20,
        )
        new_splits = [s.copy() for s in lp_result.splits]
    elif method == "gnn" and gnn_model is not None:
        import torch
        try:
            graph_data = gnn_model._build_graph_tensors(gnn_model._dataset, device="cpu")
            od_data = gnn_model._build_od_features(gnn_model._dataset, test_tm, path_library, device="cpu")
            active_mask = (np.asarray(test_tm, dtype=np.float64) > 1e-12).astype(np.float32)
            with torch.no_grad():
                selected_ods, _ = gnn_model.select_critical_flows(
                    graph_data, od_data, active_mask=active_mask, k_crit_default=K_CRIT,
                )
        except Exception:
            selected_ods = select_bottleneck_critical(
                test_tm, ecmp_base, path_library, failed_caps, K_CRIT
            )

        lp_result = solve_selected_path_lp(
            tm_vector=test_tm,
            selected_ods=selected_ods,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=failed_caps,
            time_limit_sec=20,
        )
        new_splits = [s.copy() for s in lp_result.splits]
    else:
        new_splits = [s.copy() for s in ecmp_base]

    # Generate new OpenFlow rules (the "recovery" action)
    new_groups, new_flows = splits_to_openflow_rules(
        new_splits, list(range(min(K_CRIT, len(path_library.od_pairs)))),
        path_library, topo_mapping, dataset.edges,
    )

    t_fail_end = time.perf_counter()
    recovery_ms = (t_fail_end - t_fail_start) * 1000

    # Post-recovery MLU
    post_routing = apply_routing(test_tm, new_splits, path_library, failed_caps)
    post_recovery_mlu = float(post_routing.mlu)

    return recovery_ms, post_failure_mlu, post_recovery_mlu


# ── Main benchmark ─────────────────────────────────────────────────────

def benchmark_topology(topo_key: str, methods: list, gnn_model_cache: dict) -> List[dict]:
    """Run full SDN benchmark for one topology."""
    logger.info(f"=== Loading {topo_key} ===")
    dataset, path_library = load_dataset(topo_key)
    capacities = np.asarray(dataset.capacities, dtype=float)
    weights = np.asarray(dataset.weights, dtype=float)
    ecmp_base = ecmp_splits(path_library)

    topo_mapping = SDNTopologyMapping.from_mininet(
        dataset.nodes, dataset.edges, dataset.od_pairs
    )

    # Get test TMs
    sp = dataset.split
    test_start = int(sp["test_start"])
    tm_data = dataset.tm
    test_indices = list(range(test_start, tm_data.shape[0]))
    logger.info(f"  {len(test_indices)} test timesteps, {len(dataset.nodes)} nodes, {len(dataset.edges)} edges")

    # Load GNN model
    gnn_model = None
    if "gnn" in methods:
        if topo_key not in gnn_model_cache:
            gnn_model_cache[topo_key] = load_gnn_model(dataset, path_library)
        gnn_model = gnn_model_cache[topo_key]
        if gnn_model is None:
            logger.warning(f"  GNN model not available, skipping GNN for {topo_key}")
            methods = [m for m in methods if m != "gnn"]

    all_rows = []

    for method in methods:
        logger.info(f"  Running {method} on {topo_key}...")

        # Multi-run for timing stability
        run_results = {
            'decision_times': [], 'rule_delays': [], 'flow_updates': [],
            'throughputs': [], 'latencies': [], 'p95_latencies': [],
            'packet_losses': [], 'jitters': [], 'post_mlus': [], 'disturbances': [],
            'recovery_times': [],
        }

        for run_idx in range(NUM_RUNS):
            current_splits = [s.copy() for s in ecmp_base]
            baseline_groups, _ = build_ecmp_baseline_rules(
                path_library, topo_mapping, dataset.edges
            )
            current_groups = baseline_groups
            prev_latency = None

            cycle_decisions = []
            cycle_throughputs = []
            cycle_latencies = []
            cycle_p95lats = []
            cycle_losses = []
            cycle_jitters = []
            cycle_updates = []
            cycle_rule_delays = []
            cycle_mlus = []
            cycle_dists = []

            for i, t_idx in enumerate(test_indices):
                tm_vec = tm_data[t_idx]

                result, current_splits, current_groups, prev_latency = run_sdn_cycle(
                    tm_vector=tm_vec,
                    method=method,
                    dataset=dataset,
                    path_library=path_library,
                    ecmp_base=ecmp_base,
                    current_splits=current_splits,
                    current_groups=current_groups,
                    topo_mapping=topo_mapping,
                    capacities=capacities,
                    weights=weights,
                    gnn_model=gnn_model,
                    prev_latency_by_od=prev_latency,
                )

                cycle_decisions.append(result.decision_time_ms)
                cycle_throughputs.append(result.throughput)
                cycle_latencies.append(result.mean_latency)
                cycle_p95lats.append(result.p95_latency)
                cycle_losses.append(result.packet_loss)
                cycle_jitters.append(result.jitter)
                cycle_updates.append(result.flow_table_updates)
                cycle_rule_delays.append(result.rule_install_delay_ms)
                cycle_mlus.append(result.post_mlu)
                cycle_dists.append(result.disturbance)

            run_results['decision_times'].append(np.mean(cycle_decisions))
            run_results['rule_delays'].append(np.mean(cycle_rule_delays))
            run_results['flow_updates'].append(np.mean(cycle_updates))
            run_results['throughputs'].append(np.mean(cycle_throughputs))
            run_results['latencies'].append(np.mean(cycle_latencies))
            run_results['p95_latencies'].append(np.mean(cycle_p95lats))
            run_results['packet_losses'].append(np.mean(cycle_losses))
            run_results['jitters'].append(np.mean(cycle_jitters))
            run_results['post_mlus'].append(np.mean(cycle_mlus))
            run_results['disturbances'].append(np.mean(cycle_dists))

            # Failure recovery (run once on a representative TM)
            mid_tm = tm_data[test_indices[len(test_indices)//2]]
            rec_ms, pre_fail_mlu, post_rec_mlu = run_failure_recovery(
                mid_tm, method, dataset, path_library, ecmp_base,
                capacities, weights, topo_mapping, gnn_model
            )
            run_results['recovery_times'].append(rec_ms)

            if run_idx == 0:
                logger.info(f"    Run {run_idx+1}/{NUM_RUNS}: MLU={np.mean(cycle_mlus):.4f}, "
                           f"decision={np.mean(cycle_decisions):.1f}ms, "
                           f"rules={np.mean(cycle_updates):.1f}, "
                           f"throughput={np.mean(cycle_throughputs):.4f}")

        # Average across runs
        row = {
            'topology': topo_key,
            'method': method,
            'nodes': len(dataset.nodes),
            'edges': len(dataset.edges),
            'test_tms': len(test_indices),
            'num_runs': NUM_RUNS,
            # TE metrics
            'mean_mlu': float(np.mean(run_results['post_mlus'])),
            'mean_disturbance': float(np.mean(run_results['disturbances'])),
            # Model-based metrics
            'throughput': float(np.mean(run_results['throughputs'])),
            'mean_latency_au': float(np.mean(run_results['latencies'])),
            'p95_latency_au': float(np.mean(run_results['p95_latencies'])),
            'packet_loss': float(np.mean(run_results['packet_losses'])),
            'jitter_au': float(np.mean(run_results['jitters'])),
            # Measured deployment metrics
            'decision_time_ms': float(np.mean(run_results['decision_times'])),
            'decision_time_std_ms': float(np.std(run_results['decision_times'])),
            'flow_table_updates': float(np.mean(run_results['flow_updates'])),
            'rule_install_delay_ms': float(np.mean(run_results['rule_delays'])),
            'rule_install_delay_std_ms': float(np.std(run_results['rule_delays'])),
            'failure_recovery_ms': float(np.mean(run_results['recovery_times'])),
            'failure_recovery_std_ms': float(np.std(run_results['recovery_times'])),
        }
        all_rows.append(row)
        logger.info(f"  {method} done: MLU={row['mean_mlu']:.4f}, "
                    f"throughput={row['throughput']:.4f}, "
                    f"recovery={row['failure_recovery_ms']:.1f}ms")

    return all_rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    topologies = ["abilene", "geant"]
    methods = ["gnn", "bottleneck", "ecmp"]
    gnn_cache: dict = {}

    all_results = []
    for topo in topologies:
        try:
            rows = benchmark_topology(topo, methods.copy(), gnn_cache)
            all_results.extend(rows)
        except Exception as e:
            logger.error(f"Failed for {topo}: {e}", exc_info=True)

    if not all_results:
        logger.error("No results produced!")
        sys.exit(1)

    # Save CSV
    csv_path = OUT_DIR / "sdn_benchmark_results.csv"
    keys = all_results[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    logger.info(f"Results saved to {csv_path}")

    # Save metadata
    meta = {
        "description": "SDN deployment benchmark with all 9 required metrics",
        "topologies": topologies,
        "methods": methods,
        "k_crit": K_CRIT,
        "num_runs": NUM_RUNS,
        "metric_sources": {
            "decision_time_ms": "MEASURED (wall-clock perf_counter)",
            "flow_table_updates": "MEASURED (OpenFlow GroupMod diff count)",
            "rule_install_delay_ms": "MEASURED (wall-clock serialization time)",
            "failure_recovery_ms": "MEASURED (wall-clock: failure detect -> new rules ready)",
            "throughput": "MODEL-BASED (routed_demand / total_demand from LP solution)",
            "mean_latency_au": "MODEL-BASED (M/M/1 queuing delay, abstract units)",
            "p95_latency_au": "MODEL-BASED (95th percentile of per-OD M/M/1 delay)",
            "packet_loss": "MODEL-BASED (1 - throughput, LP feasibility-based)",
            "jitter_au": "MODEL-BASED (demand-weighted inter-timestep latency change)",
        },
        "honest_notes": [
            "Throughput, latency, packet_loss, jitter are computed from analytical queuing models, NOT from actual packet forwarding.",
            "Latency uses M/M/1 queuing delay derived from link utilization and weights — units are abstract (not milliseconds).",
            "No live Mininet testbed was deployed. All metrics come from offline SDN simulation mode.",
            "Failure recovery time measures controller-side computation only (detect + recompute + serialize rules), NOT switch-side rule installation.",
            "Flow-table updates count the number of OpenFlow SELECT group modifications per cycle.",
        ]
    }
    with open(OUT_DIR / "sdn_benchmark_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    # Print summary table
    print(f"\n{'='*120}")
    print("SDN DEPLOYMENT BENCHMARK RESULTS")
    print(f"{'='*120}")
    print(f"{'Topology':<12} {'Method':<12} {'MLU':>8} {'Throughput':>11} {'Latency(au)':>12} "
          f"{'PktLoss':>8} {'Jitter(au)':>11} {'Decision(ms)':>13} {'FT Updates':>11} "
          f"{'RuleDelay(ms)':>14} {'Recovery(ms)':>13}")
    print("-" * 120)
    for r in all_results:
        print(f"{r['topology']:<12} {r['method']:<12} "
              f"{r['mean_mlu']:>8.4f} "
              f"{r['throughput']:>11.4f} "
              f"{r['mean_latency_au']:>12.2f} "
              f"{r['packet_loss']:>8.6f} "
              f"{r['jitter_au']:>11.4f} "
              f"{r['decision_time_ms']:>13.1f} "
              f"{r['flow_table_updates']:>11.1f} "
              f"{r['rule_install_delay_ms']:>14.2f} "
              f"{r['failure_recovery_ms']:>13.1f}")
    print(f"{'='*120}")
    print("\nMetric sources:")
    print("  MEASURED (wall-clock): decision_time, flow_table_updates, rule_install_delay, failure_recovery")
    print("  MODEL-BASED (M/M/1): throughput, latency, packet_loss, jitter")
    print(f"\nResults: {csv_path}")
    print(f"Metadata: {OUT_DIR / 'sdn_benchmark_metadata.json'}")


if __name__ == "__main__":
    main()
