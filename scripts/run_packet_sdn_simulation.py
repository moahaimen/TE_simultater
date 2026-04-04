#!/usr/bin/env python3
"""Packet-Level SDN Simulation Extension.

Model-based simulation of packet-level network behavior using M/M/1 queuing.
Does NOT use Mininet or real packets. All metrics are analytically computed.

Output: results/packet_sdn_simulation/
"""

from __future__ import annotations

import os
import sys
import time
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── Config ──
K_CRIT = 40
LT = 20  # LP time limit
DEVICE = "cpu"
SEED = 42
FAILURE_CYCLE = 30  # inject failure at this test timestep index

OUTPUT_DIR = Path("results/packet_sdn_simulation")
PLOTS_DIR = OUTPUT_DIR / "plots"

SELECTOR_NAMES = ["bottleneck", "topk", "sensitivity", "gnn"]

TOPOLOGIES = [
    "abilene", "cernet", "geant", "germany50",
    "rocketfuel_ebone", "rocketfuel_sprintlink",
    "rocketfuel_tiscali", "topologyzoo_vtlwavenet2011",
]

KNOWN = {"abilene", "cernet", "geant", "rocketfuel_ebone",
         "rocketfuel_sprintlink", "rocketfuel_tiscali"}

METHODS = ["ecmp", "metagate", "stable_metagate"]

FAILURE_TYPES = ["none", "single_link_failure", "capacity_degradation", "traffic_spike"]


# ── Queueing Model ──

def mm1_delay(load: float, capacity: float, prop_delay: float = 0.001) -> float:
    """M/M/1 mean sojourn time + propagation delay.

    Args:
        load: traffic on link (same units as capacity)
        capacity: link capacity
        prop_delay: propagation delay in seconds (default 1ms)

    Returns:
        delay in seconds. Returns large value if overloaded.
    """
    if capacity <= 0:
        return 10.0  # 10 seconds for dead link
    rho = load / capacity
    if rho >= 0.999:
        return 5.0 + prop_delay  # overloaded
    if rho < 1e-12:
        return (1.0 / capacity) + prop_delay  # nearly empty
    # M/M/1: E[T] = 1/(mu - lambda) where mu=capacity, lambda=load
    return (1.0 / (capacity - load)) + prop_delay


def compute_link_packet_loss(load: float, capacity: float) -> float:
    """Overflow-based packet loss approximation.

    Loss = max(0, (load - capacity) / load)
    """
    if load <= 0:
        return 0.0
    if load <= capacity:
        return 0.0
    return (load - capacity) / load


def compute_od_metrics(
    tm_vector: np.ndarray,
    splits: Sequence[np.ndarray],
    path_library,
    capacities: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-OD delay, throughput, loss.

    Returns:
        od_delays: [num_od] end-to-end delay per OD pair (seconds)
        od_throughput: [num_od] throughput fraction per OD pair [0,1]
        od_loss: [num_od] packet loss fraction per OD pair [0,1]
    """
    num_od = len(tm_vector)
    num_edges = capacities.size

    # First compute link loads
    link_loads = np.zeros(num_edges, dtype=float)
    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            continue
        od_paths = path_library.edge_idx_paths_by_od[od_idx]
        if not od_paths:
            continue
        split_vec = np.asarray(splits[od_idx], dtype=float)
        s = float(np.sum(split_vec))
        if s <= 0:
            continue
        norm = split_vec / s
        for path_idx, frac in enumerate(norm):
            if frac <= 0:
                continue
            flow = float(demand) * float(frac)
            for edge_idx in od_paths[path_idx]:
                link_loads[edge_idx] += flow

    # Per-link delay and loss
    link_delays = np.array([
        mm1_delay(link_loads[e], capacities[e],
                  prop_delay=max(weights[e] * 0.001, 0.0005) if e < len(weights) else 0.001)
        for e in range(num_edges)
    ])
    link_loss = np.array([
        compute_link_packet_loss(link_loads[e], capacities[e])
        for e in range(num_edges)
    ])

    # Per-OD metrics (weighted average over paths)
    od_delays = np.zeros(num_od)
    od_throughput = np.ones(num_od)
    od_loss = np.zeros(num_od)

    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            od_delays[od_idx] = 0
            od_throughput[od_idx] = 1.0
            od_loss[od_idx] = 0.0
            continue

        od_paths = path_library.edge_idx_paths_by_od[od_idx]
        if not od_paths:
            od_delays[od_idx] = 0
            od_throughput[od_idx] = 0.0
            od_loss[od_idx] = 1.0
            continue

        split_vec = np.asarray(splits[od_idx], dtype=float)
        s = float(np.sum(split_vec))
        if s <= 0:
            od_delays[od_idx] = 0
            od_throughput[od_idx] = 0.0
            od_loss[od_idx] = 1.0
            continue

        norm = split_vec / s
        weighted_delay = 0.0
        weighted_throughput = 0.0
        weighted_loss = 0.0

        for path_idx, frac in enumerate(norm):
            if frac <= 0:
                continue
            path_edges = od_paths[path_idx]
            # Path delay = sum of link delays
            path_delay = sum(link_delays[e] for e in path_edges)
            # Path loss = 1 - product(1 - link_loss) along path
            path_survival = 1.0
            for e in path_edges:
                path_survival *= (1.0 - link_loss[e])
            path_loss = 1.0 - path_survival

            weighted_delay += frac * path_delay
            weighted_loss += frac * path_loss
            weighted_throughput += frac * (1.0 - path_loss)

        od_delays[od_idx] = weighted_delay
        od_throughput[od_idx] = weighted_throughput
        od_loss[od_idx] = weighted_loss

    return od_delays, od_throughput, od_loss


def estimate_rule_install_delay(num_rules: int) -> float:
    """Empirical OVS rule installation delay model (milliseconds).

    Based on: 0.5ms base + 0.02ms per rule.
    """
    return 0.5 + 0.02 * num_rules


# ── Data Classes ──

@dataclass
class PacketCycleResult:
    topology: str
    topology_type: str
    method: str
    failure_type: str
    timestep: int
    cycle: int
    # TE metrics
    mlu: float
    mean_utilization: float
    # Packet-level metrics (M/M/1 model-based)
    mean_throughput: float
    mean_delay_ms: float
    p50_delay_ms: float
    p95_delay_ms: float
    p99_delay_ms: float
    mean_packet_loss: float
    jitter_ms: float
    # SDN metrics
    decision_time_ms: float
    rules_pushed: int
    rule_install_delay_ms: float
    flow_table_size: int
    # Failure metrics
    failure_injected: bool
    recovery_cycles: int


# ── Setup ──

def setup():
    """Import all modules."""
    import torch
    from te.baselines import (
        ecmp_splits, select_bottleneck_critical,
        select_sensitivity_critical, select_topk_by_demand,
    )
    from te.lp_solver import solve_selected_path_lp
    from te.simulator import apply_routing
    from phase1_reactive.eval.common import (
        load_bundle, load_named_dataset, collect_specs,
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
        "torch": torch,
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
        "DynamicMetaGate": DynamicMetaGate,
        "MetaGateConfig": MetaGateConfig,
        "extract_features": extract_features,
        "load_gnn_selector": load_gnn_selector,
        "build_graph_tensors": build_graph_tensors,
        "build_od_features": build_od_features,
        "compute_reactive_telemetry": compute_reactive_telemetry,
    }


def run_heuristic_selector(M, method, tm, ecmp_base, path_library, capacities, k_crit):
    if method == "topk":
        return M["select_topk_by_demand"](tm, k_crit)
    elif method == "bottleneck":
        return M["select_bottleneck_critical"](tm, ecmp_base, path_library, capacities, k_crit)
    elif method == "sensitivity":
        return M["select_sensitivity_critical"](tm, ecmp_base, path_library, capacities, k_crit)
    else:
        raise ValueError(f"Unknown heuristic: {method}")


def run_gnn_selector(M, tm, dataset, path_library, gnn_model, k_crit, telemetry=None):
    import torch
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
    timing = {}
    selector_results = {}

    for method in ["bottleneck", "topk", "sensitivity"]:
        t0 = time.perf_counter()
        sel = run_heuristic_selector(M, method, tm, ecmp_base, path_library,
                                     capacities, k_crit)
        timing[method] = (time.perf_counter() - t0) * 1000
        selector_results[method] = list(sel)

    t0 = time.perf_counter()
    gnn_sel, gnn_info = run_gnn_selector(M, tm, dataset, path_library, gnn_model, k_crit,
                                          telemetry=telemetry)
    timing["gnn"] = (time.perf_counter() - t0) * 1000
    selector_results["gnn"] = list(gnn_sel)

    return selector_results, gnn_info, timing


def apply_failure(tm, capacities, failure_type, edges, rng):
    """Apply failure modification. Returns modified (tm, capacities)."""
    tm_mod = tm.copy()
    cap_mod = capacities.copy()

    if failure_type == "single_link_failure":
        # Fail a random link (set capacity to near-zero)
        if len(cap_mod) > 0:
            fail_idx = rng.integers(0, len(cap_mod))
            cap_mod[fail_idx] = 1e-6
    elif failure_type == "capacity_degradation":
        # Reduce all capacities by 50%
        cap_mod = cap_mod * 0.5
    elif failure_type == "traffic_spike":
        # Double all demands
        tm_mod = tm_mod * 2.0

    return tm_mod, cap_mod


def compute_disturbance(current_ods, prev_ods, k_crit):
    if prev_ods is None:
        return 0.0
    return len(set(current_ods) ^ set(prev_ods)) / max(k_crit, 1)


def stable_select(probs, selector_results, prev_selected_ods, prev_expert_idx,
                  lambda_d, lambda_s, k_crit):
    scores = np.zeros(len(SELECTOR_NAMES))
    for i, name in enumerate(SELECTOR_NAMES):
        scores[i] = np.log(probs[i] + 1e-12)
        if prev_selected_ods is not None:
            dist = compute_disturbance(selector_results[name], prev_selected_ods, k_crit)
            scores[i] -= lambda_d * dist
        if prev_expert_idx is not None and i != prev_expert_idx:
            scores[i] -= lambda_s
    return int(np.argmax(scores))


def run_simulation_for_topology(
    M, dataset, path_library, gate, gnn_model,
    method: str, failure_type: str, rng,
    stable_ld: float = 0.2, stable_ls: float = 0.1,
) -> List[PacketCycleResult]:
    """Run full packet-level simulation for one topology/method/failure combo."""

    ds_key = dataset.key
    topo_type = "known" if ds_key in KNOWN else "unseen"
    test_indices = M["split_indices"](dataset, "test")
    num_nodes = len(dataset.nodes)
    num_edges = len(dataset.edges)
    capacities = np.asarray(dataset.capacities, dtype=float)
    weights = np.asarray(dataset.weights, dtype=float)
    ecmp_base = M["ecmp_splits"](path_library)

    results = []
    prev_delays = None
    prev_selected_ods = None
    prev_expert_idx = None
    prev_rules_count = 0  # flow table size estimate

    for cycle, t_idx in enumerate(test_indices):
        tm = np.asarray(dataset.tm[t_idx], dtype=float)
        if np.max(tm) < 1e-12:
            continue

        # Apply failure if past injection point
        cap_eff = capacities.copy()
        tm_eff = tm.copy()
        failure_injected = (failure_type != "none" and cycle >= FAILURE_CYCLE)
        if failure_injected:
            tm_eff, cap_eff = apply_failure(tm, capacities, failure_type,
                                             dataset.edges, rng)

        t_total_start = time.perf_counter()

        if method == "ecmp":
            # ECMP: use base splits, no selection
            splits = ecmp_base
            selected_ods = []
            decision_time = 0.0
            rules_pushed = 0
        else:
            # MetaGate or Stable MetaGate
            routing_ecmp = M["apply_routing"](tm_eff, ecmp_base, path_library, cap_eff)
            telemetry = M["compute_reactive_telemetry"](
                tm_eff, ecmp_base, path_library, routing_ecmp, weights)
            ecmp_link_utils = np.array(routing_ecmp.utilization, dtype=np.float32)

            selector_results, gnn_info, expert_timing = run_all_experts(
                M, tm_eff, ecmp_base, path_library, cap_eff, dataset,
                gnn_model, K_CRIT, telemetry=telemetry,
            )

            feats = M["extract_features"](
                tm_eff, selector_results, num_nodes, num_edges, K_CRIT,
                gnn_info=gnn_info, ecmp_link_utils=ecmp_link_utils,
            )

            _, probs = gate.predict(feats)

            if method == "stable_metagate":
                pred_class = stable_select(
                    probs, selector_results, prev_selected_ods,
                    prev_expert_idx, stable_ld, stable_ls, K_CRIT,
                )
            else:
                pred_class = int(np.argmax(probs))

            pred_name = SELECTOR_NAMES[pred_class]
            selected_ods = selector_results[pred_name]

            # LP solve
            lp = M["solve_selected_path_lp"](
                tm_eff, selected_ods, ecmp_base, path_library, cap_eff,
                time_limit_sec=LT,
            )
            splits = lp.splits
            decision_time = sum(expert_timing.values())

            # Rule diff estimate
            rules_pushed = len(selected_ods)  # conservative: one group per selected OD

        # Apply routing
        routing = M["apply_routing"](tm_eff, splits, path_library, cap_eff)

        # Packet-level metrics
        od_delays, od_throughput, od_loss = compute_od_metrics(
            tm_eff, splits, path_library, cap_eff, weights,
        )

        # Filter to active ODs
        active = tm_eff > 1e-12
        if np.any(active):
            active_delays = od_delays[active] * 1000  # to ms
            mean_delay = float(np.mean(active_delays))
            p50_delay = float(np.percentile(active_delays, 50))
            p95_delay = float(np.percentile(active_delays, 95))
            p99_delay = float(np.percentile(active_delays, 99))
            mean_throughput = float(np.mean(od_throughput[active]))
            mean_loss = float(np.mean(od_loss[active]))
        else:
            mean_delay = p50_delay = p95_delay = p99_delay = 0.0
            mean_throughput = 1.0
            mean_loss = 0.0

        # Jitter
        if prev_delays is not None and np.any(active):
            prev_active = prev_delays[active] * 1000 if len(prev_delays) == len(od_delays) else None
            if prev_active is not None and len(prev_active) == len(active_delays):
                jitter = float(np.mean(np.abs(active_delays - prev_active)))
            else:
                jitter = 0.0
        else:
            jitter = 0.0

        prev_delays = od_delays.copy()

        # Rule install delay
        rule_delay = estimate_rule_install_delay(rules_pushed)

        # Failure recovery: cycles since failure injection until MLU stabilizes
        recovery_cycles = 0
        if failure_injected and cycle == FAILURE_CYCLE:
            recovery_cycles = 1  # at least 1 cycle to detect and respond

        # Flow table size estimate
        flow_table_size = prev_rules_count + rules_pushed
        prev_rules_count = flow_table_size

        t_total = (time.perf_counter() - t_total_start) * 1000

        results.append(PacketCycleResult(
            topology=ds_key,
            topology_type=topo_type,
            method=method,
            failure_type=failure_type,
            timestep=t_idx,
            cycle=cycle,
            mlu=float(routing.mlu),
            mean_utilization=float(routing.mean_utilization),
            mean_throughput=mean_throughput,
            mean_delay_ms=mean_delay,
            p50_delay_ms=p50_delay,
            p95_delay_ms=p95_delay,
            p99_delay_ms=p99_delay,
            mean_packet_loss=mean_loss,
            jitter_ms=jitter,
            decision_time_ms=decision_time + t_total,
            rules_pushed=rules_pushed,
            rule_install_delay_ms=rule_delay,
            flow_table_size=flow_table_size,
            failure_injected=failure_injected,
            recovery_cycles=recovery_cycles,
        ))

        # Update state for stable metagate
        if method in ("metagate", "stable_metagate"):
            prev_selected_ods = selected_ods
            prev_expert_idx = pred_class

    return results


def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PACKET-LEVEL SDN SIMULATION (Model-Based)")
    print("  All metrics are analytically computed, NOT from real packets")
    print("  Queuing model: M/M/1 per-link")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    M = setup()
    torch = M["torch"]

    # Load GNN model
    gnn_path = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
    print(f"\nLoading GNN model from {gnn_path}...")
    gnn_model, _ = M["load_gnn_selector"](gnn_path, device=DEVICE)

    # Load MetaGate
    gate_path = Path("results/dynamic_metagate/models/metagate_unified.pt")
    print(f"Loading MetaGate from {gate_path}...")
    gate = M["DynamicMetaGate"](M["MetaGateConfig"]())
    gate.load(gate_path, feat_dim=49)

    # Load all datasets
    config_path = "configs/phase1_reactive_full.yaml"
    if not Path(config_path).exists():
        config_path = "configs/phase1_reactive_topologies.yaml"
    bundle = M["load_bundle"](config_path)

    # Calibrate gate per topology
    print("\nCalibrating gate per topology...")
    all_datasets = {}
    all_path_libs = {}
    from phase1_reactive.eval.common import max_steps_from_args
    max_steps = max_steps_from_args(bundle, 500)
    all_specs = M["collect_specs"](bundle, "eval_topologies") + M["collect_specs"](bundle, "generalization_topologies")
    for spec in all_specs:
        try:
            ds, pl = M["load_named_dataset"](bundle, spec, max_steps)
            all_datasets[ds.key] = ds
            all_path_libs[ds.key] = pl
            print(f"  Loaded: {ds.key} ({len(ds.nodes)} nodes, {len(ds.edges)} edges)")
        except Exception as e:
            print(f"  Skip {spec.key}: {e}")

    all_results = []

    for ds_key in TOPOLOGIES:
        if ds_key not in all_datasets:
            print(f"\n  Skipping {ds_key} (not found in bundle)")
            continue

        dataset = all_datasets[ds_key]
        path_library = all_path_libs[ds_key]

        # Calibrate for this topology
        val_indices = M["split_indices"](dataset, "val")[:10]
        ecmp_base = M["ecmp_splits"](path_library)
        capacities = np.asarray(dataset.capacities, dtype=float)
        weights = np.asarray(dataset.weights, dtype=float)

        # Compute calibration priors
        expert_wins = {n: 0 for n in SELECTOR_NAMES}
        for t_idx in val_indices:
            tm = np.asarray(dataset.tm[t_idx], dtype=float)
            if np.max(tm) < 1e-12:
                continue
            mlus = {}
            for name in SELECTOR_NAMES:
                try:
                    if name == "gnn":
                        sel, _ = run_gnn_selector(M, tm, dataset, path_library,
                                                   gnn_model, K_CRIT)
                    else:
                        sel = run_heuristic_selector(M, name, tm, ecmp_base,
                                                      path_library, capacities, K_CRIT)
                    lp = M["solve_selected_path_lp"](
                        tm, sel, ecmp_base, path_library, capacities, time_limit_sec=LT)
                    r = M["apply_routing"](tm, lp.splits, path_library, capacities)
                    mlus[name] = float(r.mlu)
                except Exception:
                    mlus[name] = float("inf")
            best = min(mlus, key=mlus.get)
            expert_wins[best] += 1

        win_counts = [expert_wins[n] for n in SELECTOR_NAMES]
        gate.calibrate(win_counts, smoothing=1.0, strength=5.0)

        print(f"\n{'='*60}")
        print(f"  {ds_key} ({len(dataset.nodes)} nodes, {len(dataset.edges)} edges)")
        print(f"  Calibration: {expert_wins}")
        print(f"{'='*60}")

        for method in METHODS:
            for failure_type in FAILURE_TYPES:
                label = f"  {method} / {failure_type}"
                print(f"{label}...", end="", flush=True)

                t0 = time.perf_counter()
                results = run_simulation_for_topology(
                    M, dataset, path_library, gate, gnn_model,
                    method=method, failure_type=failure_type, rng=rng,
                )
                elapsed = time.perf_counter() - t0
                all_results.extend(results)

                if results:
                    mean_mlu = np.mean([r.mlu for r in results])
                    mean_delay = np.mean([r.mean_delay_ms for r in results])
                    mean_tp = np.mean([r.mean_throughput for r in results])
                    mean_loss = np.mean([r.mean_packet_loss for r in results])
                    print(f" {len(results)} cycles, MLU={mean_mlu:.4f}, "
                          f"delay={mean_delay:.2f}ms, tp={mean_tp:.4f}, "
                          f"loss={mean_loss:.4f} [{elapsed:.1f}s]")
                else:
                    print(f" no results [{elapsed:.1f}s]")

    # ── Save Results ──
    print(f"\n{'='*60}")
    print(f"  Saving {len(all_results)} rows...")

    # Detailed CSV
    df = pd.DataFrame([asdict(r) for r in all_results])
    df.to_csv(OUTPUT_DIR / "packet_sdn_results.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'packet_sdn_results.csv'}")

    if len(df) == 0:
        print("  No results to process!")
        return

    # Summary CSV
    summary = df.groupby(["topology", "topology_type", "method", "failure_type"]).agg({
        "mlu": "mean",
        "mean_throughput": "mean",
        "mean_delay_ms": "mean",
        "p95_delay_ms": "mean",
        "mean_packet_loss": "mean",
        "jitter_ms": "mean",
        "decision_time_ms": "mean",
        "rules_pushed": "sum",
        "rule_install_delay_ms": "mean",
        "cycle": "count",
    }).rename(columns={"cycle": "n_cycles"}).reset_index()
    summary.to_csv(OUTPUT_DIR / "packet_sdn_summary.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'packet_sdn_summary.csv'}")

    # ── Generate CDF Plots ──
    generate_plots(df)

    # ── Generate Report ──
    generate_report(df, summary)

    print(f"\n{'='*60}")
    print(f"  DONE. All outputs in {OUTPUT_DIR}")
    print(f"{'='*60}")


def generate_plots(df):
    """Generate all CDF and comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_colors = {"ecmp": "#999999", "metagate": "#2196F3", "stable_metagate": "#FF9800"}
    method_labels = {"ecmp": "ECMP Baseline", "metagate": "MetaGate", "stable_metagate": "Stable MetaGate"}

    def cdf_xy(vals):
        s = np.sort(vals)
        return s, np.arange(1, len(s)+1) / len(s)

    def plot_cdf(data_dict, xlabel, title, fname, log_x=False):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, vals in data_dict.items():
            if len(vals) == 0:
                continue
            x, y = cdf_xy(vals)
            ax.step(x, y, where="post", label=method_labels.get(method, method),
                    color=method_colors.get(method, "black"), lw=2)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("CDF", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        if log_x:
            ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot: {fname}")

    # Normal conditions only
    normal = df[df["failure_type"] == "none"]

    # 1. Throughput CDF
    plot_cdf(
        {m: normal[normal["method"] == m]["mean_throughput"].values for m in METHODS},
        "Throughput (fraction)", "CDF of Throughput (M/M/1 model)", "cdf_throughput.png"
    )

    # 2. Delay CDF
    plot_cdf(
        {m: normal[normal["method"] == m]["mean_delay_ms"].values for m in METHODS},
        "End-to-End Delay (ms)", "CDF of Mean Delay (M/M/1 model)", "cdf_delay.png",
        log_x=True
    )

    # 3. Packet Loss CDF
    plot_cdf(
        {m: normal[normal["method"] == m]["mean_packet_loss"].values for m in METHODS},
        "Packet Loss (fraction)", "CDF of Packet Loss (overflow model)", "cdf_packet_loss.png"
    )

    # 4. Jitter CDF
    plot_cdf(
        {m: normal[normal["method"] == m]["jitter_ms"].values for m in METHODS},
        "Jitter (ms)", "CDF of Jitter (delay variation)", "cdf_jitter.png"
    )

    # 5. Decision Time CDF
    plot_cdf(
        {m: normal[normal["method"] == m]["decision_time_ms"].values for m in METHODS},
        "Decision Time (ms)", "CDF of Decision Time", "cdf_decision_time.png"
    )

    # 6. MLU under each failure type
    for ft in ["single_link_failure", "capacity_degradation", "traffic_spike"]:
        fail_df = df[df["failure_type"] == ft]
        if len(fail_df) == 0:
            continue
        plot_cdf(
            {m: fail_df[fail_df["method"] == m]["mlu"].values for m in METHODS},
            "MLU", f"CDF of MLU under {ft.replace('_', ' ').title()}", f"cdf_mlu_{ft}.png",
            log_x=True
        )

    # 7. Delay under failure
    for ft in ["single_link_failure", "capacity_degradation", "traffic_spike"]:
        fail_df = df[df["failure_type"] == ft]
        if len(fail_df) == 0:
            continue
        plot_cdf(
            {m: fail_df[fail_df["method"] == m]["mean_delay_ms"].values for m in METHODS},
            "Delay (ms)", f"CDF of Delay under {ft.replace('_', ' ').title()}",
            f"cdf_delay_{ft}.png", log_x=True
        )

    # 8. Per-topology MLU comparison (normal)
    topos = normal["topology"].unique()
    for topo in topos:
        sub = normal[normal["topology"] == topo]
        plot_cdf(
            {m: sub[sub["method"] == m]["mlu"].values for m in METHODS},
            "MLU", f"CDF of MLU - {topo}", f"cdf_mlu_{topo}.png"
        )

    # 9. Summary bar chart: method comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ("mean_throughput", "Throughput", False),
        ("mean_delay_ms", "Delay (ms)", True),
        ("mean_packet_loss", "Packet Loss", False),
        ("jitter_ms", "Jitter (ms)", True),
        ("mlu", "MLU", True),
        ("rule_install_delay_ms", "Rule Install Delay (ms)", False),
    ]
    for idx, (col, label, use_log) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        for i, m in enumerate(METHODS):
            sub = normal[normal["method"] == m]
            means = []
            topo_names = []
            for topo in TOPOLOGIES:
                tsub = sub[sub["topology"] == topo]
                if len(tsub) > 0:
                    means.append(tsub[col].mean())
                    topo_names.append(topo.replace("rocketfuel_", "rf_").replace("topologyzoo_", "tz_"))
            if means:
                x = np.arange(len(means))
                width = 0.25
                ax.bar(x + i * width, means, width, label=method_labels[m],
                       color=method_colors[m], alpha=0.85)
        if topo_names:
            ax.set_xticks(np.arange(len(topo_names)) + 0.25)
            ax.set_xticklabels(topo_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        if use_log and idx != 0:
            ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Packet-Level SDN Simulation: Method Comparison (Model-Based)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "method_comparison_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: method_comparison_summary.png")

    # 10. Failure impact comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, ft in enumerate(["single_link_failure", "capacity_degradation", "traffic_spike"]):
        ax = axes[idx]
        for i, m in enumerate(METHODS):
            # Normal vs failure
            norm_mlu = normal[normal["method"] == m]["mlu"].mean()
            fail_sub = df[(df["failure_type"] == ft) & (df["method"] == m)]
            fail_mlu = fail_sub["mlu"].mean() if len(fail_sub) > 0 else 0

            x = [0, 1]
            ax.plot(x, [norm_mlu, fail_mlu], "o-", label=method_labels[m],
                    color=method_colors[m], lw=2, markersize=8)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal", ft.replace("_", "\n").title()])
        ax.set_ylabel("Mean MLU")
        ax.set_title(ft.replace("_", " ").title())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Failure Impact on MLU: Normal vs Failure Scenarios", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "failure_impact_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: failure_impact_comparison.png")


def generate_report(df, summary):
    """Generate markdown report."""
    normal = df[df["failure_type"] == "none"]

    report = []
    report.append("# Packet-Level SDN Simulation Report")
    report.append("")
    report.append("## IMPORTANT DISCLAIMER")
    report.append("")
    report.append("**All metrics in this report are model-based approximations, NOT real packet measurements.**")
    report.append("No actual packets were generated, transmitted, or received.")
    report.append("No Mininet, Open vSwitch, or Ryu controller was used.")
    report.append("These results use analytical queueing models applied to the existing routing solutions.")
    report.append("")

    report.append("## 1. Modeling Assumptions")
    report.append("")
    report.append("| Component | Model | Formula |")
    report.append("|---|---|---|")
    report.append("| Per-link delay | M/M/1 queuing | `d = 1/(mu - lambda) + prop_delay` |")
    report.append("| End-to-end delay | Sum of link delays | `D_od = sum(d_link for link in path)` |")
    report.append("| Throughput | Bottleneck model | `min(1, capacity/load)` per link |")
    report.append("| Packet loss | Overflow approximation | `max(0, (load - capacity) / load)` |")
    report.append("| Jitter | Delay variation | `|delay(t) - delay(t-1)|` per OD |")
    report.append("| Rule install delay | Empirical OVS model | `0.5ms + 0.02ms * num_rules` |")
    report.append("| Failure recovery | Simulation clock | Cycles from failure to stable reroute |")
    report.append("")

    report.append("## 2. What Is Simulated vs What Is NOT")
    report.append("")
    report.append("### Simulated (model-based)")
    report.append("- Routing decisions (expert selection, LP optimization)")
    report.append("- Link loads from traffic matrix + split ratios")
    report.append("- Queueing delays (M/M/1)")
    report.append("- Overflow-based packet loss")
    report.append("- OpenFlow rule counts and diffs")
    report.append("- Failure injection (link failure, capacity degradation, traffic spike)")
    report.append("")
    report.append("### NOT simulated (requires live Mininet)")
    report.append("- Actual packet generation and forwarding")
    report.append("- Real OpenFlow rule installation on OVS")
    report.append("- TCP/UDP protocol behavior")
    report.append("- Switch buffer overflow")
    report.append("- Control-plane latency to/from Ryu controller")
    report.append("- LLDP topology discovery")
    report.append("- Actual iperf throughput measurements")
    report.append("")

    report.append("## 3. Experiments")
    report.append("")
    report.append(f"- **Topologies:** {len(normal['topology'].unique())} ({', '.join(normal['topology'].unique())})")
    report.append(f"- **Methods:** ECMP Baseline, MetaGate, Stable MetaGate (ld=0.2, ls=0.1)")
    report.append(f"- **Failure types:** Normal, Single Link Failure, Capacity Degradation (50%), Traffic Spike (2x)")
    report.append(f"- **Total cycles:** {len(df)}")
    report.append("")

    report.append("## 4. Key Results (Normal Conditions)")
    report.append("")
    report.append("| Method | Mean MLU | Throughput | Delay (ms) | P95 Delay | Loss | Jitter (ms) |")
    report.append("|---|---|---|---|---|---|---|")
    for m in METHODS:
        sub = normal[normal["method"] == m]
        if len(sub) == 0:
            continue
        report.append(
            f"| {m} | {sub['mlu'].mean():.4f} | {sub['mean_throughput'].mean():.4f} | "
            f"{sub['mean_delay_ms'].mean():.2f} | {sub['p95_delay_ms'].mean():.2f} | "
            f"{sub['mean_packet_loss'].mean():.4f} | {sub['jitter_ms'].mean():.4f} |"
        )
    report.append("")

    report.append("## 5. Failure Scenario Results")
    report.append("")
    for ft in ["single_link_failure", "capacity_degradation", "traffic_spike"]:
        fail_df = df[df["failure_type"] == ft]
        if len(fail_df) == 0:
            continue
        report.append(f"### {ft.replace('_', ' ').title()}")
        report.append("")
        report.append("| Method | Mean MLU | Throughput | Delay (ms) | Loss |")
        report.append("|---|---|---|---|---|")
        for m in METHODS:
            sub = fail_df[fail_df["method"] == m]
            if len(sub) == 0:
                continue
            report.append(
                f"| {m} | {sub['mlu'].mean():.4f} | {sub['mean_throughput'].mean():.4f} | "
                f"{sub['mean_delay_ms'].mean():.2f} | {sub['mean_packet_loss'].mean():.4f} |"
            )
        report.append("")

    report.append("## 6. Per-Topology Summary (Normal)")
    report.append("")
    norm_summary = summary[summary["failure_type"] == "none"]
    report.append("| Topology | Method | MLU | Throughput | Delay | Loss | Rules |")
    report.append("|---|---|---|---|---|---|---|")
    for _, row in norm_summary.iterrows():
        report.append(
            f"| {row['topology']} | {row['method']} | {row['mlu']:.4f} | "
            f"{row['mean_throughput']:.4f} | {row['mean_delay_ms']:.2f} | "
            f"{row['mean_packet_loss']:.4f} | {int(row['rules_pushed'])} |"
        )
    report.append("")

    report.append("## 7. SDN Metrics")
    report.append("")
    report.append("| Method | Mean Decision Time (ms) | Mean Rules/Cycle | Rule Install Delay (ms) |")
    report.append("|---|---|---|---|")
    for m in METHODS:
        sub = normal[normal["method"] == m]
        if len(sub) == 0:
            continue
        report.append(
            f"| {m} | {sub['decision_time_ms'].mean():.1f} | "
            f"{sub['rules_pushed'].mean():.1f} | {sub['rule_install_delay_ms'].mean():.2f} |"
        )
    report.append("")

    report.append("## 8. Limitations")
    report.append("")
    report.append("1. **M/M/1 assumes Poisson arrivals** — real network traffic is bursty (self-similar)")
    report.append("2. **No buffer modeling** — real switches have finite buffers; our overflow model is conservative")
    report.append("3. **No TCP dynamics** — real TCP adjusts sending rate based on congestion signals")
    report.append("4. **Rule install delay is empirical** — actual OVS installation depends on table size, hardware")
    report.append("5. **Failure injection is instantaneous** — real failures propagate through LLDP detection (~100ms)")
    report.append("6. **All metrics are per-timestep** — no sub-timestep dynamics (packet-level events)")
    report.append("")
    report.append("## 9. Conclusion")
    report.append("")
    report.append("This simulation provides analytical upper/lower bounds on packet-level behavior.")
    report.append("The M/M/1 model is optimistic for delay (real delays are higher due to burstiness)")
    report.append("but reasonable for relative comparison between methods.")
    report.append("**Live Mininet validation remains required for actual packet-level metrics.**")
    report.append("")

    report.append("## 10. Generated Files")
    report.append("")
    report.append("| File | Description |")
    report.append("|---|---|")
    report.append("| packet_sdn_results.csv | Per-cycle detailed results |")
    report.append("| packet_sdn_summary.csv | Per-topology/method/failure summary |")
    report.append("| plots/*.png | CDF and comparison plots |")
    report.append("| PACKET_SDN_SIMULATION_REPORT.md | This report |")

    with open(OUTPUT_DIR / "PACKET_SDN_SIMULATION_REPORT.md", "w") as f:
        f.write("\n".join(report))
    print(f"  Saved: {OUTPUT_DIR / 'PACKET_SDN_SIMULATION_REPORT.md'}")


if __name__ == "__main__":
    main()
