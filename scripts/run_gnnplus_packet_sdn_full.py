#!/usr/bin/env python3
"""
GNN+ Packet-Level SDN Simulation Report — Full 8-Topology Evaluation

Based on run_sdn_benchmark.py, extended to:
- 8 topologies (6 known + 2 unseen)
- GNN+ method with enriched features, dropout=0.2, fixed K=40
- All required failure scenarios
- Full SDN metrics per old Packet_SDN_Simulation_Report.docx formulas

Output: results/gnnplus_packet_sdn_report/
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple
from collections import defaultdict

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
from phase1_reactive.routing.path_cache import (
    assert_selected_ods_have_paths,
    build_modified_paths,
    surviving_od_mask,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Output directory
OUT_DIR = project_root / "results" / "gnnplus_packet_sdn_report"
K_CRIT = 40
K_PATHS = 3
NUM_RUNS = 3  # For timing stability
SEEDS = [42, 43, 44, 45, 46]

# Topologies: 6 known + 2 unseen
TOPOLOGIES = {
    'abilene': {'status': 'known', 'key': 'abilene'},
    'cernet': {'status': 'known', 'key': 'cernet'},
    'geant': {'status': 'known', 'key': 'geant'},
    'ebone': {'status': 'known', 'key': 'ebone'},
    'sprintlink': {'status': 'known', 'key': 'sprintlink'},
    'tiscali': {'status': 'known', 'key': 'tiscali'},
    'germany50': {'status': 'unseen', 'key': 'germany50'},
    'vtlwavenet2011': {'status': 'unseen', 'key': 'vtlwavenet2011'},
}

# Methods to evaluate
CORE_METHODS = ["ecmp", "bottleneck", "gnn", "gnnplus"]
OPTIONAL_METHODS = ["ospf", "topk", "sensitivity", "metagate", "stable_metagate"]
PAPER_BASELINES = ["flexdate", "flexentry", "erodrl"]

# Failure scenarios
FAILURE_SCENARIOS = [
    "normal",
    "single_link_failure",  # highest utilization link
    "random_link_failure_1",  # random single link
    "random_link_failure_2",  # two random links
    "capacity_degradation_50",
    "traffic_spike_2x"
]


# ── GNN+ Model Definition ─────────────────────────────────────────────

class GNNPlusScorer(nn.Module):
    """GNN+ with enriched features and dropout."""
    
    def __init__(self, in_channels=30, hidden_channels=64, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        self.conv1 = nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, hidden_channels)
        
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, x, edge_index, od_pairs_batch):
        h = self.node_encoder(x)
        h = F.relu(self.conv1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h)
        
        src_features = h[od_pairs_batch[:, 0]]
        dst_features = h[od_pairs_batch[:, 1]]
        edge_features = torch.cat([src_features, dst_features], dim=-1)
        scores = self.edge_scorer(edge_features).squeeze(-1)
        return scores


def load_gnnplus_model(dataset, path_library, device="cpu"):
    """Load GNN+ model using the correct architecture from gnn_plus_selector."""
    try:
        from phase1_reactive.drl.gnn_plus_selector import load_gnn_plus, GNNPlusConfig
        
        ckpt = project_root / "results/gnn_plus_retrained_fixedk40/gnn_plus_fixed_k40.pt"
        if ckpt.exists():
            model, cfg = load_gnn_plus(ckpt, device=device)
            logger.info(f"Loaded GNN+ model from {ckpt}")
            return model
        else:
            logger.warning("No GNN+ checkpoint found")
            return None
    except Exception as e:
        logger.warning(f"Could not load GNN+ model: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def gnnplus_select_critical(model, dataset, path_library, tm_vector, k_crit=K_CRIT, device="cpu"):
    """Select critical flows using GNN+."""
    try:
        from phase1_reactive.drl.gnn_plus_selector import (
            build_graph_tensors_plus, build_od_features_plus
        )

        graph_data = build_graph_tensors_plus(
            dataset,
            tm_vector=tm_vector,
            path_library=path_library,
            device=device,
        )
        od_data = build_od_features_plus(
            dataset, tm_vector, path_library, device=device
        )

        tm_arr = np.asarray(tm_vector, dtype=np.float64)
        active_mask = (tm_arr > 1e-12) & surviving_od_mask(path_library)

        selected, info = model.select_critical_flows(
            graph_data=graph_data,
            od_data=od_data,
            active_mask=active_mask,
            k_crit_default=k_crit,
            force_default_k=True,
        )

        logger.debug(f"GNN+ selected {len(selected)} flows, k_used={info.get('k_used', 'N/A')}")
        assert_selected_ods_have_paths(path_library, selected, context=f"{dataset.key}:gnnplus")
        return selected
    except Exception as e:
        logger.warning(f"GNN+ selection failed: {e}, falling back to bottleneck")
        import traceback
        logger.debug(traceback.format_exc())
        # Fallback to bottleneck
        return select_bottleneck_critical(
            tm_vector, ecmp_splits(path_library), path_library,
            dataset.capacities, k_crit
        )


# ── Dataset Loading ────────────────────────────────────────────────────

def load_dataset(key: str) -> Tuple[TEDataset, PathLibrary]:
    """Load a dataset from the processed NPZ files."""
    from phase1_reactive.eval.common import load_bundle, collect_specs, load_named_dataset, max_steps_from_args
    config_path = project_root / "configs" / "phase1_reactive_full.yaml"
    bundle = load_bundle(config_path)

    for field_name in ["eval_topologies", "generalization_topologies", "train_topologies"]:
        try:
            specs = collect_specs(bundle, field_name)
            max_steps = max_steps_from_args(bundle, 500)
            for spec in specs:
                if key.lower() in spec.key.lower():
                    try:
                        dataset, pl = load_named_dataset(bundle, spec, max_steps)
                        return dataset, pl
                    except Exception as e:
                        logger.warning(f"Failed to load {spec.key}: {e}")
        except Exception:
            continue

    raise ValueError(f"Dataset '{key}' not found")


def load_gnn_model(dataset, path_library):
    """Try to load trained Original GNN model."""
    try:
        import torch
        from phase1_reactive.drl.gnn_selector import load_gnn_selector, build_graph_tensors, build_od_features

        ckpt = project_root / "results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt"
        if not ckpt.exists():
            logger.warning(f"No GNN checkpoint at {ckpt}")
            return None

        model, cfg = load_gnn_selector(str(ckpt), device="cpu")
        model.eval()
        model._dataset = dataset
        model._build_graph_tensors = build_graph_tensors
        model._build_od_features = build_od_features
        logger.info(f"Loaded Original GNN model from {ckpt}")
        return model
    except Exception as e:
        logger.warning(f"Could not load GNN model: {e}")
        return None


def gnn_select_critical(model, dataset, path_library, tm_vector, k_crit=K_CRIT):
    """Select critical flows using the original GNN with path-valid active masking."""
    try:
        import torch

        graph_data = model._build_graph_tensors(dataset, device="cpu")
        od_data = model._build_od_features(dataset, tm_vector, path_library, device="cpu")
        active_mask = (
            (np.asarray(tm_vector, dtype=np.float64) > 1e-12)
            & surviving_od_mask(path_library)
        ).astype(np.float32)
        with torch.no_grad():
            selected, _ = model.select_critical_flows(
                graph_data,
                od_data,
                active_mask=active_mask,
                k_crit_default=k_crit,
            )
        assert_selected_ods_have_paths(path_library, selected, context=f"{dataset.key}:gnn")
        return selected
    except Exception as e:
        logger.debug(f"GNN fallback: {e}")
        selected = select_bottleneck_critical(tm_vector, ecmp_splits(path_library), path_library, dataset.capacities, k_crit)
        assert_selected_ods_have_paths(path_library, selected, context=f"{dataset.key}:gnn_fallback")
        return selected


def solve_selected_path_lp_safe(
    *,
    tm_vector,
    selected_ods,
    base_splits,
    path_library,
    capacities,
    time_limit_sec,
    context,
    warm_start_splits=None,
):
    assert_selected_ods_have_paths(path_library, selected_ods, context=context)
    return solve_selected_path_lp(
        tm_vector=tm_vector,
        selected_ods=selected_ods,
        base_splits=base_splits,
        path_library=path_library,
        capacities=capacities,
        warm_start_splits=warm_start_splits,
        time_limit_sec=time_limit_sec,
    )


def _build_dataset_view(dataset: TEDataset, *, edges, capacities, weights):
    return SimpleNamespace(
        key=dataset.key,
        name=dataset.name,
        nodes=dataset.nodes,
        edges=list(edges),
        capacities=np.asarray(capacities, dtype=float),
        weights=np.asarray(weights, dtype=float),
        od_pairs=dataset.od_pairs,
        metadata=dataset.metadata,
    )


def _build_failure_execution_state(
    *,
    scenario: str,
    tm_vector: np.ndarray,
    dataset: TEDataset,
    path_library: PathLibrary,
    capacities: np.ndarray,
    weights: np.ndarray,
    normal_routing,
) -> dict:
    failure_mask = np.ones(len(capacities), dtype=float)
    failed_edges: list[int] = []
    effective_tm = np.asarray(tm_vector, dtype=float)
    effective_path_library = path_library
    effective_caps = np.asarray(capacities, dtype=float)
    effective_weights = np.asarray(weights, dtype=float)
    effective_dataset = dataset

    if scenario == "single_link_failure":
        fail_idx = int(np.argmax(np.asarray(normal_routing.utilization)))
        failed_edges = [fail_idx]
    elif scenario == "random_link_failure_1":
        failed_edges = [random.randint(0, len(capacities) - 1)]
    elif scenario == "random_link_failure_2":
        failed_edges = random.sample(range(len(capacities)), min(2, len(capacities)))
    elif scenario == "capacity_degradation_50":
        util = np.asarray(normal_routing.utilization)
        degraded = np.where(util > 0.5)[0].tolist()
        for idx in degraded:
            failure_mask[idx] = 0.5
        effective_caps = np.asarray(capacities, dtype=float) * failure_mask
        effective_dataset = _build_dataset_view(
            dataset,
            edges=dataset.edges,
            capacities=effective_caps,
            weights=weights,
        )
    elif scenario == "traffic_spike_2x":
        tm_spike = np.asarray(tm_vector, dtype=float).copy()
        top_demands = np.argsort(tm_vector)[-K_CRIT:]
        tm_spike[top_demands] *= 2.0
        effective_tm = tm_spike
    else:
        raise ValueError(f"Unsupported failure scenario: {scenario}")

    if failed_edges:
        for idx in failed_edges:
            failure_mask[int(idx)] = 0.0
        keep = [idx for idx in range(len(dataset.edges)) if idx not in set(int(x) for x in failed_edges)]
        kept_edges = [dataset.edges[idx] for idx in keep]
        kept_caps = np.asarray([capacities[idx] for idx in keep], dtype=float)
        kept_weights = np.asarray([weights[idx] for idx in keep], dtype=float)
        effective_path_library = build_modified_paths(
            dataset.nodes,
            kept_edges,
            kept_weights,
            dataset.od_pairs,
            k_paths=K_PATHS,
        )
        effective_caps = kept_caps
        effective_weights = kept_weights
        effective_dataset = _build_dataset_view(
            dataset,
            edges=kept_edges,
            capacities=kept_caps,
            weights=kept_weights,
        )

    effective_ecmp = ecmp_splits(effective_path_library)
    return {
        "effective_tm": effective_tm,
        "effective_path_library": effective_path_library,
        "effective_caps": effective_caps,
        "effective_weights": effective_weights,
        "effective_dataset": effective_dataset,
        "effective_ecmp": effective_ecmp,
        "failure_mask": failure_mask,
        "failed_edges": failed_edges,
    }


# ── SDN Benchmark Core ─────────────────────────────────────────────────

@dataclass
class SDNCycleResult:
    """Result of one SDN control cycle with all 9 metrics."""
    cycle: int
    method: str
    topology: str
    scenario: str = "normal"
    # Primary TE metrics
    pre_mlu: float = 0.0
    post_mlu: float = 0.0
    disturbance: float = 0.0
    # Model-based network metrics
    throughput: float = 0.0
    mean_latency: float = 0.0
    p95_latency: float = 0.0
    packet_loss: float = 0.0
    jitter: float = 0.0
    # Measured SDN deployment metrics
    decision_time_ms: float = 0.0
    flow_table_updates: int = 0
    rule_install_delay_ms: float = 0.0
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
    gnnplus_model=None,
    prev_latency_by_od=None,
    failure_mask=None,
) -> Tuple[SDNCycleResult, list, List[OFGroupMod], np.ndarray]:
    """Run one SDN TE cycle and measure all metrics."""

    t_total_start = time.perf_counter()

    # Apply failure mask if provided
    effective_caps = capacities.copy()
    if failure_mask is not None:
        effective_caps = effective_caps * failure_mask

    # ── 1. OBSERVE ──
    routing_pre = apply_routing(tm_vector, current_splits, path_library, effective_caps)
    pre_mlu = float(routing_pre.mlu)

    # ── 2. SELECT + OPTIMIZE ──
    selected_ods = []
    
    if method == "ecmp":
        new_splits = [s.copy() for s in ecmp_base]
    elif method == "ospf":
        from te.baselines import ospf_splits
        new_splits = ospf_splits(path_library)
    elif method == "bottleneck":
        selected_ods = select_bottleneck_critical(tm_vector, ecmp_base, path_library, effective_caps, K_CRIT)
        lp_result = solve_selected_path_lp_safe(
            tm_vector=tm_vector, selected_ods=selected_ods, base_splits=ecmp_base,
            path_library=path_library, capacities=effective_caps, time_limit_sec=20,
            context=f"{dataset.key}:{method}:normal_cycle",
        )
        new_splits = [s.copy() for s in lp_result.splits]
    elif method == "topk":
        selected_ods = select_topk_by_demand(tm_vector, K_CRIT)
        lp_result = solve_selected_path_lp_safe(
            tm_vector=tm_vector, selected_ods=selected_ods, base_splits=ecmp_base,
            path_library=path_library, capacities=effective_caps, time_limit_sec=20,
            context=f"{dataset.key}:{method}:normal_cycle",
        )
        new_splits = [s.copy() for s in lp_result.splits]
    elif method == "sensitivity":
        selected_ods = select_sensitivity_critical(tm_vector, ecmp_base, path_library, effective_caps, K_CRIT)
        lp_result = solve_selected_path_lp_safe(
            tm_vector=tm_vector, selected_ods=selected_ods, base_splits=ecmp_base,
            path_library=path_library, capacities=effective_caps, time_limit_sec=20,
            context=f"{dataset.key}:{method}:normal_cycle",
        )
        new_splits = [s.copy() for s in lp_result.splits]
    elif method == "gnn" and gnn_model is not None:
        selected_ods = gnn_select_critical(gnn_model, dataset, path_library, tm_vector, K_CRIT)
        lp_result = solve_selected_path_lp_safe(
            tm_vector=tm_vector, selected_ods=selected_ods, base_splits=ecmp_base,
            path_library=path_library, capacities=effective_caps, time_limit_sec=20,
            context=f"{dataset.key}:{method}:normal_cycle",
        )
        new_splits = [s.copy() for s in lp_result.splits]
    elif method == "gnnplus" and gnnplus_model is not None:
        selected_ods = gnnplus_select_critical(gnnplus_model, dataset, path_library, tm_vector, K_CRIT)
        lp_result = solve_selected_path_lp_safe(
            tm_vector=tm_vector, selected_ods=selected_ods, base_splits=ecmp_base,
            path_library=path_library, capacities=effective_caps, time_limit_sec=20,
            context=f"{dataset.key}:{method}:normal_cycle",
        )
        new_splits = [s.copy() for s in lp_result.splits]
    else:
        # Fallback to bottleneck
        selected_ods = select_bottleneck_critical(tm_vector, ecmp_base, path_library, effective_caps, K_CRIT)
        lp_result = solve_selected_path_lp_safe(
            tm_vector=tm_vector, selected_ods=selected_ods, base_splits=ecmp_base,
            path_library=path_library, capacities=effective_caps, time_limit_sec=20,
            context=f"{dataset.key}:{method}:normal_cycle",
        )
        new_splits = [s.copy() for s in lp_result.splits]

    t_decision_end = time.perf_counter()
    decision_time_ms = (t_decision_end - t_total_start) * 1000

    # ── 3. APPLY ──
    t_rule_start = time.perf_counter()

    if method == "ecmp":
        new_groups, new_flows = build_ecmp_baseline_rules(path_library, topo_mapping, dataset.edges)
    else:
        new_groups, new_flows = splits_to_openflow_rules(
            new_splits, selected_ods, path_library, topo_mapping, dataset.edges
        )

    changed_groups = compute_rule_diff(current_groups, new_groups)
    flow_table_updates = len(changed_groups)

    t_rule_end = time.perf_counter()
    rule_install_delay_ms = (t_rule_end - t_rule_start) * 1000

    # ── 4. MEASURE ──
    routing_post = apply_routing(tm_vector, new_splits, path_library, effective_caps)
    post_mlu = float(routing_post.mlu)

    dist = compute_disturbance(current_splits, new_splits, tm_vector)

    telemetry = compute_telemetry(
        tm_vector=tm_vector, splits=new_splits, path_library=path_library,
        routing=routing_post, weights=weights, prev_latency_by_od=prev_latency_by_od,
    )

    result = SDNCycleResult(
        cycle=0, method=method, topology=dataset.key,
        pre_mlu=pre_mlu, post_mlu=post_mlu, disturbance=float(dist),
        throughput=telemetry.throughput, mean_latency=telemetry.mean_latency,
        p95_latency=telemetry.p95_latency, packet_loss=telemetry.packet_loss,
        jitter=telemetry.jitter, decision_time_ms=decision_time_ms,
        flow_table_updates=flow_table_updates, rule_install_delay_ms=rule_install_delay_ms,
    )

    return result, new_splits, new_groups, telemetry.latency_by_od


def run_failure_scenario(
    scenario: str,
    tm_vector: np.ndarray,
    method: str,
    dataset: TEDataset,
    path_library: PathLibrary,
    ecmp_base: list,
    capacities: np.ndarray,
    weights: np.ndarray,
    topo_mapping: SDNTopologyMapping,
    gnn_model=None,
    gnnplus_model=None,
) -> Tuple[float, float, float, np.ndarray]:
    """Run a failure scenario and return recovery metrics.
    
    Returns: (recovery_time_ms, pre_failure_mlu, post_recovery_mlu, failure_mask)
    """
    # Compute normal MLU first
    normal_routing = apply_routing(tm_vector, ecmp_base, path_library, capacities)
    pre_failure_mlu = float(normal_routing.mlu)
    
    failure_state = _build_failure_execution_state(
        scenario=scenario,
        tm_vector=tm_vector,
        dataset=dataset,
        path_library=path_library,
        capacities=capacities,
        weights=weights,
        normal_routing=normal_routing,
    )
    effective_tm = failure_state["effective_tm"]
    effective_caps = failure_state["effective_caps"]
    effective_path_library = failure_state["effective_path_library"]
    effective_dataset = failure_state["effective_dataset"]
    effective_ecmp = failure_state["effective_ecmp"]
    failure_mask = failure_state["failure_mask"]

    # Compute post-failure MLU (without recovery) on the surviving topology state.
    post_failure_routing = apply_routing(effective_tm, effective_ecmp, effective_path_library, effective_caps)
    
    # Run recovery
    t_start = time.perf_counter()
    
    # Run method under failure
    if method == "ecmp":
        recovery_splits = [s.copy() for s in effective_ecmp]
    elif method == "bottleneck":
        selected = select_bottleneck_critical(effective_tm, effective_ecmp, effective_path_library, effective_caps, K_CRIT)
        lp_result = solve_selected_path_lp_safe(
            tm_vector=effective_tm, selected_ods=selected, base_splits=effective_ecmp,
            path_library=effective_path_library, capacities=effective_caps, time_limit_sec=20,
            context=f"{dataset.key}:{scenario}:{method}",
        )
        recovery_splits = [s.copy() for s in lp_result.splits]
    elif method == "gnn" and gnn_model is not None:
        selected = gnn_select_critical(gnn_model, effective_dataset, effective_path_library, effective_tm, K_CRIT)
        lp_result = solve_selected_path_lp_safe(
            tm_vector=effective_tm, selected_ods=selected, base_splits=effective_ecmp,
            path_library=effective_path_library, capacities=effective_caps, time_limit_sec=20,
            context=f"{dataset.key}:{scenario}:{method}",
        )
        recovery_splits = [s.copy() for s in lp_result.splits]
    elif method == "gnnplus" and gnnplus_model is not None:
        selected = gnnplus_select_critical(gnnplus_model, effective_dataset, effective_path_library, effective_tm, K_CRIT)
        lp_result = solve_selected_path_lp_safe(
            tm_vector=effective_tm, selected_ods=selected, base_splits=effective_ecmp,
            path_library=effective_path_library, capacities=effective_caps, time_limit_sec=20,
            context=f"{dataset.key}:{scenario}:{method}",
        )
        recovery_splits = [s.copy() for s in lp_result.splits]
    else:
        recovery_splits = [s.copy() for s in effective_ecmp]
    
    t_end = time.perf_counter()
    recovery_ms = (t_end - t_start) * 1000
    
    # Post-recovery MLU
    post_routing = apply_routing(effective_tm, recovery_splits, effective_path_library, effective_caps)
    post_recovery_mlu = float(post_routing.mlu)
    
    return recovery_ms, pre_failure_mlu, post_recovery_mlu, failure_mask


# ── Main Benchmark ─────────────────────────────────────────────────────

def benchmark_topology_normal(topo_key: str, methods: list, gnn_cache: dict, gnnplus_cache: dict) -> List[dict]:
    """Run normal (no failure) benchmark for one topology."""
    logger.info(f"=== Loading {topo_key} ===")
    dataset, path_library = load_dataset(topo_key)
    capacities = np.asarray(dataset.capacities, dtype=float)
    weights = np.asarray(dataset.weights, dtype=float)
    ecmp_base = ecmp_splits(path_library)

    topo_mapping = SDNTopologyMapping.from_mininet(dataset.nodes, dataset.edges, dataset.od_pairs)

    sp = dataset.split
    test_start = int(sp["test_start"])
    tm_data = dataset.tm
    test_indices = list(range(test_start, tm_data.shape[0]))
    logger.info(f"  {len(test_indices)} test timesteps, {len(dataset.nodes)} nodes, {len(dataset.edges)} edges")

    # Load models
    gnn_model = None
    if "gnn" in methods:
        if topo_key not in gnn_cache:
            gnn_cache[topo_key] = load_gnn_model(dataset, path_library)
        gnn_model = gnn_cache[topo_key]
        if gnn_model is None:
            logger.warning(f"  GNN model not available for {topo_key}")
            methods = [m for m in methods if m != "gnn"]

    gnnplus_model = None
    if "gnnplus" in methods:
        if topo_key not in gnnplus_cache:
            gnnplus_cache[topo_key] = load_gnnplus_model(dataset, path_library)
        gnnplus_model = gnnplus_cache[topo_key]
        if gnnplus_model is None:
            logger.warning(f"  GNN+ model not available for {topo_key}")
            methods = [m for m in methods if m != "gnnplus"]

    all_rows = []

    for method in methods:
        logger.info(f"  Running {method} on {topo_key} (normal)...")

        run_results = defaultdict(list)

        for run_idx in range(NUM_RUNS):
            current_splits = [s.copy() for s in ecmp_base]
            baseline_groups, _ = build_ecmp_baseline_rules(path_library, topo_mapping, dataset.edges)
            current_groups = baseline_groups
            prev_latency = None

            for i, t_idx in enumerate(test_indices):
                tm_vec = tm_data[t_idx]

                result, current_splits, current_groups, prev_latency = run_sdn_cycle(
                    tm_vector=tm_vec, method=method, dataset=dataset, path_library=path_library,
                    ecmp_base=ecmp_base, current_splits=current_splits, current_groups=current_groups,
                    topo_mapping=topo_mapping, capacities=capacities, weights=weights,
                    gnn_model=gnn_model, gnnplus_model=gnnplus_model, prev_latency_by_od=prev_latency,
                )

                run_results['post_mlus'].append(result.post_mlu)
                run_results['disturbances'].append(result.disturbance)
                run_results['throughputs'].append(result.throughput)
                run_results['latencies'].append(result.mean_latency)
                run_results['p95_latencies'].append(result.p95_latency)
                run_results['packet_losses'].append(result.packet_loss)
                run_results['jitters'].append(result.jitter)
                run_results['decision_times'].append(result.decision_time_ms)
                run_results['flow_updates'].append(result.flow_table_updates)
                run_results['rule_delays'].append(result.rule_install_delay_ms)

        row = {
            'topology': topo_key,
            'status': TOPOLOGIES[topo_key]['status'],
            'method': method,
            'scenario': 'normal',
            'nodes': len(dataset.nodes),
            'edges': len(dataset.edges),
            'mean_mlu': float(np.mean(run_results['post_mlus'])),
            'mean_disturbance': float(np.mean(run_results['disturbances'])),
            'throughput': float(np.mean(run_results['throughputs'])),
            'mean_latency_au': float(np.mean(run_results['latencies'])),
            'p95_latency_au': float(np.mean(run_results['p95_latencies'])),
            'packet_loss': float(np.mean(run_results['packet_losses'])),
            'jitter_au': float(np.mean(run_results['jitters'])),
            'decision_time_ms': float(np.mean(run_results['decision_times'])),
            'flow_table_updates': float(np.mean(run_results['flow_updates'])),
            'rule_install_delay_ms': float(np.mean(run_results['rule_delays'])),
        }
        all_rows.append(row)
        logger.info(f"    {method}: MLU={row['mean_mlu']:.4f}, throughput={row['throughput']:.4f}")

    return all_rows


def benchmark_topology_failures(topo_key: str, methods: list, gnn_cache: dict, gnnplus_cache: dict) -> List[dict]:
    """Run failure scenario benchmark for one topology."""
    logger.info(f"=== Loading {topo_key} (failures) ===")
    dataset, path_library = load_dataset(topo_key)
    capacities = np.asarray(dataset.capacities, dtype=float)
    weights = np.asarray(dataset.weights, dtype=float)
    ecmp_base = ecmp_splits(path_library)

    topo_mapping = SDNTopologyMapping.from_mininet(dataset.nodes, dataset.edges, dataset.od_pairs)

    sp = dataset.split
    test_start = int(sp["test_start"])
    tm_data = dataset.tm
    test_indices = list(range(test_start, tm_data.shape[0]))

    # Load models
    gnn_model = gnn_cache.get(topo_key)
    gnnplus_model = gnnplus_cache.get(topo_key)

    all_rows = []

    for scenario in FAILURE_SCENARIOS:
        if scenario == "normal":
            continue  # Already done

        for method in methods:
            logger.info(f"  Running {method} on {topo_key} ({scenario})...")

            run_results = defaultdict(list)

            # Sample timesteps for failure testing
            sample_indices = test_indices[::max(1, len(test_indices)//10)]

            for t_idx in sample_indices:
                tm_vec = tm_data[t_idx]

                recovery_ms, pre_mlu, post_mlu, failure_mask = run_failure_scenario(
                    scenario=scenario, tm_vector=tm_vec, method=method, dataset=dataset,
                    path_library=path_library, ecmp_base=ecmp_base, capacities=capacities,
                    weights=weights, topo_mapping=topo_mapping,
                    gnn_model=gnn_model, gnnplus_model=gnnplus_model,
                )

                run_results['recovery_times'].append(recovery_ms)
                run_results['pre_mlus'].append(pre_mlu)
                run_results['post_mlus'].append(post_mlu)

            row = {
                'topology': topo_key,
                'status': TOPOLOGIES[topo_key]['status'],
                'method': method,
                'scenario': scenario,
                'nodes': len(dataset.nodes),
                'edges': len(dataset.edges),
                'mean_mlu': float(np.mean(run_results['post_mlus'])),
                'pre_failure_mlu': float(np.mean(run_results['pre_mlus'])),
                'failure_recovery_ms': float(np.mean(run_results['recovery_times'])),
            }
            all_rows.append(row)
            logger.info(f"    {method} ({scenario}): MLU={row['mean_mlu']:.4f}, recovery={row['failure_recovery_ms']:.1f}ms")

    return all_rows


def main():
    """Main entry point — smoke test then full evaluation."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── Phase 1: Smoke Test ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1: SMOKE TEST (Abilene + Germany50)")
    logger.info("=" * 60)
    
    smoke_topologies = ["abilene", "germany50"]
    smoke_methods = ["ecmp", "bottleneck", "gnn", "gnnplus"]
    gnn_cache = {}
    gnnplus_cache = {}
    
    smoke_results = []
    for topo in smoke_topologies:
        try:
            rows = benchmark_topology_normal(topo, smoke_methods.copy(), gnn_cache, gnnplus_cache)
            smoke_results.extend(rows)
        except Exception as e:
            logger.error(f"Smoke test failed for {topo}: {e}", exc_info=True)
            print(f"\n*** SMOKE TEST FAILED for {topo} ***")
            print(f"Error: {e}")
            print("Cannot proceed with full evaluation.")
            return 1
    
    logger.info("=" * 60)
    logger.info("SMOKE TEST PASSED — proceeding to full evaluation")
    logger.info("=" * 60)
    
    # ── Phase 2: Full 8-Topology Normal Evaluation ────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: FULL NORMAL EVALUATION (8 topologies)")
    logger.info("=" * 60)
    
    full_topologies = list(TOPOLOGIES.keys())
    full_methods = ["ecmp", "bottleneck", "gnn", "gnnplus"]  # Core methods
    
    normal_results = []
    for topo in full_topologies:
        try:
            rows = benchmark_topology_normal(topo, full_methods.copy(), gnn_cache, gnnplus_cache)
            normal_results.extend(rows)
        except Exception as e:
            logger.error(f"Failed for {topo}: {e}", exc_info=True)
    
    # Save normal results
    if normal_results:
        import pandas as pd
        df_normal = pd.DataFrame(normal_results)
        df_normal.to_csv(OUT_DIR / "packet_sdn_summary.csv", index=False)
        logger.info(f"Normal results saved: {OUT_DIR / 'packet_sdn_summary.csv'}")
    
    # ── Phase 3: Failure Evaluation ───────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: FAILURE SCENARIO EVALUATION")
    logger.info("=" * 60)
    
    failure_results = []
    for topo in full_topologies:
        try:
            rows = benchmark_topology_failures(topo, full_methods.copy(), gnn_cache, gnnplus_cache)
            failure_results.extend(rows)
        except Exception as e:
            logger.error(f"Failure eval failed for {topo}: {e}", exc_info=True)
    
    if failure_results:
        import pandas as pd
        df_failure = pd.DataFrame(failure_results)
        df_failure.to_csv(OUT_DIR / "packet_sdn_failure.csv", index=False)
        logger.info(f"Failure results saved: {OUT_DIR / 'packet_sdn_failure.csv'}")
    
    # ── Phase 4: Summary ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results directory: {OUT_DIR}")
    logger.info(f"Normal results: {len(normal_results)} rows")
    logger.info(f"Failure results: {len(failure_results)} rows")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
