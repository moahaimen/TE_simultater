"""LP solvers for hybrid path-based TE and full MCF reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pulp

from te.baselines import clone_splits
from te.paths import PathLibrary
from te.simulator import RoutingResult, apply_routing

EPS = 1e-12


@dataclass
class HybridLPResult:
    splits: List[np.ndarray]
    routing: RoutingResult
    status: str


@dataclass
class FullMCFResult:
    mlu: float
    link_loads: np.ndarray
    status: str
    edge_flows_by_od: List[Dict[int, float]]


def _build_background_load(
    tm_vector: np.ndarray,
    base_splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    selected_set: set[int],
    num_edges: int,
) -> np.ndarray:
    load = np.zeros(num_edges, dtype=float)
    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0 or od_idx in selected_set:
            continue

        paths = path_library.edge_idx_paths_by_od[od_idx]
        if not paths:
            continue

        splits = np.asarray(base_splits[od_idx], dtype=float)
        if splits.size == 0:
            continue

        split_sum = float(np.sum(splits))
        if split_sum <= EPS:
            continue

        splits = splits / split_sum
        for path_idx, frac in enumerate(splits):
            if frac <= 0:
                continue
            flow = float(demand) * float(frac)
            for edge_idx in paths[path_idx]:
                load[edge_idx] += flow

    return load


def solve_selected_path_lp(
    tm_vector: np.ndarray,
    selected_ods: Sequence[int],
    base_splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
    time_limit_sec: int = 20,
    solver_msg: bool = False,
) -> HybridLPResult:
    """
    Hybrid LP: optimize selected OD flows across K paths, with non-selected ODs fixed to base policy.
    """
    num_edges = int(capacities.size)
    selected_set = {
        int(od_idx)
        for od_idx in selected_ods
        if 0 <= int(od_idx) < len(tm_vector) and tm_vector[int(od_idx)] > 0
    }

    if not selected_set:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status="NoSelection")

    background = _build_background_load(tm_vector, base_splits, path_library, selected_set, num_edges)

    model = pulp.LpProblem("hybrid_te_selected_lp", pulp.LpMinimize)
    U = pulp.LpVariable("U", lowBound=0.0)

    flow_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
    incidence: List[List[pulp.LpVariable]] = [[] for _ in range(num_edges)]

    for od_idx in sorted(selected_set):
        demand = float(tm_vector[od_idx])
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if demand <= 0 or not paths:
            continue

        per_od_vars = []
        for path_idx, edge_path in enumerate(paths):
            var = pulp.LpVariable(f"f_{od_idx}_{path_idx}", lowBound=0.0)
            flow_vars[(od_idx, path_idx)] = var
            per_od_vars.append(var)
            for edge_idx in edge_path:
                incidence[edge_idx].append(var)

        model += pulp.lpSum(per_od_vars) == demand, f"demand_{od_idx}"

    for edge_idx in range(num_edges):
        model += (
            background[edge_idx] + pulp.lpSum(incidence[edge_idx]) <= U * float(capacities[edge_idx]),
            f"cap_{edge_idx}",
        )

    model += U

    solver = pulp.PULP_CBC_CMD(msg=solver_msg, timeLimit=int(time_limit_sec), threads=1)
    status_code = model.solve(solver)
    status = pulp.LpStatus.get(status_code, "Unknown")

    splits = clone_splits(base_splits)

    if status not in {"Optimal", "Not Solved", "Undefined"}:
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status=status)

    for od_idx in sorted(selected_set):
        demand = float(tm_vector[od_idx])
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if demand <= 0 or not paths:
            continue

        vec = np.zeros(len(paths), dtype=float)
        for path_idx in range(len(paths)):
            var = flow_vars.get((od_idx, path_idx))
            if var is None:
                continue
            vec[path_idx] = max(float(var.value() or 0.0), 0.0)

        if demand > EPS:
            vec /= demand

        vec_sum = float(np.sum(vec))
        if vec_sum > EPS:
            vec /= vec_sum
            splits[od_idx] = vec

    routing = apply_routing(tm_vector, splits, path_library, capacities)
    return HybridLPResult(splits=splits, routing=routing, status=status)


def solve_full_mcf_min_mlu(
    tm_vector: np.ndarray,
    od_pairs: Sequence[Tuple[str, str]],
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    capacities: np.ndarray,
    time_limit_sec: int = 60,
    solver_msg: bool = False,
) -> FullMCFResult:
    """Full multi-commodity flow LP minimizing MLU (M2 reference)."""
    active_ods = [idx for idx, demand in enumerate(tm_vector) if demand > 0]
    num_edges = len(edges)

    if not active_ods:
        return FullMCFResult(
            mlu=0.0,
            link_loads=np.zeros(num_edges, dtype=float),
            status="NoDemand",
            edge_flows_by_od=[{} for _ in od_pairs],
        )

    node_to_out: Dict[str, List[int]] = {node: [] for node in nodes}
    node_to_in: Dict[str, List[int]] = {node: [] for node in nodes}
    for edge_idx, (src, dst) in enumerate(edges):
        node_to_out[src].append(edge_idx)
        node_to_in[dst].append(edge_idx)

    model = pulp.LpProblem("full_mcf_min_mlu", pulp.LpMinimize)
    U = pulp.LpVariable("U", lowBound=0.0)

    x: Dict[Tuple[int, int], pulp.LpVariable] = {}
    edge_to_vars: List[List[pulp.LpVariable]] = [[] for _ in range(num_edges)]

    for od_idx in active_ods:
        for edge_idx in range(num_edges):
            var = pulp.LpVariable(f"x_{od_idx}_{edge_idx}", lowBound=0.0)
            x[(od_idx, edge_idx)] = var
            edge_to_vars[edge_idx].append(var)

    for edge_idx in range(num_edges):
        model += pulp.lpSum(edge_to_vars[edge_idx]) <= U * float(capacities[edge_idx]), f"cap_{edge_idx}"

    for od_idx in active_ods:
        src, dst = od_pairs[od_idx]
        demand = float(tm_vector[od_idx])

        for node in nodes:
            out_flow = pulp.lpSum(x[(od_idx, e_idx)] for e_idx in node_to_out[node])
            in_flow = pulp.lpSum(x[(od_idx, e_idx)] for e_idx in node_to_in[node])

            rhs = 0.0
            if node == src:
                rhs = demand
            elif node == dst:
                rhs = -demand

            model += out_flow - in_flow == rhs, f"flow_{od_idx}_{node}"

    model += U

    solver = pulp.PULP_CBC_CMD(msg=solver_msg, timeLimit=int(time_limit_sec), threads=1)
    status_code = model.solve(solver)
    status = pulp.LpStatus.get(status_code, "Unknown")

    link_loads = np.zeros(num_edges, dtype=float)
    edge_flows_by_od: List[Dict[int, float]] = [{} for _ in od_pairs]

    for od_idx in active_ods:
        od_map: Dict[int, float] = {}
        for edge_idx in range(num_edges):
            var = x.get((od_idx, edge_idx))
            if var is None:
                continue
            value = max(float(var.value() or 0.0), 0.0)
            if value > EPS:
                od_map[edge_idx] = value
                link_loads[edge_idx] += value
        edge_flows_by_od[od_idx] = od_map

    util = link_loads / np.maximum(capacities, EPS)
    if status in {"Optimal", "Not Solved", "Undefined"}:
        mlu = float(np.max(util)) if util.size else 0.0
    else:
        mlu = float("inf")

    return FullMCFResult(
        mlu=mlu,
        link_loads=link_loads,
        status=status,
        edge_flows_by_od=edge_flows_by_od,
    )
