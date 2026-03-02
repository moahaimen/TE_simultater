"""Shared evaluation helpers for Phase-3 generalization and failure experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from te.baselines import (
    clone_splits,
    ecmp_splits,
    ospf_splits,
    project_edge_flows_to_k_path_splits,
    select_bottleneck_critical,
    select_topk_by_demand,
)
from te.disturbance import compute_disturbance
from te.lp_solver import solve_full_mcf_min_mlu, solve_selected_path_lp
from te.simulator import apply_routing


@dataclass
class Phase3RunResult:
    summary_rows: list[dict[str, object]]
    timeseries_rows: list[dict[str, object]]


def _stretch_metric(tm_vector: np.ndarray, splits: list[np.ndarray], path_library) -> float:
    num = 0.0
    den = 0.0
    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            continue
        costs = path_library.costs_by_od[od_idx]
        if not costs:
            continue
        shortest = float(np.min(costs))
        if shortest <= 0:
            continue
        vec = np.asarray(splits[od_idx], dtype=float)
        if vec.size == 0:
            continue
        s = float(np.sum(vec))
        if s <= 0:
            continue
        vec = vec / s
        expected = float(np.sum(vec * np.asarray(costs[: vec.size], dtype=float)))
        num += float(demand) * (expected / shortest)
        den += float(demand)
    return 1.0 if den <= 0 else num / den


def run_methods_on_dataset(
    dataset,
    tm: np.ndarray,
    methods: Sequence[str],
    path_library,
    k_crit: int,
    lp_time_limit_sec: int,
    full_mcf_time_limit_sec: int,
    capacity_fn: Callable[[int], np.ndarray] | None = None,
) -> Phase3RunResult:
    summary_rows: list[dict[str, object]] = []
    timeseries_rows: list[dict[str, object]] = []

    test_indices = list(range(dataset.split["test_start"], tm.shape[0]))
    ecmp_base = ecmp_splits(path_library)
    ospf_base = ospf_splits(path_library)

    for method in methods:
        prev_splits = None
        method_rows: list[dict[str, object]] = []

        for test_step, t_idx in enumerate(test_indices):
            step_tm = tm[t_idx]
            capacities = np.asarray(capacity_fn(t_idx) if capacity_fn is not None else dataset.capacities, dtype=float)
            t0 = time.perf_counter()

            if method == "ospf":
                splits = clone_splits(ospf_base)
                routing = apply_routing(step_tm, splits, path_library, capacities)
                status = "Static"

            elif method == "ecmp":
                splits = clone_splits(ecmp_base)
                routing = apply_routing(step_tm, splits, path_library, capacities)
                status = "Static"

            elif method == "topk":
                selected = select_topk_by_demand(step_tm, k_crit=k_crit)
                lp = solve_selected_path_lp(
                    tm_vector=step_tm,
                    selected_ods=selected,
                    base_splits=ecmp_base,
                    path_library=path_library,
                    capacities=capacities,
                    time_limit_sec=lp_time_limit_sec,
                )
                splits = lp.splits
                routing = lp.routing
                status = lp.status

            elif method == "bottleneck":
                selected = select_bottleneck_critical(
                    tm_vector=step_tm,
                    ecmp_policy=ecmp_base,
                    path_library=path_library,
                    capacities=capacities,
                    k_crit=k_crit,
                )
                lp = solve_selected_path_lp(
                    tm_vector=step_tm,
                    selected_ods=selected,
                    base_splits=ecmp_base,
                    path_library=path_library,
                    capacities=capacities,
                    time_limit_sec=lp_time_limit_sec,
                )
                splits = lp.splits
                routing = lp.routing
                status = lp.status

            elif method == "lp_optimal":
                full = solve_full_mcf_min_mlu(
                    tm_vector=step_tm,
                    od_pairs=dataset.od_pairs,
                    nodes=dataset.nodes,
                    edges=dataset.edges,
                    capacities=capacities,
                    time_limit_sec=full_mcf_time_limit_sec,
                )
                splits = project_edge_flows_to_k_path_splits(full.edge_flows_by_od, path_library)
                routing = apply_routing(step_tm, splits, path_library, capacities)
                status = full.status
            else:
                raise ValueError(f"Unsupported method '{method}'")

            runtime_sec = time.perf_counter() - t0
            disturbance = compute_disturbance(prev_splits, splits, step_tm)
            stretch = _stretch_metric(step_tm, splits, path_library)
            prev_splits = clone_splits(splits)

            row = {
                "dataset": dataset.key,
                "method": method,
                "timestep": int(t_idx),
                "test_step": int(test_step),
                "mlu": float(routing.mlu),
                "mean_utilization": float(routing.mean_utilization),
                "disturbance": float(disturbance),
                "stretch": float(stretch),
                "runtime_sec": float(runtime_sec),
                "solver_status": status,
            }
            method_rows.append(row)
            timeseries_rows.append(row)

        arr_mlu = np.asarray([r["mlu"] for r in method_rows], dtype=float)
        arr_dist = np.asarray([r["disturbance"] for r in method_rows], dtype=float)
        arr_run = np.asarray([r["runtime_sec"] for r in method_rows], dtype=float)
        arr_stretch = np.asarray([r["stretch"] for r in method_rows], dtype=float)

        summary_rows.append(
            {
                "dataset": dataset.key,
                "method": method,
                "mean_mlu": float(np.mean(arr_mlu)) if arr_mlu.size else np.nan,
                "p95_mlu": float(np.quantile(arr_mlu, 0.95)) if arr_mlu.size else np.nan,
                "mean_disturbance": float(np.mean(arr_dist)) if arr_dist.size else np.nan,
                "p95_disturbance": float(np.quantile(arr_dist, 0.95)) if arr_dist.size else np.nan,
                "mean_runtime_sec": float(np.mean(arr_run)) if arr_run.size else np.nan,
                "mean_stretch": float(np.mean(arr_stretch)) if arr_stretch.size else np.nan,
                "num_test_steps": int(arr_mlu.size),
            }
        )

    return Phase3RunResult(summary_rows=summary_rows, timeseries_rows=timeseries_rows)
