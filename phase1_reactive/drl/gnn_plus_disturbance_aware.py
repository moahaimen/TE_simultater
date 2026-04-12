"""Disturbance-aware inference helpers for the standalone GNN+ selector.

Section 7 adds three stability mechanisms without changing the legacy GNN+ path:
  1. Temporal GNN+ features via ``section7_temporal``
  2. A continuity bonus for previously selected ODs
  3. Rollout bookkeeping for previous selection, disturbance, TM, and utilization
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from phase1_reactive.drl.gnn_plus_selector import (
    build_graph_tensors_plus,
    build_od_features_plus,
    load_gnn_plus,
)

DA_GNNPLUS_METHOD = "our_da_gnnplus_selector"


@dataclass
class DAGNNPlusConfig:
    lambda_cont: float = 0.10
    feature_variant: str = "section7_temporal"
    force_default_k: bool = True


def _selection_indicator(num_od: int, selected: list[int]) -> np.ndarray:
    out = np.zeros(int(num_od), dtype=np.float32)
    if selected:
        out[np.asarray(selected, dtype=int)] = 1.0
    return out


def choose_da_gnnplus_selector(
    env,
    gnnplus_model,
    *,
    prev_tm=None,
    prev_util=None,
    prev_selected_indicator=None,
    prev_disturbance: float = 0.0,
    da_cfg: DAGNNPlusConfig | None = None,
    device: str = "cpu",
) -> tuple[list[int], dict]:
    """Single-step disturbance-aware GNN+ inference on a reactive env."""
    da_cfg = da_cfg or DAGNNPlusConfig()
    obs = env.current_obs
    current_tm = np.asarray(obs.current_tm, dtype=np.float64)
    telemetry = obs.telemetry

    graph_data = build_graph_tensors_plus(
        env.dataset,
        tm_vector=current_tm,
        path_library=env.path_library,
        telemetry=telemetry,
        failure_mask=getattr(obs, "failure_mask", None),
        prev_util=prev_util,
        prev_tm=prev_tm,
        prev_selected_indicator=prev_selected_indicator,
        prev_disturbance=prev_disturbance,
        feature_variant=da_cfg.feature_variant,
        device=device,
    )
    od_data = build_od_features_plus(
        env.dataset,
        current_tm,
        env.path_library,
        telemetry=telemetry,
        prev_tm=prev_tm,
        prev_util=prev_util,
        prev_selected_indicator=prev_selected_indicator,
        prev_disturbance=prev_disturbance,
        feature_variant=da_cfg.feature_variant,
        device=device,
    )

    active_mask = np.asarray(getattr(obs, "active_mask", current_tm > 1e-12), dtype=bool)
    selected, info = gnnplus_model.select_critical_flows(
        graph_data=graph_data,
        od_data=od_data,
        active_mask=active_mask,
        k_crit_default=env.k_crit,
        force_default_k=bool(da_cfg.force_default_k),
        prev_selected_indicator=prev_selected_indicator,
        continuity_bonus=float(da_cfg.lambda_cont),
    )
    info = dict(info)
    info["feature_variant"] = str(da_cfg.feature_variant)
    return selected, info


def rollout_da_gnnplus_selector(
    env,
    gnnplus_model,
    *,
    da_cfg: DAGNNPlusConfig | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Full reactive rollout using the disturbance-aware GNN+ inference path."""
    da_cfg = da_cfg or DAGNNPlusConfig()
    if isinstance(gnnplus_model, (str, Path)):
        gnnplus_model, _ = load_gnn_plus(Path(gnnplus_model), device=device)

    gnnplus_model.eval()
    env.reset()
    rows = []
    done = False
    prev_tm = None
    prev_util = None
    prev_selected_indicator = np.zeros(len(env.dataset.od_pairs), dtype=np.float32)
    prev_disturbance = 0.0

    while not done:
        obs = env.current_obs
        current_tm = np.asarray(obs.current_tm, dtype=np.float64)
        telemetry = obs.telemetry
        decision_start = time.perf_counter()
        selected, da_info = choose_da_gnnplus_selector(
            env,
            gnnplus_model,
            prev_tm=prev_tm,
            prev_util=prev_util,
            prev_selected_indicator=prev_selected_indicator,
            prev_disturbance=prev_disturbance,
            da_cfg=da_cfg,
            device=device,
        )
        inference_latency = time.perf_counter() - decision_start

        _, reward, done, info = env.step(selected)
        info = dict(info)
        info["reward"] = float(reward)
        info["inference_latency_sec"] = float(inference_latency)
        info["decision_time_ms"] = float((time.perf_counter() - decision_start) * 1000.0)
        info["method"] = DA_GNNPLUS_METHOD
        info["selected_count"] = int(len(selected))
        info.update({f"da_gnnplus_{key}": value for key, value in da_info.items()})
        rows.append(info)

        prev_tm = current_tm
        prev_util = np.asarray(telemetry.utilization, dtype=np.float64)
        prev_selected_indicator = _selection_indicator(len(env.dataset.od_pairs), selected)
        prev_disturbance = float(info.get("disturbance", 0.0))

    return pd.DataFrame(rows)
