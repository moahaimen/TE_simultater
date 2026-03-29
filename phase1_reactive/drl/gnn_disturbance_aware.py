"""Disturbance-Aware GNN Selector.

Extends the base GNN selector with continuity-weighted scoring
to balance MLU optimization with routing stability (low disturbance).

Formulation:
  final_score(od) = gnn_score(od) + lambda_cont * continuity_bonus(od)

where continuity_bonus(od) = 1 if od was selected in previous timestep, 0 otherwise.

lambda_cont is tuned on validation to minimize:
  Score = normalized_MLU + lambda_db * normalized_Disturbance

This gives a clean tradeoff: lambda_cont=0 recovers pure GNN (MLU-optimal),
larger lambda_cont increases routing stability at some MLU cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from phase1_reactive.drl.gnn_selector import (
    GNNFlowSelector,
    build_graph_tensors,
    build_od_features,
    load_gnn_selector,
)
from te.baselines import ecmp_splits, clone_splits
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.simulator import apply_routing
from phase1_reactive.drl.state_builder import compute_reactive_telemetry


DA_GNN_METHOD = "our_da_gnn_selector"


@dataclass
class DAGNNConfig:
    """Configuration for Disturbance-Aware GNN inference."""
    lambda_cont: float = 0.3       # continuity bonus weight
    normalize_scores: bool = True   # normalize GNN scores before adding bonus


def choose_da_gnn_selector(
    env, gnn_model, *,
    prev_selected: set[int] | None = None,
    da_cfg: DAGNNConfig | None = None,
    device: str = "cpu",
) -> tuple[list[int], dict]:
    """Single-step disturbance-aware GNN inference.

    Blends GNN OD scores with a continuity bonus for previously-selected ODs.

    Returns:
      selected: list[int] - selected OD indices
      info: dict - diagnostics
    """
    if da_cfg is None:
        da_cfg = DAGNNConfig()

    obs = env.current_obs
    graph_data = build_graph_tensors(
        env.dataset,
        telemetry=obs.telemetry,
        failure_mask=getattr(obs, "failure_mask", None),
        device=device,
    )
    od_data = build_od_features(
        env.dataset,
        obs.current_tm,
        env.path_library,
        telemetry=obs.telemetry,
        device=device,
    )

    # Get base GNN scores
    with torch.no_grad():
        scores_raw, k_pred, model_info = gnn_model(graph_data, od_data)
    scores = scores_raw.cpu().numpy()
    num_od = len(scores)

    # Normalize scores to [0, 1] range
    if da_cfg.normalize_scores and scores.max() > scores.min():
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores_norm = scores.copy()

    # Add continuity bonus
    continuity_bonus = np.zeros(num_od, dtype=np.float64)
    if prev_selected is not None:
        for od in prev_selected:
            if od < num_od:
                continuity_bonus[od] = 1.0

    final_scores = scores_norm + da_cfg.lambda_cont * continuity_bonus

    # Select top-k from active ODs
    active_mask = np.asarray(obs.active_mask, dtype=bool) if hasattr(obs, "active_mask") else np.ones(num_od, dtype=bool)
    k = env.k_crit

    # Mask inactive ODs
    final_scores[~active_mask] = -np.inf

    # Select top-k
    if k >= active_mask.sum():
        selected = np.where(active_mask)[0].tolist()
    else:
        selected = np.argsort(final_scores)[::-1][:k].tolist()

    info = {
        "lambda_cont": da_cfg.lambda_cont,
        "continuity_bonus_active": int(continuity_bonus.sum()),
        "continuity_kept": int(sum(1 for od in selected if prev_selected and od in prev_selected)),
        "k_pred": float(k_pred) if k_pred is not None else float(k),
    }
    return selected, info


def rollout_da_gnn_selector(
    env, gnn_model, *,
    da_cfg: DAGNNConfig | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Full rollout of disturbance-aware GNN selector.

    Tracks previous selections and uses continuity bonus for stability.
    """
    if da_cfg is None:
        da_cfg = DAGNNConfig()

    if isinstance(gnn_model, (str,)):
        from pathlib import Path
        gnn_model, _ = load_gnn_selector(Path(gnn_model), device=device)

    gnn_model.eval()
    env.reset()
    rows = []
    done = False
    prev_selected_set = None
    prev_splits = None

    while not done:
        decision_start = time.perf_counter()

        selected, da_info = choose_da_gnn_selector(
            env, gnn_model,
            prev_selected=prev_selected_set,
            da_cfg=da_cfg,
            device=device,
        )
        inference_latency = time.perf_counter() - decision_start

        _, reward, done, info = env.step(selected)
        info = dict(info)

        decision_ms = (time.perf_counter() - decision_start) * 1000.0
        info["reward"] = float(reward)
        info["inference_latency_sec"] = float(inference_latency)
        info["decision_time_ms"] = float(decision_ms)
        info["method"] = DA_GNN_METHOD
        info["selected_count"] = len(selected)
        info.update({f"da_{k}": v for k, v in da_info.items()})

        rows.append(info)

        # Update state for next timestep
        prev_selected_set = set(selected)

    return pd.DataFrame(rows)


def tune_lambda_cont(
    env, gnn_model, *,
    lambda_candidates: list[float] | None = None,
    lambda_db: float = 1.0,
    device: str = "cpu",
) -> tuple[float, dict]:
    """Tune lambda_cont on validation split to minimize MLU + lambda_db * Disturbance.

    Returns:
        best_lambda: float
        results: dict[float, dict] - per-lambda results
    """
    if lambda_candidates is None:
        lambda_candidates = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    results = {}
    best_score = float("inf")
    best_lambda = 0.0

    for lam in lambda_candidates:
        da_cfg = DAGNNConfig(lambda_cont=lam)
        df = rollout_da_gnn_selector(env, gnn_model, da_cfg=da_cfg, device=device)

        mean_mlu = df["mlu"].mean()
        mean_db = df.get("disturbance", pd.Series([0.0])).mean()

        # Normalize: use lambda=0 MLU as reference
        score = mean_mlu + lambda_db * mean_db

        results[lam] = {
            "mean_mlu": float(mean_mlu),
            "mean_disturbance": float(mean_db),
            "combined_score": float(score),
        }

        if score < best_score:
            best_score = score
            best_lambda = lam

    return best_lambda, results
