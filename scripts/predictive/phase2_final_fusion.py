"""Phase 2 Final — Predictive GNN+ Sticky composition module.

This module is the heart of Phase 2 Final. It composes:
  - Phase 1 GNN+ score (kept dominant via alpha)
  - Predictive features built from a GRU forecast of link utilization
  - The Phase 1 sticky filter, LP, and do-no-harm gate (called unmodified)

Flow:
  1. gnnplus_score_only(...)             -- get raw GNN+ scores (no top-K yet)
  2. predictive_rerank_top_k(...)        -- fuse scores + features, return top-K
  3. apply_do_no_harm_gate(...)          -- Phase 1 sticky + LP + DNH (unchanged)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────
# Default fusion weights (per professor's spec)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class FusionWeights:
    """Score-level fusion coefficients. Phase 1 (current state only) is
    recovered when alpha=1 and beta=gamma=delta=eta=rho=0.
    """
    alpha: float = 0.45     # GNN+ current score (dominant — preserves Phase 1)
    beta: float = 0.20      # predicted_bottleneck_score
    gamma: float = 0.15     # predicted_hotspot_score_on_path
    delta: float = 0.10     # predicted_failure_risk_on_path
    eta: float = 0.10       # alternative_path_gain_predicted
    rho: float = 0.05       # predicted_demand_growth

    @classmethod
    def conservative(cls) -> "FusionWeights":
        # heavy on GNN, light on prediction
        return cls(alpha=0.70, beta=0.10, gamma=0.08, delta=0.05, eta=0.05, rho=0.02)

    @classmethod
    def balanced(cls) -> "FusionWeights":
        return cls()

    @classmethod
    def proactive(cls) -> "FusionWeights":
        return cls(alpha=0.30, beta=0.25, gamma=0.20, delta=0.12, eta=0.10, rho=0.03)

    @classmethod
    def phase1_only(cls) -> "FusionWeights":
        # Recovers Phase 1 GNN+ Sticky (no prediction influence)
        return cls(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, eta=0.0, rho=0.0)


# ─────────────────────────────────────────────────────────────────────
# Predictive features over the candidate pool
# ─────────────────────────────────────────────────────────────────────
def _path_max_util(util: np.ndarray, edge_idx_path) -> float:
    if edge_idx_path is None or len(edge_idx_path) == 0:
        return 0.0
    return float(util[np.asarray(edge_idx_path, dtype=np.int64)].max())


def compute_predictive_features(
    *,
    candidate_indices: np.ndarray,
    tm_now: np.ndarray,
    tm_prev: np.ndarray | None,
    predicted_util_horizon: np.ndarray,        # (H, num_links)
    failure_risk_per_link: np.ndarray | None,  # (num_links,) in [0, 1] OR None
    path_library,
    hotspot_threshold: float = 0.7,
) -> dict[str, np.ndarray]:
    """Build per-candidate-OD predictive features.

    Returns dict of arrays of length len(candidate_indices), in the SAME
    order as candidate_indices. All features are pre-normalization.
    """
    n = len(candidate_indices)
    bn_pred = np.zeros(n, dtype=np.float64)
    hotspot_score = np.zeros(n, dtype=np.float64)
    failure_risk = np.zeros(n, dtype=np.float64)
    alt_path_gain = np.zeros(n, dtype=np.float64)
    demand_growth = np.zeros(n, dtype=np.float64)

    pred_util_max_over_h = predicted_util_horizon.max(axis=0)  # (num_links,)

    for i, od in enumerate(candidate_indices):
        od = int(od)
        paths = path_library.edge_idx_paths_by_od[od] if od < len(path_library.edge_idx_paths_by_od) else []
        if not paths:
            continue
        primary = paths[0]
        # 1. predicted_bottleneck_score on primary path (max over horizon)
        primary_pred = _path_max_util(pred_util_max_over_h, primary)
        bn_pred[i] = float(tm_now[od]) * primary_pred

        # 2. hotspot score on path: count of edges predicted above threshold
        if primary:
            edges = np.asarray(primary, dtype=np.int64)
            hotspot_count = float((pred_util_max_over_h[edges] > hotspot_threshold).sum())
            hotspot_score[i] = float(tm_now[od]) * hotspot_count

        # 3. failure risk on path: max predicted failure risk along the path
        if failure_risk_per_link is not None and primary:
            edges = np.asarray(primary, dtype=np.int64)
            failure_risk[i] = float(failure_risk_per_link[edges].max())

        # 4. alternative_path_gain (predicted)
        if len(paths) > 1:
            alt_costs = [_path_max_util(pred_util_max_over_h, p) for p in paths[1:]]
            best_alt = float(min(alt_costs)) if alt_costs else primary_pred
            gain = primary_pred - best_alt
            alt_path_gain[i] = max(gain, 0.0)

        # 5. predicted demand growth: tm[t]/(prev_tm[t-1]+eps) - 1
        if tm_prev is not None and tm_prev[od] > 1e-9:
            demand_growth[i] = max(float(tm_now[od]) / float(tm_prev[od]) - 1.0, 0.0)
        else:
            demand_growth[i] = 0.0

    return {
        "bn_pred": bn_pred,
        "hotspot_score": hotspot_score,
        "failure_risk_on_path": failure_risk,
        "alt_path_gain": alt_path_gain,
        "demand_growth": demand_growth,
    }


# ─────────────────────────────────────────────────────────────────────
# Score fusion + reselection
# ─────────────────────────────────────────────────────────────────────
def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    denom = float(np.abs(x).max())
    if denom < 1e-12:
        return np.zeros_like(x)
    return x / denom


def predictive_rerank_top_k(
    *,
    gnn_scores: np.ndarray,            # (num_candidates,) — raw GNN+ scores
    candidate_indices: np.ndarray,     # (num_candidates,) — OD indices
    active_mask_full: np.ndarray,      # (num_od,) — which ODs have demand & paths
    num_od: int,
    predictive_features: dict[str, np.ndarray],
    k_crit: int,
    weights: FusionWeights,
) -> tuple[list[int], np.ndarray]:
    """Fuse Phase 1 GNN+ scores with predictive features and select top-K.

    Returns (selected_ods, full_score_vector).

    The fusion lives ONLY on the candidate pool (the GNN+ already selected
    a candidate pool). For ODs outside the pool, score is -inf.
    """
    n_cand = len(candidate_indices)
    gnn_norm = _normalize(np.asarray(gnn_scores, dtype=np.float64))
    bn_norm = _normalize(predictive_features["bn_pred"])
    hot_norm = _normalize(predictive_features["hotspot_score"])
    fail_norm = _normalize(predictive_features["failure_risk_on_path"])
    alt_norm = _normalize(predictive_features["alt_path_gain"])
    dg_norm = _normalize(predictive_features["demand_growth"])

    fused = (weights.alpha * gnn_norm
             + weights.beta * bn_norm
             + weights.gamma * hot_norm
             + weights.delta * fail_norm
             + weights.eta * alt_norm
             + weights.rho * dg_norm)

    # Place fused values into a num_od-length vector for top-K selection
    full_score = np.full(num_od, -np.inf, dtype=np.float64)
    for i, od in enumerate(candidate_indices):
        full_score[int(od)] = float(fused[i])

    # Mask out inactive ODs
    full_score[~active_mask_full] = -np.inf

    # Top-K (no continuity bonus here — sticky filter handles continuity later)
    finite_count = int(np.isfinite(full_score).sum())
    if finite_count == 0:
        return [], full_score
    k_eff = min(k_crit, finite_count)
    if k_eff >= num_od:
        order = list(np.argsort(-full_score).tolist())
    else:
        idx = np.argpartition(-full_score, k_eff)[:k_eff]
        order = list(idx[np.argsort(-full_score[idx])].tolist())
    return [int(o) for o in order if np.isfinite(full_score[int(o)])], full_score


# ─────────────────────────────────────────────────────────────────────
# Failure-risk derivation (no separate ML model — use predicted util directly)
# ─────────────────────────────────────────────────────────────────────
def compute_effective_capacities(
    capacities: np.ndarray,
    predicted_util_horizon: np.ndarray,
    *, mpc_lambda: float = 0.5, threshold: float = 0.7,
) -> np.ndarray:
    """Path 2 — Predictive MPC at the LP level.

    Returns effective capacities that are SHRUNK on links predicted to be
    hot in the next horizon. The LP then minimizes max(load/effective_cap)
    and naturally routes AWAY from predicted-hot links, even when the
    OD selection itself is unchanged.

    formula:
        effective[link] = capacity[link] / (1 + lambda * max(predicted_util_max[link] - threshold, 0))

    With lambda=0, effective_capacities == capacities (Phase 1 fallback).
    With lambda>0, links predicted above the threshold are penalized
    proportionally to how hot they are predicted to get.
    """
    if mpc_lambda <= 0.0:
        return np.asarray(capacities, dtype=np.float64)
    util_max = np.asarray(predicted_util_horizon, dtype=np.float64).max(axis=0)
    # If util is in raw bandwidth units (not fraction), normalize relative
    # to the per-link mean so the threshold is interpretable.
    if util_max.max() > 1.5:
        util_max = util_max / max(util_max.max(), 1e-9)
    excess = np.maximum(util_max - threshold, 0.0)
    penalty = 1.0 + mpc_lambda * excess
    return np.asarray(capacities, dtype=np.float64) / penalty


def derive_failure_risk_from_predicted_util(
    predicted_util_horizon: np.ndarray,
    *, hotspot_threshold: float = 0.7, overload_threshold: float = 0.95,
) -> np.ndarray:
    """Per-link failure risk in [0, 1] derived from predicted utilization.

    Heuristic mapping:
      0       if max predicted util on link < hotspot_threshold
      linear  in [0, 1] between hotspot_threshold and overload_threshold
      1       if max predicted util on link >= overload_threshold

    This is a deliberately simple proxy for failure risk. A separate
    ML classifier (RandomForest or small GRU) could replace this — see
    train_failure_risk_predictor.py — but the heuristic is used by
    default to keep Phase 2 Final reproducible.
    """
    util_max = predicted_util_horizon.max(axis=0)
    hot = max(hotspot_threshold, 1e-9)
    over = max(overload_threshold, hot + 1e-6)
    risk = np.clip((util_max - hot) / (over - hot), 0.0, 1.0)
    return risk
