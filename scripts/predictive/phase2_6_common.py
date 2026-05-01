"""Phase 2.6 common utilities: scoring, scenario generation, metrics.

Shared by:
  - eval_phase2_6_predictive_cfs.py
  - eval_phase2_6_predictive_mpc.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys
from typing import Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "predictive"))

from train_gru_forecaster import GRUForecaster  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data" / "forecasting"
GRU_LINKUTIL_ROOT = PROJECT_ROOT / "results" / "predictive_phase2_gru_linkutil"


# ─────────────────────────────────────────────────────────────────────
# Forecaster wrapper
# ─────────────────────────────────────────────────────────────────────
@dataclass
class LoadedForecaster:
    model: GRUForecaster
    feat_mean: np.ndarray   # shape (1, 1, num_links)
    feat_std: np.ndarray
    window: int
    num_links: int


def load_linkutil_forecaster(topo: str, num_links: int) -> LoadedForecaster:
    ckpt_path = GRU_LINKUTIL_ROOT / topo / "gru_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No GRU link-util checkpoint at {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = GRUForecaster(num_od=num_links, hidden=cfg["hidden"],
                          layers=cfg["layers"], dropout=0.0)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return LoadedForecaster(
        model=model,
        feat_mean=payload["feat_mean"],
        feat_std=payload["feat_std"],
        window=int(cfg["window"]),
        num_links=int(cfg.get("num_links", num_links)),
    )


def predict_linkutil(fc: LoadedForecaster, util_history: np.ndarray) -> np.ndarray:
    """util_history: (window, num_links). Returns predicted next-step util in real-space."""
    util_log = np.log1p(util_history)
    fm = fc.feat_mean.squeeze(axis=(0, 1))
    fs = fc.feat_std.squeeze(axis=(0, 1))
    x_norm = (util_log - fm) / fs
    x = torch.from_numpy(x_norm).float().unsqueeze(0)
    with torch.no_grad():
        out_norm = fc.model(x).cpu().numpy().squeeze(0)
    out_log = out_norm * fs + fm
    return np.expm1(out_log)


def predict_linkutil_horizon(fc: LoadedForecaster, util_history: np.ndarray,
                             horizon: int) -> np.ndarray:
    """Roll the forecaster forward `horizon` steps. Returns (horizon, num_links)."""
    history = util_history.copy()
    preds = []
    for _ in range(max(1, horizon)):
        nxt = predict_linkutil(fc, history)
        preds.append(nxt)
        history = np.concatenate([history[1:], nxt[None, :]], axis=0)
    return np.stack(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────
# Predictive-CFS scoring (corrected per audit)
# ─────────────────────────────────────────────────────────────────────
def compute_path_max_util(util: np.ndarray, edge_idx_path: Sequence[int]) -> float:
    if not edge_idx_path:
        return 0.0
    return float(util[np.asarray(edge_idx_path, dtype=np.int64)].max())


def compute_path_cost_predicted(predicted_util_horizon: np.ndarray,
                                edge_idx_path: Sequence[int]) -> float:
    """Path cost = max over horizon of (max util on path).

    Captures the worst-case predicted load on this path within the horizon.
    """
    if not edge_idx_path:
        return 0.0
    arr = np.asarray(edge_idx_path, dtype=np.int64)
    # predicted_util_horizon: (horizon, num_links)
    return float(predicted_util_horizon[:, arr].max())


def predictive_cfs_score(
    *,
    tm_vector: np.ndarray,
    predicted_util_horizon: np.ndarray,
    path_library,
    num_od: int,
    require_positive_gain: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corrected predictive scoring.

    score(OD) = demand × predicted_bottleneck_risk × max(alt_path_gain, 0)

    where:
      predicted_bottleneck_risk = max predicted utilization on the OD's
        primary (cheapest) path over the horizon.
      alt_path_gain = predicted_path_cost(primary) - predicted_path_cost(best alt)
        i.e. the gain from rerouting away from the primary.
        Negative gain means alternates are no better => OD is not useful
        to select.

    Returns (score, bottleneck_risk, alt_path_gain) — all shape (num_od,).
    """
    score = np.zeros(num_od, dtype=np.float64)
    bottleneck_risk = np.zeros(num_od, dtype=np.float64)
    alt_path_gain = np.zeros(num_od, dtype=np.float64)

    for od in range(num_od):
        paths = path_library.edge_idx_paths_by_od[od] if od < len(path_library.edge_idx_paths_by_od) else []
        if not paths or tm_vector[od] <= 1e-12:
            continue

        primary = paths[0]
        primary_cost = compute_path_cost_predicted(predicted_util_horizon, primary)
        bottleneck_risk[od] = primary_cost

        if len(paths) == 1:
            # No alternates -> gain = 0; OD cannot be usefully rerouted
            alt_path_gain[od] = 0.0
        else:
            alt_costs = [
                compute_path_cost_predicted(predicted_util_horizon, p)
                for p in paths[1:]
            ]
            best_alt_cost = float(min(alt_costs)) if alt_costs else primary_cost
            alt_path_gain[od] = primary_cost - best_alt_cost

        gain = max(alt_path_gain[od], 0.0) if require_positive_gain else alt_path_gain[od]
        score[od] = float(tm_vector[od]) * primary_cost * gain

    return score, bottleneck_risk, alt_path_gain


def reactive_cfs_score(
    *,
    tm_vector: np.ndarray,
    current_util: np.ndarray,
    path_library,
    num_od: int,
) -> np.ndarray:
    """Standard bottleneck-style: tm × max(current util on primary path).

    No alt-path-gain filter (matches the existing bottleneck baseline)."""
    score = np.zeros(num_od, dtype=np.float64)
    for od in range(num_od):
        paths = path_library.edge_idx_paths_by_od[od] if od < len(path_library.edge_idx_paths_by_od) else []
        if not paths or tm_vector[od] <= 1e-12:
            continue
        primary = paths[0]
        score[od] = float(tm_vector[od]) * compute_path_max_util(current_util, primary)
    return score


def select_topk(score: np.ndarray, k: int) -> list[int]:
    if k >= len(score):
        return list(np.argsort(-score).tolist())
    idx = np.argpartition(-score, k)[:k]
    return list(idx[np.argsort(-score[idx])].tolist())


# ─────────────────────────────────────────────────────────────────────
# Dynamic stress scenario generators
# ─────────────────────────────────────────────────────────────────────
def gen_normal(tm: np.ndarray, **kwargs) -> np.ndarray:
    return tm.copy()


def gen_traffic_spike(tm: np.ndarray, *, spike_factor: float = 2.0,
                      test_start: int | None = None, spike_offset: int = 5,
                      spike_len: int = 8, top_k: int = 5,
                      seed: int = 1234) -> np.ndarray:
    """Multiply the top-k highest-demand ODs by spike_factor for spike_len cycles.

    Perturbation is placed INSIDE the test window so the simulator actually
    sees it during evaluation. Default: starts spike_offset cycles after
    test_start (or 85% of series length when test_start unspecified).
    """
    rng = np.random.default_rng(seed)
    out = tm.copy()
    n_steps = tm.shape[0]
    if test_start is None:
        test_start = int(n_steps * 0.85)
    spike_start = test_start + spike_offset
    spike_end = min(spike_start + spike_len, n_steps)
    if spike_start >= n_steps:
        return out
    mean_demand = tm.mean(axis=0)
    top_ods = np.argsort(-mean_demand)[:top_k]
    out[spike_start:spike_end, top_ods] *= float(spike_factor)
    return out


def gen_ramp_up(tm: np.ndarray, *, ramp_factor: float = 2.0,
                test_start: int | None = None, ramp_offset: int = 5,
                ramp_len: int = 20, top_k: int = 10) -> np.ndarray:
    """Linearly ramp up the top-k ODs from 1x to ramp_factor over ramp_len cycles."""
    out = tm.copy()
    n_steps = tm.shape[0]
    if test_start is None:
        test_start = int(n_steps * 0.85)
    ramp_start = test_start + ramp_offset
    ramp_end = min(ramp_start + ramp_len, n_steps)
    if ramp_start >= n_steps:
        return out
    mean_demand = tm.mean(axis=0)
    top_ods = np.argsort(-mean_demand)[:top_k]
    for i, t in enumerate(range(ramp_start, ramp_end)):
        progress = (i + 1) / ramp_len
        factor = 1.0 + (ramp_factor - 1.0) * progress
        out[t, top_ods] *= factor
    if ramp_end < n_steps:
        out[ramp_end:, top_ods] *= float(ramp_factor)
    return out


def gen_ramp_down(tm: np.ndarray, *, ramp_factor: float = 2.0,
                  test_start: int | None = None, ramp_offset: int = 5,
                  ramp_len: int = 20, top_k: int = 10) -> np.ndarray:
    """Starts at ramp_factor at test_start, linearly decays to 1x over ramp_len."""
    out = tm.copy()
    n_steps = tm.shape[0]
    if test_start is None:
        test_start = int(n_steps * 0.85)
    ramp_start = test_start + ramp_offset
    ramp_end = min(ramp_start + ramp_len, n_steps)
    if ramp_start >= n_steps:
        return out
    mean_demand = tm.mean(axis=0)
    top_ods = np.argsort(-mean_demand)[:top_k]
    out[ramp_start:ramp_end, top_ods] = out[ramp_start:ramp_end, top_ods].astype(np.float64)
    for i, t in enumerate(range(ramp_start, ramp_end)):
        progress = (i + 1) / ramp_len
        factor = ramp_factor + (1.0 - ramp_factor) * progress
        out[t, top_ods] *= factor
    return out


def gen_flash_crowd(tm: np.ndarray, *, flash_factor: float = 5.0,
                    test_start: int | None = None, flash_offset: int = 10,
                    flash_len: int = 3, seed: int = 1234) -> np.ndarray:
    """Brief, dramatic spike on a few random ODs, within the test window."""
    rng = np.random.default_rng(seed)
    out = tm.copy()
    n_steps, n_od = tm.shape
    if test_start is None:
        test_start = int(n_steps * 0.85)
    flash_start = test_start + flash_offset
    flash_end = min(flash_start + flash_len, n_steps)
    if flash_start >= n_steps:
        return out
    flash_ods = rng.choice(n_od, size=min(3, n_od), replace=False)
    out[flash_start:flash_end, flash_ods] *= float(flash_factor)
    return out


def make_scenario_fn(name: str, test_start: int | None = None):
    """Return a callable that applies the named scenario to a TM, anchored
    to the given test_start so perturbations land in the evaluation window.
    """
    if name == "normal":
        return lambda tm: gen_normal(tm)
    if name == "spike_1p5x":
        return lambda tm: gen_traffic_spike(tm, spike_factor=1.5, test_start=test_start)
    if name == "spike_2x":
        return lambda tm: gen_traffic_spike(tm, spike_factor=2.0, test_start=test_start)
    if name == "ramp_up":
        return lambda tm: gen_ramp_up(tm, test_start=test_start)
    if name == "ramp_down":
        return lambda tm: gen_ramp_down(tm, test_start=test_start)
    if name == "flash_crowd":
        return lambda tm: gen_flash_crowd(tm, test_start=test_start)
    raise ValueError(f"unknown scenario {name}")


SCENARIOS = {
    "normal":           make_scenario_fn("normal"),
    "spike_1p5x":       make_scenario_fn("spike_1p5x"),
    "spike_2x":         make_scenario_fn("spike_2x"),
    "ramp_up":          make_scenario_fn("ramp_up"),
    "ramp_down":        make_scenario_fn("ramp_down"),
    "flash_crowd":      make_scenario_fn("flash_crowd"),
}


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────
def overload_duration(mlu_series: np.ndarray, threshold: float) -> int:
    """Number of cycles with MLU > threshold (for util-fraction topologies).

    For raw-bandwidth topologies, threshold is interpreted relative to the max.
    """
    if mlu_series.size == 0:
        return 0
    if mlu_series.max() <= 1.0 + 1e-9:
        # util-fraction topology
        return int((mlu_series > threshold).sum())
    # raw-bandwidth: threshold is fractional vs max in series
    cap = mlu_series.max()
    return int((mlu_series > threshold * cap).sum())


def recovery_time(mlu_series: np.ndarray, peak_idx: int, baseline: float,
                  tol: float = 1.05) -> int:
    """Cycles between the peak and return to within tol×baseline. -1 if never."""
    if peak_idx >= mlu_series.size - 1:
        return -1
    target = baseline * tol
    for i in range(peak_idx + 1, mlu_series.size):
        if mlu_series[i] <= target:
            return i - peak_idx
    return -1


def summarize_mlu(mlu_series: np.ndarray) -> dict:
    if mlu_series.size == 0:
        return {"mean": float("nan"), "p95": float("nan"), "peak": float("nan"),
                "overload_0p7": 0, "overload_0p9": 0}
    return {
        "mean": float(mlu_series.mean()),
        "p95": float(np.percentile(mlu_series, 95)),
        "peak": float(mlu_series.max()),
        "overload_0p7": overload_duration(mlu_series, 0.7),
        "overload_0p9": overload_duration(mlu_series, 0.9),
    }


def compute_disturbance_series(splits_history: list, tm_history: np.ndarray) -> np.ndarray:
    """Per-cycle disturbance vs previous cycle. Returns array of length len(splits_history)-1."""
    from te.disturbance import compute_disturbance
    out = np.zeros(max(0, len(splits_history) - 1), dtype=np.float64)
    for i in range(1, len(splits_history)):
        try:
            d = compute_disturbance(splits_history[i - 1], splits_history[i], tm_history[i])
            out[i - 1] = float(d)
        except Exception:
            out[i - 1] = 0.0
    return out
