"""Traffic-matrix predictors for Phase-2 proactive TE experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass
class PredictionMetrics:
    mae: float
    rmse: float
    smape: float


class BaseTMPredictor:
    """Base interface for one-step TM predictors."""

    name: str = "base"

    def required_history(self) -> int:
        return 1

    def fit(self, tm_train: np.ndarray) -> None:
        _ = tm_train

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NaiveLastPredictor(BaseTMPredictor):
    """Persistence baseline: next demand equals the last observed demand."""

    name = "naive_last"

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history, dtype=float)
        if hist.ndim != 2 or hist.shape[0] == 0:
            raise ValueError("history must be a non-empty [T, |OD|] matrix")
        return hist[-1].copy()


class MovingAveragePredictor(BaseTMPredictor):
    """Rolling mean predictor over the most recent window."""

    name = "moving_avg"

    def __init__(self, window: int = 4):
        if window <= 0:
            raise ValueError("window must be >= 1")
        self.window = int(window)

    def required_history(self) -> int:
        return self.window

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history, dtype=float)
        if hist.ndim != 2 or hist.shape[0] == 0:
            raise ValueError("history must be a non-empty [T, |OD|] matrix")
        w = min(self.window, hist.shape[0])
        return np.mean(hist[-w:], axis=0)


class RidgeAutoRegressivePredictor(BaseTMPredictor):
    """
    Per-OD ridge autoregression using only that OD's own lagged values.

    This keeps training cheap even when |OD| is large, while still capturing
    short-term temporal structure beyond naive persistence.
    """

    name = "ar_ridge"

    def __init__(self, window: int = 6, alpha: float = 1e-2):
        if window <= 0:
            raise ValueError("window must be >= 1")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.window = int(window)
        self.alpha = float(alpha)
        self.weights: np.ndarray | None = None  # [|OD|, window]
        self.bias: np.ndarray | None = None  # [|OD|]

    def required_history(self) -> int:
        return self.window

    def fit(self, tm_train: np.ndarray) -> None:
        tm = np.asarray(tm_train, dtype=float)
        if tm.ndim != 2 or tm.shape[0] < 2:
            raise ValueError("tm_train must be [T, |OD|] with T >= 2")

        num_steps, num_od = tm.shape
        w = self.window

        weights = np.zeros((num_od, w), dtype=float)
        bias = np.zeros(num_od, dtype=float)

        # We fit one compact model per OD to avoid high-dimensional global regression.
        for od_idx in range(num_od):
            series = tm[:, od_idx]

            x_rows = []
            y_vals = []
            for t in range(w, num_steps):
                # Lag order: [t-1, t-2, ..., t-w]
                lag_vec = series[t - w : t][::-1]
                x_rows.append(lag_vec)
                y_vals.append(series[t])

            if not x_rows:
                # Not enough history: fallback to constant predictor.
                bias[od_idx] = float(series[-1])
                continue

            X = np.asarray(x_rows, dtype=float)
            y = np.asarray(y_vals, dtype=float)
            X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=float)], axis=1)

            reg = self.alpha * np.eye(w + 1, dtype=float)
            reg[-1, -1] = 0.0  # do not regularize intercept

            lhs = X_aug.T @ X_aug + reg
            rhs = X_aug.T @ y

            try:
                beta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(lhs) @ rhs

            weights[od_idx] = beta[:w]
            bias[od_idx] = float(beta[-1])

        self.weights = weights
        self.bias = bias

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history, dtype=float)
        if hist.ndim != 2 or hist.shape[0] == 0:
            raise ValueError("history must be a non-empty [T, |OD|] matrix")
        if self.weights is None or self.bias is None:
            raise RuntimeError("predictor must be fitted before predict_next")

        num_od = hist.shape[1]
        if num_od != self.weights.shape[0]:
            raise ValueError("OD dimension mismatch between fit data and history")

        w = self.window
        out = np.zeros(num_od, dtype=float)
        for od_idx in range(num_od):
            series = hist[:, od_idx]
            if series.shape[0] >= w:
                lag_vec = series[-w:][::-1]
            else:
                pad = np.zeros(w, dtype=float)
                pad[: series.shape[0]] = series[::-1]
                lag_vec = pad

            pred = float(np.dot(self.weights[od_idx], lag_vec) + self.bias[od_idx])
            out[od_idx] = max(pred, 0.0)

        return out


def build_predictor(name: str, window: int = 6, alpha: float = 1e-2) -> BaseTMPredictor:
    key = str(name).strip().lower()
    if key in {"naive", "naive_last", "last"}:
        return NaiveLastPredictor()
    if key in {"ma", "moving_avg", "moving_average"}:
        return MovingAveragePredictor(window=window)
    if key in {"ar", "ar_ridge", "ridge_ar"}:
        return RidgeAutoRegressivePredictor(window=window, alpha=alpha)
    raise ValueError(f"Unknown predictor '{name}'")


def compute_prediction_metrics(pred: np.ndarray, actual: np.ndarray) -> PredictionMetrics:
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    if p.shape != a.shape:
        raise ValueError("pred and actual must have same shape")

    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))

    denom = np.abs(p) + np.abs(a) + EPS
    smape = float(np.mean(2.0 * np.abs(err) / denom))

    return PredictionMetrics(mae=mae, rmse=rmse, smape=smape)
