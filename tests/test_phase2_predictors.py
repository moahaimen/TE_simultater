import numpy as np

from phase2.predictors import (
    MovingAveragePredictor,
    NaiveLastPredictor,
    RidgeAutoRegressivePredictor,
    compute_prediction_metrics,
)


def test_naive_last_predictor_returns_last_row() -> None:
    tm = np.array([[1.0, 2.0], [3.0, 4.0], [7.0, 9.0]], dtype=float)
    pred = NaiveLastPredictor().predict_next(tm)
    assert np.allclose(pred, tm[-1])


def test_moving_average_predictor_window() -> None:
    tm = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]], dtype=float)
    pred = MovingAveragePredictor(window=2).predict_next(tm)
    assert np.allclose(pred, np.array([4.0, 6.0]))


def test_ridge_ar_predictor_shapes_and_finite() -> None:
    # OD0 increases linearly, OD1 stays constant.
    tm = np.array(
        [
            [1.0, 3.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [4.0, 3.0],
            [5.0, 3.0],
        ],
        dtype=float,
    )
    model = RidgeAutoRegressivePredictor(window=2, alpha=1e-2)
    model.fit(tm)
    pred = model.predict_next(tm)
    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0.0)


def test_prediction_metrics_non_negative() -> None:
    pred = np.array([1.0, 3.0, 5.0], dtype=float)
    actual = np.array([2.0, 3.0, 4.0], dtype=float)
    m = compute_prediction_metrics(pred, actual)
    assert m.mae >= 0.0
    assert m.rmse >= 0.0
    assert m.smape >= 0.0
