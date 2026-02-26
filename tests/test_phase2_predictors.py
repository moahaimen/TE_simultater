import numpy as np

from phase2.predictors import (
    EnsemblePredictor,
    LSTMPredictor,
    MovingAveragePredictor,
    NaiveLastPredictor,
    RidgeAutoRegressivePredictor,
    SeasonalNaivePredictor,
    compute_prediction_metrics,
)


def test_naive_last_predictor_returns_last_row() -> None:
    tm = np.array([[1.0, 2.0], [3.0, 4.0], [7.0, 9.0]], dtype=float)
    pred = NaiveLastPredictor().predict_next(tm)
    assert np.allclose(pred, tm[-1])


def test_seasonal_naive_fallback_and_lag() -> None:
    model = SeasonalNaivePredictor(season_lag=4)

    short = np.array([[1.0], [2.0], [3.0]], dtype=float)
    assert np.allclose(model.predict_next(short), np.array([3.0]))

    long = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=float)
    # len=6 and lag=4 => prediction uses entry index -4 == value 3.0
    assert np.allclose(model.predict_next(long), np.array([3.0]))


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


def test_lstm_predictor_shapes_and_finite() -> None:
    tm = np.array(
        [
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2],
            [1.3, 2.3],
            [1.4, 2.4],
            [1.5, 2.5],
            [1.6, 2.6],
            [1.7, 2.7],
        ],
        dtype=float,
    )
    model = LSTMPredictor(window=3, hidden_dim=16, num_layers=1, epochs=2, patience=2)
    model.fit(tm[:6], tm[6:], seed=7)
    pred = model.predict_next(tm[:7])
    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0.0)


def test_ensemble_weights_sum_to_one() -> None:
    rng = np.random.default_rng(13)
    tm = rng.uniform(0.0, 5.0, size=(24, 3)).astype(float)
    model = EnsemblePredictor(
        season_lag=4,
        ridge_window=3,
        ridge_alpha=1e-2,
        lstm_window=3,
        lstm_hidden_dim=8,
        lstm_layers=1,
        lstm_epochs=2,
        lstm_patience=2,
    )
    model.fit(tm[:18], tm[18:], seed=9)
    assert np.isclose(float(np.sum(model.weights)), 1.0, atol=1e-6)
    pred = model.predict_next(tm[:20])
    assert pred.shape == (3,)
    assert np.all(pred >= 0.0)


def test_prediction_metrics_non_negative() -> None:
    pred = np.array([1.0, 3.0, 5.0], dtype=float)
    actual = np.array([2.0, 3.0, 4.0], dtype=float)
    m = compute_prediction_metrics(pred, actual)
    assert m.mae >= 0.0
    assert m.rmse >= 0.0
    assert m.smape >= 0.0
