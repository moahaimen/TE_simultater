import numpy as np

from te.disturbance import compute_disturbance


def test_disturbance_formula_sanity():
    prev = [np.array([1.0, 0.0])]
    curr = [np.array([0.5, 0.5])]
    demand = np.array([10.0])

    # L1 = 1.0, half-L1 = 0.5, weighted by demand then normalized => 0.5
    value = compute_disturbance(prev, curr, demand)
    assert np.isclose(value, 0.5)
