import numpy as np
from pathlib import Path

from phase3.tm_mgm import generate_mgm_tm
from phase3.topology_sources import parse_rocketfuel_topology, parse_topologyzoo_topology


def test_mgm_shape_and_nonnegative() -> None:
    od_pairs = [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A")]
    tm = generate_mgm_tm(od_pairs=od_pairs, steps=12, seed=7)
    assert tm.shape == (12, 4)
    assert np.all(np.isfinite(tm))
    assert np.all(tm >= 0)


def test_parse_rocketfuel_sample() -> None:
    topo = parse_rocketfuel_topology(Path("data/samples/rocketfuel_sample.txt"))
    assert len(topo.nodes) >= 2
    assert len(topo.edges) > 0
    assert topo.capacities.shape[0] == len(topo.edges)


def test_parse_topozoo_sample() -> None:
    topo = parse_topologyzoo_topology(Path("data/samples/topologyzoo_sample.graphml"))
    assert len(topo.nodes) >= 2
    assert len(topo.edges) > 0
    assert topo.weights.shape[0] == len(topo.edges)
