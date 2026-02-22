import numpy as np
import networkx as nx

from te.paths import build_k_shortest_paths
from te.simulator import apply_routing


def test_apply_routing_mlu():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=1.0)
    graph.add_edge("B", "C", weight=1.0)

    edge_to_idx = {edge: idx for idx, edge in enumerate(graph.edges())}
    path_lib = build_k_shortest_paths(graph, [("A", "C")], edge_to_idx=edge_to_idx, k=1)

    tm = np.array([10.0])
    splits = [np.array([1.0])]
    capacities = np.array([5.0, 10.0])

    result = apply_routing(tm, splits, path_lib, capacities)
    assert np.isclose(result.link_loads[0], 10.0)
    assert np.isclose(result.link_loads[1], 10.0)
    assert np.isclose(result.mlu, 2.0)
