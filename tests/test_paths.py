import networkx as nx

from te.paths import build_k_shortest_paths


def test_build_k_shortest_paths_orders_by_cost():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=1.0)
    graph.add_edge("B", "D", weight=1.0)
    graph.add_edge("A", "C", weight=1.0)
    graph.add_edge("C", "D", weight=2.0)
    graph.add_edge("A", "D", weight=5.0)

    edge_to_idx = {edge: idx for idx, edge in enumerate(graph.edges())}
    lib = build_k_shortest_paths(graph, [("A", "D")], edge_to_idx=edge_to_idx, k=3)

    costs = lib.costs_by_od[0]
    assert len(costs) == 3
    assert costs[0] <= costs[1] <= costs[2]
    assert lib.node_paths_by_od[0][0] == ["A", "B", "D"]
