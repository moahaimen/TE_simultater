"""GNN+ enhanced critical flow selector — screening extension.

Upgrades over the original GNN selector (gnn_selector.py):
  1. Richer node features: fills 4 placeholder zeros with real features
  2. Richer edge features: 8 → 12 dimensions (+4 new)
  3. Richer OD features: 10 → 18 dimensions (+8 new)
  4. Dynamic bounded K: K = max(15, min(K_pred, 40))

This file does NOT modify the original gnn_selector.py.
It imports the original model class and provides new feature builders
plus a thin wrapper that adjusts input dimensions.

Created as part of gnn-plus-extension branch.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1_reactive.drl.gnn_selector import (
    GNNSelectorConfig,
    GNNFlowSelector,
    GraphSAGELayer,
)


# ---------------------------------------------------------------------------
#  GNN+ Configuration
# ---------------------------------------------------------------------------

@dataclass
class GNNPlusConfig(GNNSelectorConfig):
    """Extended config for GNN+ with richer features."""
    node_dim: int = 16       # same dim, but 4 zeros replaced with real features
    edge_dim: int = 12       # 8 original + 4 new
    od_dim: int = 18         # 10 original + 8 new
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    residual_alpha_init: float = 0.1
    learn_k_crit: bool = True
    k_crit_min: int = 15     # bounded: dynamic K ∈ [15, 40]
    k_crit_max: int = 40     # matches all prior experiments
    device: str = "cpu"


_PLUS_TOPOLOGY_CACHE: Dict[tuple, Dict[str, object]] = {}


def _topology_cache_key(dataset, path_library) -> tuple:
    return (
        getattr(dataset, "key", "unknown"),
        id(path_library),
        len(dataset.nodes),
        len(dataset.edges),
        len(path_library.od_pairs),
    )


def _compute_clustering(adjacency: np.ndarray) -> np.ndarray:
    num_nodes = adjacency.shape[0]
    clustering = np.zeros(num_nodes, dtype=np.float64)
    for node_idx in range(num_nodes):
        nbrs = np.flatnonzero(adjacency[node_idx] > 0)
        k = int(nbrs.size)
        if k < 2:
            continue
        subgraph = adjacency[np.ix_(nbrs, nbrs)]
        triangles = float(np.sum(subgraph) / 2.0)
        clustering[node_idx] = 2.0 * triangles / (k * (k - 1))
    return clustering


def _get_plus_topology_cache(dataset, path_library) -> Dict[str, object]:
    key = _topology_cache_key(dataset, path_library)
    cached = _PLUS_TOPOLOGY_CACHE.get(key)
    if cached is not None:
        return cached

    node_to_idx = {node: idx for idx, node in enumerate(dataset.nodes)}
    num_nodes = len(dataset.nodes)
    num_edges = len(dataset.edges)
    num_od = len(path_library.od_pairs)

    src_idx = np.asarray([node_to_idx[src] for src, _ in dataset.edges], dtype=np.int64)
    dst_idx = np.asarray([node_to_idx[dst] for _, dst in dataset.edges], dtype=np.int64)
    edge_index_np = np.stack([src_idx, dst_idx], axis=0)

    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adjacency[src_idx, dst_idx] = 1.0
    adjacency[dst_idx, src_idx] = 1.0
    neighbor_counts = np.sum(adjacency, axis=1)
    clustering = _compute_clustering(adjacency)

    od_src = np.asarray([node_to_idx[src] for src, _ in path_library.od_pairs], dtype=np.int64)
    od_dst = np.asarray([node_to_idx[dst] for _, dst in path_library.od_pairs], dtype=np.int64)

    path_costs = np.zeros(num_od, dtype=np.float64)
    num_paths = np.zeros(num_od, dtype=np.float64)
    hop_count = np.zeros(num_od, dtype=np.float64)
    best_path_edges_list: List[List[int]] = [[] for _ in range(num_od)]
    alt_edge_paths: List[List[List[int]]] = [[] for _ in range(num_od)]
    best_path_incidence = np.zeros((num_od, num_edges), dtype=np.float32)
    all_path_edge_count_by_od = np.zeros((num_od, num_edges), dtype=np.float32)

    for od_idx in range(num_od):
        costs = path_library.costs_by_od[od_idx]
        edge_paths = path_library.edge_idx_paths_by_od[od_idx]
        num_paths[od_idx] = len(costs)
        if not costs:
            continue
        best_path_idx = int(np.argmin(costs))
        path_costs[od_idx] = float(costs[best_path_idx])
        best_edges = list(edge_paths[best_path_idx]) if best_path_idx < len(edge_paths) else []
        best_path_edges_list[od_idx] = best_edges
        hop_count[od_idx] = float(len(best_edges))
        if best_edges:
            best_path_incidence[od_idx, np.asarray(best_edges, dtype=np.int64)] = 1.0
        alt_edge_paths[od_idx] = [list(path) for path_idx, path in enumerate(edge_paths) if path_idx != best_path_idx and path]
        for edge_path in edge_paths:
            if edge_path:
                np.add.at(all_path_edge_count_by_od[od_idx], np.asarray(edge_path, dtype=np.int64), 1.0)

    cached = {
        "node_to_idx": node_to_idx,
        "edge_index_np": edge_index_np,
        "adjacency": adjacency,
        "neighbor_counts": neighbor_counts,
        "clustering": clustering,
        "od_src": od_src,
        "od_dst": od_dst,
        "path_costs": path_costs,
        "num_paths": num_paths,
        "hop_count": hop_count,
        "best_path_edges_list": best_path_edges_list,
        "best_path_incidence": best_path_incidence,
        "all_path_edge_count_by_od": all_path_edge_count_by_od,
        "alt_edge_paths": alt_edge_paths,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }
    _PLUS_TOPOLOGY_CACHE[key] = cached
    return cached


# ---------------------------------------------------------------------------
#  Enhanced feature builders
# ---------------------------------------------------------------------------

def build_graph_tensors_plus(
    dataset, tm_vector=None, path_library=None,
    telemetry=None, failure_mask=None,
    prev_util=None,
    device="cpu",
):
    """Build enhanced graph tensors with richer node and edge features.

    Node features [V, 16]:
      0-11: same as original
      12: ecmp_demand_through_node (normalized) — total demand routed through this node
      13: congested_neighbor_fraction — fraction of neighbors with util > 0.8
      14: max_residual_capacity — max(cap - load) on incident edges, normalized
      15: clustering_coefficient_proxy — triangles / possible triangles

    Edge features [E, 12]:
      0-7: same as original (cap, log_cap, weight, util, delay, congested, headroom, fail)
      8: num_od_sharing_edge (normalized) — how many active OD paths use this edge
      9: residual_capacity_abs — (cap - load), normalized
      10: load_change_ratio — load_t / load_{t-1}, normalized to [0, 1]
      11: is_bottleneck — binary: this edge is bottleneck of ≥1 active OD
    """
    if path_library is None:
        raise ValueError("build_graph_tensors_plus requires a path_library for cached topology features")

    dev = torch.device(device)
    capacities = np.asarray(dataset.capacities, dtype=np.float64)
    weights = np.asarray(dataset.weights, dtype=np.float64)
    cache = _get_plus_topology_cache(dataset, path_library)
    num_nodes = int(cache["num_nodes"])
    num_edges = int(cache["num_edges"])
    node_to_idx = cache["node_to_idx"]
    src_idx = np.asarray(cache["edge_index_np"][0], dtype=np.int64)
    dst_idx = np.asarray(cache["edge_index_np"][1], dtype=np.int64)
    edge_index = torch.tensor(cache["edge_index_np"], dtype=torch.long, device=dev)
    adjacency = np.asarray(cache["adjacency"], dtype=np.float64)
    neighbor_counts = np.asarray(cache["neighbor_counts"], dtype=np.float64)

    # --- Base edge features (same as original) ---
    cap_norm = capacities / (np.max(capacities) + 1e-12)
    weight_norm = weights / (np.max(weights) + 1e-12)

    util = np.zeros(num_edges, dtype=np.float64)
    delay = np.zeros(num_edges, dtype=np.float64)
    fail = np.zeros(num_edges, dtype=np.float64)
    if telemetry is not None:
        util_src = np.asarray(telemetry.utilization, dtype=np.float64)
        delay_src = np.asarray(telemetry.link_delay, dtype=np.float64)
        take = min(num_edges, util_src.size)
        util[:take] = util_src[:take]
        delay[:take] = delay_src[: min(num_edges, delay_src.size)]
        delay = delay / (np.max(delay) + 1e-12)
    if failure_mask is not None:
        fail_src = np.asarray(failure_mask, dtype=np.float64)
        fail[: min(num_edges, fail_src.size)] = fail_src[: min(num_edges, fail_src.size)]

    congested = (util > 0.9).astype(np.float64)
    headroom = np.clip(1.0 - util, 0.0, 1.0)
    log_cap = np.log1p(capacities) / (np.log1p(np.max(capacities)) + 1e-12)

    # --- NEW edge features ---
    # 8: num_od_sharing_edge
    od_per_edge = np.zeros(num_edges, dtype=np.float64)
    if path_library is not None and tm_vector is not None:
        tm = np.asarray(tm_vector, dtype=np.float64)
        active_od = (tm > 1e-12).astype(np.float32)
        od_per_edge = active_od @ np.asarray(cache["all_path_edge_count_by_od"], dtype=np.float32)
    od_per_edge_norm = od_per_edge / (np.max(od_per_edge) + 1e-12)

    # 9: residual_capacity_abs
    load = util * capacities
    residual_cap = np.clip(capacities - load, 0, None)
    residual_cap_norm = residual_cap / (np.max(capacities) + 1e-12)

    # 10: load_change_ratio
    load_change = np.ones(num_edges, dtype=np.float64) * 0.5  # default = no change (0.5 after norm)
    if prev_util is not None:
        prev_load = np.asarray(prev_util, dtype=np.float64)[:num_edges] * capacities
        ratio = load / (prev_load + 1e-12)
        # Clip to [0.5, 2.0] then normalize to [0, 1]
        ratio_clipped = np.clip(ratio, 0.5, 2.0)
        load_change = (ratio_clipped - 0.5) / 1.5  # maps [0.5, 2.0] -> [0, 1]

    # 11: is_bottleneck — edge is the bottleneck for at least one active OD
    is_bottleneck = np.zeros(num_edges, dtype=np.float64)
    if path_library is not None and tm_vector is not None:
        tm = np.asarray(tm_vector, dtype=np.float64)
        for od_idx in np.flatnonzero(tm > 1e-12):
            path_edges = cache["best_path_edges_list"][int(od_idx)]
            if not path_edges:
                continue
            path_utils = util[np.asarray(path_edges, dtype=np.int64)]
            if path_utils.size == 0:
                continue
            max_u = float(np.max(path_utils))
            for eidx in path_edges:
                if util[int(eidx)] >= max_u - 1e-12:
                    is_bottleneck[int(eidx)] = 1.0

    edge_feat = np.stack([
        cap_norm, log_cap, weight_norm, util, delay, congested, headroom, fail,
        od_per_edge_norm, residual_cap_norm, load_change, is_bottleneck,
    ], axis=1).astype(np.float32)
    edge_features = torch.tensor(edge_feat, dtype=torch.float32, device=dev)

    # --- Node features (fill placeholders 12-15 with real features) ---
    in_degree = np.bincount(dst_idx, minlength=num_nodes).astype(np.float64)
    out_degree = np.bincount(src_idx, minlength=num_nodes).astype(np.float64)
    sum_util_in = np.bincount(dst_idx, weights=util, minlength=num_nodes).astype(np.float64)
    sum_util_out = np.bincount(src_idx, weights=util, minlength=num_nodes).astype(np.float64)
    sum_cap_in = np.bincount(dst_idx, weights=capacities, minlength=num_nodes).astype(np.float64)
    sum_cap_out = np.bincount(src_idx, weights=capacities, minlength=num_nodes).astype(np.float64)
    max_util_in = np.zeros(num_nodes, dtype=np.float64)
    max_util_out = np.zeros(num_nodes, dtype=np.float64)
    np.maximum.at(max_util_out, src_idx, util)
    np.maximum.at(max_util_in, dst_idx, util)

    total_degree = in_degree + out_degree
    degree_norm = total_degree / (np.max(total_degree) + 1e-12)
    mean_util_in = sum_util_in / (in_degree + 1e-12)
    mean_util_out = sum_util_out / (out_degree + 1e-12)
    mean_cap = (sum_cap_in + sum_cap_out) / (total_degree + 1e-12)
    mean_cap_norm = mean_cap / (np.max(mean_cap) + 1e-12)
    hub_proxy = total_degree / float(num_nodes)
    log_degree = np.log1p(total_degree) / (np.log1p(np.max(total_degree)) + 1e-12)

    fail_exposure = np.zeros(num_nodes, dtype=np.float64)
    if failure_mask is not None:
        np.add.at(fail_exposure, src_idx, (fail > 0.5).astype(np.float64))
        np.add.at(fail_exposure, dst_idx, (fail > 0.5).astype(np.float64))
        fail_exposure = fail_exposure / (total_degree + 1e-12)

    # --- NEW node features (replacing placeholders 12-15) ---

    # 12: ecmp_demand_through_node — total demand for ODs where this node is src or dst
    demand_through_node = np.zeros(num_nodes, dtype=np.float64)
    if tm_vector is not None:
        tm = np.asarray(tm_vector, dtype=np.float64)
        np.add.at(demand_through_node, np.asarray(cache["od_src"], dtype=np.int64), tm)
        np.add.at(demand_through_node, np.asarray(cache["od_dst"], dtype=np.int64), tm)
    demand_through_node_norm = demand_through_node / (np.max(demand_through_node) + 1e-12)

    # 13: congested_neighbor_fraction
    congested_nodes = ((max_util_in > 0.8) | (max_util_out > 0.8)).astype(np.float64)
    congested_neighbor_frac = (adjacency @ congested_nodes) / np.maximum(neighbor_counts, 1.0)

    # 14: max_residual_capacity on incident edges
    max_residual_cap = np.zeros(num_nodes, dtype=np.float64)
    np.maximum.at(max_residual_cap, src_idx, residual_cap)
    np.maximum.at(max_residual_cap, dst_idx, residual_cap)
    max_residual_cap_norm = max_residual_cap / (np.max(max_residual_cap) + 1e-12)

    # 15: clustering coefficient proxy (undirected)
    clustering = np.asarray(cache["clustering"], dtype=np.float64)

    node_feat = np.stack([
        degree_norm,                                                        # 0
        log_degree,                                                         # 1
        np.minimum(in_degree, out_degree) / (np.max(total_degree) + 1e-12), # 2
        hub_proxy,                                                          # 3
        mean_util_in,                                                       # 4
        mean_util_out,                                                      # 5
        max_util_in,                                                        # 6
        max_util_out,                                                       # 7
        mean_cap_norm,                                                      # 8
        np.log1p(mean_cap) / (np.log1p(np.max(mean_cap)) + 1e-12),         # 9
        fail_exposure,                                                      # 10
        (in_degree / (out_degree + 1e-12)).clip(0, 5) / 5.0,               # 11
        demand_through_node_norm,                                           # 12 NEW
        congested_neighbor_frac,                                            # 13 NEW
        max_residual_cap_norm,                                              # 14 NEW
        clustering,                                                         # 15 NEW
    ], axis=1)[:, :16].astype(np.float32)

    node_features = torch.tensor(node_feat, dtype=torch.float32, device=dev)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "node_to_idx": node_to_idx,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }


def build_od_features_plus(
    dataset, tm_vector, path_library,
    telemetry=None, prev_tm=None,
    device="cpu",
):
    """Build enhanced per-OD features with 18 dimensions.

    OD features [num_od, 18]:
      0-9: same as original
      10: hop_count — shortest path hop count, normalized
      11: demand_change_ratio — tm_t / tm_{t-1}, normalized to [0, 1]
      12: src_congestion — max_util_out of source node
      13: dst_congestion — max_util_in of dest node
      14: path_overlap_score — fraction of path edges shared with other active ODs
      15: ecmp_contribution — demand * bottleneck_util / sum(all demands * bottleneck_utils)
      16: alternative_path_headroom — max headroom across non-best paths
      17: demand_x_hop — demand * hop_count, normalized
    """
    dev = torch.device(device)
    cache = _get_plus_topology_cache(dataset, path_library)
    num_od = len(path_library.od_pairs)
    num_edges = len(dataset.edges)
    num_nodes = len(dataset.nodes)

    tm = np.asarray(tm_vector, dtype=np.float64)
    tm_norm = tm / (np.max(tm) + 1e-12)

    od_src = np.asarray(cache["od_src"], dtype=np.int64)
    od_dst = np.asarray(cache["od_dst"], dtype=np.int64)

    # --- Original features ---
    sensitivity_scores = np.zeros(num_od, dtype=np.float32)
    path_costs = np.asarray(cache["path_costs"], dtype=np.float64).copy()
    num_paths = np.asarray(cache["num_paths"], dtype=np.float64).copy()
    bottleneck_util = np.zeros(num_od, dtype=np.float64)
    mean_path_util = np.zeros(num_od, dtype=np.float64)
    hop_count = np.asarray(cache["hop_count"], dtype=np.float64).copy()
    best_path_edges_list = cache["best_path_edges_list"]

    util = np.zeros(num_edges, dtype=np.float64)
    if telemetry is not None:
        util = np.asarray(telemetry.utilization, dtype=np.float64)[:num_edges]

    # Node-level congestion (for src/dst congestion features)
    max_util_in = np.zeros(num_nodes, dtype=np.float64)
    max_util_out = np.zeros(num_nodes, dtype=np.float64)
    src_idx = np.asarray(cache["edge_index_np"][0], dtype=np.int64)
    dst_idx = np.asarray(cache["edge_index_np"][1], dtype=np.int64)
    np.maximum.at(max_util_out, src_idx, util)
    np.maximum.at(max_util_in, dst_idx, util)

    # Alternative path headroom
    alt_path_headroom = np.zeros(num_od, dtype=np.float64)

    for od_idx in range(num_od):
        if path_costs[od_idx] > 0.0:
            sensitivity_scores[od_idx] = float(tm[od_idx]) * float(path_costs[od_idx])
        path_edges = best_path_edges_list[od_idx]
        if path_edges:
            path_utils = util[np.asarray(path_edges, dtype=np.int64)]
            bottleneck_util[od_idx] = float(np.max(path_utils)) if path_utils.size else 0.0
            mean_path_util[od_idx] = float(np.mean(path_utils)) if path_utils.size else 0.0

        for alt_edges in cache["alt_edge_paths"][od_idx]:
            if not alt_edges:
                continue
            path_utils = util[np.asarray(alt_edges, dtype=np.int64)]
            if path_utils.size:
                alt_path_headroom[od_idx] = max(alt_path_headroom[od_idx], 1.0 - float(np.max(path_utils)))

    # Normalize original features
    path_costs_norm = path_costs / (np.max(path_costs) + 1e-12)
    num_paths_norm = num_paths / (np.max(num_paths) + 1e-12)
    sensitivity_norm = sensitivity_scores / (np.max(np.abs(sensitivity_scores)) + 1e-12)
    active = (tm > 0).astype(np.float64)
    headroom = np.clip(1.0 - bottleneck_util, 0.0, 1.0)

    demand_rank = np.zeros(num_od, dtype=np.float64)
    sorted_idx = np.argsort(tm)
    for rank, idx in enumerate(sorted_idx):
        demand_rank[idx] = float(rank) / max(float(num_od - 1), 1.0)

    # --- NEW features ---

    # 10: hop_count normalized
    hop_count_norm = hop_count / (np.max(hop_count) + 1e-12)

    # 11: demand_change_ratio
    demand_change = np.ones(num_od, dtype=np.float64) * 0.5
    if prev_tm is not None:
        prev = np.asarray(prev_tm, dtype=np.float64)
        ratio = tm / (prev + 1e-12)
        ratio_clipped = np.clip(ratio, 0.5, 2.0)
        demand_change = (ratio_clipped - 0.5) / 1.5  # [0, 1]

    # 12: src_congestion
    src_congestion = max_util_out[od_src]

    # 13: dst_congestion
    dst_congestion = max_util_in[od_dst]

    # 14: path_overlap_score — fraction of best-path edges shared with other active ODs
    # Build edge usage count across active ODs
    edge_od_count = ((tm > 1e-12).astype(np.float32) @ np.asarray(cache["best_path_incidence"], dtype=np.float32)).astype(np.float64)
    path_overlap = np.zeros(num_od, dtype=np.float64)
    shared_edge_mask = (edge_od_count > 1.0).astype(np.float32)
    overlap_mass = np.asarray(cache["best_path_incidence"], dtype=np.float32) @ shared_edge_mask
    valid_hops = hop_count > 0
    path_overlap[valid_hops] = overlap_mass[valid_hops] / hop_count[valid_hops]

    # 15: ecmp_contribution — this OD's share of total congestion pressure
    bn_scores = tm * bottleneck_util
    total_bn = np.sum(bn_scores) + 1e-12
    ecmp_contribution = bn_scores / total_bn

    # 16: alt_path_headroom already computed above

    # 17: demand_x_hop
    demand_x_hop = tm * hop_count
    demand_x_hop_norm = demand_x_hop / (np.max(demand_x_hop) + 1e-12)

    # Stack all 18 features
    od_feat = np.stack([
        tm_norm,               # 0
        path_costs_norm,       # 1
        num_paths_norm,        # 2
        bottleneck_util,       # 3
        mean_path_util,        # 4
        headroom,              # 5
        sensitivity_norm,      # 6
        active,                # 7
        demand_rank,           # 8
        np.log1p(tm) / (np.log1p(np.max(tm)) + 1e-12),  # 9
        hop_count_norm,        # 10 NEW
        demand_change,         # 11 NEW
        src_congestion,        # 12 NEW
        dst_congestion,        # 13 NEW
        path_overlap,          # 14 NEW
        ecmp_contribution,     # 15 NEW
        alt_path_headroom,     # 16 NEW
        demand_x_hop_norm,     # 17 NEW
    ], axis=1).astype(np.float32)

    bottleneck_scores = tm.astype(np.float32) * bottleneck_util.astype(np.float32)

    return {
        "od_features": torch.tensor(od_feat, dtype=torch.float32, device=dev),
        "od_src_idx": torch.tensor(od_src, dtype=torch.long, device=dev),
        "od_dst_idx": torch.tensor(od_dst, dtype=torch.long, device=dev),
        "sensitivity_scores": torch.tensor(sensitivity_scores, dtype=torch.float32, device=dev),
        "bottleneck_scores": torch.tensor(bottleneck_scores, dtype=torch.float32, device=dev),
    }


# ---------------------------------------------------------------------------
#  GNN+ Model (same architecture, adjusted input dims)
# ---------------------------------------------------------------------------

class GNNPlusFlowSelector(GNNFlowSelector):
    """GNN+ with richer features and dynamic bounded K.

    Inherits full architecture from GNNFlowSelector.
    Only difference: adjusted input dimensions and bounded k_pred.
    """

    def __init__(self, cfg: GNNPlusConfig):
        # GNNFlowSelector.__init__ handles everything
        super().__init__(cfg)
        # Override k bounds for safety
        self._k_min = 15
        self._k_max = 40

    def select_critical_flows(self, graph_data, od_data, active_mask, k_crit_default=40,
                              path_library=None, telemetry=None, force_default_k=False):
        """Select critical flows with dynamic bounded K.

        Key difference from parent: force_default_k defaults to False,
        so K_pred is used: K = max(15, min(K_pred, 40))
        """
        with torch.no_grad():
            scores, k_pred, info = self.forward(graph_data, od_data)

        if force_default_k or k_pred is None:
            k = k_crit_default
        else:
            k = max(self._k_min, min(k_pred, self._k_max))

        scores_np = scores.detach().cpu().numpy().astype(np.float32)
        active = np.asarray(active_mask, dtype=bool)
        active_indices = np.where(active)[0]

        if active_indices.size == 0 or k <= 0:
            info["k_used"] = 0
            info["k_default"] = k_crit_default
            info["k_dynamic"] = k_pred
            return [], info

        take = min(k, active_indices.size)
        active_scores = scores_np[active_indices]
        top_local = np.argsort(-active_scores, kind="mergesort")[:take]
        selected = [int(active_indices[i]) for i in top_local]

        info["k_used"] = take
        info["k_default"] = k_crit_default
        info["k_dynamic"] = k_pred
        return selected, info


def save_gnn_plus(model: GNNPlusFlowSelector, path, extra_meta=None):
    """Save GNN+ checkpoint."""
    cfg = model.cfg
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "node_dim": cfg.node_dim,
            "edge_dim": cfg.edge_dim,
            "od_dim": cfg.od_dim,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
            "residual_alpha_init": cfg.residual_alpha_init,
            "learn_k_crit": cfg.learn_k_crit,
            "k_crit_min": cfg.k_crit_min,
            "k_crit_max": cfg.k_crit_max,
        },
        "model_type": "gnn_plus",
    }
    if extra_meta:
        payload.update(extra_meta)
    torch.save(payload, str(path))


def load_gnn_plus(path, device="cpu"):
    """Load GNN+ checkpoint."""
    payload = torch.load(str(path), map_location=torch.device(device), weights_only=False)
    cfg = GNNPlusConfig(**payload["config"])
    cfg.device = device
    model = GNNPlusFlowSelector(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg
