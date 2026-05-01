"""Phase 2.0: Dump per-topology forecasting datasets.

Pulls TM time series + link utilization (under ECMP routing) from the
existing TEDataset loader, and writes per-topology .npz files ready for
forecasting models.

Usage:
    python scripts/predictive/dump_forecasting_data.py
    python scripts/predictive/dump_forecasting_data.py --topology abilene

Output:
    data/forecasting/<topo>/tm_series.npz
    data/forecasting/<topo>/link_util_series.npz   (under ECMP)
    data/forecasting/<topo>/split_indices.json
    data/forecasting/<topo>/topology_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_gnnplus_packet_sdn_full import load_dataset

ALL_TOPOLOGIES = [
    "abilene",
    "cernet",
    "ebone",
    "geant",
    "sprintlink",
    "tiscali",
    "germany50",
    "vtlwavenet2011",
]


def compute_ecmp_link_util(dataset, path_library) -> np.ndarray:
    """Per-cycle link utilization under ECMP. Returns (num_steps, num_edges).

    PathLibrary exposes edge_idx_paths_by_od: list of OD entries, each being
    a list of paths, each being a list of edge-indices into dataset.edges.
    Under ECMP, demand[OD] is split equally across the available paths.
    """
    num_edges = len(dataset.edges)
    capacities = np.asarray(dataset.capacities, dtype=np.float64)
    num_od = len(dataset.od_pairs)

    edge_idx_paths_by_od = getattr(path_library, "edge_idx_paths_by_od", None)
    if edge_idx_paths_by_od is None:
        raise AttributeError(
            f"PathLibrary missing edge_idx_paths_by_od (have: {dir(path_library)})"
        )

    od_edge_share = np.zeros((num_od, num_edges), dtype=np.float64)
    for od_idx in range(num_od):
        paths = edge_idx_paths_by_od[od_idx] if od_idx < len(edge_idx_paths_by_od) else []
        if not paths:
            continue
        share = 1.0 / len(paths)
        for path_edge_indices in paths:
            for eidx in path_edge_indices:
                if 0 <= int(eidx) < num_edges:
                    od_edge_share[od_idx, int(eidx)] += share

    # Per-cycle util: (num_steps, num_od) @ (num_od, num_edges) / capacities
    link_loads = dataset.tm @ od_edge_share              # (num_steps, num_edges)
    link_util = link_loads / np.maximum(capacities, 1e-9)
    return link_util.astype(np.float64)


def dump_topology(topo: str, out_root: Path) -> dict:
    print(f"[dump] {topo} ...", flush=True)
    dataset, path_library = load_dataset(topo)

    out_dir = out_root / topo
    out_dir.mkdir(parents=True, exist_ok=True)

    tm = np.asarray(dataset.tm, dtype=np.float64)
    np.savez_compressed(out_dir / "tm_series.npz", tm=tm)

    try:
        link_util = compute_ecmp_link_util(dataset, path_library)
        np.savez_compressed(out_dir / "link_util_series.npz", util=link_util)
        link_status = "ok"
    except Exception as exc:
        print(f"  warning: link util failed for {topo}: {exc}", flush=True)
        link_util = None
        link_status = f"failed: {exc}"

    # Split indices (chronological).
    split = dict(dataset.split)
    (out_dir / "split_indices.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )

    # Topology metadata.
    meta = {
        "topology": topo,
        "num_nodes": len(dataset.nodes),
        "num_edges": len(dataset.edges),
        "num_od_pairs": len(dataset.od_pairs),
        "num_timesteps": int(tm.shape[0]),
        "tm_min": float(tm.min()),
        "tm_max": float(tm.max()),
        "tm_mean": float(tm.mean()),
        "link_util_status": link_status,
    }
    (out_dir / "topology_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"  done. "
        f"timesteps={meta['num_timesteps']} "
        f"od={meta['num_od_pairs']} "
        f"edges={meta['num_edges']} "
        f"link_util={link_status}",
        flush=True,
    )
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="all", help="topology key or 'all'")
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "data" / "forecasting"),
        help="output root directory",
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    topos = ALL_TOPOLOGIES if args.topology == "all" else [args.topology]
    summaries = []
    for topo in topos:
        try:
            meta = dump_topology(topo, out_root)
            summaries.append(meta)
        except Exception as exc:
            print(f"[dump] {topo} FAILED: {exc}", flush=True)
            summaries.append({"topology": topo, "error": str(exc)})

    (out_root / "dump_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[dump] wrote summary to {out_root}/dump_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
