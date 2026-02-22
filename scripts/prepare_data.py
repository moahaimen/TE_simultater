#!/usr/bin/env python3
"""Prepare SNDlib dynamic traffic matrices into compact .npz files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from te.parser_sndlib import (
    DATASET_SPECS,
    build_tm_matrix,
    discover_tm_files,
    parse_native_topology,
    safe_extract_tgz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SNDlib datasets for TE simulator")
    parser.add_argument("--data_dir", default="data", help="Base data directory")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "abilene", "geant"],
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Use only first N timesteps (chronological)",
    )
    parser.add_argument(
        "--force_extract",
        action="store_true",
        help="Re-extract dynamic archive even if extracted dir already exists",
    )
    return parser.parse_args()


def _resolve_topology_path(data_dir: Path, dataset_key: str, topology_name: str, extracted_dir: Path) -> Path:
    explicit = data_dir / "raw" / "topology" / topology_name
    if explicit.exists():
        return explicit

    in_extracted = list(extracted_dir.rglob(topology_name))
    if in_extracted:
        return in_extracted[0]

    raise FileNotFoundError(
        f"Missing topology file '{topology_name}' for dataset '{dataset_key}'. "
        "Run scripts/download_sndlib.sh first."
    )


def _build_od_pairs(nodes: List[str]) -> List[Tuple[str, str]]:
    return [(src, dst) for src in nodes for dst in nodes if src != dst]


def prepare_one_dataset(data_dir: Path, dataset_key: str, max_steps: int | None, force_extract: bool) -> Path:
    spec = DATASET_SPECS[dataset_key]

    archive_path = data_dir / "raw" / "archives" / spec["dynamic_archive"]
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Dynamic archive not found for dataset '{dataset_key}': {archive_path}. "
            "Run scripts/download_sndlib.sh first."
        )

    extracted_dir = data_dir / "raw" / "extracted" / dataset_key
    safe_extract_tgz(archive_path, extracted_dir, force=force_extract)

    topology_path = _resolve_topology_path(data_dir, dataset_key, spec["topology_file"], extracted_dir)
    topology = parse_native_topology(topology_path)

    tm_files = discover_tm_files(extracted_dir)
    if not tm_files:
        raise RuntimeError(
            f"No TM snapshot files discovered under {extracted_dir}. "
            "Check archive contents and parser assumptions."
        )

    od_pairs = _build_od_pairs(topology.nodes)
    tm, selected_tm_files = build_tm_matrix(tm_files, od_pairs=od_pairs, max_steps=max_steps)

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / f"{dataset_key}.npz"

    edge_src = np.array([link.src for link in topology.links])
    edge_dst = np.array([link.dst for link in topology.links])
    capacities = np.array([link.capacity for link in topology.links], dtype=np.float32)
    weights = np.array([link.weight for link in topology.links], dtype=np.float32)

    od_src = np.array([src for src, _ in od_pairs])
    od_dst = np.array([dst for _, dst in od_pairs])

    metadata: Dict[str, object] = {
        "dataset_key": dataset_key,
        "dynamic_archive": spec["dynamic_archive"],
        "topology_file": str(topology_path),
        "normalization_rule": topology.normalization_rule,
        "num_tm_files_discovered": len(tm_files),
        "num_tm_files_used": len(selected_tm_files),
        "max_steps": max_steps,
    }

    np.savez_compressed(
        output_path,
        nodes=np.array(topology.nodes),
        edge_src=edge_src,
        edge_dst=edge_dst,
        capacities=capacities,
        weights=weights,
        od_src=od_src,
        od_dst=od_dst,
        tm=tm,
        tm_files=np.array([str(x) for x in selected_tm_files]),
        metadata_json=np.array(json.dumps(metadata)),
    )

    print(
        "Prepared dataset:",
        dataset_key,
        f"steps={tm.shape[0]}",
        f"od_pairs={tm.shape[1]}",
        f"edges={len(topology.links)}",
        f"-> {output_path}",
    )

    return output_path


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    targets = [args.dataset] if args.dataset != "all" else list(DATASET_SPECS.keys())
    outputs = []
    for dataset_key in targets:
        outputs.append(
            prepare_one_dataset(
                data_dir=data_dir,
                dataset_key=dataset_key,
                max_steps=args.max_steps,
                force_extract=args.force_extract,
            )
        )

    print("Prepared files:")
    for output in outputs:
        print(f"  - {output}")


if __name__ == "__main__":
    main()
