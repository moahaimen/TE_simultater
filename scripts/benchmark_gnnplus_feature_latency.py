#!/usr/bin/env python3
"""Benchmark GNN+ feature extraction before vs after Phase A caching/vectorization."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "results" / "professor_clean_gnnplus_zeroshot"
OLD_BRANCH = "only-GNN+"
OLD_SELECTOR_PATH = "phase1_reactive/drl/gnn_plus_selector.py"


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_old_selector_module():
    blob = subprocess.check_output(
        ["git", "show", f"{OLD_BRANCH}:{OLD_SELECTOR_PATH}"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    tmp = tempfile.NamedTemporaryFile("w", suffix="_gnn_plus_selector_old.py", delete=False)
    with tmp:
        tmp.write(blob)
        tmp_path = Path(tmp.name)
    return _load_module_from_path("gnn_plus_selector_old", tmp_path)


def _prepare_case(topo_key: str):
    from scripts.run_gnnplus_packet_sdn_full import load_dataset
    from te.baselines import ecmp_splits
    from te.simulator import apply_routing
    from phase3.state_builder import compute_telemetry

    dataset, path_library = load_dataset(topo_key)
    tm = np.asarray(dataset.tm[int(dataset.split["test_start"])], dtype=float)
    capacities = np.asarray(dataset.capacities, dtype=float)
    weights = np.asarray(dataset.weights, dtype=float)
    ecmp = ecmp_splits(path_library)
    routing = apply_routing(tm, ecmp, path_library, capacities)
    telemetry = compute_telemetry(tm, ecmp, path_library, routing, weights)
    return dataset, path_library, tm, telemetry


def _time_once(fn, *args, **kwargs) -> float:
    start = time.perf_counter()
    fn(*args, **kwargs)
    return (time.perf_counter() - start) * 1000.0


def _time_repeated(fn, repeats: int, *args, **kwargs) -> float:
    times = []
    for _ in range(repeats):
        times.append(_time_once(fn, *args, **kwargs))
    return float(np.mean(times))


def benchmark_case(topo_key: str, repeats: int = 20) -> list[dict]:
    import phase1_reactive.drl.gnn_plus_selector as current_selector

    old_selector = _load_old_selector_module()
    dataset, path_library, tm, telemetry = _prepare_case(topo_key)

    rows: list[dict] = []

    current_selector._PLUS_TOPOLOGY_CACHE.clear()
    cold_graph_ms = _time_once(
        current_selector.build_graph_tensors_plus,
        dataset,
        tm_vector=tm,
        path_library=path_library,
        telemetry=telemetry,
        device="cpu",
    )
    cold_od_ms = _time_once(
        current_selector.build_od_features_plus,
        dataset,
        tm,
        path_library,
        telemetry=telemetry,
        device="cpu",
    )
    warm_graph_ms = _time_repeated(
        current_selector.build_graph_tensors_plus,
        repeats,
        dataset,
        tm_vector=tm,
        path_library=path_library,
        telemetry=telemetry,
        device="cpu",
    )
    warm_od_ms = _time_repeated(
        current_selector.build_od_features_plus,
        repeats,
        dataset,
        tm,
        path_library,
        telemetry=telemetry,
        device="cpu",
    )
    old_graph_ms = _time_repeated(
        old_selector.build_graph_tensors_plus,
        repeats,
        dataset,
        tm_vector=tm,
        path_library=path_library,
        telemetry=telemetry,
        device="cpu",
    )
    old_od_ms = _time_repeated(
        old_selector.build_od_features_plus,
        repeats,
        dataset,
        tm,
        path_library,
        telemetry=telemetry,
        device="cpu",
    )

    rows.extend(
        [
            {
                "topology": topo_key,
                "stage": "graph",
                "old_mean_ms": old_graph_ms,
                "new_cold_ms": cold_graph_ms,
                "new_warm_mean_ms": warm_graph_ms,
                "warm_speedup_x": old_graph_ms / max(warm_graph_ms, 1e-9),
            },
            {
                "topology": topo_key,
                "stage": "od",
                "old_mean_ms": old_od_ms,
                "new_cold_ms": cold_od_ms,
                "new_warm_mean_ms": warm_od_ms,
                "warm_speedup_x": old_od_ms / max(warm_od_ms, 1e-9),
            },
        ]
    )
    return rows


def write_summary(df: pd.DataFrame) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_DIR / "gnnplus_feature_latency_benchmark.csv"
    md_path = RESULT_DIR / "gnnplus_feature_latency_benchmark.md"
    df.to_csv(csv_path, index=False)

    lines = [
        "# GNN+ Feature Latency Benchmark",
        "",
        f"- Compared current `phaseA-gnnplus-correctness` selector against `{OLD_BRANCH}`.",
        "- `old_mean_ms`: previous implementation average over warm repeats.",
        "- `new_cold_ms`: first call after clearing the topology cache.",
        "- `new_warm_mean_ms`: average over warm repeats after cache build.",
        "",
        df.to_markdown(index=False),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = []
    for topo_key in ["abilene", "germany50"]:
        rows.extend(benchmark_case(topo_key))
    df = pd.DataFrame(rows)
    write_summary(df)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
