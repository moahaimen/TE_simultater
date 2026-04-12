#!/usr/bin/env python3
"""Run only the old-branch failure benchmark for seminar comparison."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd


OLD_WORKTREE = Path("/tmp/only_gnn_old_worktree")
OLD_RUNNER = OLD_WORKTREE / "scripts" / "run_gnnplus_packet_sdn_full.py"
OLD_OUT_DIR = OLD_WORKTREE / "results" / "professor_clean_gnnplus_zeroshot"


def load_old_runner():
    if str(OLD_WORKTREE) not in sys.path:
        sys.path.insert(0, str(OLD_WORKTREE))
    spec = importlib.util.spec_from_file_location("old_run_gnnplus_packet_sdn_full", OLD_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load old runner from {OLD_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    os.chdir(OLD_WORKTREE)
    OLD_OUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = load_old_runner()

    gnn_cache = {}
    gnnplus_cache = {}
    full_topologies = list(runner.TOPOLOGIES.keys())
    full_methods = ["ecmp", "bottleneck", "gnn", "gnnplus"]

    failure_results = []
    for topo in full_topologies:
        rows = runner.benchmark_topology_failures(topo, full_methods.copy(), gnn_cache, gnnplus_cache)
        failure_results.extend(rows)

    df_failure = pd.DataFrame(failure_results)
    out_path = OLD_OUT_DIR / "packet_sdn_failure.csv"
    df_failure.to_csv(out_path, index=False)
    print(out_path)
    print(f"rows={len(df_failure)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
