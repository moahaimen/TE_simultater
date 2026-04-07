#!/usr/bin/env python3
"""Run the professor-requested clean 4-method GNN/GNN+ zero-shot evaluation.

This wrapper intentionally keeps the original packet-SDN runner untouched.
It reuses the same direct 4-method path:
  - ECMP
  - Bottleneck
  - Original GNN
  - GNN+

It does NOT use:
  - MetaGate
  - Stable MetaGate
  - Bayesian calibration
  - per-topology adaptation

Outputs are written to a separate bundle:
  results/professor_clean_gnnplus_zeroshot/
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_gnnplus_packet_sdn_full.py"
OUTPUT_DIR = PROJECT_ROOT / "results" / "professor_clean_gnnplus_zeroshot"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_gnnplus_packet_sdn_full", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runner = load_runner()
    runner.OUT_DIR = OUTPUT_DIR

    runner.logger.info("=" * 72)
    runner.logger.info("PROFESSOR CLEAN GNN/GNN+ ZERO-SHOT RERUN")
    runner.logger.info("Scope: {ECMP, Bottleneck, Original GNN, GNN+}")
    runner.logger.info("No MetaGate, no Stable MetaGate, no calibration, no adaptation")
    runner.logger.info("Results directory: %s", OUTPUT_DIR)
    runner.logger.info("=" * 72)

    return int(runner.main())


if __name__ == "__main__":
    raise SystemExit(main())
