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
import json
import os
import sys
from pathlib import Path

from phase1_reactive.eval.common import collect_specs, load_bundle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase1_reactive_full.yaml"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_gnnplus_packet_sdn_full.py"
OUTPUT_DIR = PROJECT_ROOT / "results" / "professor_clean_gnnplus_zeroshot"
UNSEEN_KEYS = {"germany50_real", "vtlwavenet2011"}


def load_runner():
    spec = importlib.util.spec_from_file_location("run_gnnplus_packet_sdn_full", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_split_manifest() -> dict:
    bundle = load_bundle(CONFIG_PATH)
    train_specs = collect_specs(bundle, "train_topologies")
    eval_specs = collect_specs(bundle, "eval_topologies")
    generalization_specs = collect_specs(bundle, "generalization_topologies")

    train_keys = [spec.key for spec in train_specs]
    eval_keys = [spec.key for spec in eval_specs]
    generalization_keys = [spec.key for spec in generalization_specs]

    overlap = sorted(UNSEEN_KEYS.intersection(train_keys))
    if overlap:
        raise AssertionError(f"Zero-shot violation: unseen topologies found in training set: {overlap}")
    missing_unseen = sorted(UNSEEN_KEYS.difference(generalization_keys))
    if missing_unseen:
        raise AssertionError(f"Zero-shot violation: expected unseen topologies missing from generalization set: {missing_unseen}")

    exp = bundle.raw.get("experiment", {}) if isinstance(bundle.raw.get("experiment"), dict) else {}
    split_cfg = exp.get("split", {}) if isinstance(exp.get("split"), dict) else {}

    gnn_ckpt = PROJECT_ROOT / "results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt"
    gnnplus_ckpt = PROJECT_ROOT / "results/gnn_plus_retrained_fixedk40/gnn_plus_fixed_k40.pt"

    manifest = {
        "config_path": str(CONFIG_PATH),
        "train_topologies": train_keys,
        "validation_topologies": train_keys,
        "evaluation_topologies": eval_keys,
        "generalization_topologies": generalization_keys,
        "split_policy": {
            "train": float(split_cfg.get("train", 0.70)),
            "val": float(split_cfg.get("val", 0.15)),
            "test": float(split_cfg.get("test", 0.15)),
            "validation_note": "Validation is chronological and contained inside the training topologies only.",
        },
        "checkpoint_manifest": {
            "original_gnn": str(gnn_ckpt),
            "gnnplus": str(gnnplus_ckpt),
        },
        "zero_shot_guards": {
            "blocked_from_training": sorted(UNSEEN_KEYS),
            "assertion_passed": True,
            "bayesian_calibration_used": False,
            "per_topology_adaptation_used": False,
        },
    }
    return manifest


def write_split_manifest(manifest: dict) -> None:
    (OUTPUT_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_split_manifest()
    write_split_manifest(manifest)

    runner = load_runner()
    runner.OUT_DIR = OUTPUT_DIR

    runner.logger.info("=" * 72)
    runner.logger.info("PROFESSOR CLEAN GNN/GNN+ ZERO-SHOT RERUN")
    runner.logger.info("Scope: {ECMP, Bottleneck, Original GNN, GNN+}")
    runner.logger.info("No MetaGate, no Stable MetaGate, no calibration, no adaptation")
    runner.logger.info("Train topologies: %s", ", ".join(manifest["train_topologies"]))
    runner.logger.info("Validation topologies: %s", ", ".join(manifest["validation_topologies"]))
    runner.logger.info("Evaluation topologies: %s", ", ".join(manifest["evaluation_topologies"]))
    runner.logger.info("Generalization topologies: %s", ", ".join(manifest["generalization_topologies"]))
    runner.logger.info("Split manifest: %s", OUTPUT_DIR / "split_manifest.json")
    runner.logger.info("Results directory: %s", OUTPUT_DIR)
    runner.logger.info("=" * 72)

    return int(runner.main())


if __name__ == "__main__":
    raise SystemExit(main())
