#!/usr/bin/env python3
"""Generate a zero-shot unseen-topology summary for MetaGate+GNN+."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from phase1_reactive.drl.dynamic_meta_gate import DynamicMetaGate, MetaGateConfig
from scripts.run_dynamic_metagate_gnnplus_eval import (
    setup,
    CONFIG_PATH,
    MAX_STEPS,
    DEVICE,
    K_CRIT,
    GNNPLUS_CHECKPOINT,
    evaluate_on_topology,
)

OUTPUT_DIR = Path(
    os.environ.get(
        "METAGATE_GNNPLUS_OUTPUT_DIR",
        str(PROJECT_ROOT / "results" / "dynamic_metagate_gnnplus"),
    )
).resolve()
MODEL_PATH = OUTPUT_DIR / "models" / "metagate_gnnplus_unified.pt"
ORACLE_CSV = OUTPUT_DIR / "test_oracle.csv"
ZERO_SHOT_RESULTS_CSV = OUTPUT_DIR / "zero_shot_unseen_results.csv"
ZERO_SHOT_SUMMARY_CSV = OUTPUT_DIR / "zero_shot_unseen_summary.csv"
ZERO_SHOT_AUDIT_JSON = OUTPUT_DIR / "zero_shot_unseen_summary.json"


def main():
    M = setup()
    gnnplus_model, _ = M["load_gnn_plus"](str(GNNPLUS_CHECKPOINT), device=DEVICE)
    gate = DynamicMetaGate(
        MetaGateConfig(
            hidden_dim=128,
            dropout=0.3,
            learning_rate=5e-4,
            num_epochs=300,
            batch_size=64,
        )
    )
    gate.load(MODEL_PATH, feat_dim=49)

    oracle_all = pd.read_csv(ORACLE_CSV)
    bundle = M["load_bundle"](CONFIG_PATH)

    all_results = []
    for spec in M["collect_specs"](bundle, "generalization_topologies"):
        dataset, path_library = M["load_named_dataset"](
            bundle,
            spec,
            M["max_steps_from_args"](bundle, MAX_STEPS),
        )
        oracle_df = oracle_all[oracle_all["dataset"] == dataset.key].copy()
        gate.clear_calibration()
        results, _ = evaluate_on_topology(M, dataset, path_library, gate, gnnplus_model, K_CRIT, oracle_df)
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    summary_df = (
        results_df.groupby(["dataset", "topology_type"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            metagate_mlu=("metagate_mlu", "mean"),
            oracle_mlu=("oracle_mlu", "mean"),
            n_timesteps=("timestep", "count"),
        )
    )
    summary_df["oracle_gap_pct"] = (
        (summary_df["metagate_mlu"] - summary_df["oracle_mlu"]) / summary_df["oracle_mlu"] * 100.0
    )
    summary_df["mode"] = "zero_shot"

    results_df.to_csv(ZERO_SHOT_RESULTS_CSV, index=False)
    summary_df.to_csv(ZERO_SHOT_SUMMARY_CSV, index=False)
    ZERO_SHOT_AUDIT_JSON.write_text(
        json.dumps(
            {
                "rows": int(len(results_df)),
                "datasets": summary_df["dataset"].astype(str).tolist(),
                "mean_accuracy": float(results_df["correct"].mean()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved zero-shot unseen summary to {ZERO_SHOT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
