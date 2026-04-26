#!/usr/bin/env python3
"""Build seminar-friendly old-vs-new failure comparison tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEW_DIR = PROJECT_ROOT / "results" / "professor_clean_gnnplus_zeroshot"
OLD_DIR = Path("/tmp/only_gnn_old_worktree/results/professor_clean_gnnplus_zeroshot")
OUT_DIR = NEW_DIR / "seminar_failure_comparison"

SCENARIO_DISPLAY = {
    "single_link_failure": "Single Link Failure",
    "multiple_link_failure": "Multiple Link Failure (2 Links)",
    "three_link_failure": "3-Link Failure",
    "capacity_degradation_50": "Capacity Degradation (50%)",
    "traffic_spike_2x": "Traffic Spike (2x)",
}

TOPOLOGY_DISPLAY = {
    "abilene": "Abilene",
    "cernet": "CERNET",
    "geant": "GEANT",
    "ebone": "Ebone",
    "sprintlink": "Sprintlink",
    "tiscali": "Tiscali",
    "germany50": "Germany50",
    "vtlwavenet2011": "VtlWavenet2011",
}


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    old_df = pd.read_csv(OLD_DIR / "packet_sdn_failure.csv")
    new_df = pd.read_csv(NEW_DIR / "packet_sdn_failure.csv")
    return old_df, new_df


def build_full_table(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["topology", "scenario", "method"]
    merged = old_df.merge(
        new_df,
        on=keys,
        suffixes=("_old", "_new"),
        how="inner",
    )
    merged["mlu_delta"] = merged["mean_mlu_new"] - merged["mean_mlu_old"]
    merged["recovery_delta_ms"] = merged["failure_recovery_ms_new"] - merged["failure_recovery_ms_old"]
    merged["Topology"] = merged["topology"].map(TOPOLOGY_DISPLAY)
    merged["Scenario"] = merged["scenario"].map(SCENARIO_DISPLAY)
    merged["Method"] = merged["method"].str.upper().replace({"ECMP": "ECMP", "BOTTLENECK": "Bottleneck", "GNN": "GNN", "GNNPLUS": "GNN+"})
    cols = [
        "Topology",
        "Scenario",
        "Method",
        "mean_mlu_old",
        "mean_mlu_new",
        "mlu_delta",
        "failure_recovery_ms_old",
        "failure_recovery_ms_new",
        "recovery_delta_ms",
    ]
    return merged[cols].rename(
        columns={
            "mean_mlu_old": "Old MLU",
            "mean_mlu_new": "New MLU",
            "mlu_delta": "Delta MLU",
            "failure_recovery_ms_old": "Old Recovery (ms)",
            "failure_recovery_ms_new": "New Recovery (ms)",
            "recovery_delta_ms": "Delta Recovery (ms)",
        }
    )


def build_key_table(full_df: pd.DataFrame) -> pd.DataFrame:
    picks = [
        ("Abilene", "Random Link Failure (2)"),
        ("Sprintlink", "Random Link Failure (1)"),
        ("Germany50", "Single Link Failure"),
        ("Germany50", "Random Link Failure (2)"),
        ("VtlWavenet2011", "Random Link Failure (2)"),
    ]
    frames = []
    for topo, scenario in picks:
        sub = full_df[(full_df["Topology"] == topo) & (full_df["Scenario"] == scenario)].copy()
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def write_outputs(full_df: pd.DataFrame, key_df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_csv = OUT_DIR / "failure_old_vs_new_full.csv"
    key_csv = OUT_DIR / "failure_old_vs_new_key_cases.csv"
    md = OUT_DIR / "failure_old_vs_new_seminar_table.md"

    full_df.to_csv(full_csv, index=False)
    key_df.to_csv(key_csv, index=False)

    lines = [
        "# Failure Old vs New Comparison",
        "",
        "- Old: pre-fix `only-GNN+` branch failure pipeline.",
        "- New: `phaseA-gnnplus-correctness` with rebuilt post-failure path libraries and path-validity guards.",
        "",
        "## Key Seminar Cases",
        "",
        key_df.to_markdown(index=False),
        "",
        "## Full Failure Comparison",
        "",
        full_df.to_markdown(index=False),
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    old_df, new_df = _load()
    full_df = build_full_table(old_df, new_df)
    key_df = build_key_table(full_df)
    write_outputs(full_df, key_df)
    print(key_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
