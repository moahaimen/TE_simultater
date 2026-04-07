#!/usr/bin/env python3
"""Build a synthetic practice bundle where displayed GNN+ values copy Bottleneck values.

This is NOT a real experiment. It exists only for presentation practice.
The output report is explicitly marked synthetic.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "results" / "professor_gnnplus_baselines_zeroshot"
MOCK_DIR = PROJECT_ROOT / "results" / "professor_gnnplus_baselines_mock_bottleneck_swap"
SUMMARY_CSV = SOURCE_DIR / "packet_sdn_summary.csv"
FAILURE_CSV = SOURCE_DIR / "packet_sdn_failure.csv"
SDN_CSV = SOURCE_DIR / "packet_sdn_sdn_metrics.csv"


def copy_bottleneck_into_gnnplus(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    bottleneck = df[df["method"] == "bottleneck"].copy()
    gnnplus = df[df["method"] == "gnnplus"].copy()
    merged = gnnplus[key_cols].merge(
        bottleneck.drop(columns=["method"]),
        on=key_cols,
        how="left",
        suffixes=("", "_bn"),
    )
    if merged.isnull().any().any():
        raise RuntimeError("Could not find matching bottleneck rows for all synthetic GNN+ replacements")
    merged["method"] = "gnnplus"

    keep_cols = list(df.columns)
    gnnplus_synth = merged[keep_cols]
    other = df[df["method"] != "gnnplus"].copy()
    out = pd.concat([other, gnnplus_synth], ignore_index=True)
    return out.sort_values(key_cols + ["method"]).reset_index(drop=True)


def main() -> int:
    os.chdir(PROJECT_ROOT)
    MOCK_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY_CSV)
    failure = pd.read_csv(FAILURE_CSV)
    sdn = pd.read_csv(SDN_CSV)

    summary_mock = copy_bottleneck_into_gnnplus(summary, ["topology"])
    failure_mock = copy_bottleneck_into_gnnplus(failure, ["topology", "scenario"])
    sdn_mock = copy_bottleneck_into_gnnplus(sdn, ["topology"])

    summary_mock.to_csv(MOCK_DIR / "packet_sdn_summary.csv", index=False)
    failure_mock.to_csv(MOCK_DIR / "packet_sdn_failure.csv", index=False)
    sdn_mock.to_csv(MOCK_DIR / "packet_sdn_sdn_metrics.csv", index=False)

    env = os.environ.copy()
    env["PROF_GNNPLUS_BASELINE_INPUT_DIR"] = str(MOCK_DIR)
    env["PROF_GNNPLUS_BASELINE_REPORT_NAME"] = "GNNPLUS_SARAH_STYLE_BASELINE_REPORT_SYNTHETIC_BOTTLENECK_SWAP.docx"
    env["PROF_GNNPLUS_BASELINE_AUDIT_NAME"] = "report_audit_gnnplus_sarah_baselines_synthetic_swap.md"
    env["PROF_GNNPLUS_BASELINE_TITLE"] = "GNN+ Baseline Comparison Report (Synthetic Practice Version)"
    env["PROF_GNNPLUS_BASELINE_SUBTITLE"] = (
        "Sarah-Style Practice Version\n"
        "Displayed GNN+ rows are synthetically replaced with Bottleneck values\n"
        "For presentation practice only"
    )
    env["PROF_GNNPLUS_BASELINE_SYNTHETIC_WARNING"] = (
        "SYNTHETIC PRACTICE ONLY: in this report, the displayed GNN+ values are copied from Bottleneck. "
        "This is not a real experiment and must not be used as scientific evidence."
    )

    subprocess.run(
        ["python", "scripts/build_professor_gnnplus_baseline_report.py"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
