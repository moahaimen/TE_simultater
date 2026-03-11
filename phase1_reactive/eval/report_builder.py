"""Markdown report builder for reactive Phase-1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phase1_reactive.baselines.literature_baselines import BASELINE_NOTES


METHOD_LABELS = {
    "lp_optimal": "LP-optimal upper bound (sampled, runtime-capped)",
}
HEURISTIC_METHODS = {"topk", "bottleneck", "sensitivity", "erodrl", "flexdate", "cfrrl", "flexentry"}


def _render_cell(col: str, val):
    if col == "method":
        return METHOD_LABELS.get(str(val), str(val))
    if isinstance(val, float):
        return f"{val:.6f}" if np.isfinite(val) else "nan"
    return str(val)


def _table_from_df(df: pd.DataFrame, cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row.get(col)
            vals.append(_render_cell(col, val))
        out.append("| " + " | ".join(vals) + " |")
    return out


def _load_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _improvement_table(base_dir: Path) -> pd.DataFrame:
    comp = _load_optional_csv(base_dir / "eval" / "drl_improvement_comparison.csv")
    if comp.empty:
        return pd.DataFrame()
    agg = comp.groupby("variant", as_index=False).agg(
        avg_mean_mlu=("mean_mlu", "mean"),
        avg_p95_mlu=("p95_mlu", "mean"),
        avg_mean_delay=("mean_delay", "mean"),
        avg_mean_disturbance=("mean_disturbance", "mean"),
    )
    order = [
        "PPO baseline",
        "DQN baseline",
        "PPO + pretraining",
        "DQN + pretraining",
        "PPO + pretraining + curriculum",
        "DQN + pretraining + curriculum",
        "Dual-Gate final",
    ]
    agg["rank"] = agg["variant"].map({name: idx for idx, name in enumerate(order)})
    return agg.sort_values("rank").drop(columns=["rank"]) if not agg.empty else agg


def _dual_gate_wins(summary_df: pd.DataFrame) -> pd.DataFrame:
    methods = ["our_drl_ppo", "our_drl_dqn", "our_drl_dual_gate"]
    if summary_df is None or summary_df.empty or "method" not in summary_df.columns:
        return pd.DataFrame()
    sub = summary_df[summary_df["method"].isin(methods)].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for dataset, grp in sub.groupby("dataset"):
        dual = grp[grp["method"] == "our_drl_dual_gate"]
        ppo = grp[grp["method"] == "our_drl_ppo"]
        dqn = grp[grp["method"] == "our_drl_dqn"]
        if dual.empty:
            continue
        dual_mlu = float(dual["mean_mlu"].iloc[0])
        rows.append(
            {
                "dataset": dataset,
                "dual_gate_mean_mlu": dual_mlu,
                "ppo_mean_mlu": float(ppo["mean_mlu"].iloc[0]) if not ppo.empty else np.nan,
                "dqn_mean_mlu": float(dqn["mean_mlu"].iloc[0]) if not dqn.empty else np.nan,
                "dual_beats_ppo": bool(not ppo.empty and dual_mlu < float(ppo["mean_mlu"].iloc[0])),
                "dual_beats_dqn": bool(not dqn.empty and dual_mlu < float(dqn["mean_mlu"].iloc[0])),
            }
        )
    return pd.DataFrame(rows)


def _best_heuristic_gap(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or summary_df.empty or "method" not in summary_df.columns:
        return pd.DataFrame()
    rows = []
    for dataset, grp in summary_df.groupby("dataset"):
        dual = grp[grp["method"] == "our_drl_dual_gate"]
        heur = grp[grp["method"].isin(HEURISTIC_METHODS)]
        if dual.empty or heur.empty:
            continue
        dual_mlu = float(dual["mean_mlu"].iloc[0])
        best_heur = heur.sort_values("mean_mlu").iloc[0]
        gap_pct = (dual_mlu - float(best_heur["mean_mlu"])) / max(float(best_heur["mean_mlu"]), 1e-12) * 100.0
        rows.append(
            {
                "dataset": dataset,
                "dual_gate_mean_mlu": dual_mlu,
                "best_heuristic": str(best_heur["method"]),
                "best_heuristic_mean_mlu": float(best_heur["mean_mlu"]),
                "gap_pct_vs_best_heuristic": float(gap_pct),
            }
        )
    return pd.DataFrame(rows)


def build_phase1_report(
    *,
    summary_df: pd.DataFrame,
    failure_df: pd.DataFrame | None,
    generalization_df: pd.DataFrame | None,
    output_path: Path,
) -> None:
    base_dir = output_path.parent
    lines = ["# Phase 1: Reactive Traffic Engineering", "", "## Main Benchmark", ""]
    main_cols = [
        "dataset",
        "display_name",
        "method",
        "mean_mlu",
        "p95_mlu",
        "mean_delay",
        "throughput",
        "mean_disturbance",
        "mean_gap_pct",
        "mean_achieved_pct",
        "decision_time_ms",
        "training_time_sec",
        "convergence_rate",
    ]
    if summary_df is not None and not summary_df.empty:
        lines.extend(_table_from_df(summary_df.sort_values(["dataset", "mean_mlu"]), [c for c in main_cols if c in summary_df.columns]))
    else:
        lines.append("No in-domain benchmark table was provided to this report builder.")

    lines.extend([
        "",
        "## Improved DRL Selector Framework",
        "",
        "- Teacher-guided pretraining: PPO and DQN are first warm-started from offline teacher labels built from Top-K, bottleneck, sensitivity, and sampled LP-optimal signals.",
        "- Curriculum congestion training: training proceeds through C2, then C3, then mixed C1/C2/C3 stages with progressively richer reward terms.",
        "- Dual-gate inference: PPO and DQN each propose Top-Kcrit OD selections; both are evaluated through the same LP layer; the lower-MLU candidate is chosen, with disturbance and delay tie-breakers.",
    ])

    improvement = _improvement_table(base_dir)
    if not improvement.empty:
        lines.extend(["", "## DRL Ablation Table", ""])
        lines.extend(_table_from_df(improvement, ["variant", "avg_mean_mlu", "avg_p95_mlu", "avg_mean_delay", "avg_mean_disturbance"]))

    dual_wins = _dual_gate_wins(summary_df if summary_df is not None else pd.DataFrame())
    if not dual_wins.empty:
        lines.extend(["", "## Dual-Gate Win Table", ""])
        lines.extend(_table_from_df(dual_wins, ["dataset", "dual_gate_mean_mlu", "ppo_mean_mlu", "dqn_mean_mlu", "dual_beats_ppo", "dual_beats_dqn"]))

    heuristic_gap = _best_heuristic_gap(summary_df if summary_df is not None else pd.DataFrame())
    if not heuristic_gap.empty:
        lines.extend(["", "## Gap To Best Heuristic+LP Baseline", ""])
        lines.extend(_table_from_df(heuristic_gap, ["dataset", "dual_gate_mean_mlu", "best_heuristic", "best_heuristic_mean_mlu", "gap_pct_vs_best_heuristic"]))

    lines.extend(["", "## LP-optimal Interpretation", "", "- Use the wording `sampled LP-optimal upper bound` or `runtime-capped LP upper bound` in the thesis/report.", "- This row comes from a full multicommodity-flow LP on sampled evaluation steps only, with a CBC runtime cap. In the full Phase-1 run it is executed only on tractable SNDlib topologies (Abilene and GEANT).", "", "## Reproduced Literature Baselines", "", "- `ERODRL`, `FlexDATE`, `CFR-RL`, and `FlexEntry` are faithful simplified reproductions, not exact official implementations."])
    notes_df = pd.DataFrame([
        {"method": key, "note": spec.note} for key, spec in sorted(BASELINE_NOTES.items())
    ])
    lines.extend(_table_from_df(notes_df, ["method", "note"]))

    if failure_df is not None and not failure_df.empty:
        lines.extend(["", "## Failure Scenarios", ""])
        fcols = [
            "dataset",
            "failure_type",
            "method",
            "pre_failure_mean_mlu",
            "post_failure_peak_mlu",
            "post_failure_mean_mlu",
            "failover_convergence_time_steps",
            "route_change_frequency",
        ]
        lines.extend(_table_from_df(failure_df.sort_values(["dataset", "failure_type", "post_failure_mean_mlu"]), [c for c in fcols if c in failure_df.columns]))

    if generalization_df is not None and not generalization_df.empty:
        lines.extend(["", "## Dataset Execution Status", "", "- `germany50_real` was executed as the unseen generalization topology.", "- `germany50_topologyzoo_real` was not executed because the external real traffic matrix file is missing.", "- `cernet_real` was not executed because the external real traffic matrix file is missing.", "", "## Generalization on Unseen Topologies", "", "- Final generalization uses the corrected unseen split; `germany50_real` does not appear in `train_topologies` or `eval_topologies` in the corrected full config."])
        gcols = ["train_scope", "dataset", "display_name", "method", "mean_mlu", "mean_delay", "mean_disturbance"]
        lines.extend(_table_from_df(generalization_df.sort_values(["dataset", "mean_mlu"]), [c for c in gcols if c in generalization_df.columns]))

    # Honest conclusion block.
    if summary_df is not None and not summary_df.empty:
        best_drl = summary_df[summary_df["method"].isin(["our_drl_ppo", "our_drl_dqn", "our_drl_dual_gate"])].sort_values("mean_mlu").groupby("dataset", as_index=False).first()
        compare_rows = []
        for dataset, grp in summary_df.groupby("dataset"):
            drl = best_drl[best_drl["dataset"] == dataset]
            if drl.empty:
                continue
            drl_row = drl.iloc[0]
            ospf = grp[grp["method"] == "ospf"]
            ecmp = grp[grp["method"] == "ecmp"]
            heur = grp[grp["method"].isin(HEURISTIC_METHODS)].sort_values("mean_mlu")
            compare_rows.append(
                {
                    "dataset": dataset,
                    "best_drl_method": str(drl_row["method"]),
                    "best_drl_mean_mlu": float(drl_row["mean_mlu"]),
                    "beats_ospf": bool(not ospf.empty and float(drl_row["mean_mlu"]) < float(ospf["mean_mlu"].iloc[0])),
                    "beats_ecmp": bool(not ecmp.empty and float(drl_row["mean_mlu"]) < float(ecmp["mean_mlu"].iloc[0])),
                    "beats_best_heuristic": bool(not heur.empty and float(drl_row["mean_mlu"]) < float(heur["mean_mlu"].iloc[0])),
                }
            )
        compare_df = pd.DataFrame(compare_rows)
        if not compare_df.empty:
            lines.extend(["", "## Honest Outcome Summary", ""])
            lines.extend(_table_from_df(compare_df, ["dataset", "best_drl_method", "best_drl_mean_mlu", "beats_ospf", "beats_ecmp", "beats_best_heuristic"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
