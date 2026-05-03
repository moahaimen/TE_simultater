"""Phase 2 Final plot generator. Reads
results/phase2_final/phase2_final_routing_results.csv and produces
8 publication-style figures comparing Phase 1 GNN+ Sticky vs Phase 2
Predictive GNN+ Sticky vs ablations.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_DIR = PROJECT_ROOT / "results" / "phase2_final"
FIG_ROOT = PHASE_DIR / "figures"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = PHASE_DIR / "phase2_final_routing_results.csv"

SCENARIO_ORDER = ["normal", "spike_2x", "ramp_up", "flash_crowd"]
DELAY_ORDER = [0, 1, 2]
METHOD_ORDER = ["phase1", "current_apg", "predictive", "oracle"]
METHOD_LABELS = {
    "phase1": "Phase 1 GNN+ Sticky",
    "current_apg": "GNN+ Sticky + APG (current)",
    "predictive": "Predictive GNN+ Sticky",
    "oracle": "Oracle GNN+ Sticky",
}
METHOD_COLORS = {
    "phase1": "#888888",
    "current_apg": "#F39C12",
    "predictive": "#2E86AB",
    "oracle": "#2E7D32",
}


def load() -> list[dict]:
    rows = []
    with open(RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            for k in ["mean_mlu", "p95_mlu", "peak_mlu", "mean_disturb",
                      "mean_decision_ms", "delta_mean_pct", "delta_p95_pct",
                      "delta_peak_pct", "delta_disturb_pct",
                      "abl_delta_mean_vs_currapg_pct",
                      "abl_delta_p95_vs_currapg_pct",
                      "abl_delta_peak_vs_currapg_pct"]:
                if k in r:
                    try: r[k] = float(r[k])
                    except (ValueError, TypeError): r[k] = float("nan")
            r["k"] = int(r["k"]); r["delay"] = int(r["delay"])
            rows.append(r)
    return rows


def by_method(rows: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        out[r["method"]].append(r)
    return out


def plot_phase1_vs_phase2_metric(rows: list[dict], metric_field: str,
                                  ylabel: str, title: str, fname: str):
    """Group by (scenario, delay), plot mean of each method as bars."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[(r["scenario"], r["delay"])][r["method"]].append(r[metric_field])

    fig, axes = plt.subplots(1, len(DELAY_ORDER), figsize=(14, 4.5), sharey=True)
    if len(DELAY_ORDER) == 1:
        axes = [axes]
    width = 0.2
    xs = np.arange(len(SCENARIO_ORDER))
    for ax_i, d in enumerate(DELAY_ORDER):
        ax = axes[ax_i]
        for j, m in enumerate(METHOD_ORDER):
            vals = [np.mean(grouped[(s, d)].get(m, [np.nan])) for s in SCENARIO_ORDER]
            offset = (j - (len(METHOD_ORDER) - 1) / 2) * width
            ax.bar(xs + offset, vals, width, label=METHOD_LABELS[m], color=METHOD_COLORS[m])
        ax.set_xticks(xs)
        ax.set_xticklabels(SCENARIO_ORDER, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"delay = {d}")
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=12)
    axes[-1].legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    out = FIG_ROOT / fname
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  wrote {out}")


def plot_predictive_delta_vs_phase1(rows: list[dict], metric_field: str,
                                     ylabel: str, title: str, fname: str):
    """Bars: delta of predictive vs phase1, faceted by scenario × delay."""
    grouped = defaultdict(list)
    for r in rows:
        if r["method"] == "predictive":
            grouped[(r["scenario"], r["delay"])].append(r[metric_field])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.2
    xs = np.arange(len(DELAY_ORDER))
    for i, scen in enumerate(SCENARIO_ORDER):
        means = [np.mean(grouped.get((scen, d), [0])) for d in DELAY_ORDER]
        offset = (i - (len(SCENARIO_ORDER) - 1) / 2) * width
        bars = ax.bar(xs + offset, means, width, label=scen)
        for b, v in zip(bars, means):
            if abs(v) > 0.5:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.1f}",
                        ha="center", va="bottom" if v > 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"d={d}" for d in DELAY_ORDER])
    ax.set_xlabel("Actuation delay (cycles)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="scenario", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = FIG_ROOT / fname
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  wrote {out}")


def plot_ablation_pred_vs_currapg(rows: list[dict], fname: str):
    """The KEY plot: predictive delta vs current_apg."""
    grouped = defaultdict(list)
    for r in rows:
        if r["method"] == "predictive":
            grouped[(r["scenario"], r["delay"])].append(r["abl_delta_mean_vs_currapg_pct"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.2
    xs = np.arange(len(DELAY_ORDER))
    for i, scen in enumerate(SCENARIO_ORDER):
        means = [np.mean(grouped.get((scen, d), [0])) for d in DELAY_ORDER]
        offset = (i - (len(SCENARIO_ORDER) - 1) / 2) * width
        bars = ax.bar(xs + offset, means, width, label=scen)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.2f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"d={d}" for d in DELAY_ORDER])
    ax.set_xlabel("Actuation delay (cycles)")
    ax.set_ylabel("Predictive Δ mean MLU vs Current-State CFS+APG (%)")
    ax.set_title("ABLATION: Does AI prediction add value over current-state selection?")
    ax.legend(title="scenario", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = FIG_ROOT / fname
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  wrote {out}")


def plot_decision_time(rows: list[dict]):
    by_m = by_method(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [[r["mean_decision_ms"] for r in by_m[m] if not np.isnan(r["mean_decision_ms"])]
            for m in METHOD_ORDER]
    bp = ax.boxplot(data, tick_labels=[METHOD_LABELS[m] for m in METHOD_ORDER],
                    patch_artist=True)
    for patch, m in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[m])
    ax.set_ylabel("Mean decision time (ms)")
    ax.set_title("Decision-time distribution per method (Phase 2 Final)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=10, ha="right", fontsize=8)
    plt.tight_layout()
    out = FIG_ROOT / "decision_time_by_method.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  wrote {out}")


def plot_disturbance(rows: list[dict]):
    plot_phase1_vs_phase2_metric(
        rows, "mean_disturb", "Mean disturbance (DB)",
        "Disturbance per method, by scenario × delay",
        "phase1_vs_phase2_disturbance.png",
    )


def main() -> int:
    rows = load()
    print(f"Loaded {len(rows)} result rows.")

    plot_phase1_vs_phase2_metric(
        rows, "mean_mlu", "Mean MLU",
        "Mean MLU — Phase 1 vs Phase 2 GNN+ Sticky variants",
        "phase1_vs_phase2_mean_mlu.png",
    )
    plot_phase1_vs_phase2_metric(
        rows, "p95_mlu", "p95 MLU",
        "p95 MLU — Phase 1 vs Phase 2 variants",
        "phase1_vs_phase2_p95_mlu.png",
    )
    plot_phase1_vs_phase2_metric(
        rows, "peak_mlu", "Peak MLU",
        "Peak MLU — Phase 1 vs Phase 2 variants",
        "phase1_vs_phase2_peak_mlu.png",
    )
    plot_disturbance(rows)
    plot_predictive_delta_vs_phase1(
        rows, "delta_mean_pct",
        "Predictive Δ mean MLU vs Phase 1 GNN+ Sticky (%)",
        "Predictive GNN+ Sticky improvement over Phase 1, by scenario × delay",
        "predictive_delta_vs_phase1_mean.png",
    )
    plot_predictive_delta_vs_phase1(
        rows, "delta_p95_pct",
        "Predictive Δ p95 MLU vs Phase 1 (%)",
        "Predictive p95 MLU improvement over Phase 1",
        "predictive_delta_vs_phase1_p95.png",
    )
    plot_ablation_pred_vs_currapg(rows, "ablation_predictive_vs_currapg.png")
    plot_decision_time(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
