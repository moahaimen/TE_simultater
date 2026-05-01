"""Phase 2.6 plot generator.

Reads results/phase2_6/phase2_6_predictive_cfs_results.csv and produces
plots in results/phase2_6/figures/.

Plots:
  - mean_mlu_by_delay.png        (mean MLU delta vs delay, faceted by scenario)
  - p95_mlu_by_delay.png         (p95 MLU delta vs delay)
  - peak_mlu_by_delay.png        (peak MLU delta vs delay)
  - overload_07_by_delay.png     (overload@0.7 delta)
  - disturbance_by_delay.png     (disturbance delta)
  - decision_time_by_method.png  (mean decision time per method)
  - oracle_vs_predictive_gap.png (how close GRU is to oracle)
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
PHASE2_6 = PROJECT_ROOT / "results" / "phase2_6"
FIG_ROOT = PHASE2_6 / "figures"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = PHASE2_6 / "phase2_6_predictive_cfs_results.csv"

SCENARIO_ORDER = ["normal", "spike_2x", "ramp_up", "flash_crowd"]
DELAY_ORDER = [0, 1, 2]
K_ORDER = [5, 10, 20, 40]


def load() -> list[dict]:
    rows = []
    with open(RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            r["k"] = int(r["k"]); r["delay"] = int(r["delay"])
            r["mean_mlu"] = float(r["mean_mlu"])
            r["p95_mlu"] = float(r["p95_mlu"])
            r["peak_mlu"] = float(r["peak_mlu"])
            r["overload_0p7"] = int(r["overload_0p7"])
            r["overload_0p9"] = int(r["overload_0p9"])
            r["delta_mean_pct"] = float(r["delta_mean_pct"])
            r["delta_p95_pct"] = float(r["delta_p95_pct"])
            r["delta_peak_pct"] = float(r["delta_peak_pct"])
            r["delta_disturb_pct"] = float(r["delta_disturb_pct"])
            r["mean_decision_ms"] = float(r["mean_decision_ms"])
            r["delta_overload_0p7"] = int(r["delta_overload_0p7"])
            rows.append(r)
    return rows


def aggregate_predictive_delta(
    rows: list[dict], metric_field: str,
) -> dict[tuple[str, int], list[float]]:
    """Group predictive deltas by (scenario, delay), averaging across topo + K."""
    out = defaultdict(list)
    for r in rows:
        if r["method"] != "predictive":
            continue
        out[(r["scenario"], r["delay"])].append(r[metric_field])
    return out


def plot_delta_by_delay(rows: list[dict], metric_field: str, ylabel: str,
                        title: str, fname: str, ymin=None, ymax=None):
    grouped = aggregate_predictive_delta(rows, metric_field)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.18
    xs = np.arange(len(DELAY_ORDER))

    for i, scen in enumerate(SCENARIO_ORDER):
        means = []
        for d in DELAY_ORDER:
            vals = grouped.get((scen, d), [])
            means.append(np.mean(vals) if vals else 0.0)
        offset = (i - (len(SCENARIO_ORDER) - 1) / 2) * width
        bars = ax.bar(xs + offset, means, width, label=scen)
        for b, v in zip(bars, means):
            if abs(v) > 0.5:
                ax.text(b.get_x() + b.get_width() / 2, v,
                        f"{v:+.1f}", ha="center",
                        va="bottom" if v > 0 else "top", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"d={d}" for d in DELAY_ORDER])
    ax.set_xlabel("Actuation delay (cycles)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)
    ax.legend(title="scenario", loc="best", fontsize=9)
    plt.tight_layout()
    out = FIG_ROOT / fname
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  wrote {out}")


def plot_decision_time(rows: list[dict]):
    methods = ["reactive", "predictive", "oracle"]
    means = {m: [] for m in methods}
    for m in methods:
        for r in rows:
            if r["method"] == m:
                means[m].append(r["mean_decision_ms"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot([means[m] for m in methods], labels=methods, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#999999", "#2E86AB", "#2E7D32"]):
        patch.set_facecolor(color)
    ax.set_ylabel("Mean decision time (ms) — across all cells")
    ax.set_title("Decision time per method (Phase 2.6 sweep)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = FIG_ROOT / "decision_time_by_method.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  wrote {out}")


def plot_oracle_gap(rows: list[dict]):
    """Predictive delta_mean_pct vs Oracle delta_mean_pct, scatter."""
    by_cell: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        key = (r["topology"], r["scenario"], r["k"], r["delay"])
        by_cell[key][r["method"]] = r
    pred_d = []; orac_d = []; colors = []
    color_map = {"normal": "#888", "spike_2x": "#D32F2F",
                 "ramp_up": "#1976D2", "flash_crowd": "#F57C00"}
    for k, c in by_cell.items():
        if "predictive" in c and "oracle" in c:
            pred_d.append(c["predictive"]["delta_mean_pct"])
            orac_d.append(c["oracle"]["delta_mean_pct"])
            colors.append(color_map.get(k[1], "#888"))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for color, lab in color_map.items():
        idx = [i for i, c in enumerate(colors) if c == color]
        if idx:
            ax.scatter([orac_d[i] for i in idx], [pred_d[i] for i in idx],
                       c=color, label=color, alpha=0.6, s=32)
    lo = min(min(pred_d, default=0), min(orac_d, default=0))
    hi = max(max(pred_d, default=0), max(orac_d, default=0))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1, label="y = x (perfect)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Oracle Δ mean MLU vs Reactive (%)")
    ax.set_ylabel("Predictive Δ mean MLU vs Reactive (%)")
    ax.set_title("How close is the GRU to oracle? (per cell)")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_lab = dict(zip(labels, handles))
    ax.legend(by_lab.values(), by_lab.keys(), title="scenario")
    plt.tight_layout()
    out = FIG_ROOT / "oracle_vs_predictive_gap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  wrote {out}")


def main() -> int:
    rows = load()
    print(f"Loaded {len(rows)} result rows.")

    plot_delta_by_delay(
        rows, "delta_mean_pct",
        "Predictive Δ mean MLU vs Reactive (%)",
        "Predictive vs Reactive — mean MLU (negative = improvement)",
        "mean_mlu_by_delay.png",
    )
    plot_delta_by_delay(
        rows, "delta_p95_pct",
        "Predictive Δ p95 MLU vs Reactive (%)",
        "Predictive vs Reactive — p95 MLU (negative = improvement)",
        "p95_mlu_by_delay.png",
    )
    plot_delta_by_delay(
        rows, "delta_peak_pct",
        "Predictive Δ peak MLU vs Reactive (%)",
        "Predictive vs Reactive — peak MLU (negative = improvement)",
        "peak_mlu_by_delay.png",
    )
    plot_delta_by_delay(
        rows, "delta_disturb_pct",
        "Predictive Δ disturbance vs Reactive (%)",
        "Predictive vs Reactive — disturbance (negative = improvement)",
        "disturbance_by_delay.png",
    )
    plot_decision_time(rows)
    plot_oracle_gap(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
