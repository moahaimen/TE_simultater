#!/usr/bin/env python3
"""Generate final CDF plots and complexity metrics for the paper."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "results" / "phase1_reactive" / "final_benchmark"
OUT = BENCH / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

PALETTE = {
    "our_unified_meta":        "#d62728",   # red — ours
    "bottleneck":              "#1f77b4",   # blue
    "flexdate":                "#2ca02c",   # green
    "sensitivity":             "#9467bd",   # purple
    "cfrrl":                   "#8c564b",   # brown
    "our_hybrid_moe_gate_v3":  "#ff7f0e",   # orange — MoE v3
    "ecmp":                    "#7f7f7f",   # grey
    "ospf":                    "#bcbd22",   # olive
    "erodrl":                  "#17becf",   # cyan
}

DISPLAY = {
    "our_unified_meta":        "Unified Meta (Ours)",
    "bottleneck":              "Bottleneck",
    "flexdate":                "FlexDATE",
    "sensitivity":             "Sensitivity",
    "cfrrl":                   "CFRRL",
    "our_hybrid_moe_gate_v3":  "MoE v3 (Ours)",
    "ecmp":                    "ECMP",
    "ospf":                    "OSPF",
    "erodrl":                  "ERODRL",
    "topk":                    "Top-k",
    "flexentry":               "FlexEntry",
}

TOPO_DISPLAY = {
    "abilene":               "Abilene (12N)",
    "geant":                 "GEANT (22N)",
    "rocketfuel_ebone":      "Ebone (23N)",
    "rocketfuel_sprintlink": "Sprintlink (44N)",
    "rocketfuel_tiscali":    "Tiscali (49N)",
    "germany50":             "Germany50 (50N, unseen)",
}

# Methods to show in CDF plots (paper-facing)
PAPER_METHODS = [
    "our_unified_meta", "bottleneck", "flexdate", "sensitivity",
    "our_hybrid_moe_gate_v3", "ecmp", "ospf",
]

# For failure CDFs (only methods that were evaluated)
FAIL_METHODS = ["our_unified_meta", "bottleneck", "flexdate", "ecmp"]


# ── Helpers ──────────────────────────────────────────────────────────
def _cdf(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr, arr
    arr = np.sort(arr)
    y = np.arange(1, arr.size + 1) / arr.size
    return arr, y


def _save(fig, stem):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {stem}.png / .pdf")


# ── CDF MLU: Seen topologies (2×3 grid) ─────────────────────────────
def plot_mlu_cdfs_seen(eval_ts):
    print("\n[1] CDF of MLU — Seen Topologies")
    datasets = ["abilene", "geant", "rocketfuel_ebone",
                "rocketfuel_sprintlink", "rocketfuel_tiscali"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    axes_flat = axes.flatten()

    for i, ds in enumerate(datasets):
        ax = axes_flat[i]
        sub = eval_ts[eval_ts["dataset"] == ds]
        for method in PAPER_METHODS:
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            x, y = _cdf(msub["mlu"].values)
            if x.size == 0:
                continue
            lw = 2.5 if method == "our_unified_meta" else 1.5
            ls = "-" if method == "our_unified_meta" else "--"
            ax.plot(x, y, label=DISPLAY.get(method, method),
                    color=PALETTE.get(method), linewidth=lw, linestyle=ls)
        ax.set_title(TOPO_DISPLAY.get(ds, ds))
        ax.set_xlabel("MLU")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)

    axes_flat[5].axis("off")  # empty 6th panel
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.08),
               ncol=2, frameon=True, fontsize=10)
    fig.suptitle("CDF of Maximum Link Utilization — Seen Topologies", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    _save(fig, "cdf_mlu_seen")


# ── CDF MLU: Germany50 unseen ───────────────────────────────────────
def plot_mlu_cdf_germany50(gen_ts):
    print("\n[2] CDF of MLU — Germany50 (Unseen)")
    gen_methods = [m for m in PAPER_METHODS if m != "our_hybrid_moe_gate_v3"]
    sub = gen_ts[gen_ts["dataset"] == "germany50"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in gen_methods:
        msub = sub[sub["method"] == method]
        if msub.empty:
            continue
        x, y = _cdf(msub["mlu"].values)
        if x.size == 0:
            continue
        lw = 2.5 if method == "our_unified_meta" else 1.5
        ls = "-" if method == "our_unified_meta" else "--"
        ax.plot(x, y, label=DISPLAY.get(method, method),
                color=PALETTE.get(method), linewidth=lw, linestyle=ls)
    ax.set_title("Germany50 (50 nodes, unseen topology)")
    ax.set_xlabel("MLU")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    _save(fig, "cdf_mlu_germany50")


# ── CDF Disturbance: Seen topologies ────────────────────────────────
def plot_disturbance_cdfs(eval_ts):
    print("\n[3] CDF of Disturbance — Seen Topologies")
    datasets = ["abilene", "geant", "rocketfuel_ebone",
                "rocketfuel_sprintlink", "rocketfuel_tiscali"]
    # Only methods that do routing changes (not ECMP/OSPF which have 0 disturbance)
    dist_methods = ["our_unified_meta", "bottleneck", "flexdate",
                    "sensitivity", "our_hybrid_moe_gate_v3"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    axes_flat = axes.flatten()

    for i, ds in enumerate(datasets):
        ax = axes_flat[i]
        sub = eval_ts[eval_ts["dataset"] == ds]
        for method in dist_methods:
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            vals = msub["disturbance"].dropna().values
            if len(vals) == 0:
                continue
            x, y = _cdf(vals)
            lw = 2.5 if method == "our_unified_meta" else 1.5
            ls = "-" if method == "our_unified_meta" else "--"
            ax.plot(x, y, label=DISPLAY.get(method, method),
                    color=PALETTE.get(method), linewidth=lw, linestyle=ls)
        ax.set_title(TOPO_DISPLAY.get(ds, ds))
        ax.set_xlabel("Route Disturbance")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)

    axes_flat[5].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.08),
               ncol=2, frameon=True, fontsize=10)
    fig.suptitle("CDF of Route Disturbance — Seen Topologies", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    _save(fig, "cdf_disturbance_seen")


# ── CDF MLU: Failure scenarios ──────────────────────────────────────
def plot_failure_cdfs(fail_ts):
    print("\n[4] CDF of MLU — Failure Scenarios")
    datasets = ["abilene", "geant", "rocketfuel_ebone",
                "rocketfuel_sprintlink", "rocketfuel_tiscali"]
    fail_types = sorted(fail_ts["failure_type"].dropna().unique())
    nft = len(fail_types)
    fig, axes = plt.subplots(nft, len(datasets), figsize=(4 * len(datasets), 4 * nft),
                             squeeze=False)
    for j, ds in enumerate(datasets):
        for i, ft in enumerate(fail_types):
            ax = axes[i][j]
            sub = fail_ts[(fail_ts["dataset"] == ds) & (fail_ts["failure_type"] == ft)]
            for method in FAIL_METHODS:
                msub = sub[sub["method"] == method]
                if msub.empty:
                    continue
                x, y = _cdf(msub["mlu"].values)
                if x.size == 0:
                    continue
                lw = 2.5 if method == "our_unified_meta" else 1.5
                ls = "-" if method == "our_unified_meta" else "--"
                ax.plot(x, y, label=DISPLAY.get(method, method),
                        color=PALETTE.get(method), linewidth=lw, linestyle=ls)
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(TOPO_DISPLAY.get(ds, ds), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{ft.replace('_', ' ').title()}\nCDF", fontsize=10)
            if i == nft - 1:
                ax.set_xlabel("Post-Failure MLU")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(FAIL_METHODS),
                   frameon=True, fontsize=10)
    fig.suptitle("CDF of Post-Failure MLU — All Topologies × Failure Types", fontsize=14, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    _save(fig, "cdf_mlu_failures")


# ── CDF Decision Time: All topologies ───────────────────────────────
def plot_decision_time_cdfs(eval_ts):
    print("\n[5] CDF of Decision Time — Seen Topologies")
    datasets = ["abilene", "geant", "rocketfuel_ebone",
                "rocketfuel_sprintlink", "rocketfuel_tiscali"]
    time_methods = [m for m in PAPER_METHODS if m not in ("ecmp", "ospf")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    axes_flat = axes.flatten()

    for i, ds in enumerate(datasets):
        ax = axes_flat[i]
        sub = eval_ts[eval_ts["dataset"] == ds]
        for method in time_methods:
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            vals = msub["decision_time_ms"].dropna().values
            vals = vals[vals > 0]
            if len(vals) == 0:
                continue
            x, y = _cdf(vals)
            lw = 2.5 if method == "our_unified_meta" else 1.5
            ls = "-" if method == "our_unified_meta" else "--"
            ax.plot(x, y, label=DISPLAY.get(method, method),
                    color=PALETTE.get(method), linewidth=lw, linestyle=ls)
        ax.set_title(TOPO_DISPLAY.get(ds, ds))
        ax.set_xlabel("Decision Time (ms)")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)

    axes_flat[5].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.08),
               ncol=2, frameon=True, fontsize=10)
    fig.suptitle("CDF of Decision Time — Seen Topologies", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    _save(fig, "cdf_decision_time_seen")


# ── Complexity Metrics Table ─────────────────────────────────────────
def build_complexity_table(eval_summary):
    """
    Complexity metric per method. For heuristics: 0 learnable params, O(1) selector.
    For neural methods: count params + inference time.
    We build from eval_summary decision_time_ms averages.
    """
    print("\n[6] Complexity Metrics Table")

    # Method info: (learnable_params, selector_type, LP_required, model_components)
    method_info = {
        "ecmp":                    (0,       "None (static)",   False, "—"),
        "ospf":                    (0,       "None (static)",   False, "—"),
        "bottleneck":              (0,       "O(E) heuristic",  True,  "LP solver"),
        "flexdate":                (0,       "O(E) heuristic",  True,  "LP solver"),
        "sensitivity":             (0,       "O(F·E) heuristic",True,  "LP solver"),
        "cfrrl":                   (0,       "O(E) heuristic",  True,  "LP solver"),
        "erodrl":                  (0,       "O(E) heuristic",  True,  "LP solver"),
        "our_hybrid_moe_gate_v3":  ("~45K",  "MLP gate + 3 experts", True, "GNN + MoE gate + LP"),
        "our_unified_meta":        ("~38K",  "Per-topology lookup + GNN", True, "GNN + lookup table + LP"),
    }

    # Average decision time from eval across topologies
    rows = []
    for method, (params, selector, uses_lp, components) in method_info.items():
        sub = eval_summary[eval_summary["method"] == method]
        if sub.empty:
            continue
        mean_dt = sub["decision_time_ms"].mean()
        max_dt = sub["decision_time_ms"].max()
        mean_mlu_all = sub["mean_mlu"].mean()
        rows.append({
            "Method": DISPLAY.get(method, method),
            "Learnable Params": params if isinstance(params, str) else f"{params:,}",
            "Selector Complexity": selector,
            "LP Required": "✓" if uses_lp else "✗",
            "Components": components,
            "Avg Decision (ms)": f"{mean_dt:.1f}",
            "Max Decision (ms)": f"{max_dt:.1f}",
            "Avg MLU (all topos)": f"{mean_mlu_all:.2f}",
        })
    df = pd.DataFrame(rows)

    # Save as CSV
    df.to_csv(OUT / "complexity_table.csv", index=False)

    # Save as markdown
    md = df.to_markdown(index=False)
    (OUT / "complexity_table.md").write_text(md + "\n")
    print(f"  → complexity_table.csv / .md")
    return df


# ── Complexity bar chart ─────────────────────────────────────────────
def plot_complexity_bars(eval_summary):
    print("\n[7] Complexity vs Performance Scatter")
    methods_order = ["ecmp", "ospf", "bottleneck", "flexdate", "sensitivity",
                     "our_hybrid_moe_gate_v3", "our_unified_meta"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: avg decision time
    names = []
    times = []
    colors = []
    for m in methods_order:
        sub = eval_summary[eval_summary["method"] == m]
        if sub.empty:
            continue
        names.append(DISPLAY.get(m, m))
        times.append(sub["decision_time_ms"].mean())
        colors.append(PALETTE.get(m, "#333333"))

    bars = ax1.barh(names, times, color=colors, edgecolor="white", height=0.6)
    ax1.set_xlabel("Avg Decision Time (ms)")
    ax1.set_title("Computational Cost")
    ax1.grid(True, axis="x", alpha=0.3)
    ax1.invert_yaxis()
    for bar, t in zip(bars, times):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{t:.1f}", va="center", fontsize=9)

    # Scatter: decision time vs mean MLU (avg across topologies)
    for m in methods_order:
        sub = eval_summary[eval_summary["method"] == m]
        if sub.empty:
            continue
        mean_mlu = sub["mean_mlu"].mean()
        mean_dt = sub["decision_time_ms"].mean()
        ax2.scatter(mean_dt, mean_mlu, s=120, c=PALETTE.get(m, "#333333"),
                    edgecolors="black", linewidth=0.5, zorder=5)
        ax2.annotate(DISPLAY.get(m, m), (mean_dt, mean_mlu),
                     textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax2.set_xlabel("Avg Decision Time (ms)")
    ax2.set_ylabel("Avg Mean MLU (across 5 topologies)")
    ax2.set_title("Efficiency–Performance Trade-off")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "complexity_tradeoff")


# ── Time-series MLU plot (for one representative topology) ───────────
def plot_mlu_timeseries(eval_ts):
    print("\n[8] MLU Time-Series — Sprintlink")
    ds = "rocketfuel_sprintlink"
    methods_show = ["our_unified_meta", "bottleneck", "flexdate", "ecmp"]
    sub = eval_ts[eval_ts["dataset"] == ds].sort_values("timestep")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for method in methods_show:
        msub = sub[sub["method"] == method]
        if msub.empty:
            continue
        lw = 2.0 if method == "our_unified_meta" else 1.2
        alpha = 1.0 if method == "our_unified_meta" else 0.7
        ax.plot(msub["timestep"].values, msub["mlu"].values,
                label=DISPLAY.get(method, method),
                color=PALETTE.get(method), linewidth=lw, alpha=alpha)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("MLU")
    ax.set_title(f"MLU Over Time — {TOPO_DISPLAY.get(ds, ds)}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "timeseries_mlu_sprintlink")


# ── Summary radar / improvement bar chart ────────────────────────────
def plot_improvement_bars(eval_summary, gen_summary):
    print("\n[9] Improvement Over Best Baseline (bar chart)")
    datasets = ["abilene", "geant", "rocketfuel_ebone",
                "rocketfuel_sprintlink", "rocketfuel_tiscali"]
    all_ds = datasets + ["germany50"]
    our_mlu = []
    best_baseline_mlu = []
    labels = []

    for ds in datasets:
        sub = eval_summary[eval_summary["dataset"] == ds]
        ours = sub[sub["method"] == "our_unified_meta"]["mean_mlu"].values
        baselines = sub[sub["method"].isin(["bottleneck", "flexdate", "sensitivity",
                                            "cfrrl", "erodrl"])]["mean_mlu"]
        if len(ours) > 0 and len(baselines) > 0:
            our_mlu.append(ours[0])
            best_baseline_mlu.append(baselines.min())
            labels.append(TOPO_DISPLAY.get(ds, ds))

    # Germany50
    gsub = gen_summary[gen_summary["dataset"] == "germany50"]
    ours_g = gsub[gsub["method"] == "our_unified_meta"]["mean_mlu"].values
    base_g = gsub[gsub["method"].isin(["bottleneck", "flexdate", "sensitivity"])]["mean_mlu"]
    if len(ours_g) > 0 and len(base_g) > 0:
        our_mlu.append(ours_g[0])
        best_baseline_mlu.append(base_g.min())
        labels.append(TOPO_DISPLAY.get("germany50", "Germany50"))

    improvement_pct = [(b - o) / b * 100 for o, b in zip(our_mlu, best_baseline_mlu)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in improvement_pct]
    bars = ax.bar(x, improvement_pct, color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Improvement Over Best Baseline (%)")
    ax.set_title("Unified Meta — % Improvement in Mean MLU vs Best Published Baseline")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, improvement_pct):
        yoff = 0.3 if v >= 0 else -0.5
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + yoff,
                f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    fig.tight_layout()
    _save(fig, "improvement_vs_baseline")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Final Plots & Complexity Metrics")
    print("=" * 60)

    eval_ts = pd.read_csv(BENCH / "eval_timeseries.csv")
    eval_summary = pd.read_csv(BENCH / "eval_summary.csv")
    gen_ts = pd.read_csv(BENCH / "gen_timeseries.csv")
    gen_summary = pd.read_csv(BENCH / "gen_summary.csv")
    fail_ts = pd.read_csv(BENCH / "failure_timeseries.csv")

    plot_mlu_cdfs_seen(eval_ts)
    plot_mlu_cdf_germany50(gen_ts)
    plot_disturbance_cdfs(eval_ts)
    plot_failure_cdfs(fail_ts)
    plot_decision_time_cdfs(eval_ts)
    complexity_df = build_complexity_table(eval_summary)
    plot_complexity_bars(eval_summary)
    plot_mlu_timeseries(eval_ts)
    plot_improvement_bars(eval_summary, gen_summary)

    print("\n" + "=" * 60)
    print("COMPLEXITY TABLE")
    print("=" * 60)
    print(complexity_df.to_string(index=False))

    print(f"\nAll plots saved to: {OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
