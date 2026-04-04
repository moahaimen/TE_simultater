#!/usr/bin/env python3
"""Generate CDF comparison plots: Baseline MetaGate vs Stable MetaGate.

Outputs to results/dynamic_metagate/stable/plots/
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

BASELINE_RESULTS = Path("results/dynamic_metagate/metagate_results.csv")
BASELINE_SUMMARY = Path("results/dynamic_metagate/metagate_summary.csv")
STABLE_RESULTS = Path("results/dynamic_metagate/stable/stable_metagate_results.csv")
STABLE_SUMMARY = Path("results/dynamic_metagate/stable/stable_metagate_summary.csv")
SWEEP_SUMMARY = Path("results/dynamic_metagate/stable/parameter_sweep_summary.csv")
OUTPUT_DIR = Path("results/dynamic_metagate/stable/plots")

# Best stable config (lambda_d=0.2, lambda_s=0.1) — lowest disturbance + fewest switches
BEST_LD = 0.2
BEST_LS = 0.1

TOPOLOGIES = [
    "abilene", "cernet", "geant", "germany50",
    "rocketfuel_ebone", "rocketfuel_sprintlink",
    "rocketfuel_tiscali", "topologyzoo_vtlwavenet2011",
]

KNOWN = {"abilene", "cernet", "geant", "rocketfuel_ebone",
         "rocketfuel_sprintlink", "rocketfuel_tiscali"}


def cdf_xy(values):
    """Return sorted x, y for CDF plot."""
    s = np.sort(values)
    y = np.arange(1, len(s) + 1) / len(s)
    return s, y


def plot_cdf_comparison(baseline_vals, stable_vals, title, xlabel, outpath,
                        log_x=False):
    """Plot CDF of baseline vs stable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(baseline_vals) > 0:
        x, y = cdf_xy(baseline_vals)
        ax.step(x, y, where="post", label="Baseline MetaGate", color="tab:blue", lw=2)
    if len(stable_vals) > 0:
        x, y = cdf_xy(stable_vals)
        ax.step(x, y, where="post", label=f"Stable (λ_d={BEST_LD}, λ_s={BEST_LS})",
                color="tab:orange", lw=2, ls="--")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    bl = pd.read_csv(BASELINE_RESULTS)
    st = pd.read_csv(STABLE_RESULTS)
    bl_sum = pd.read_csv(BASELINE_SUMMARY)
    st_sum = pd.read_csv(STABLE_SUMMARY)
    sweep = pd.read_csv(SWEEP_SUMMARY)

    # Filter stable to best config
    st_best = st[(st["lambda_d"] == BEST_LD) & (st["lambda_s"] == BEST_LS)]
    st_sum_best = st_sum[(st_sum["lambda_d"] == BEST_LD) & (st_sum["lambda_s"] == BEST_LS)]

    print(f"Baseline rows: {len(bl)}, Stable (best) rows: {len(st_best)}")
    print(f"Baseline topologies: {bl['dataset'].nunique()}, Stable: {st_best['dataset'].nunique()}")

    # ─── 1. Global CDF: MLU ───
    plot_cdf_comparison(
        bl["metagate_mlu"].values,
        st_best["metagate_mlu"].values,
        "CDF of Per-Timestep MLU: Baseline vs Stable MetaGate",
        "Maximum Link Utilization (MLU)",
        OUTPUT_DIR / "cdf_mlu_global.png",
        log_x=True,
    )

    # ─── 2. Global CDF: Decision Time ───
    if "t_decision_ms" in bl.columns and "t_decision_ms" in st_best.columns:
        plot_cdf_comparison(
            bl["t_decision_ms"].values,
            st_best["t_decision_ms"].values,
            "CDF of Decision Time: Baseline vs Stable MetaGate",
            "Decision Time (ms)",
            OUTPUT_DIR / "cdf_decision_time_global.png",
        )

    # ─── 3. Global CDF: Total Time ───
    if "t_total_ms" in bl.columns and "t_total_ms" in st_best.columns:
        plot_cdf_comparison(
            bl["t_total_ms"].values,
            st_best["t_total_ms"].values,
            "CDF of Total End-to-End Time: Baseline vs Stable MetaGate",
            "Total Time (ms)",
            OUTPUT_DIR / "cdf_total_time_global.png",
        )

    # ─── 4. CDF: Disturbance (stable only, since baseline doesn't track per-timestep) ───
    if "disturbance" in st_best.columns:
        dist_vals = st_best["disturbance"].dropna().values
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot all 9 configs
        colors = plt.cm.viridis(np.linspace(0, 1, len(sweep)))
        for idx, (_, row) in enumerate(sweep.iterrows()):
            ld, ls = row["lambda_d"], row["lambda_s"]
            sub = st[(st["lambda_d"] == ld) & (st["lambda_s"] == ls)]
            if "disturbance" in sub.columns:
                vals = sub["disturbance"].dropna().values
                x, y = cdf_xy(vals)
                lbl = f"λ_d={ld}, λ_s={ls}"
                ax.step(x, y, where="post", label=lbl, color=colors[idx], lw=1.5,
                        alpha=0.8)
        ax.set_xlabel("Routing Disturbance (sym. diff / K_crit)", fontsize=12)
        ax.set_ylabel("CDF", fontsize=12)
        ax.set_title("CDF of Routing Disturbance Across Parameter Sweep", fontsize=13)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "cdf_disturbance_sweep.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / 'cdf_disturbance_sweep.png'}")

    # ─── 5. Per-Topology CDF: MLU ───
    for topo in TOPOLOGIES:
        bl_sub = bl[bl["dataset"] == topo]["metagate_mlu"].values
        st_sub = st_best[st_best["dataset"] == topo]["metagate_mlu"].values
        if len(bl_sub) > 0 or len(st_sub) > 0:
            ttype = "known" if topo in KNOWN else "unseen"
            plot_cdf_comparison(
                bl_sub, st_sub,
                f"CDF of MLU — {topo} ({ttype})",
                "MLU",
                OUTPUT_DIR / f"cdf_mlu_{topo}.png",
            )

    # ─── 6. Per-Topology CDF: Disturbance ───
    if "disturbance" in st_best.columns:
        for topo in TOPOLOGIES:
            st_sub = st_best[st_best["dataset"] == topo]["disturbance"].dropna().values
            if len(st_sub) > 0:
                ttype = "known" if topo in KNOWN else "unseen"
                fig, ax = plt.subplots(figsize=(8, 5))
                x, y = cdf_xy(st_sub)
                ax.step(x, y, where="post", label=f"Stable (λ_d={BEST_LD}, λ_s={BEST_LS})",
                        color="tab:orange", lw=2)
                ax.set_xlabel("Routing Disturbance", fontsize=12)
                ax.set_ylabel("CDF", fontsize=12)
                ax.set_title(f"CDF of Disturbance — {topo} ({ttype})", fontsize=13)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1.05)
                fig.tight_layout()
                fig.savefig(OUTPUT_DIR / f"cdf_disturbance_{topo}.png", dpi=150,
                            bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {OUTPUT_DIR / f'cdf_disturbance_{topo}.png'}")

    # ─── 7. Summary Comparison Bar Chart ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Mean disturbance by topology
    ax = axes[0]
    topos_short = [t.replace("rocketfuel_", "rf_").replace("topologyzoo_", "tz_")
                   for t in TOPOLOGIES]
    bl_dist = []
    st_dist = []
    for topo in TOPOLOGIES:
        # Baseline disturbance — compute from results
        bl_sub = bl[bl["dataset"] == topo]
        if len(bl_sub) > 0 and "selector" in bl_sub.columns:
            # Estimate disturbance from switch rate
            bl_dist.append(np.nan)  # baseline doesn't have disturbance column
        else:
            bl_dist.append(np.nan)

        st_row = st_sum_best[st_sum_best["dataset"] == topo]
        if len(st_row) > 0:
            st_dist.append(float(st_row.iloc[0]["mean_disturbance"]))
        else:
            st_dist.append(np.nan)

    x_pos = np.arange(len(TOPOLOGIES))
    ax.bar(x_pos, st_dist, 0.6, label=f"Stable (λ_d={BEST_LD}, λ_s={BEST_LS})",
           color="tab:orange", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(topos_short, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Routing Disturbance")
    ax.set_title("Mean Disturbance per Topology")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Switch rate by topology
    ax = axes[1]
    st_switch = []
    for topo in TOPOLOGIES:
        st_row = st_sum_best[st_sum_best["dataset"] == topo]
        if len(st_row) > 0:
            st_switch.append(float(st_row.iloc[0]["switch_rate"]))
        else:
            st_switch.append(0)
    ax.bar(x_pos, [s * 100 for s in st_switch], 0.6,
           color="tab:orange", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(topos_short, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Expert Switch Rate (%)")
    ax.set_title(f"Switch Rate per Topology (λ_d={BEST_LD}, λ_s={BEST_LS})")
    ax.grid(axis="y", alpha=0.3)

    # MLU comparison
    ax = axes[2]
    bl_mlu_vals = []
    st_mlu_vals = []
    for topo in TOPOLOGIES:
        bl_row = bl_sum[bl_sum["dataset"] == topo]
        if len(bl_row) > 0:
            bl_mlu_vals.append(float(bl_row.iloc[0]["metagate_mlu"]))
        else:
            bl_mlu_vals.append(np.nan)
        st_row = st_sum_best[st_sum_best["dataset"] == topo]
        if len(st_row) > 0:
            st_mlu_vals.append(float(st_row.iloc[0]["metagate_mlu"]))
        else:
            st_mlu_vals.append(np.nan)

    width = 0.35
    ax.bar(x_pos - width/2, bl_mlu_vals, width, label="Baseline", color="tab:blue", alpha=0.8)
    ax.bar(x_pos + width/2, st_mlu_vals, width,
           label=f"Stable (λ_d={BEST_LD}, λ_s={BEST_LS})", color="tab:orange", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(topos_short, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean MLU")
    ax.set_title("Mean MLU: Baseline vs Stable")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    fig.suptitle("Stable MetaGate Extension — Summary Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summary_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'summary_comparison.png'}")

    # ─── 8. Parameter Sweep Heatmaps ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ld_vals = sorted(sweep["lambda_d"].unique())
    ls_vals = sorted(sweep["lambda_s"].unique())

    for idx, (metric, title, cmap) in enumerate([
        ("mean_disturbance", "Mean Disturbance", "YlOrRd_r"),
        ("switch_rate", "Expert Switch Rate", "YlOrRd_r"),
        ("mean_mlu", "Mean MLU", "YlOrRd_r"),
    ]):
        ax = axes[idx]
        grid = np.zeros((len(ld_vals), len(ls_vals)))
        for i, ld in enumerate(ld_vals):
            for j, ls in enumerate(ls_vals):
                row = sweep[(sweep["lambda_d"] == ld) & (sweep["lambda_s"] == ls)]
                if len(row) > 0:
                    grid[i, j] = float(row.iloc[0][metric])
        im = ax.imshow(grid, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(ls_vals)))
        ax.set_xticklabels([f"{v}" for v in ls_vals])
        ax.set_yticks(range(len(ld_vals)))
        ax.set_yticklabels([f"{v}" for v in ld_vals])
        ax.set_xlabel("λ_s (switch penalty)")
        ax.set_ylabel("λ_d (disturbance penalty)")
        ax.set_title(title)
        # Annotate cells
        for i in range(len(ld_vals)):
            for j in range(len(ls_vals)):
                val = grid[i, j]
                if metric == "switch_rate":
                    text = f"{val:.1%}"
                elif metric == "mean_mlu":
                    text = f"{val:.2f}"
                else:
                    text = f"{val:.4f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Parameter Sweep: Stability vs Performance Trade-off", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "parameter_sweep_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'parameter_sweep_heatmap.png'}")

    # ─── 9. Expert Distribution Comparison ───
    sel_col_bl = "metagate_selector" if "metagate_selector" in bl.columns else "selector"
    sel_col_st = "metagate_selector" if "metagate_selector" in st_best.columns else "selector"
    if sel_col_bl in bl.columns and sel_col_st in st_best.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, df, label in [
            (axes[0], bl, "Baseline"),
            (axes[1], st_best, f"Stable (λ_d={BEST_LD}, λ_s={BEST_LS})"),
        ]:
            # Count expert usage per topology
            for topo in TOPOLOGIES:
                sub = df[df["dataset"] == topo]
                if len(sub) == 0:
                    continue
                col = sel_col_bl if label == "Baseline" else sel_col_st
                counts = sub[col].value_counts(normalize=True)
                topo_short = topo.replace("rocketfuel_", "rf_").replace("topologyzoo_", "tz_")
                for expert in ["bottleneck", "topk", "sensitivity", "gnn"]:
                    frac = counts.get(expert, 0)
                    # will be plotted as stacked bar

            # Simple stacked bar
            experts = ["bottleneck", "topk", "sensitivity", "gnn"]
            expert_colors = {"bottleneck": "#2196F3", "topk": "#FF9800",
                             "sensitivity": "#4CAF50", "gnn": "#E91E63"}
            bottoms = np.zeros(len(TOPOLOGIES))
            for expert in experts:
                heights = []
                for topo in TOPOLOGIES:
                    sub = df[df["dataset"] == topo]
                    if len(sub) == 0:
                        heights.append(0)
                    else:
                        col = sel_col_bl if label == "Baseline" else sel_col_st
                        heights.append((sub[col] == expert).mean())
                ax.bar(range(len(TOPOLOGIES)), heights, bottom=bottoms,
                       label=expert, color=expert_colors[expert], alpha=0.85)
                bottoms += heights

            ax.set_xticks(range(len(TOPOLOGIES)))
            ax.set_xticklabels(topos_short, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Expert Selection Fraction")
            ax.set_title(label)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_ylim(0, 1.1)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Expert Selection Distribution: Baseline vs Stable", fontsize=14)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "expert_distribution_comparison.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / 'expert_distribution_comparison.png'}")

    # ─── Print Summary Table ───
    print("\n" + "=" * 80)
    print("  BASELINE vs STABLE MetaGate — Summary Comparison")
    print("=" * 80)
    print(f"\n{'Metric':<35} {'Baseline':>15} {'Stable (best)':>15} {'Delta':>10}")
    print("-" * 80)

    bl_mean_mlu = bl["metagate_mlu"].mean()
    st_mean_mlu = st_best["metagate_mlu"].mean()
    print(f"{'Mean MLU':<35} {bl_mean_mlu:>15.4f} {st_mean_mlu:>15.4f} {(st_mean_mlu - bl_mean_mlu):>10.4f}")

    if "t_decision_ms" in bl.columns:
        bl_dec = bl["t_decision_ms"].mean()
        st_dec = st_best["t_decision_ms"].mean() if "t_decision_ms" in st_best.columns else np.nan
        print(f"{'Mean Decision Time (ms)':<35} {bl_dec:>15.2f} {st_dec:>15.2f} {(st_dec - bl_dec):>+10.2f}")

    if "disturbance" in st_best.columns:
        st_mean_dist = st_best["disturbance"].mean()
        print(f"{'Mean Disturbance':<35} {'N/A':>15} {st_mean_dist:>15.4f} {'—':>10}")

    # Switch rate
    sw_col = "expert_switch" if "expert_switch" in st_best.columns else "switch"
    if sw_col in st_best.columns:
        st_switch_rate = st_best[sw_col].mean()
        print(f"{'Expert Switch Rate':<35} {'N/A':>15} {st_switch_rate:>15.1%} {'—':>10}")

    # Accuracy
    bl_acc_vals = []
    st_acc_vals = []
    for topo in TOPOLOGIES:
        bl_row = bl_sum[bl_sum["dataset"] == topo]
        if len(bl_row) > 0:
            bl_acc_vals.append(float(bl_row.iloc[0]["accuracy"]))
        st_row = st_sum_best[st_sum_best["dataset"] == topo]
        if len(st_row) > 0:
            st_acc_vals.append(float(st_row.iloc[0]["accuracy"]))

    if bl_acc_vals and st_acc_vals:
        bl_acc = np.mean(bl_acc_vals)
        st_acc = np.mean(st_acc_vals)
        print(f"{'Mean Accuracy':<35} {bl_acc:>15.1%} {st_acc:>15.1%} {(st_acc - bl_acc):>+10.1%}")

    print("\n" + "=" * 80)
    print(f"  Total plots saved: {len(list(OUTPUT_DIR.glob('*.png')))}")
    print("=" * 80)


if __name__ == "__main__":
    main()
