"""
Mechanism-level analysis for the fair DRL ablation study.

Produces:
  A. Global baselines: Bottleneck-only, GNN-only, Unified Meta on all topologies
  B. Critical-flow coverage per topology and per TM
  C. Per-TM congestion localization (bottleneck link, contributors, overlap)
  D. Figures: per-TM MLU trace and coverage plot
"""
import sys, os, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

from phase1_reactive.eval.common import (
    load_bundle, collect_specs, load_named_dataset,
    max_steps_from_args, resolve_phase1_k_crit,
)
from phase1_reactive.env.offline_env import ReactiveRoutingEnv, ReactiveEnvConfig
from phase1_reactive.drl.moe_features import (
    ppo_raw_scores, dqn_raw_scores, topk_from_scores,
    bottleneck_scores, sensitivity_scores,
)
from phase1_reactive.drl.gnn_inference import choose_gnn_selector, load_gnn_selector
from phase1_reactive.drl.drl_selector import load_trained_ppo
from phase1_reactive.drl.dqn_selector import load_trained_dqn
from te.baselines import ecmp_splits
from te.simulator import apply_routing

CONFIG = "configs/phase1_reactive_full.yaml"
OUT_DIR = Path("results/drl_lookup_ablation")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

bundle = load_bundle(CONFIG)
max_steps = max_steps_from_args(bundle, 500)

# Load models
ppo_model = None
try:
    ppo_model = load_trained_ppo("results/phase1_reactive/train/ppo/policy.pt", device="cpu")
    ppo_model.eval()
except: pass

dqn_model = None
try:
    dqn_model = load_trained_dqn("results/phase1_reactive/train/dqn/qnet.pt", device="cpu")
    dqn_model.eval()
except: pass

gnn_model = None
try:
    gnn_model, _ = load_gnn_selector(
        "results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt", device="cpu")
    gnn_model.eval()
except: pass

# Load lookup
with open(OUT_DIR / "lookup_all.json") as f:
    lookup = json.load(f)
all_lookup = lookup["all_expert_lookup"]
topo_node_counts = lookup["topo_node_counts"]


def make_env(ds, pl, k_crit, split_name):
    cfg = ReactiveEnvConfig(k_crit=k_crit, lp_time_limit_sec=20)
    return ReactiveRoutingEnv(dataset=ds, tm_data=ds.tm, path_library=pl,
                              split_name=split_name, cfg=cfg)


def get_selected(env, obs, expert_name):
    """Return selected OD indices for a given expert."""
    if expert_name == "bottleneck":
        sc = bottleneck_scores(obs.current_tm, env.ecmp_base, env.path_library, env.capacities)
        return topk_from_scores(sc, obs.active_mask, env.k_crit)
    elif expert_name == "gnn" and gnn_model is not None:
        sel, _ = choose_gnn_selector(env, gnn_model, device="cpu")
        return sel
    elif expert_name == "ppo" and ppo_model is not None:
        sc = ppo_raw_scores(ppo_model, obs, device="cpu")
        return topk_from_scores(sc, obs.active_mask, env.k_crit)
    elif expert_name == "dqn" and dqn_model is not None:
        sc = dqn_raw_scores(dqn_model, obs, device="cpu")
        return topk_from_scores(sc, obs.active_mask, env.k_crit)
    else:
        sc = bottleneck_scores(obs.current_tm, env.ecmp_base, env.path_library, env.capacities)
        return topk_from_scores(sc, obs.active_mask, env.k_crit)


def compute_od_link_contrib(tm_vector, splits, path_library, capacities):
    """Compute per-OD contributions to each link and find bottleneck."""
    num_edges = capacities.size
    link_loads = np.zeros(num_edges, dtype=float)
    od_edge_contrib = [dict() for _ in range(len(tm_vector))]

    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0:
            continue
        od_paths = path_library.edge_idx_paths_by_od[od_idx]
        if not od_paths:
            continue
        sv = np.asarray(splits[od_idx], dtype=float)
        ss = float(np.sum(sv))
        if ss <= 0:
            continue
        norm = sv / ss
        for pi, frac in enumerate(norm):
            if frac <= 0 or pi >= len(od_paths):
                continue
            flow = float(demand) * float(frac)
            for eidx in od_paths[pi]:
                link_loads[eidx] += flow
                od_edge_contrib[od_idx][eidx] = od_edge_contrib[od_idx].get(eidx, 0.0) + flow

    util = link_loads / np.maximum(capacities, 1e-12)
    mlu = float(np.max(util)) if util.size else 0.0
    bn_edge = int(np.argmax(util))

    # Rank ODs by contribution to bottleneck link
    bn_contribs = [(od_edge_contrib[od].get(bn_edge, 0.0), od) for od in range(len(tm_vector))]
    bn_contribs.sort(key=lambda x: -x[0])

    return {
        "link_loads": link_loads, "utilization": util, "mlu": mlu,
        "bn_edge": bn_edge, "bn_util": float(util[bn_edge]),
        "bn_contribs": bn_contribs,  # [(flow, od_idx), ...]
    }


# ═══════════════════════════════════════════════════════
# COLLECT DATA
# ═══════════════════════════════════════════════════════

METHODS = ["bottleneck", "gnn"]
if ppo_model:
    METHODS.append("ppo")
if dqn_model:
    METHODS.append("dqn")

print("=" * 70)
print("MECHANISM-LEVEL ANALYSIS")
print("=" * 70)

coverage_rows = []
congestion_rows = []
global_baseline_rows = []
per_tm_rows = []

for field_name in ["eval_topologies", "generalization_topologies"]:
    specs = collect_specs(bundle, field_name)
    for spec in specs:
        try:
            ds, pl = load_named_dataset(bundle, spec, max_steps)
        except Exception as e:
            print(f"  SKIP {spec}: {e}")
            continue

        k_crit = resolve_phase1_k_crit(bundle, ds)
        topo = ds.key
        n_nodes = len(ds.nodes)
        n_od = len(ds.od_pairs)
        n_edges = len(ds.edges)
        ecmp_base = ecmp_splits(pl)

        # Determine unified meta choice
        if topo in all_lookup:
            meta_expert = all_lookup[topo]
        else:
            closest = min(topo_node_counts, key=lambda k: abs(int(topo_node_counts[k]) - n_nodes))
            meta_expert = all_lookup.get(closest, "bottleneck")

        print(f"\n  Topology: {topo} ({n_nodes}N, {n_od} OD pairs, {n_edges} edges, k_crit={k_crit})")
        print(f"    Meta-Selector picks: {meta_expert}")

        # ── B. Coverage summary ──
        coverage_rows.append({
            "topology": topo, "nodes": n_nodes, "total_od": n_od,
            "k_crit": k_crit, "k_over_total_pct": k_crit / n_od * 100 if n_od > 0 else 0,
            "edges": n_edges,
        })

        # ── Run each method through the test env ──
        for method in METHODS:
            if method == "gnn" and gnn_model is None:
                continue
            if method == "ppo" and ppo_model is None:
                continue
            if method == "dqn" and dqn_model is None:
                continue

            try:
                env = make_env(ds, pl, k_crit, "test")
                obs = env.reset()
                done = False
                step_idx = 0

                method_mlus = []
                method_dists = []

                while not done:
                    tm_now = obs.current_tm
                    total_demand = float(np.sum(tm_now))

                    # ECMP-only (before optimization)
                    ecmp_info = compute_od_link_contrib(tm_now, ecmp_base, pl, ds.capacities)
                    mlu_before = ecmp_info["mlu"]
                    bn_edge_before = ecmp_info["bn_edge"]

                    # Top 5 contributors to bottleneck before optimization
                    top5_bn = ecmp_info["bn_contribs"][:5]
                    top5_bn_ods = [od for _, od in top5_bn]

                    # Get expert selection
                    selected = get_selected(env, obs, method)
                    selected_set = set(selected)

                    # Selected demand share
                    sel_demand = float(sum(tm_now[od] for od in selected if od < len(tm_now)))
                    sel_demand_pct = sel_demand / total_demand * 100 if total_demand > 0 else 0

                    # Overlap: how many of top-5 bottleneck contributors were selected?
                    overlap = len(selected_set.intersection(top5_bn_ods))

                    # Step env (runs LP)
                    next_obs, reward, done, info = env.step(selected)
                    mlu_after = info["mlu"]
                    disturbance = info.get("disturbance", 0.0)

                    method_mlus.append(mlu_after)
                    method_dists.append(disturbance)

                    # Find bottleneck after
                    # We can approximate from info
                    delta_mlu = mlu_before - mlu_after

                    # Per-TM trace
                    per_tm_rows.append({
                        "topology": topo, "tm_idx": step_idx, "method": method,
                        "mlu_after": mlu_after, "mlu_before": mlu_before,
                        "disturbance": disturbance, "delta_mlu": delta_mlu,
                        "k_selected": len(selected),
                        "selected_demand_pct": sel_demand_pct,
                    })

                    # Congestion localization (first 20 TMs + every 10th after)
                    if step_idx < 20 or step_idx % 10 == 0:
                        edge_pair = ds.edges[bn_edge_before] if bn_edge_before < len(ds.edges) else ("?", "?")
                        congestion_rows.append({
                            "topology": topo, "tm_idx": step_idx, "method": method,
                            "bn_link_before": f"{edge_pair[0]}->{edge_pair[1]}",
                            "bn_edge_idx": bn_edge_before,
                            "mlu_before": mlu_before,
                            "top5_contributors": str(top5_bn_ods),
                            "selected_critical": str(list(selected)[:10]),
                            "overlap_count": overlap,
                            "mlu_after": mlu_after,
                            "delta_mlu": delta_mlu,
                            "k_selected": len(selected),
                            "selected_demand_pct": sel_demand_pct,
                        })

                    obs = next_obs
                    step_idx += 1

                # Global baseline summary
                mean_mlu = np.mean(method_mlus)
                p95_mlu = np.percentile(method_mlus, 95)
                mean_dist = np.mean(method_dists)
                is_meta = (method == meta_expert)

                global_baseline_rows.append({
                    "topology": topo, "method": method,
                    "mean_mlu": mean_mlu, "p95_mlu": p95_mlu,
                    "mean_disturbance": mean_dist,
                    "is_meta_choice": is_meta,
                    "n_steps": len(method_mlus),
                })
                marker = " <-- META" if is_meta else ""
                print(f"    {method:15s} mean_MLU={mean_mlu:.6f}  p95={p95_mlu:.6f}  "
                      f"dist={mean_dist:.4f}{marker}")

            except Exception as e:
                print(f"    {method:15s} FAILED: {e}")
                import traceback; traceback.print_exc()

# Save CSVs
pd.DataFrame(coverage_rows).to_csv(OUT_DIR / "coverage_summary.csv", index=False)
pd.DataFrame(congestion_rows).to_csv(OUT_DIR / "congestion_localization.csv", index=False)
pd.DataFrame(global_baseline_rows).to_csv(OUT_DIR / "global_baselines.csv", index=False)
pd.DataFrame(per_tm_rows).to_csv(OUT_DIR / "per_tm_trace.csv", index=False)


# ═══════════════════════════════════════════════════════
# PRINT TABLES
# ═══════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print("TABLE: GLOBAL BASELINES (Bottleneck-only vs GNN-only vs Unified Meta)")
print(f"{'='*80}")
gb_df = pd.DataFrame(global_baseline_rows)
for topo in gb_df["topology"].unique():
    td = gb_df[gb_df["topology"] == topo]
    print(f"\n  {topo}:")
    for _, r in td.iterrows():
        m = " <-- META" if r["is_meta_choice"] else ""
        print(f"    {r['method']:15s} MLU={r['mean_mlu']:.6f}  p95={r['p95_mlu']:.6f}  "
              f"dist={r['mean_disturbance']:.4f}{m}")

    # Comment
    methods = td.set_index("method")
    bn_mlu = methods.loc["bottleneck", "mean_mlu"] if "bottleneck" in methods.index else None
    gnn_mlu = methods.loc["gnn", "mean_mlu"] if "gnn" in methods.index else None
    if bn_mlu is not None and gnn_mlu is not None:
        gap = (bn_mlu - gnn_mlu) / gnn_mlu * 100 if gnn_mlu > 0 else 0
        if abs(gap) < 0.5:
            print(f"    >> Flat regime: BN and GNN within {abs(gap):.2f}% — selector choice barely matters")
        else:
            print(f"    >> GNN-needed regime: GNN is {gap:.2f}% better than Bottleneck")


print(f"\n\n{'='*80}")
print("TABLE: CRITICAL-FLOW COVERAGE")
print(f"{'='*80}")
cov_df = pd.DataFrame(coverage_rows)
print(f"  {'Topology':<15} {'Nodes':>6} {'Total OD':>10} {'k_crit':>8} {'k/OD%':>8} {'Edges':>7}")
print(f"  {'-'*60}")
for _, r in cov_df.iterrows():
    print(f"  {r['topology']:<15} {r['nodes']:>6} {r['total_od']:>10} {r['k_crit']:>8} "
          f"{r['k_over_total_pct']:>7.1f}% {r['edges']:>7}")

# Per-TM demand coverage
ptm_df = pd.DataFrame(per_tm_rows)
print(f"\n  Per-TM selected demand share (mean across TMs):")
for topo in ptm_df["topology"].unique():
    for method in ptm_df["method"].unique():
        td = ptm_df[(ptm_df["topology"] == topo) & (ptm_df["method"] == method)]
        if td.empty:
            continue
        print(f"    {topo:15s} {method:15s} demand_share={td['selected_demand_pct'].mean():.1f}%  "
              f"k_selected={td['k_selected'].mean():.1f}")


# ═══════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════

print(f"\n\nGenerating figures...")

# Figure 1: Per-TM MLU trace
for topo in ptm_df["topology"].unique():
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    colors = {"bottleneck": "#2196F3", "gnn": "#4CAF50", "ppo": "#FF9800", "dqn": "#F44336"}
    for method in ["bottleneck", "gnn", "ppo", "dqn"]:
        td = ptm_df[(ptm_df["topology"] == topo) & (ptm_df["method"] == method)]
        if td.empty:
            continue
        ax1.plot(td["tm_idx"], td["mlu_after"], label=f"{method} MLU",
                 color=colors.get(method, "gray"), linewidth=1.5)
        ax2.plot(td["tm_idx"], td["disturbance"], label=f"{method} dist.",
                 color=colors.get(method, "gray"), linewidth=0.8, linestyle="--", alpha=0.5)

    ax1.set_xlabel("Traffic Matrix Index", fontsize=11)
    ax1.set_ylabel("MLU after optimization", fontsize=11, color="black")
    ax2.set_ylabel("Disturbance", fontsize=11, color="gray")
    ax1.set_title(f"{topo.title()} — Per-TM MLU and Disturbance", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = FIG_DIR / f"fig1_mlu_trace_{topo}.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved: {fpath}")

# Figure 2: Coverage — selected demand % vs achieved MLU per method
for topo in ptm_df["topology"].unique():
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"bottleneck": "#2196F3", "gnn": "#4CAF50", "ppo": "#FF9800", "dqn": "#F44336"}
    for method in ["bottleneck", "gnn", "ppo", "dqn"]:
        td = ptm_df[(ptm_df["topology"] == topo) & (ptm_df["method"] == method)]
        if td.empty:
            continue
        ax.scatter(td["selected_demand_pct"], td["mlu_after"],
                   label=method, color=colors.get(method, "gray"), alpha=0.6, s=20)

    ax.set_xlabel("Selected Demand Share (%)", fontsize=11)
    ax.set_ylabel("MLU after optimization", fontsize=11)
    ax.set_title(f"{topo.title()} — Coverage vs MLU", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = FIG_DIR / f"fig2_coverage_vs_mlu_{topo}.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved: {fpath}")


# Figure 3: Overlap — how well does each selector target bottleneck contributors?
cong_df = pd.DataFrame(congestion_rows)
for topo in cong_df["topology"].unique():
    td = cong_df[cong_df["topology"] == topo]
    fig, ax = plt.subplots(figsize=(10, 4))
    for method in ["bottleneck", "gnn", "ppo", "dqn"]:
        md = td[td["method"] == method]
        if md.empty:
            continue
        ax.plot(md["tm_idx"], md["overlap_count"], label=method,
                color=colors.get(method, "gray"), marker="o", markersize=3, linewidth=1)

    ax.set_xlabel("Traffic Matrix Index", fontsize=11)
    ax.set_ylabel("Overlap with top-5 bottleneck contributors", fontsize=11)
    ax.set_title(f"{topo.title()} — Bottleneck Targeting Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.5, 5.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = FIG_DIR / f"fig3_overlap_{topo}.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved: {fpath}")


print(f"\n\nAll mechanism analysis saved to: {OUT_DIR}/")
print("Done.")
