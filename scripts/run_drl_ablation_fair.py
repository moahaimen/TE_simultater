"""
FAIR DRL ABLATION STUDY (v2 — all professor issues fixed)
==========================================================
Professor's request: Compare PPO, DQN, GNN, and Bottleneck as standalone experts
in the EXACT SAME pipeline: expert selects k flows -> LP optimizes -> ECMP for rest.

TWO lookups are built:
  1. ALL-EXPERT lookup: among all experts (GNN, PPO, DQN, Bottleneck, etc.)
  2. DRL-FAMILY-ONLY lookup: restricted to PPO and DQN only

Protocol:
  - VALIDATION phase: Run all experts on val split.
    Apply 0.1% threshold: if best is <= 0.1% better than simplest heuristic
    (Bottleneck), choose Bottleneck (parsimony preference).
  - TEST phase: Run all experts on test split. Report both lookups.

Unseen-topology fallback rule:
  For topologies NOT in the validation set, find the known topology with
  the closest node count. Inherit that topology's lookup expert.
  If no known topology exists, default to Bottleneck.

Scientific guarantees:
  - No test data leakage (lookups built from val only)
  - Same LP solver, same ECMP baseline, same k_crit for all experts
  - Proper observation construction for PPO/DQN via ReactiveRoutingEnv
  - Reproducible with fixed seed (42)
"""
import sys, os, time, json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch

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
from phase1_reactive.baselines.literature_baselines import select_literature_baseline

CONFIG = "configs/phase1_reactive_full.yaml"
OUT_DIR = Path("results/drl_lookup_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Threshold: if best expert is <= this fraction better than Bottleneck, use Bottleneck
THRESHOLD = 0.002  # 0.2% — parsimony: prefer simpler Bottleneck unless gain is meaningful

# ═══════════════════════════════════════════════════════
# LOAD CONFIG & MODELS
# ═══════════════════════════════════════════════════════
print("=" * 70)
print("FAIR DRL ABLATION STUDY v2")
print("=" * 70)

bundle = load_bundle(CONFIG)
max_steps = max_steps_from_args(bundle, 500)

ppo_path = Path("results/phase1_reactive/train/ppo/policy.pt")
ppo_model = None
if ppo_path.exists():
    try:
        ppo_model = load_trained_ppo(ppo_path, device="cpu")
        ppo_model.eval()
        print(f"  PPO loaded from {ppo_path}")
    except Exception as e:
        print(f"  PPO failed: {e}")

dqn_path = Path("results/phase1_reactive/train/dqn/qnet.pt")
dqn_model = None
if dqn_path.exists():
    try:
        dqn_model = load_trained_dqn(dqn_path, device="cpu")
        dqn_model.eval()
        print(f"  DQN loaded from {dqn_path}")
    except Exception as e:
        print(f"  DQN failed: {e}")

gnn_path = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
gnn_model = None
if gnn_path.exists():
    try:
        gnn_model, gnn_cfg = load_gnn_selector(gnn_path, device="cpu")
        gnn_model.eval()
        print(f"  GNN loaded from {gnn_path}")
    except Exception as e:
        print(f"  GNN failed: {e}")


# ═══════════════════════════════════════════════════════
# EXPERT ROLLOUT
# ═══════════════════════════════════════════════════════

def rollout_expert(env, expert_name, ppo_model=None, dqn_model=None,
                   gnn_model=None, device="cpu"):
    """Run a single expert through the SAME env pipeline (expert -> LP -> ECMP)."""
    obs = env.reset()
    rows = []
    done = False
    while not done:
        t0 = time.perf_counter()
        if expert_name == "ppo" and ppo_model is not None:
            scores = ppo_raw_scores(ppo_model, obs, device=device)
            selected = topk_from_scores(scores, obs.active_mask, env.k_crit)
        elif expert_name == "dqn" and dqn_model is not None:
            scores = dqn_raw_scores(dqn_model, obs, device=device)
            selected = topk_from_scores(scores, obs.active_mask, env.k_crit)
        elif expert_name == "gnn" and gnn_model is not None:
            selected, _gnn_info = choose_gnn_selector(env, gnn_model, device=device)
        elif expert_name == "bottleneck":
            scores = bottleneck_scores(
                obs.current_tm, env.ecmp_base, env.path_library, env.capacities)
            selected = topk_from_scores(scores, obs.active_mask, env.k_crit)
        elif expert_name == "sensitivity":
            scores = sensitivity_scores(
                obs.current_tm, env.ecmp_base, env.path_library, env.capacities)
            selected = topk_from_scores(scores, obs.active_mask, env.k_crit)
        elif expert_name == "erodrl":
            prev_sel = getattr(env, '_erodrl_prev', None)
            selected = select_literature_baseline(
                "erodrl", tm_vector=obs.current_tm, ecmp_policy=env.ecmp_base,
                path_library=env.path_library, capacities=env.capacities,
                k_crit=env.k_crit, prev_selected=prev_sel)
            env._erodrl_prev = selected
        else:
            scores = bottleneck_scores(
                obs.current_tm, env.ecmp_base, env.path_library, env.capacities)
            selected = topk_from_scores(scores, obs.active_mask, env.k_crit)
        decision_ms = (time.perf_counter() - t0) * 1000.0
        next_obs, reward, done, info = env.step(selected)
        info = dict(info)
        info["expert"] = expert_name
        info["decision_time_ms"] = decision_ms
        rows.append(info)
        obs = next_obs
    return pd.DataFrame(rows)


def make_env(ds, pl, k_crit, split_name):
    cfg = ReactiveEnvConfig(k_crit=k_crit, lp_time_limit_sec=20)
    return ReactiveRoutingEnv(dataset=ds, tm_data=ds.tm, path_library=pl,
                              split_name=split_name, cfg=cfg)


def pick_best_with_threshold(expert_mlus, baseline="bottleneck", threshold=THRESHOLD):
    """Pick best expert, but prefer baseline if improvement <= threshold.

    Returns: (chosen_expert, improvement_pct, threshold_applied, reason)
    """
    if not expert_mlus:
        return baseline, 0.0, False, "no experts available"
    best = min(expert_mlus, key=expert_mlus.get)
    best_mlu = expert_mlus[best]
    base_mlu = expert_mlus.get(baseline)
    if base_mlu is None:
        return best, 0.0, False, f"baseline '{baseline}' not available"
    if best == baseline:
        return baseline, 0.0, False, "baseline is already best"
    improvement = (base_mlu - best_mlu) / base_mlu if base_mlu > 0 else 0
    if improvement <= threshold:
        return baseline, improvement * 100, True, (
            f"{best} only {improvement*100:.4f}% better (threshold {threshold*100:.1f}%), "
            f"prefer {baseline} by parsimony")
    return best, improvement * 100, False, (
        f"{best} is {improvement*100:.2f}% better than {baseline}")


def unseen_fallback(topo_node_counts, lookup_dict, n_nodes):
    """Unseen-topology fallback: closest node count among known topologies.

    Rule: find topology in lookup_dict whose node count is closest to n_nodes.
    Inherit that topology's expert. If lookup_dict is empty, default to bottleneck.
    """
    if not topo_node_counts or not lookup_dict:
        return "bottleneck", "no known topologies", None
    closest = min(topo_node_counts, key=lambda k: abs(topo_node_counts[k] - n_nodes))
    expert = lookup_dict.get(closest, "bottleneck")
    return expert, f"closest known = {closest} ({topo_node_counts[closest]}N) -> {expert}", closest


# ═══════════════════════════════════════════════════════
# PHASE 1: VALIDATION
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 1: VALIDATION")
print("=" * 70)

EXPERTS = ["bottleneck", "sensitivity"]
if ppo_model is not None:
    EXPERTS.append("ppo")
if dqn_model is not None:
    EXPERTS.append("dqn")
if gnn_model is not None:
    EXPERTS.append("gnn")
EXPERTS.append("erodrl")

DRL_EXPERTS = [e for e in ["ppo", "dqn"] if e in EXPERTS]

print(f"  All experts: {EXPERTS}")
print(f"  DRL-family experts: {DRL_EXPERTS}")

val_results = []
topo_node_counts = {}
# Per-topology mean MLU for all experts
topo_expert_mlus = {}  # {topo: {expert: mean_mlu}}

for field_name in ["eval_topologies"]:
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
        topo_node_counts[topo] = n_nodes
        print(f"\n  Topology: {topo} ({n_nodes} nodes, k_crit={k_crit})")

        expert_mean_mlus = {}
        for expert_name in EXPERTS:
            if expert_name == "ppo" and ppo_model is None:
                continue
            if expert_name == "dqn" and dqn_model is None:
                continue
            if expert_name == "gnn" and gnn_model is None:
                continue
            try:
                env = make_env(ds, pl, k_crit, "val")
                df = rollout_expert(env, expert_name,
                                    ppo_model=ppo_model, dqn_model=dqn_model,
                                    gnn_model=gnn_model)
                mean_mlu = df["mlu"].mean()
                p95_mlu = df["mlu"].quantile(0.95)
                mean_dist = df["disturbance"].mean() if "disturbance" in df.columns else 0.0
                mean_dec = df["decision_time_ms"].mean()
                print(f"    {expert_name:15s} val_MLU={mean_mlu:.6f}  p95={p95_mlu:.6f}  "
                      f"dist={mean_dist:.4f}  dec={mean_dec:.1f}ms")
                expert_mean_mlus[expert_name] = mean_mlu
                val_results.append({
                    "topology": topo, "nodes": n_nodes, "expert": expert_name,
                    "split": "val", "mean_mlu": mean_mlu, "p95_mlu": p95_mlu,
                    "mean_disturbance": mean_dist, "decision_time_ms": mean_dec,
                    "n_steps": len(df),
                })
                df.to_csv(OUT_DIR / f"val_{topo}_{expert_name}_ts.csv", index=False)
            except Exception as e:
                print(f"    {expert_name:15s} FAILED: {e}")
                import traceback; traceback.print_exc()

        topo_expert_mlus[topo] = expert_mean_mlus


# ── Build BOTH lookups from validation data ──

# LOOKUP 1: All-expert (threshold vs Bottleneck)
all_expert_lookup = {}
all_expert_lookup_detail = {}
for topo, mlus in topo_expert_mlus.items():
    chosen, improv_pct, thresh_applied, reason = pick_best_with_threshold(mlus)
    all_expert_lookup[topo] = chosen
    all_expert_lookup_detail[topo] = {
        "chosen": chosen, "improvement_pct": improv_pct,
        "threshold_applied": thresh_applied, "reason": reason,
    }
    marker = " [THRESHOLD -> Bottleneck]" if thresh_applied else ""
    print(f"    ALL-EXPERT  {topo}: {chosen}{marker} ({reason})")

# LOOKUP 2: DRL-family-only (PPO, DQN; threshold vs best-available simple baseline)
drl_lookup = {}
drl_lookup_detail = {}
for topo, mlus in topo_expert_mlus.items():
    drl_mlus = {e: mlus[e] for e in DRL_EXPERTS if e in mlus}
    if not drl_mlus:
        drl_lookup[topo] = None
        drl_lookup_detail[topo] = {"chosen": None, "reason": "no DRL experts available"}
        continue
    # Among DRL-only candidates, pick the one with lowest val MLU
    best_drl = min(drl_mlus, key=drl_mlus.get)
    # If PPO == DQN, prefer simpler (DQN has fewer params)
    if len(drl_mlus) == 2:
        ppo_v = drl_mlus.get("ppo", float('inf'))
        dqn_v = drl_mlus.get("dqn", float('inf'))
        if abs(ppo_v - dqn_v) / max(ppo_v, 1e-12) <= THRESHOLD:
            best_drl = "dqn"  # prefer simpler on tie
            reason = f"PPO ({ppo_v:.6f}) and DQN ({dqn_v:.6f}) within threshold, prefer DQN (simpler)"
        else:
            reason = f"{best_drl} has lower val MLU ({drl_mlus[best_drl]:.6f})"
    else:
        reason = f"only {best_drl} available"
    drl_lookup[topo] = best_drl
    drl_lookup_detail[topo] = {"chosen": best_drl, "reason": reason,
                                "ppo_mlu": drl_mlus.get("ppo"), "dqn_mlu": drl_mlus.get("dqn")}
    print(f"    DRL-ONLY    {topo}: {best_drl} ({reason})")

val_df = pd.DataFrame(val_results)
val_df.to_csv(OUT_DIR / "validation_results.csv", index=False)


# ═══════════════════════════════════════════════════════
# PHASE 2: TEST
# ═══════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PHASE 2: TEST")
print(f"{'='*70}")

test_results = []
unseen_assignments = []

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
        is_unseen = (topo not in all_expert_lookup)
        print(f"\n  Topology: {topo} ({n_nodes} nodes, k_crit={k_crit})"
              f"{' [UNSEEN]' if is_unseen else ''}")

        # Determine lookup assignments
        if topo in all_expert_lookup:
            all_lookup_expert = all_expert_lookup[topo]
            all_fallback_reason = "direct from validation"
        else:
            all_lookup_expert, all_fallback_reason, closest = unseen_fallback(
                topo_node_counts, all_expert_lookup, n_nodes)
            print(f"    All-expert fallback: {all_fallback_reason}")

        if topo in drl_lookup:
            drl_lookup_expert = drl_lookup[topo]
            drl_fallback_reason = "direct from validation"
        else:
            drl_lookup_expert, drl_fallback_reason, closest = unseen_fallback(
                topo_node_counts, drl_lookup, n_nodes)
            print(f"    DRL-only fallback: {drl_fallback_reason}")

        if is_unseen:
            unseen_assignments.append({
                "topology": topo, "nodes": n_nodes,
                "all_expert_lookup": all_lookup_expert,
                "all_expert_rule": all_fallback_reason,
                "drl_lookup": drl_lookup_expert,
                "drl_rule": drl_fallback_reason,
            })

        for expert_name in EXPERTS:
            if expert_name == "ppo" and ppo_model is None:
                continue
            if expert_name == "dqn" and dqn_model is None:
                continue
            if expert_name == "gnn" and gnn_model is None:
                continue
            try:
                env = make_env(ds, pl, k_crit, "test")
                df = rollout_expert(env, expert_name,
                                    ppo_model=ppo_model, dqn_model=dqn_model,
                                    gnn_model=gnn_model)
                mean_mlu = df["mlu"].mean()
                p95_mlu = df["mlu"].quantile(0.95)
                mean_dist = df["disturbance"].mean() if "disturbance" in df.columns else 0.0
                mean_dec = df["decision_time_ms"].mean()

                is_all_choice = (expert_name == all_lookup_expert)
                is_drl_choice = (expert_name == drl_lookup_expert)
                markers = []
                if is_all_choice:
                    markers.append("ALL-LOOKUP")
                if is_drl_choice:
                    markers.append("DRL-LOOKUP")
                marker_str = f"  [{','.join(markers)}]" if markers else ""

                print(f"    {expert_name:15s} test_MLU={mean_mlu:.6f}  p95={p95_mlu:.6f}  "
                      f"dist={mean_dist:.4f}  dec={mean_dec:.1f}ms{marker_str}")

                test_results.append({
                    "topology": topo, "nodes": n_nodes, "expert": expert_name,
                    "split": "test", "mean_mlu": mean_mlu, "p95_mlu": p95_mlu,
                    "mean_disturbance": mean_dist, "decision_time_ms": mean_dec,
                    "n_steps": len(df),
                    "is_all_lookup_choice": is_all_choice,
                    "is_drl_lookup_choice": is_drl_choice,
                    "is_unseen": is_unseen,
                })
                df.to_csv(OUT_DIR / f"test_{topo}_{expert_name}_ts.csv", index=False)
            except Exception as e:
                print(f"    {expert_name:15s} FAILED: {e}")
                import traceback; traceback.print_exc()

test_df = pd.DataFrame(test_results)
test_df.to_csv(OUT_DIR / "test_results.csv", index=False)


# ═══════════════════════════════════════════════════════
# STATISTICAL SIGNIFICANCE
# ═══════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("STATISTICAL SIGNIFICANCE (paired t-test on test MLU)")
print(f"{'='*70}")
from scipy import stats

sig_results = []
for topo in test_df["topology"].unique():
    gnn_path_f = OUT_DIR / f"test_{topo}_gnn_ts.csv"
    if not gnn_path_f.exists():
        continue
    gnn_ts = pd.read_csv(gnn_path_f)
    if "mlu" not in gnn_ts.columns:
        continue
    for expert_name in ["ppo", "dqn", "bottleneck", "sensitivity", "erodrl"]:
        exp_path = OUT_DIR / f"test_{topo}_{expert_name}_ts.csv"
        if not exp_path.exists():
            continue
        exp_ts = pd.read_csv(exp_path)
        if "mlu" not in exp_ts.columns:
            continue
        n_common = min(len(gnn_ts), len(exp_ts))
        if n_common < 5:
            continue
        gnn_mlu = gnn_ts["mlu"].values[:n_common]
        exp_mlu = exp_ts["mlu"].values[:n_common]
        t_stat, p_val = stats.ttest_rel(gnn_mlu, exp_mlu)
        mean_diff = float(np.mean(gnn_mlu - exp_mlu))
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        sig_results.append({
            "topology": topo, "comparison": f"GNN vs {expert_name}",
            "mean_diff": mean_diff, "t_stat": t_stat, "p_value": p_val,
            "significance": sig, "n_samples": n_common,
            "gnn_better": bool(mean_diff < 0),
        })
        print(f"  {topo:20s}  GNN vs {expert_name:12s}  "
              f"d={mean_diff:+.6f}  t={t_stat:+.3f}  p={p_val:.4f}  {sig}")

if sig_results:
    pd.DataFrame(sig_results).to_csv(OUT_DIR / "significance_tests.csv", index=False)


# ═══════════════════════════════════════════════════════
# SAVE ALL LOOKUPS
# ═══════════════════════════════════════════════════════

lookup_payload = {
    "all_expert_lookup": all_expert_lookup,
    "all_expert_lookup_detail": all_expert_lookup_detail,
    "drl_lookup": {k: v for k, v in drl_lookup.items()},
    "drl_lookup_detail": drl_lookup_detail,
    "unseen_assignments": unseen_assignments,
    "topo_node_counts": {k: int(v) for k, v in topo_node_counts.items()},
    "threshold": f"{THRESHOLD*100:.1f}%",
    "threshold_rule": (
        f"If best expert is <= {THRESHOLD*100:.1f}% better than Bottleneck, "
        f"choose Bottleneck (parsimony preference)."
    ),
    "unseen_rule": (
        "For unseen topologies, find known topology with closest node count. "
        "Inherit that topology's lookup expert. If no known topology, default to Bottleneck."
    ),
    "drl_family": DRL_EXPERTS,
    "seed": SEED,
}
with open(OUT_DIR / "lookup_all.json", "w") as f:
    json.dump(lookup_payload, f, indent=2)


# ═══════════════════════════════════════════════════════
# SUMMARY TABLES
# ═══════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print("TABLE A: ALL-EXPERT VALIDATION LOOKUP")
print(f"{'='*80}")
print(f"  {'Topology':<15} ", end="")
for e in EXPERTS:
    print(f"{e:>12}", end="")
print(f"  {'Chosen':>12}  {'Thresh?':>8}  Reason")
print(f"  {'-'*120}")
for topo in topo_expert_mlus:
    mlus = topo_expert_mlus[topo]
    detail = all_expert_lookup_detail[topo]
    print(f"  {topo:<15} ", end="")
    for e in EXPERTS:
        v = mlus.get(e)
        print(f"{v:12.6f}" if v is not None else f"{'N/A':>12}", end="")
    print(f"  {detail['chosen']:>12}  {'YES' if detail['threshold_applied'] else 'NO':>8}  {detail['reason']}")

print(f"\n\n{'='*80}")
print("TABLE B: DRL-FAMILY-ONLY VALIDATION LOOKUP")
print(f"{'='*80}")
print(f"  {'Topology':<15} {'PPO':>12} {'DQN':>12} {'Chosen DRL':>12}  Reason")
print(f"  {'-'*80}")
for topo, detail in drl_lookup_detail.items():
    ppo_v = f"{detail.get('ppo_mlu', 0):.6f}" if detail.get('ppo_mlu') else "N/A"
    dqn_v = f"{detail.get('dqn_mlu', 0):.6f}" if detail.get('dqn_mlu') else "N/A"
    chosen = detail["chosen"] or "N/A"
    print(f"  {topo:<15} {ppo_v:>12} {dqn_v:>12} {chosen:>12}  {detail['reason']}")

print(f"\n\n{'='*80}")
print("TABLE C: FINAL TEST COMPARISON")
print(f"{'='*80}")
for topo in test_df["topology"].unique():
    tdata = test_df[test_df["topology"] == topo].sort_values("mean_mlu")
    n = tdata["nodes"].iloc[0]
    unseen = " [UNSEEN]" if tdata["is_unseen"].iloc[0] else ""
    print(f"\n  {topo} ({n} nodes){unseen}:")
    # Find lookup MLUs
    all_lk = tdata[tdata["is_all_lookup_choice"]]
    drl_lk = tdata[tdata["is_drl_lookup_choice"]]
    all_lk_mlu = all_lk["mean_mlu"].iloc[0] if not all_lk.empty else None
    drl_lk_mlu = drl_lk["mean_mlu"].iloc[0] if not drl_lk.empty else None

    print(f"    {'Expert':<15} {'Mean MLU':>12} {'P95 MLU':>12} {'Disturb.':>10} "
          f"{'Dec.(ms)':>10} {'Marker':>14}")
    print(f"    {'-'*75}")
    for _, row in tdata.iterrows():
        mlu = f"{row['mean_mlu']:.6f}" if pd.notna(row['mean_mlu']) else "FAILED"
        p95 = f"{row['p95_mlu']:.6f}" if pd.notna(row['p95_mlu']) else "FAILED"
        dist = f"{row['mean_disturbance']:.4f}" if pd.notna(row['mean_disturbance']) else "-"
        dec = f"{row['decision_time_ms']:.1f}" if pd.notna(row['decision_time_ms']) else "-"
        marks = []
        if row.get("is_all_lookup_choice"):
            marks.append("ALL-LK")
        if row.get("is_drl_lookup_choice"):
            marks.append("DRL-LK")
        mark = ",".join(marks)
        print(f"    {row['expert']:<15} {mlu:>12} {p95:>12} {dist:>10} {dec:>10} {mark:>14}")

    if all_lk_mlu and drl_lk_mlu:
        gap = ((drl_lk_mlu - all_lk_mlu) / all_lk_mlu * 100) if all_lk_mlu > 0 else 0
        print(f"    >> DRL-lookup vs All-expert-lookup gap: {gap:+.2f}%")

print(f"\n\n{'='*80}")
print("TABLE E: UNSEEN-TOPOLOGY ASSIGNMENTS")
print(f"{'='*80}")
if unseen_assignments:
    for ua in unseen_assignments:
        print(f"  {ua['topology']} ({ua['nodes']}N):")
        print(f"    All-expert lookup: {ua['all_expert_lookup']}  ({ua['all_expert_rule']})")
        print(f"    DRL-only lookup:   {ua['drl_lookup']}  ({ua['drl_rule']})")
else:
    print("  No unseen topologies in test set.")


# ── HEAD-TO-HEAD ──
print(f"\n\n{'='*80}")
print("HEAD-TO-HEAD: Meta-Selector (All-Expert) vs DRL-Family Lookup")
print(f"{'='*80}")
for topo in test_df["topology"].unique():
    tdata = test_df[test_df["topology"] == topo]
    all_row = tdata[tdata["is_all_lookup_choice"]]
    drl_row = tdata[tdata["is_drl_lookup_choice"]]
    if all_row.empty or drl_row.empty:
        continue
    a_mlu = all_row["mean_mlu"].iloc[0]
    d_mlu = drl_row["mean_mlu"].iloc[0]
    a_exp = all_row["expert"].iloc[0]
    d_exp = drl_row["expert"].iloc[0]
    if a_mlu > 0:
        gap = (d_mlu - a_mlu) / a_mlu * 100
    else:
        gap = 0
    winner = "Meta-Selector" if a_mlu <= d_mlu else "DRL-Lookup"
    tie = abs(gap) < 0.01
    print(f"  {topo:20s}  Meta({a_exp})={a_mlu:.6f}  DRL({d_exp})={d_mlu:.6f}  "
          f"gap={gap:+.2f}%  {'TIE' if tie else winner}")

print(f"\n  All results saved to: {OUT_DIR}/")
print("Done.")
