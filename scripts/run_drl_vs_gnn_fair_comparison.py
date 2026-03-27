"""
Fair comparison: GNN vs DRL (PPO, DQN) in the SAME pipeline.
All experts use the same fixed dictionary lookup + LP optimizer.
No MoE gate, no learned gate. Just: expert selects k flows -> LP optimizes.

This answers the professor's question:
"Why not try DRL with fixed Dictionary Lookup first, to show its limitations?"
"""
import sys, os, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
import torch

from phase1_reactive.eval.common import (
    load_bundle, collect_specs, load_named_dataset,
    max_steps_from_args, resolve_phase1_k_crit,
)
from phase1_reactive.eval.core import run_selector_lp_method, run_static_method
from phase1_reactive.eval.metrics import summarize_timeseries
from phase1_reactive.baselines.literature_baselines import (
    select_bottleneck_critical, select_topk_by_demand,
    select_sensitivity_critical,
)

CONFIG = "configs/phase1_reactive_full.yaml"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

bundle = load_bundle(CONFIG)
max_steps = max_steps_from_args(bundle, 500)

# ── Load trained models ──
print("Loading models...")

# PPO
from phase1_reactive.drl.drl_selector import load_trained_ppo
ppo_path = Path("results/phase1_reactive/train/ppo/policy.pt")
ppo_model = None
if ppo_path.exists():
    try:
        ppo_model = load_trained_ppo(ppo_path, device="cpu")
        print(f"  PPO loaded from {ppo_path}")
    except Exception as e:
        print(f"  PPO load failed: {e}")

# DQN
from phase1_reactive.drl.dqn_selector import load_trained_dqn
dqn_path = Path("results/phase1_reactive/train/dqn/qnet.pt")
dqn_model = None
if dqn_path.exists():
    try:
        dqn_model = load_trained_dqn(dqn_path, device="cpu")
        print(f"  DQN loaded from {dqn_path}")
    except Exception as e:
        print(f"  DQN load failed: {e}")

# GNN
gnn_model = None
try:
    from phase1_reactive.drl.gnn_selector import load_trained_gnn
    gnn_path = Path("results/phase1_reactive/train/gnn/gnn_selector.pt")
    if gnn_path.exists():
        gnn_model = load_trained_gnn(gnn_path, device="cpu")
        print(f"  GNN loaded from {gnn_path}")
except Exception as e:
    print(f"  GNN load failed: {e}")


def build_ppo_selector(ppo_model, env, k_crit):
    """Build PPO flow selector function with same signature as bottleneck."""
    def ppo_fn(tm, ecmp_base, caps, cur_paths, cur_wts, prev_sel=None, fail_mask=None):
        try:
            from phase1_reactive.drl.moe_features import ppo_raw_scores, topk_from_scores
            obs = env._build_obs(tm) if hasattr(env, '_build_obs') else None
            if obs is None:
                # Manual feature construction
                num_flows = tm.shape[0] * (tm.shape[0] - 1) if tm.ndim == 1 else tm.shape[0]
                od_features = torch.zeros(num_flows, ppo_model.od_dim if hasattr(ppo_model, 'od_dim') else 8)
                gf = torch.zeros(ppo_model.global_dim if hasattr(ppo_model, 'global_dim') else 16)

                # Build OD features from traffic matrix
                if tm.ndim == 2:
                    n = tm.shape[0]
                    demands = []
                    for i in range(n):
                        for j in range(n):
                            if i != j:
                                demands.append(tm[i, j])
                    demands = np.array(demands)
                else:
                    demands = tm.copy()

                if len(demands) > 0:
                    d_max = demands.max() if demands.max() > 0 else 1.0
                    normed = demands / d_max
                    od_dim = od_features.shape[1]
                    od_features[:len(demands), 0] = torch.from_numpy(normed).float()

                mask = torch.ones(od_features.shape[0], dtype=torch.bool)
                selected, _, _, _ = ppo_model.act(
                    od_features.unsqueeze(0),
                    gf.unsqueeze(0),
                    mask.unsqueeze(0),
                    k_crit,
                    deterministic=True,
                )
                return selected[0].tolist() if hasattr(selected[0], 'tolist') else list(selected[0])
            else:
                od_t = torch.from_numpy(obs["od_features"]).float().unsqueeze(0)
                gf_t = torch.from_numpy(obs["global_features"]).float().unsqueeze(0)
                mask_t = torch.from_numpy(obs["active_mask"]).bool().unsqueeze(0)
                selected, _, _, _ = ppo_model.act(od_t, gf_t, mask_t, k_crit, deterministic=True)
                return selected[0].tolist() if hasattr(selected[0], 'tolist') else list(selected[0])
        except Exception as e:
            # Fallback to bottleneck
            return select_bottleneck_critical(tm, ecmp_base, cur_paths, caps, k_crit)
    return ppo_fn


def build_dqn_selector(dqn_model, env, k_crit):
    """Build DQN flow selector function with same signature as bottleneck."""
    def dqn_fn(tm, ecmp_base, caps, cur_paths, cur_wts, prev_sel=None, fail_mask=None):
        try:
            num_flows = tm.shape[0] * (tm.shape[0] - 1) if tm.ndim == 2 else tm.shape[0]
            od_dim = dqn_model.od_dim if hasattr(dqn_model, 'od_dim') else 8
            gf_dim = dqn_model.global_dim if hasattr(dqn_model, 'global_dim') else 16

            if tm.ndim == 2:
                n = tm.shape[0]
                demands = []
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            demands.append(tm[i, j])
                demands = np.array(demands)
            else:
                demands = tm.copy()

            od_features = torch.zeros(num_flows, od_dim)
            gf = torch.zeros(gf_dim)

            if len(demands) > 0:
                d_max = demands.max() if demands.max() > 0 else 1.0
                normed = demands / d_max
                od_features[:len(demands), 0] = torch.from_numpy(normed).float()

            q_scores = dqn_model.q_scores(od_features, gf.unsqueeze(0).expand(num_flows, -1))
            q_scores = q_scores.squeeze(-1)

            mask = torch.ones(num_flows, dtype=torch.bool)
            masked_scores = q_scores.clone()
            masked_scores[~mask] = -float('inf')
            _, topk_idx = masked_scores.topk(min(k_crit, num_flows))
            return topk_idx.tolist()
        except Exception as e:
            return select_bottleneck_critical(tm, ecmp_base, cur_paths, caps, k_crit)
    return dqn_fn


def build_gnn_selector(gnn_model, dataset, k_crit):
    """Build GNN flow selector (same as in unified meta)."""
    def gnn_fn(tm, ecmp_base, caps, cur_paths, cur_wts, prev_sel=None, fail_mask=None):
        try:
            from phase1_reactive.drl.gnn_inference import gnn_select_critical
            return gnn_select_critical(gnn_model, dataset, tm, ecmp_base, caps, k_crit, device="cpu")
        except Exception:
            return select_bottleneck_critical(tm, ecmp_base, cur_paths, caps, k_crit)
    return gnn_fn


def build_bottleneck_selector(k_crit):
    """Build bottleneck heuristic selector."""
    def bn_fn(tm, ecmp_base, caps, cur_paths, cur_wts, prev_sel=None, fail_mask=None):
        return select_bottleneck_critical(tm, ecmp_base, cur_paths, caps, k_crit)
    return bn_fn


# ── Run evaluation ──
all_results = []

for field in ["eval_topologies", "generalization_topologies"]:
    specs = collect_specs(bundle, field)
    for spec in specs:
        try:
            ds, pl = load_named_dataset(bundle, spec, max_steps)
        except Exception as e:
            print(f"  Skip {spec}: {e}")
            continue

        k_crit = resolve_phase1_k_crit(bundle, ds)
        topo = ds.key
        print(f"\n{'='*60}")
        print(f"Topology: {topo} ({len(ds.nodes)} nodes, k_crit={k_crit})")
        print(f"{'='*60}")

        # Build all expert selectors
        experts = {}
        experts["bottleneck"] = build_bottleneck_selector(k_crit)

        if gnn_model is not None:
            experts["gnn"] = build_gnn_selector(gnn_model, ds, k_crit)

        if ppo_model is not None:
            experts["ppo"] = build_ppo_selector(ppo_model, None, k_crit)

        if dqn_model is not None:
            experts["dqn"] = build_dqn_selector(dqn_model, None, k_crit)

        # Run each expert through the SAME LP pipeline
        for expert_name, expert_fn in experts.items():
            print(f"  Running {expert_name}...", end=" ", flush=True)
            t0 = time.perf_counter()

            try:
                ts_df = run_selector_lp_method(
                    dataset=ds,
                    path_library=pl,
                    method_name=f"expert_{expert_name}",
                    selector_fn=expert_fn,
                    k_crit=k_crit,
                    split_name="test",
                    lp_time_limit_sec=20,
                )

                if ts_df.empty:
                    print("EMPTY")
                    continue

                summary = summarize_timeseries(ts_df)
                mean_mlu = summary["mean_mlu"].values[0]
                dec_ms = summary["decision_time_ms"].values[0]
                dist = summary["mean_disturbance"].values[0]
                p95 = summary["p95_mlu"].values[0]
                elapsed = time.perf_counter() - t0

                print(f"MLU={mean_mlu:.4f}  p95={p95:.4f}  dist={dist:.4f}  "
                      f"dec={dec_ms:.1f}ms  ({elapsed:.1f}s)")

                all_results.append({
                    "topology": topo,
                    "nodes": len(ds.nodes),
                    "expert": expert_name,
                    "mean_mlu": mean_mlu,
                    "p95_mlu": p95,
                    "mean_disturbance": dist,
                    "decision_time_ms": dec_ms,
                    "pipeline": "fixed_lookup + LP",
                })

                ts_df.to_csv(f"results/final_evidence_pack/drl_fair_{topo}_{expert_name}_ts.csv", index=False)

            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"FAILED ({elapsed:.1f}s): {e}")
                all_results.append({
                    "topology": topo,
                    "nodes": len(ds.nodes),
                    "expert": expert_name,
                    "mean_mlu": float('nan'),
                    "p95_mlu": float('nan'),
                    "mean_disturbance": float('nan'),
                    "decision_time_ms": float('nan'),
                    "pipeline": "fixed_lookup + LP",
                    "error": str(e),
                })

# ── Save results ──
results_df = pd.DataFrame(all_results)
out_path = "results/final_evidence_pack/drl_vs_gnn_fair_comparison.csv"
results_df.to_csv(out_path, index=False)
print(f"\n{'='*60}")
print(f"Results saved to: {out_path}")
print(f"{'='*60}")

# ── Print summary table ──
print("\n\nFAIR COMPARISON: All experts in same pipeline (expert -> LP -> ECMP)")
print("="*80)
for topo in results_df["topology"].unique():
    tdata = results_df[results_df["topology"] == topo].sort_values("mean_mlu")
    print(f"\n{topo} ({tdata['nodes'].iloc[0]} nodes):")
    print(f"  {'Expert':<15s} {'Mean MLU':>12s} {'P95 MLU':>12s} {'Disturbance':>12s} {'Decision(ms)':>12s}")
    print(f"  {'-'*63}")
    for _, row in tdata.iterrows():
        mlu = f"{row['mean_mlu']:.4f}" if not pd.isna(row['mean_mlu']) else "FAILED"
        p95 = f"{row['p95_mlu']:.4f}" if not pd.isna(row['p95_mlu']) else "FAILED"
        dist = f"{row['mean_disturbance']:.4f}" if not pd.isna(row['mean_disturbance']) else "FAILED"
        dec = f"{row['decision_time_ms']:.1f}" if not pd.isna(row['decision_time_ms']) else "FAILED"
        print(f"  {row['expert']:<15s} {mlu:>12s} {p95:>12s} {dist:>12s} {dec:>12s}")

best_per_topo = {}
for topo in results_df["topology"].unique():
    tdata = results_df[results_df["topology"] == topo].dropna(subset=["mean_mlu"])
    if not tdata.empty:
        best = tdata.loc[tdata["mean_mlu"].idxmin()]
        best_per_topo[topo] = best["expert"]

print(f"\n\nVALIDATION LOOKUP (best expert per topology):")
for topo, expert in best_per_topo.items():
    print(f"  {topo}: {expert}")
