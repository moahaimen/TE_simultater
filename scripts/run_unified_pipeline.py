#!/usr/bin/env python3
"""Unified Meta-Selector Pipeline: train + evaluate.

Combines GNN, MoE v3, and all heuristics into a single framework.
A learned meta-gate picks the best expert per timestep.

Usage:
  python scripts/run_unified_pipeline.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from te.baselines import (
    ecmp_splits,
    select_topk_by_demand,
    select_bottleneck_critical,
    select_sensitivity_critical,
)
from te.simulator import apply_routing
from phase1_reactive.baselines.literature_baselines import select_literature_baseline
from phase1_reactive.drl.gnn_selector import load_gnn_selector, build_graph_tensors, build_od_features
from phase1_reactive.drl.gnn_inference import rollout_gnn_selector_policy, GNN_METHOD
from phase1_reactive.drl.meta_selector import (
    build_meta_features,
    collect_expert_results_per_timestep,
    train_meta_gate,
    load_meta_gate,
    rollout_unified_selector,
    rollout_oracle_selector,
    META_FEATURE_DIM,
)
from phase1_reactive.drl.state_builder import compute_reactive_telemetry
from phase1_reactive.eval.common import (
    build_reactive_env_cfg,
    load_bundle,
    load_named_dataset,
    collect_specs,
    max_steps_from_args,
    resolve_phase1_k_crit,
)
from phase1_reactive.eval.core import run_selector_lp_method, run_static_method, split_indices
from phase1_reactive.eval.metrics import summarize_timeseries
from phase1_reactive.env.offline_env import ReactiveRoutingEnv


# ============================================================
# Configuration
# ============================================================
CONFIG_PATH = "configs/phase1_reactive_full.yaml"
MAX_STEPS = 500
SEED = 42
DEVICE = "cpu"

OUTPUT_DIR = Path("results/phase1_reactive/unified_meta")
GNN_CHECKPOINT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")

# MoE v3 checkpoints
PPO_CHECKPOINT = Path("results/phase1_reactive/train/ppo/policy.pt")
DQN_CHECKPOINT = Path("results/phase1_reactive/train/dqn/qnet.pt")
MOE_GATE_CHECKPOINT = Path("results/phase1_reactive/train/moe_gate/gate.pt")

# Expert methods to include
HEURISTIC_EXPERTS = ["topk", "bottleneck", "sensitivity", "flexdate", "erodrl", "cfrrl", "flexentry"]


# ============================================================
# Helper: build expert functions for a given env
# ============================================================

def build_expert_fns(env, gnn_model=None, moe_models=None):
    """Build dict of expert_name -> fn(env) -> list[int] for a given environment."""
    ecmp_base = ecmp_splits(env.path_library)
    capacities = np.asarray(env.dataset.capacities, dtype=np.float64)
    k_crit = env.k_crit

    expert_fns = {}

    # Heuristic experts
    def make_heuristic_fn(method_name):
        def fn(env_):
            obs = env_.current_obs
            tm = obs.current_tm
            if method_name == "topk":
                return select_topk_by_demand(tm, k_crit)
            elif method_name == "bottleneck":
                return select_bottleneck_critical(tm, ecmp_base, env_.path_library, capacities, k_crit)
            elif method_name == "sensitivity":
                return select_sensitivity_critical(tm, ecmp_base, env_.path_library, capacities, k_crit)
            else:
                return select_literature_baseline(
                    method_name,
                    tm_vector=tm,
                    ecmp_policy=ecmp_base,
                    path_library=env_.path_library,
                    capacities=capacities,
                    k_crit=k_crit,
                    prev_selected=getattr(env_, 'prev_selected', None),
                    failure_mask=getattr(obs, 'failure_mask', None),
                )
        return fn

    for method in HEURISTIC_EXPERTS:
        expert_fns[method] = make_heuristic_fn(method)

    # GNN expert
    if gnn_model is not None:
        def gnn_fn(env_):
            obs = env_.current_obs
            graph_data = build_graph_tensors(
                env_.dataset, telemetry=obs.telemetry, device=DEVICE,
            )
            od_data = build_od_features(
                env_.dataset, obs.current_tm, env_.path_library,
                telemetry=obs.telemetry, device=DEVICE,
            )
            active_mask = (np.asarray(obs.current_tm, dtype=np.float64) > 1e-12).astype(np.float32)
            selected, _ = gnn_model.select_critical_flows(
                graph_data, od_data, active_mask=active_mask, k_crit_default=k_crit,
            )
            return selected
        expert_fns["gnn"] = gnn_fn

    # MoE v3 expert
    if moe_models is not None:
        ppo_model, dqn_model, gate_model = moe_models
        def moe_fn(env_):
            from phase1_reactive.drl.moe_inference import choose_moe_gate
            selected, _ = choose_moe_gate(env_, ppo_model, dqn_model, gate_model, device=DEVICE)
            return selected
        expert_fns["moe_v3"] = moe_fn

    return expert_fns


# ============================================================
# Step 1: Collect training data
# ============================================================

def collect_training_data(bundle, datasets, gnn_model, moe_models):
    """Collect per-timestep expert results for meta-gate training."""
    print("\n" + "=" * 70)
    print("STEP 1: COLLECTING PER-TIMESTEP EXPERT RESULTS")
    print("=" * 70)

    all_train_samples = []
    all_val_samples = []

    for dataset, path_library in datasets:
        k_crit = resolve_phase1_k_crit(bundle, dataset)
        ecmp_base = ecmp_splits(path_library)
        capacities = np.asarray(dataset.capacities, dtype=np.float64)

        train_indices = split_indices(dataset, "train")
        val_indices = split_indices(dataset, "val")

        # Subsample for speed
        rng = np.random.default_rng(SEED)
        if len(train_indices) > 30:
            train_indices = sorted(rng.choice(train_indices, size=30, replace=False).tolist())
        if len(val_indices) > 15:
            val_indices = sorted(rng.choice(val_indices, size=15, replace=False).tolist())

        print(f"\n  {dataset.key}: {len(train_indices)} train, {len(val_indices)} val timesteps")

        for split_name, indices, sample_list in [
            ("train", train_indices, all_train_samples),
            ("val", val_indices, all_val_samples),
        ]:
            for t_idx in indices:
                tm_vector = dataset.tm[t_idx]
                if np.max(tm_vector) < 1e-12:
                    continue

                routing = apply_routing(tm_vector, ecmp_base, path_library, capacities)
                telemetry = compute_reactive_telemetry(
                    tm_vector, ecmp_base, path_library, routing,
                    np.asarray(dataset.weights, dtype=float),
                )

                sample = collect_expert_results_per_timestep(
                    dataset, path_library, tm_vector, ecmp_base, capacities, k_crit,
                    telemetry=telemetry,
                    gnn_model=gnn_model,
                    gnn_device=DEVICE,
                    lp_time_limit_sec=15,
                )
                sample_list.append(sample)

        # Print expert wins for this topology
        topo_samples = [s for s in all_train_samples if True]  # all collected so far
        recent = all_train_samples[-len(train_indices):]
        if recent:
            wins = {}
            for s in recent:
                wins[s.best_expert] = wins.get(s.best_expert, 0) + 1
            print(f"    Expert wins: {dict(sorted(wins.items(), key=lambda x: -x[1]))}")

    print(f"\n  Total: {len(all_train_samples)} train, {len(all_val_samples)} val samples")
    return all_train_samples, all_val_samples


# ============================================================
# Step 2: Train meta-gate
# ============================================================

def train_gate(train_samples, val_samples):
    """Train the meta-gate classifier."""
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING META-GATE")
    print("=" * 70)

    # Determine expert names from samples
    all_experts = set()
    for s in train_samples + val_samples:
        all_experts.update(s.expert_mlus.keys())
    expert_names = sorted(all_experts)
    print(f"  Experts: {expert_names}")

    gate_dir = OUTPUT_DIR / "gate"
    summary = train_meta_gate(
        train_samples, val_samples, expert_names, gate_dir,
        lr=1e-3, max_epochs=200, patience=25, seed=SEED,
    )

    print(f"\n  Meta-gate trained:")
    print(f"    Best epoch: {summary.best_epoch}")
    print(f"    Best val accuracy: {summary.best_val_acc:.3f}")
    print(f"    Training time: {summary.training_time_sec:.1f}s")

    return summary


# ============================================================
# Step 3: Evaluate
# ============================================================

def evaluate_unified(bundle, eval_datasets, gnn_model, moe_models, gate_checkpoint):
    """Run unified evaluation comparing meta-selector against all baselines."""
    print("\n" + "=" * 70)
    print("STEP 3: UNIFIED EVALUATION")
    print("=" * 70)

    eval_dir = OUTPUT_DIR / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    gate, expert_names = load_meta_gate(gate_checkpoint)

    all_summaries = []

    for dataset, path_library in eval_datasets:
        print(f"\n--- Evaluating: {dataset.key} ---")
        k_crit = resolve_phase1_k_crit(bundle, dataset)
        env_cfg = build_reactive_env_cfg(bundle, k_crit_override=k_crit)

        topo_dir = eval_dir / dataset.key
        topo_dir.mkdir(parents=True, exist_ok=True)

        frames = []

        # 1. Heuristic baselines
        for method in HEURISTIC_EXPERTS:
            try:
                if method in ("ospf", "ecmp"):
                    df = run_static_method(dataset, path_library, split_name="test", method=method)
                else:
                    df = run_selector_lp_method(
                        dataset, path_library, split_name="test", method=method,
                        k_crit=k_crit, lp_time_limit_sec=20,
                    )
                frames.append(df)
                print(f"  {method:<20}: mean_mlu={df['mlu'].mean():.6f}")
            except Exception as e:
                print(f"  {method:<20}: FAILED ({e})")

        # 2. Static methods
        for method in ["ospf", "ecmp"]:
            try:
                df = run_static_method(dataset, path_library, split_name="test", method=method)
                frames.append(df)
                print(f"  {method:<20}: mean_mlu={df['mlu'].mean():.6f}")
            except Exception as e:
                print(f"  {method:<20}: FAILED ({e})")

        # 3. GNN selector
        if gnn_model is not None:
            try:
                env = ReactiveRoutingEnv(
                    dataset, dataset.tm, path_library,
                    split_name="test", cfg=env_cfg, env_name=dataset.key,
                )
                df_gnn = rollout_gnn_selector_policy(env, gnn_model, device=DEVICE)
                df_gnn["dataset"] = dataset.key
                df_gnn["method"] = GNN_METHOD
                frames.append(df_gnn)
                print(f"  {'GNN_SELECTOR':<20}: mean_mlu={df_gnn['mlu'].mean():.6f}")
            except Exception as e:
                print(f"  {'GNN_SELECTOR':<20}: FAILED ({e})")

        # 4. MoE v3
        if moe_models is not None:
            try:
                env = ReactiveRoutingEnv(
                    dataset, dataset.tm, path_library,
                    split_name="test", cfg=env_cfg, env_name=dataset.key,
                )
                from phase1_reactive.drl.moe_inference import rollout_moe_gate_policy
                df_moe = rollout_moe_gate_policy(
                    env, moe_models[0], moe_models[1], moe_models[2], device=DEVICE,
                )
                df_moe["dataset"] = dataset.key
                df_moe["method"] = "our_hybrid_moe_gate_v3"
                frames.append(df_moe)
                print(f"  {'MoE_v3':<20}: mean_mlu={df_moe['mlu'].mean():.6f}")
            except Exception as e:
                print(f"  {'MoE_v3':<20}: FAILED ({e})")

        # 5. Unified meta-selector (learned gate)
        try:
            env = ReactiveRoutingEnv(
                dataset, dataset.tm, path_library,
                split_name="test", cfg=env_cfg, env_name=dataset.key,
            )
            expert_fns = build_expert_fns(env, gnn_model=gnn_model, moe_models=moe_models)
            # Filter expert_fns to only include experts the gate knows about
            available_experts = [n for n in expert_names if n in expert_fns]
            available_fns = {n: expert_fns[n] for n in available_experts}

            df_meta = rollout_unified_selector(
                env, available_fns, gate, expert_names,
            )
            df_meta["dataset"] = dataset.key
            frames.append(df_meta)

            mean_mlu = df_meta["mlu"].mean()
            expert_counts = df_meta["expert_chosen"].value_counts().to_dict()
            print(f"  {'UNIFIED_META':<20}: mean_mlu={mean_mlu:.6f}  experts={expert_counts}")
        except Exception as e:
            print(f"  {'UNIFIED_META':<20}: FAILED ({e})")
            import traceback
            traceback.print_exc()

        # 6. Oracle (per-timestep best — upper bound)
        try:
            env = ReactiveRoutingEnv(
                dataset, dataset.tm, path_library,
                split_name="test", cfg=env_cfg, env_name=dataset.key,
            )
            expert_fns = build_expert_fns(env, gnn_model=gnn_model, moe_models=moe_models)
            df_oracle = rollout_oracle_selector(env, expert_fns, list(expert_fns.keys()))
            df_oracle["dataset"] = dataset.key
            frames.append(df_oracle)

            mean_mlu = df_oracle["mlu"].mean()
            expert_counts = df_oracle["expert_chosen"].value_counts().to_dict()
            print(f"  {'ORACLE_META':<20}: mean_mlu={mean_mlu:.6f}  experts={expert_counts}")
        except Exception as e:
            print(f"  {'ORACLE_META':<20}: FAILED ({e})")
            import traceback
            traceback.print_exc()

        # Save
        if frames:
            ts = pd.concat(frames, ignore_index=True, sort=False)
            ts.to_csv(topo_dir / "timeseries.csv", index=False)
            summary = summarize_timeseries(
                ts, group_cols=["dataset", "method"],
                training_meta={},
            )
            summary.to_csv(topo_dir / "summary.csv", index=False)
            all_summaries.append(summary)

    if all_summaries:
        summary_all = pd.concat(all_summaries, ignore_index=True, sort=False)
        summary_all.to_csv(eval_dir / "summary_all.csv", index=False)
        return summary_all
    return pd.DataFrame()


# ============================================================
# Step 4: Generalization (Germany50)
# ============================================================

def evaluate_generalization(bundle, gnn_model, moe_models, gate_checkpoint):
    """Test on unseen Germany50 topology."""
    print("\n" + "=" * 70)
    print("STEP 4: GENERALIZATION (Germany50)")
    print("=" * 70)

    gen_dir = OUTPUT_DIR / "eval" / "generalization"
    gen_dir.mkdir(parents=True, exist_ok=True)

    gate, expert_names = load_meta_gate(gate_checkpoint)
    specs = collect_specs(bundle, "generalization_topologies")
    if not specs:
        print("  No generalization topologies. Skipping.")
        return pd.DataFrame()

    frames = []
    for spec in specs:
        try:
            dataset, path_library = load_named_dataset(bundle, spec, MAX_STEPS)
            print(f"  {dataset.key}: {len(dataset.nodes)} nodes")
            k_crit = resolve_phase1_k_crit(bundle, dataset)
            env_cfg = build_reactive_env_cfg(bundle, k_crit_override=k_crit)

            # Heuristics
            for method in HEURISTIC_EXPERTS + ["ospf", "ecmp"]:
                try:
                    if method in ("ospf", "ecmp"):
                        df = run_static_method(dataset, path_library, split_name="test", method=method)
                    else:
                        df = run_selector_lp_method(
                            dataset, path_library, split_name="test", method=method,
                            k_crit=k_crit, lp_time_limit_sec=20,
                        )
                    frames.append(df)
                    print(f"    {method:<20}: mean_mlu={df['mlu'].mean():.6f}")
                except Exception as e:
                    print(f"    {method:<20}: FAILED ({e})")

            # Unified meta
            env = ReactiveRoutingEnv(
                dataset, dataset.tm, path_library,
                split_name="test", cfg=env_cfg, env_name=dataset.key,
            )
            expert_fns = build_expert_fns(env, gnn_model=gnn_model, moe_models=moe_models)
            available_experts = [n for n in expert_names if n in expert_fns]
            available_fns = {n: expert_fns[n] for n in available_experts}

            df_meta = rollout_unified_selector(env, available_fns, gate, expert_names)
            df_meta["dataset"] = dataset.key
            frames.append(df_meta)
            print(f"    {'UNIFIED_META':<20}: mean_mlu={df_meta['mlu'].mean():.6f}  "
                  f"experts={df_meta['expert_chosen'].value_counts().to_dict()}")

            # Oracle
            env = ReactiveRoutingEnv(
                dataset, dataset.tm, path_library,
                split_name="test", cfg=env_cfg, env_name=dataset.key,
            )
            expert_fns = build_expert_fns(env, gnn_model=gnn_model, moe_models=moe_models)
            df_oracle = rollout_oracle_selector(env, expert_fns, list(expert_fns.keys()))
            df_oracle["dataset"] = dataset.key
            frames.append(df_oracle)
            print(f"    {'ORACLE_META':<20}: mean_mlu={df_oracle['mlu'].mean():.6f}  "
                  f"experts={df_oracle['expert_chosen'].value_counts().to_dict()}")

        except Exception as e:
            print(f"  SKIP {spec.key}: {e}")
            import traceback
            traceback.print_exc()

    if frames:
        gen_ts = pd.concat(frames, ignore_index=True, sort=False)
        gen_ts.to_csv(gen_dir / "timeseries.csv", index=False)
        gen_summary = summarize_timeseries(
            gen_ts, group_cols=["dataset", "method"], training_meta={},
        )
        gen_summary.to_csv(gen_dir / "summary.csv", index=False)
        return gen_summary
    return pd.DataFrame()


# ============================================================
# Step 5: Build manifest
# ============================================================

def build_manifest(eval_summary, gen_summary, gate_summary):
    """Build the unified meta-selector manifest."""
    print("\n" + "=" * 70)
    print("STEP 5: BUILDING MANIFEST")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Unified Meta-Selector — Phase-1 Audit Manifest")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Seed: {SEED}")
    lines.append("")

    if gate_summary:
        lines.append("## Meta-Gate Training")
        lines.append(f"- Best epoch: {gate_summary.best_epoch}")
        lines.append(f"- Best val accuracy: {gate_summary.best_val_acc:.3f}")
        lines.append(f"- Experts: {', '.join(gate_summary.expert_names)}")
        lines.append(f"- Training time: {gate_summary.training_time_sec:.1f}s")
        lines.append("")

    if not eval_summary.empty:
        lines.append("## Standard Evaluation Results (Mean MLU)")
        lines.append("")

        methods_to_show = ["our_unified_meta", "our_unified_oracle",
                           "our_gnn_selector", "bottleneck", "cfrrl",
                           "flexdate", "sensitivity", "ecmp", "ospf"]

        header = "| Topology |"
        for m in methods_to_show:
            short = m.replace("our_unified_", "").replace("our_gnn_", "").replace("our_hybrid_moe_gate_", "")
            header += f" {short} |"
        lines.append(header)
        lines.append("|" + "---|" * (len(methods_to_show) + 1))

        for ds in eval_summary["dataset"].unique():
            ds_df = eval_summary[eval_summary["dataset"] == ds]
            row = f"| {ds} |"
            for method in methods_to_show:
                m_df = ds_df[ds_df["method"] == method]
                if not m_df.empty and "mean_mlu" in m_df.columns:
                    val = m_df["mean_mlu"].values[0]
                    row += f" {val:.6f} |"
                else:
                    row += " N/A |"
            lines.append(row)
        lines.append("")

    if not gen_summary.empty:
        lines.append("## Generalization Results (Germany50)")
        lines.append("")
        lines.append("| Method | Mean MLU |")
        lines.append("|--------|----------|")
        for method in gen_summary["method"].unique():
            m_df = gen_summary[gen_summary["method"] == method]
            if not m_df.empty and "mean_mlu" in m_df.columns:
                mlu = m_df["mean_mlu"].values[0]
                lines.append(f"| {method} | {mlu:.6f} |")
        lines.append("")

    manifest_path = OUTPUT_DIR / "unified_meta_manifest.md"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Manifest: {manifest_path}")
    return manifest_path


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("UNIFIED META-SELECTOR: FULL PIPELINE")
    print("=" * 70)
    total_start = time.perf_counter()

    # Load datasets
    print("\nLoading datasets...")
    bundle = load_bundle(CONFIG_PATH)
    max_steps = max_steps_from_args(bundle, MAX_STEPS)

    print("Training topologies:")
    train_specs = collect_specs(bundle, "train_topologies")
    train_datasets = []
    for spec in train_specs:
        try:
            dataset, path_library = load_named_dataset(bundle, spec, max_steps)
            train_datasets.append((dataset, path_library))
            print(f"  {dataset.key}: {len(dataset.nodes)} nodes, {len(dataset.od_pairs)} OD pairs")
        except Exception as e:
            print(f"  SKIP {spec.key}: {e}")

    print("Eval topologies:")
    eval_specs = collect_specs(bundle, "eval_topologies")
    eval_datasets = []
    for spec in eval_specs:
        try:
            dataset, path_library = load_named_dataset(bundle, spec, max_steps)
            eval_datasets.append((dataset, path_library))
            print(f"  {dataset.key}: {len(dataset.nodes)} nodes, {len(dataset.od_pairs)} OD pairs")
        except Exception as e:
            print(f"  SKIP {spec.key}: {e}")

    # Load GNN model
    gnn_model = None
    if GNN_CHECKPOINT.exists():
        print(f"\nLoading GNN: {GNN_CHECKPOINT}")
        gnn_model, _ = load_gnn_selector(GNN_CHECKPOINT, device=DEVICE)
        gnn_model.eval()
    else:
        print(f"\nGNN checkpoint not found: {GNN_CHECKPOINT}")

    # Load MoE v3 models
    moe_models = None
    if PPO_CHECKPOINT.exists() and DQN_CHECKPOINT.exists() and MOE_GATE_CHECKPOINT.exists():
        print(f"Loading MoE v3 models...")
        from phase1_reactive.drl.drl_selector import load_trained_ppo
        from phase1_reactive.drl.dqn_selector import load_trained_dqn
        from phase1_reactive.drl.moe_gate import MoeGateNet
        try:
            ppo = load_trained_ppo(PPO_CHECKPOINT, device=DEVICE)
            dqn = load_trained_dqn(DQN_CHECKPOINT, device=DEVICE)
            # Load MoE gate with compatibility for old checkpoint format
            ckpt = torch.load(MOE_GATE_CHECKPOINT, map_location=DEVICE)
            sd = ckpt["state_dict"]
            # Check if old sequential format (net.0.weight) vs new residual format
            if any(k.startswith("net.") for k in sd):
                # Old format: build a simple Sequential gate
                gate = nn.Sequential(
                    nn.Linear(ckpt["input_dim"], ckpt["moe_config"]["hidden_dim"]),
                    nn.ReLU(),
                    nn.Linear(ckpt["moe_config"]["hidden_dim"], ckpt["moe_config"]["hidden_dim"]),
                    nn.ReLU(),
                    nn.Linear(ckpt["moe_config"]["hidden_dim"], ckpt["num_experts"]),
                )
                gate.load_state_dict(sd)
                gate.eval()
                # Wrap to have .weights() method matching MoeGateNet interface
                class GateWrapper:
                    def __init__(self, net, num_experts):
                        self.net = net
                        self.num_experts = num_experts
                    def eval(self): self.net.eval(); return self
                    def __call__(self, x): return self.net(x)
                    def weights(self, x): return F.softmax(self.net(x), dim=-1)
                    def parameters(self): return self.net.parameters()
                gate = GateWrapper(gate, ckpt["num_experts"])
            else:
                gate = MoeGateNet(ckpt["input_dim"], ckpt["num_experts"],
                                  hidden_dim=ckpt["moe_config"]["hidden_dim"])
                gate.load_state_dict(sd)
                gate.eval()
            moe_models = (ppo, dqn, gate)
            print("  MoE v3 loaded successfully")
        except Exception as e:
            print(f"  MoE v3 load failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        missing = []
        if not PPO_CHECKPOINT.exists(): missing.append("PPO")
        if not DQN_CHECKPOINT.exists(): missing.append("DQN")
        if not MOE_GATE_CHECKPOINT.exists(): missing.append("Gate")
        print(f"  MoE v3 checkpoints missing: {missing}")

    # Step 1: Collect training data
    train_samples, val_samples = collect_training_data(
        bundle, train_datasets, gnn_model, moe_models,
    )

    if not train_samples:
        print("ERROR: No training samples collected. Aborting.")
        sys.exit(1)

    # Step 2: Train meta-gate
    gate_summary = train_gate(train_samples, val_samples)

    # Step 3: Evaluate
    eval_summary = evaluate_unified(
        bundle, eval_datasets, gnn_model, moe_models, gate_summary.checkpoint,
    )

    # Step 4: Generalization
    gen_summary = evaluate_generalization(
        bundle, gnn_model, moe_models, gate_summary.checkpoint,
    )

    # Step 5: Manifest
    manifest_path = build_manifest(eval_summary, gen_summary, gate_summary)

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 70)
    print("UNIFIED PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
