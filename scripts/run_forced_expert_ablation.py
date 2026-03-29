#!/usr/bin/env python3
"""Forced-Expert Ablation: Same unified pipeline, gate disabled.

Runs the unified pipeline with each expert forced for all timesteps.
This isolates gate contribution from pipeline-level effects.

Usage:
  python scripts/run_forced_expert_ablation.py
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
    load_meta_gate,
    rollout_unified_selector,
    rollout_forced_expert_selector,
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
from phase1_reactive.eval.core import split_indices
from phase1_reactive.eval.metrics import summarize_timeseries
from phase1_reactive.env.offline_env import ReactiveRoutingEnv

# Import build_expert_fns from the main pipeline
from scripts.run_unified_pipeline import (
    build_expert_fns,
    CONFIG_PATH,
    MAX_STEPS,
    SEED,
    DEVICE,
    OUTPUT_DIR,
    GNN_CHECKPOINT,
    PPO_CHECKPOINT,
    DQN_CHECKPOINT,
    MOE_GATE_CHECKPOINT,
    HEURISTIC_EXPERTS,
)

# Experts to force in ablation
FORCED_EXPERTS = ["bottleneck", "sensitivity", "gnn"]

ABLATION_DIR = OUTPUT_DIR / "forced_expert_ablation"


def run_ablation():
    """Run the forced-expert ablation on all eval + generalization topologies."""
    print("=" * 70)
    print("FORCED-EXPERT ABLATION: SAME PIPELINE, GATE DISABLED")
    print("=" * 70)
    total_start = time.perf_counter()

    # Load datasets
    bundle = load_bundle(CONFIG_PATH)
    max_steps = max_steps_from_args(bundle, MAX_STEPS)

    # Collect all topologies (eval + generalization)
    all_datasets = []

    print("\nEval topologies:")
    eval_specs = collect_specs(bundle, "eval_topologies")
    for spec in eval_specs:
        try:
            dataset, path_library = load_named_dataset(bundle, spec, max_steps)
            all_datasets.append((dataset, path_library))
            print(f"  {dataset.key}: {len(dataset.nodes)} nodes")
        except Exception as e:
            print(f"  SKIP {spec.key}: {e}")

    print("\nGeneralization topologies:")
    gen_specs = collect_specs(bundle, "generalization_topologies")
    for spec in gen_specs:
        try:
            dataset, path_library = load_named_dataset(bundle, spec, max_steps)
            all_datasets.append((dataset, path_library))
            print(f"  {dataset.key}: {len(dataset.nodes)} nodes")
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
        print("Loading MoE v3 models...")
        from phase1_reactive.drl.drl_selector import load_trained_ppo
        from phase1_reactive.drl.dqn_selector import load_trained_dqn
        from phase1_reactive.drl.moe_gate import MoeGateNet
        try:
            ppo = load_trained_ppo(PPO_CHECKPOINT, device=DEVICE)
            dqn = load_trained_dqn(DQN_CHECKPOINT, device=DEVICE)
            ckpt = torch.load(MOE_GATE_CHECKPOINT, map_location=DEVICE)
            sd = ckpt["state_dict"]
            if any(k.startswith("net.") for k in sd):
                gate = nn.Sequential(
                    nn.Linear(ckpt["input_dim"], ckpt["moe_config"]["hidden_dim"]),
                    nn.ReLU(),
                    nn.Linear(ckpt["moe_config"]["hidden_dim"], ckpt["moe_config"]["hidden_dim"]),
                    nn.ReLU(),
                    nn.Linear(ckpt["moe_config"]["hidden_dim"], ckpt["num_experts"]),
                )
                gate.load_state_dict(sd)
                gate.eval()
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
    else:
        print("  MoE v3 checkpoints not all available")

    # Load meta-gate for learned-gate comparison
    gate_checkpoint = OUTPUT_DIR / "gate" / "meta_gate.pt"
    meta_gate = None
    meta_expert_names = None
    if gate_checkpoint.exists():
        meta_gate, meta_expert_names = load_meta_gate(gate_checkpoint)
        print(f"\nMeta-gate loaded: {len(meta_expert_names)} experts: {meta_expert_names}")
    else:
        print(f"\nMeta-gate checkpoint not found: {gate_checkpoint}")

    # Run ablation
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for dataset, path_library in all_datasets:
        print(f"\n{'='*60}")
        print(f"TOPOLOGY: {dataset.key} ({len(dataset.nodes)} nodes)")
        print(f"{'='*60}")

        k_crit = resolve_phase1_k_crit(bundle, dataset)
        env_cfg = build_reactive_env_cfg(bundle, k_crit_override=k_crit)

        topo_results = {"topology": dataset.key, "nodes": len(dataset.nodes)}

        # Run each forced expert through unified pipeline
        for expert_name in FORCED_EXPERTS:
            try:
                env = ReactiveRoutingEnv(
                    dataset, dataset.tm, path_library,
                    split_name="test", cfg=env_cfg, env_name=dataset.key,
                )
                expert_fns = build_expert_fns(env, gnn_model=gnn_model, moe_models=moe_models)

                if expert_name not in expert_fns:
                    print(f"  forced_{expert_name:<15}: NOT AVAILABLE")
                    topo_results[f"forced_{expert_name}"] = None
                    continue

                df = rollout_forced_expert_selector(env, expert_fns, expert_name)
                mean_mlu = df["mlu"].mean()
                topo_results[f"forced_{expert_name}"] = mean_mlu
                print(f"  forced_{expert_name:<15}: mean_mlu={mean_mlu:.6f}")

                # Save timeseries
                topo_dir = ABLATION_DIR / dataset.key
                topo_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(topo_dir / f"forced_{expert_name}_timeseries.csv", index=False)

            except Exception as e:
                print(f"  forced_{expert_name:<15}: FAILED ({e})")
                topo_results[f"forced_{expert_name}"] = None
                import traceback
                traceback.print_exc()

        # Run learned gate through unified pipeline
        if meta_gate is not None and meta_expert_names is not None:
            try:
                env = ReactiveRoutingEnv(
                    dataset, dataset.tm, path_library,
                    split_name="test", cfg=env_cfg, env_name=dataset.key,
                )
                expert_fns = build_expert_fns(env, gnn_model=gnn_model, moe_models=moe_models)
                available_experts = [n for n in meta_expert_names if n in expert_fns]
                available_fns = {n: expert_fns[n] for n in available_experts}

                df = rollout_unified_selector(env, available_fns, meta_gate, meta_expert_names)
                mean_mlu = df["mlu"].mean()
                expert_counts = df["expert_chosen"].value_counts().to_dict()
                topo_results["learned_gate"] = mean_mlu
                topo_results["gate_expert_picks"] = str(expert_counts)
                print(f"  learned_gate       : mean_mlu={mean_mlu:.6f}  experts={expert_counts}")

                topo_dir = ABLATION_DIR / dataset.key
                topo_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(topo_dir / "learned_gate_timeseries.csv", index=False)

            except Exception as e:
                print(f"  learned_gate       : FAILED ({e})")
                topo_results["learned_gate"] = None
                import traceback
                traceback.print_exc()

        all_results.append(topo_results)

    # Build summary table
    print("\n" + "=" * 70)
    print("FORCED-EXPERT ABLATION RESULTS (Same Unified Pipeline)")
    print("=" * 70)

    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(ABLATION_DIR / "ablation_summary.csv", index=False)

    # Print analysis
    print("\n" + "-" * 70)
    print("GAIN ATTRIBUTION ANALYSIS")
    print("-" * 70)
    for r in all_results:
        topo = r["topology"]
        forced_bn = r.get("forced_bottleneck")
        forced_sens = r.get("forced_sensitivity")
        forced_gnn = r.get("forced_gnn")
        learned = r.get("learned_gate")

        if forced_bn is not None and learned is not None:
            forced_vals = {}
            if forced_bn is not None: forced_vals["BN"] = forced_bn
            if forced_sens is not None: forced_vals["Sens"] = forced_sens
            if forced_gnn is not None: forced_vals["GNN"] = forced_gnn

            best_forced_name = min(forced_vals, key=forced_vals.get)
            best_forced_val = forced_vals[best_forced_name]

            gate_vs_best = ((best_forced_val - learned) / best_forced_val) * 100

            print(f"\n  {topo}:")
            for name, val in forced_vals.items():
                print(f"    Forced {name:>5}: {val:.6f}")
            print(f"    Learned Gate : {learned:.6f}")
            print(f"    Best Forced  : {best_forced_name} ({best_forced_val:.6f})")
            if gate_vs_best > 0:
                print(f"    Gate Advantage: +{gate_vs_best:.2f}% (GATE WIN)")
            elif gate_vs_best < 0:
                print(f"    Gate Deficit  : {gate_vs_best:.2f}% (FORCED EXPERT WINS)")
            else:
                print(f"    Gate Advantage: 0.00% (TIE)")

    total_time = time.perf_counter() - total_start
    print(f"\nTotal ablation time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {ABLATION_DIR}")


if __name__ == "__main__":
    run_ablation()
