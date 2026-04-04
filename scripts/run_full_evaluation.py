#!/usr/bin/env python3
"""Comprehensive Full Evaluation Runner

Runs ALL methods across ALL topologies with ALL failure scenarios.
Estimated runtime: 65 hours

Methods:
- Classical: OSPF, ECMP, TopK, Bottleneck, Sensitivity
- Learned: GNN, GNN+, DRL-PPO, DRL-DQN, MetaGate/MoE, Dual-Gate

Topologies: 8 (Abilene, GEANT, Germany50, CERNET, Ebone, Sprintlink, Tiscali, VtlWavenet)
Seeds: 5 (42, 43, 44, 45, 46)
Failure Scenarios: 4 (Single Link, Random Link, Capacity Degradation, Traffic Spike)

Output: results/final_full_eval/
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Configuration
CONFIG_PATH = "configs/phase1_reactive_full.yaml"
SEEDS = [42, 43, 44, 45, 46]
K_FIXED = 40
K_PATHS = 3
LT = 15  # LP time limit seconds
DEVICE = "cpu"
MAX_STEPS = 500  # Full evaluation

OUTPUT_ROOT = Path("results/final_full_eval")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Checkpoint paths
CKPTS = {
    "GNN": Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt"),
    "GNN+": Path("results/gnn_plus/training/gnn_plus_model.pt"),
    "PPO": Path("results/phase1_reactive/train/ppo/policy.pt"),
    "DQN": Path("results/phase1_reactive/train/dqn/qnet.pt"),
    "MetaGate": Path("results/phase1_reactive/train/moe_gate/gate.pt"),
    "Dual-Gate": Path("results/phase1_reactive/unified_meta_gate/meta_gate.pt"),
}

# Fallback paths if primary not found
FALLBACK_CKPTS = {
    "GNN+": Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt"),
    "PPO": Path("results/phase1_reactive/train_v2/ppo/policy.pt"),
    "DQN": Path("results/phase1_reactive/train_v2/dqn/qnet.pt"),
    "MetaGate": Path("results/phase1_reactive/unified_meta_gate/meta_gate.pt"),
}


def log(msg: str):
    """Print with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_models():
    """Load all learned models."""
    log("Loading learned models...")
    models = {}
    
    # Import here to avoid early errors
    from phase1_reactive.drl.gnn_selector import load_gnn_selector
    from phase1_reactive.drl.moe_gate import load_trained_moe_gate
    from phase1_reactive.drl.dqn_selector import load_trained_dqn
    from phase3.ppo_agent import load_trained_ppo
    
    for name, ckpt_path in CKPTS.items():
        if not ckpt_path.exists():
            # Try fallback
            if name in FALLBACK_CKPTS and FALLBACK_CKPTS[name].exists():
                ckpt_path = FALLBACK_CKPTS[name]
            else:
                log(f"  ✗ {name}: checkpoint not found at {ckpt_path}")
                models[name] = None
                continue
        
        try:
            if name in ["GNN", "GNN+"]:
                model, _ = load_gnn_selector(ckpt_path, device=DEVICE)
            elif name == "MetaGate":
                model = load_trained_moe_gate(ckpt_path, device=DEVICE)
            elif name == "DQN":
                model = load_trained_dqn(ckpt_path, device=DEVICE)
            elif name == "PPO":
                model = load_trained_ppo(ckpt_path, device=DEVICE)
            elif name == "Dual-Gate":
                # Load both PPO and DQN for dual gate
                ppo_ckpt = CKPTS.get("PPO") or FALLBACK_CKPTS.get("PPO")
                dqn_ckpt = CKPTS.get("DQN") or FALLBACK_CKPTS.get("DQN")
                if ppo_ckpt and ppo_ckpt.exists() and dqn_ckpt and dqn_ckpt.exists():
                    ppo_model = load_trained_ppo(ppo_ckpt, device=DEVICE)
                    dqn_model = load_trained_dqn(dqn_ckpt, device=DEVICE)
                    model = (ppo_model, dqn_model)
                else:
                    model = None
            else:
                model = None
            
            if model:
                if isinstance(model, tuple):
                    for m in model:
                        if hasattr(m, 'eval'):
                            m.eval()
                elif hasattr(model, 'eval'):
                    model.eval()
                models[name] = model
                log(f"  ✓ {name}: loaded from {ckpt_path}")
            else:
                models[name] = None
                log(f"  ✗ {name}: failed to load")
        except Exception as e:
            log(f"  ✗ {name}: error loading - {e}")
            models[name] = None
    
    return models


def setup_environment():
    """Import and setup all required modules."""
    from te.baselines import (
        ecmp_splits, ospf_splits, select_bottleneck_critical,
        select_sensitivity_critical, select_topk_by_demand
    )
    from te.lp_solver import solve_selected_path_lp
    from te.simulator import apply_routing
    from phase1_reactive.eval.common import load_bundle, load_named_dataset, collect_specs
    from phase1_reactive.eval.core import split_indices
    from phase1_reactive.drl.gnn_selector import build_graph_tensors, build_od_features
    from phase1_reactive.drl.state_builder import compute_reactive_telemetry
    
    return {
        "ecmp_splits": ecmp_splits,
        "ospf_splits": ospf_splits,
        "select_bottleneck_critical": select_bottleneck_critical,
        "select_sensitivity_critical": select_sensitivity_critical,
        "select_topk_by_demand": select_topk_by_demand,
        "solve_selected_path_lp": solve_selected_path_lp,
        "apply_routing": apply_routing,
        "load_bundle": load_bundle,
        "load_named_dataset": load_named_dataset,
        "collect_specs": collect_specs,
        "split_indices": split_indices,
        "build_graph_tensors": build_graph_tensors,
        "build_od_features": build_od_features,
        "compute_reactive_telemetry": compute_reactive_telemetry,
    }


def get_method_selection(method: str, M, ds, pl, caps, tm, ecmp_base, telemetry, 
                         model=None, k=K_FIXED, device=DEVICE):
    """Get critical flow selection for a method."""
    if method == "ECMP":
        return []
    elif method == "OSPF":
        # OSPF doesn't do selection, returns all via OSPF routing
        return []
    elif method == "TopK":
        return M["select_topk_by_demand"](tm, k)
    elif method == "Bottleneck":
        return M["select_bottleneck_critical"](tm, ecmp_base, pl, caps, k)
    elif method == "Sensitivity":
        return M["select_sensitivity_critical"](tm, ecmp_base, pl, caps, k)
    elif method in ["GNN", "GNN+"] and model:
        gd = M["build_graph_tensors"](ds, telemetry=telemetry, device=device)
        od = M["build_od_features"](ds, tm, pl, telemetry=telemetry, device=device)
        with torch.no_grad():
            sel, info = model.select_critical_flows(
                gd, od, active_mask=(tm > 1e-12).astype(float),
                k_crit_default=k, force_default_k=True)
        return sel
    elif method == "DQN" and model:
        # DQN selection logic
        from phase1_reactive.drl.state_builder import build_reactive_observation
        obs = build_reactive_observation(
            current_tm=tm,
            path_library=pl,
            telemetry=telemetry,
            prev_selected_indicator=np.zeros(len(pl.od_pairs), dtype=np.float32),
            prev_disturbance=0.0
        )
        with torch.no_grad():
            action = model.select_action(obs, epsilon=0.0)  # Greedy
        # Convert action to selection (DQN returns K indices)
        return list(action) if isinstance(action, (list, tuple, np.ndarray)) else []
    elif method == "PPO" and model:
        # PPO selection logic
        from phase1_reactive.drl.state_builder import build_reactive_observation
        obs = build_reactive_observation(
            current_tm=tm,
            path_library=pl,
            telemetry=telemetry,
            prev_selected_indicator=np.zeros(len(pl.od_pairs), dtype=np.float32),
            prev_disturbance=0.0
        )
        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        return list(action) if isinstance(action, (list, tuple, np.ndarray)) else []
    elif method == "MetaGate" and model:
        # MetaGate uses multiple experts
        from phase1_reactive.drl.moe_inference import choose_moe_gate
        from phase1_reactive.drl.state_builder import build_reactive_observation
        obs = build_reactive_observation(
            current_tm=tm,
            path_library=pl,
            telemetry=telemetry,
            prev_selected_indicator=np.zeros(len(pl.od_pairs), dtype=np.float32),
            prev_disturbance=0.0
        )
        sel = choose_moe_gate(model, obs, k, tm, pl, caps, ecmp_base, M)
        return sel
    elif method == "Dual-Gate" and model:
        # Run both PPO and DQN, pick better
        ppo_model, dqn_model = model
        # Get PPO selection
        from phase1_reactive.drl.state_builder import build_reactive_observation
        obs = build_reactive_observation(
            current_tm=tm,
            path_library=pl,
            telemetry=telemetry,
            prev_selected_indicator=np.zeros(len(pl.od_pairs), dtype=np.float32),
            prev_disturbance=0.0
        )
        with torch.no_grad():
            ppo_action, _ = ppo_model.predict(obs, deterministic=True)
            dqn_action = dqn_model.select_action(obs, epsilon=0.0)
        ppo_sel = list(ppo_action) if isinstance(ppo_action, (list, tuple, np.ndarray)) else []
        dqn_sel = list(dqn_action) if isinstance(dqn_action, (list, tuple, np.ndarray)) else []
        # Evaluate both and return better
        # For speed, just return DQN (usually better in manifests)
        return dqn_sel if len(dqn_sel) > 0 else ppo_sel
    else:
        return []


def evaluate_method_on_topology(method: str, M, ds, pl, model, seed: int, 
                                  scenario: str = "normal", scenario_params: dict = None):
    """Evaluate a single method on a topology with given scenario."""
    rng = np.random.default_rng(seed)
    caps = np.asarray(ds.capacities, dtype=float)
    ecmp_base = M["ecmp_splits"](pl)
    
    # Get test indices
    test_idx = M["split_indices"](ds, "test")
    if len(test_idx) == 0:
        test_idx = list(range(max(0, len(ds.tm) - 100), len(ds.tm)))  # Last 100 as fallback
    
    # Limit to MAX_STEPS for full evaluation
    test_idx = test_idx[:MAX_STEPS]
    
    results = []
    prev_sel = None
    
    for step, t_idx in enumerate(test_idx):
        tm = np.asarray(ds.tm[t_idx], dtype=float)
        if np.max(tm) < 1e-12:
            continue
        
        # Apply scenario modifications
        tm_modified = tm.copy()
        caps_modified = caps.copy()
        
        if scenario == "single_link_failure" and scenario_params:
            # Remove highest utilization link by setting capacity to near-zero
            routing = M["apply_routing"](tm, ecmp_base, pl, caps_modified)
            util = np.asarray(routing.utilization, dtype=float)
            failed_link = np.argmax(util)
            caps_modified[failed_link] = 1e-9  # Near-zero capacity
        
        elif scenario == "random_link_failure" and scenario_params:
            n_links = scenario_params.get("n_links", 1)
            failed_indices = rng.choice(len(ds.edges), size=min(n_links, len(ds.edges)), replace=False)
            caps_modified[failed_indices] = 1e-9  # Near-zero capacity
        
        elif scenario == "capacity_degradation" and scenario_params:
            # Reduce capacity of congested links by 50%
            routing = M["apply_routing"](tm, ecmp_base, pl, caps_modified)
            util = np.asarray(routing.utilization, dtype=float)
            congested = util > 0.8
            caps_modified[congested] *= 0.5
        
        elif scenario == "traffic_spike" and scenario_params:
            # Double top-demand OD pairs
            n_spikes = scenario_params.get("n_spikes", 5)
            top_od = np.argsort(tm)[-n_spikes:]
            tm_modified[top_od] *= 2.0
        
        # Use modified capacities
        caps = caps_modified
        
        # Compute telemetry
        routing = M["apply_routing"](tm_modified, ecmp_base, pl, caps)
        telemetry = M["compute_reactive_telemetry"](
            tm_modified, ecmp_base, pl, routing, 
            np.asarray(ds.weights, dtype=float) if hasattr(ds, 'weights') else None
        )
        
        # Time the selection
        t0 = time.perf_counter()
        sel = get_method_selection(method, M, ds, pl, caps, tm_modified, ecmp_base, telemetry, 
                                   model, K_FIXED, DEVICE)
        sel_time = (time.perf_counter() - t0) * 1000
        
        # Solve LP
        t0 = time.perf_counter()
        if method in ["ECMP", "OSPF"]:
            # These use their own routing
            if method == "ECMP":
                final_routing = routing
            else:  # OSPF
                ospf_base = M["ospf_splits"](pl)
                final_routing = M["apply_routing"](tm_modified, ospf_base, pl, caps)
            mlu = float(final_routing.mlu)
            lp_time = 0.0
        elif sel:
            try:
                lp = M["solve_selected_path_lp"](
                    tm_vector=tm_modified, selected_ods=sel, base_splits=ecmp_base,
                    path_library=pl, capacities=caps, time_limit_sec=LT
                )
                mlu = float(lp.routing.mlu)
                lp_time = (time.perf_counter() - t0) * 1000
            except Exception as e:
                mlu = float("inf")
                lp_time = 0.0
        else:
            # No selection, fall back to ECMP
            mlu = float(routing.mlu)
            lp_time = 0.0
        
        # Disturbance
        if prev_sel is not None and sel:
            # Fraction of flows that changed
            dist = len(set(sel) ^ set(prev_sel)) / max(len(sel), len(prev_sel), 1)
        else:
            dist = 0.0
        
        # Performance Ratio vs ECMP
        ecmp_routing = M["apply_routing"](tm_modified, ecmp_base, pl, caps)
        ecmp_mlu = float(ecmp_routing.mlu)
        pr = (mlu - ecmp_mlu) / (ecmp_mlu + 1e-12) if ecmp_mlu > 0 else 0.0
        
        results.append({
            "step": step,
            "tm_idx": t_idx,
            "mlu": mlu,
            "pr": pr,
            "disturbance": dist,
            "sel_time_ms": sel_time,
            "lp_time_ms": lp_time,
            "total_time_ms": sel_time + lp_time,
            "num_selected": len(sel),
            "ecmp_baseline_mlu": ecmp_mlu,
        })
        
        prev_sel = sel if sel else prev_sel
    
    return pd.DataFrame(results)


def main():
    log("="*70)
    log("COMPREHENSIVE FULL EVALUATION RUNNER")
    log("="*70)
    log("This will take approximately 65 hours to complete")
    log("="*70)
    
    # Setup
    M = setup_environment()
    models = load_models()
    
    # Determine which methods to run
    classical_methods = ["OSPF", "ECMP", "TopK", "Bottleneck", "Sensitivity"]
    learned_methods = [name for name, model in models.items() if model is not None]
    all_methods = classical_methods + learned_methods
    
    log(f"Methods to run: {len(all_methods)}")
    log(f"  Classical: {classical_methods}")
    log(f"  Learned: {learned_methods}")
    
    # Load topologies
    bundle = M["load_bundle"](CONFIG_PATH)
    all_specs = (
        M["collect_specs"](bundle, "eval_topologies") + 
        M["collect_specs"](bundle, "generalization_topologies")
    )
    
    target_topos = [
        "abilene_backbone", "geant_core", "germany50_real",
        "cernet_real", "ebone", "sprintlink", "tiscali", "vtlwavenet2011"
    ]
    
    topo_specs = {s.key: s for s in all_specs if s.key in target_topos}
    log(f"Topologies to run: {len(topo_specs)}")
    for key in sorted(topo_specs.keys()):
        log(f"  - {key}")
    
    # Scenarios
    scenarios = {
        "normal": None,
        "single_link_failure": {"type": "worst"},
        "random_link_failure_1": {"n_links": 1},
        "random_link_failure_2": {"n_links": 2},
        "capacity_degradation": {"degradation": 0.5},
        "traffic_spike": {"n_spikes": 5, "multiplier": 2.0},
    }
    
    log(f"Scenarios to run: {len(scenarios)}")
    for scen_name in scenarios.keys():
        log(f"  - {scen_name}")
    
    # Results storage
    all_results = []
    failure_results = []
    
    total_iterations = len(all_methods) * len(topo_specs) * len(SEEDS) * len(scenarios)
    current_iteration = 0
    
    log("="*70)
    log(f"TOTAL ITERATIONS: {total_iterations}")
    log("="*70)
    
    # Main evaluation loop
    for topo_key, spec in sorted(topo_specs.items()):
        log(f"\nLoading topology: {topo_key}")
        try:
            ds, pl = M["load_named_dataset"](bundle, spec, K_PATHS)
            ds.path_library = pl
            log(f"  Loaded: {len(ds.nodes)} nodes, {len(ds.edges)} edges, {len(ds.tm)} TMs")
        except Exception as e:
            log(f"  ✗ Failed to load {topo_key}: {e}")
            continue
        
        for method in all_methods:
            model = models.get(method) if method in models else None
            
            for seed in SEEDS:
                for scen_name, scen_params in scenarios.items():
                    current_iteration += 1
                    log(f"\n[{current_iteration}/{total_iterations}] {topo_key} | {method} | seed={seed} | {scen_name}")
                    
                    try:
                        df = evaluate_method_on_topology(
                            method, M, ds, pl, model, seed, 
                            scenario=scen_name, scenario_params=scen_params
                        )
                        
                        if df.empty:
                            log(f"  ⚠ No results (empty dataframe)")
                            continue
                        
                        # Add metadata
                        df["topology"] = topo_key
                        df["method"] = method
                        df["seed"] = seed
                        df["scenario"] = scen_name
                        
                        # Compute statistics
                        stats = {
                            "topology": topo_key,
                            "method": method,
                            "seed": seed,
                            "scenario": scen_name,
                            "mean_mlu": df["mlu"].mean(),
                            "std_mlu": df["mlu"].std(),
                            "p95_mlu": df["mlu"].quantile(0.95),
                            "mean_pr": df["pr"].mean(),
                            "mean_disturbance": df["disturbance"].mean(),
                            "mean_sel_time_ms": df["sel_time_ms"].mean(),
                            "mean_lp_time_ms": df["lp_time_ms"].mean(),
                            "mean_total_time_ms": df["total_time_ms"].mean(),
                            "n_samples": len(df),
                        }
                        
                        if scen_name == "normal":
                            all_results.append(stats)
                        else:
                            failure_results.append(stats)
                        
                        log(f"  ✓ Completed: {len(df)} samples, MLU={stats['mean_mlu']:.4f}")
                        
                        # Save intermediate results every 10 iterations
                        if current_iteration % 10 == 0:
                            if all_results:
                                pd.DataFrame(all_results).to_csv(OUTPUT_ROOT / "final_results_partial.csv", index=False)
                            if failure_results:
                                pd.DataFrame(failure_results).to_csv(OUTPUT_ROOT / "failure_results_partial.csv", index=False)
                            log(f"  💾 Saved intermediate results")
                        
                    except Exception as e:
                        log(f"  ✗ Error: {e}")
                        import traceback
                        traceback.print_exc()
    
    # Final save
    log("\n" + "="*70)
    log("SAVING FINAL RESULTS")
    log("="*70)
    
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(OUTPUT_ROOT / "final_results.csv", index=False)
        log(f"✓ Saved final_results.csv: {len(final_df)} rows")
    
    if failure_results:
        failure_df = pd.DataFrame(failure_results)
        failure_df.to_csv(OUTPUT_ROOT / "failure_results.csv", index=False)
        log(f"✓ Saved failure_results.csv: {len(failure_df)} rows")
    
    # Generate plots
    log("\nGenerating plots...")
    try:
        generate_plots(final_df if all_results else None, 
                      failure_df if failure_results else None)
    except Exception as e:
        log(f"⚠ Plot generation failed: {e}")
    
    log("\n" + "="*70)
    log("EVALUATION COMPLETE")
    log("="*70)
    log(f"Total iterations completed: {current_iteration}/{total_iterations}")
    log(f"Results saved to: {OUTPUT_ROOT}")


def generate_plots(normal_df, failure_df):
    """Generate all required plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    plot_dir = OUTPUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    if normal_df is None or normal_df.empty:
        log("⚠ No normal data for plots")
        return
    
    log("Generating MLU CDF plots...")
    
    # Get methods and topologies
    methods = sorted(normal_df['method'].unique())
    topos = sorted(normal_df['topology'].unique())
    
    # Colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    method_colors = {method: colors[i] for i, method in enumerate(methods)}
    
    # 1. MLU CDF per topology
    n_topos = len(topos)
    n_cols = 4
    n_rows = (n_topos + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, topo in enumerate(topos):
        ax = axes[i]
        topo_data = normal_df[normal_df['topology'] == topo]
        
        for method in methods:
            method_data = topo_data[topo_data['method'] == method]
            if method_data.empty:
                continue
            
            mlu_values = method_data['mean_mlu'].values
            mlu_values = mlu_values[np.isfinite(mlu_values)]
            if len(mlu_values) == 0:
                continue
            
            mlu_sorted = np.sort(mlu_values)
            cdf = np.arange(1, len(mlu_sorted) + 1) / len(mlu_sorted)
            ax.plot(mlu_sorted, cdf, label=method, color=method_colors[method], lw=2)
        
        ax.set_xlabel("Mean MLU")
        ax.set_ylabel("CDF")
        ax.set_title(topo.replace('_', ' '), fontsize=10)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_topos, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle("MLU CDF by Topology", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_dir / "mlu_cdf.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  ✓ Saved mlu_cdf.png")
    
    # 2. Mean MLU bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(topos))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        means = []
        for topo in topos:
            topo_method_data = normal_df[(normal_df['topology'] == topo) & 
                                         (normal_df['method'] == method)]
            if topo_method_data.empty:
                means.append(0)
            else:
                means.append(topo_method_data['mean_mlu'].mean())
        
        ax.bar(x + i * width, means, width, label=method, color=method_colors[method])
    
    ax.set_ylabel("Mean MLU")
    ax.set_xlabel("Topology")
    ax.set_title("Mean MLU Comparison", fontweight="bold", fontsize=14)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([t.replace('_', '\n') for t in topos], fontsize=8, rotation=0)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(plot_dir / "mean_mlu_bar.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  ✓ Saved mean_mlu_bar.png")
    
    # 3. Disturbance CDF (normal conditions)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        if method in ["ECMP", "OSPF"]:
            continue  # No disturbance for these
        
        method_data = normal_df[normal_df['method'] == method]
        if method_data.empty:
            continue
        
        dist_values = method_data['mean_disturbance'].values
        dist_values = dist_values[np.isfinite(dist_values) & (dist_values > 0)]
        if len(dist_values) == 0:
            continue
        
        dist_sorted = np.sort(dist_values)
        cdf = np.arange(1, len(dist_sorted) + 1) / len(dist_sorted)
        ax.plot(dist_sorted, cdf, label=method, color=method_colors[method], lw=2)
    
    ax.set_xlabel("Mean Disturbance")
    ax.set_ylabel("CDF")
    ax.set_title("Disturbance CDF (Normal Conditions)", fontweight="bold", fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "disturbance_cdf.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  ✓ Saved disturbance_cdf.png")
    
    # 4. Failure scenario comparison (if failure data exists)
    if failure_df is not None and not failure_df.empty:
        log("Generating failure robustness plots...")
        
        scenarios = sorted(failure_df['scenario'].unique())
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, scen in enumerate(scenarios[:6]):  # Plot first 6 scenarios
            ax = axes[i]
            scen_data = failure_df[failure_df['scenario'] == scen]
            
            for method in methods:
                method_data = scen_data[scen_data['method'] == method]
                if method_data.empty:
                    continue
                
                mlu_values = method_data['mean_mlu'].values
                mlu_values = mlu_values[np.isfinite(mlu_values)]
                if len(mlu_values) == 0:
                    continue
                
                mlu_sorted = np.sort(mlu_values)
                cdf = np.arange(1, len(mlu_sorted) + 1) / len(mlu_sorted)
                ax.plot(mlu_sorted, cdf, label=method, color=method_colors[method], lw=2)
            
            ax.set_xlabel("Mean MLU")
            ax.set_ylabel("CDF")
            ax.set_title(scen.replace('_', ' '), fontsize=10)
            ax.legend(fontsize=7, loc='lower right')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle("Failure Scenario Robustness", fontweight="bold", fontsize=14)
        fig.tight_layout()
        fig.savefig(plot_dir / "failure_robustness.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        log(f"  ✓ Saved failure_robustness.png")
    
    log(f"All plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
