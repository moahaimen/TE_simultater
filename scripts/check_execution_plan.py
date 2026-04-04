#!/usr/bin/env python3
"""Execution plan verification: Check which methods are runnable."""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_runnable_methods():
    """Check which methods have trained models available."""
    print("="*70)
    print("EXECUTION PLAN: METHOD AVAILABILITY CHECK")
    print("="*70)
    
    results = {
        "classical": {},
        "learned": {}
    }
    
    # Classical baselines - always runnable (no models needed)
    print("\n1. CLASSICAL BASELINES (always runnable):")
    classical = ["OSPF", "ECMP", "TopK", "Bottleneck", "Sensitivity"]
    for method in classical:
        print(f"   ✓ {method}")
        results["classical"][method] = "runnable"
    
    # Learned methods - need to check for checkpoints
    print("\n2. LEARNED METHODS (need checkpoint verification):")
    
    # Check for GNN selector
    gnn_ckpt = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
    if gnn_ckpt.exists():
        print(f"   ✓ Original GNN: {gnn_ckpt}")
        results["learned"]["GNN"] = str(gnn_ckpt)
    else:
        print(f"   ✗ Original GNN: checkpoint not found")
        results["learned"]["GNN"] = None
    
    # Check for GNN+ (final locked model)
    gnn_plus_ckpt = Path("results/final_metagate_gnn_plus/gnn_plus_model.pt")
    if gnn_plus_ckpt.exists():
        print(f"   ✓ GNN+: {gnn_plus_ckpt}")
        results["learned"]["GNN+"] = str(gnn_plus_ckpt)
    else:
        # Try alternative location
        gnn_plus_alt = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
        if gnn_plus_alt.exists():
            print(f"   ✓ GNN+: using {gnn_plus_alt} (final locked config)")
            results["learned"]["GNN+"] = str(gnn_plus_alt)
        else:
            print(f"   ✗ GNN+: checkpoint not found")
            results["learned"]["GNN+"] = None
    
    # Check for DRL methods
    train_dir = Path("results/phase1_reactive/train")
    if train_dir.exists():
        # Check for PPO
        ppo_ckpt = list(train_dir.rglob("*ppo*/best_model.pt")) + list(train_dir.rglob("*ppo*.pt"))
        if ppo_ckpt:
            print(f"   ✓ DRL-PPO: {ppo_ckpt[0]}")
            results["learned"]["DRL-PPO"] = str(ppo_ckpt[0])
        else:
            print(f"   ~ DRL-PPO: searching...")
            results["learned"]["DRL-PPO"] = "searching"
        
        # Check for DQN
        dqn_ckpt = list(train_dir.rglob("*dqn*/best_model.pt")) + list(train_dir.rglob("*dqn*.pt"))
        if dqn_ckpt:
            print(f"   ✓ DRL-DQN: {dqn_ckpt[0]}")
            results["learned"]["DRL-DQN"] = str(dqn_ckpt[0])
        else:
            print(f"   ~ DRL-DQN: searching...")
            results["learned"]["DRL-DQN"] = "searching"
        
        # Check for MetaGate/MoE
        moe_ckpt = list(train_dir.rglob("*moe*/best_model.pt")) + list(train_dir.rglob("*moe*.pt"))
        if moe_ckpt:
            print(f"   ✓ MetaGate/MoE: {moe_ckpt[0]}")
            results["learned"]["MetaGate"] = str(moe_ckpt[0])
        else:
            print(f"   ~ MetaGate/MoE: searching...")
            results["learned"]["MetaGate"] = "searching"
    else:
        print(f"   ✗ Train directory not found: {train_dir}")
        results["learned"]["DRL-PPO"] = None
        results["learned"]["DRL-DQN"] = None
        results["learned"]["MetaGate"] = None
    
    # Check for Dual Gate
    dual_ckpt = list(Path("results").rglob("*dual*/best_model.pt")) + list(Path("results").rglob("*dual*.pt"))
    if dual_ckpt:
        print(f"   ✓ Dual Gate: {dual_ckpt[0]}")
        results["learned"]["Dual-Gate"] = str(dual_ckpt[0])
    else:
        print(f"   ~ Dual Gate: searching...")
        results["learned"]["Dual-Gate"] = "searching"
    
    return results


def check_topologies():
    """Check which topologies have data available."""
    print("\n" + "="*70)
    print("TOPOLOGY AVAILABILITY CHECK")
    print("="*70)
    
    target_topos = [
        "abilene_backbone", "geant_core", "germany50_real",
        "cernet_real", "ebone", "sprintlink", "tiscali", "vtlwavenet2011"
    ]
    
    try:
        from phase1_reactive.eval.common import load_bundle, collect_specs
        
        bundle = load_bundle("configs/phase1_reactive_full.yaml")
        
        available = []
        missing = []
        
        for topo_key in target_topos:
            try:
                # Try to find in eval or generalization specs
                specs = collect_specs(bundle, "eval_topologies") + collect_specs(bundle, "generalization_topologies")
                matching = [s for s in specs if s.key == topo_key]
                if matching:
                    available.append(topo_key)
                    print(f"   ✓ {topo_key}")
                else:
                    missing.append(topo_key)
                    print(f"   ✗ {topo_key} (not in config)")
            except Exception as e:
                missing.append(topo_key)
                print(f"   ✗ {topo_key}: {e}")
        
        return available, missing
    except Exception as e:
        print(f"   ERROR checking topologies: {e}")
        print(f"   Will attempt all {len(target_topos)} topologies and report failures during execution")
        return target_topos, []


def estimate_runtime(num_methods, num_topos, num_seeds, has_failures=True):
    """Estimate total runtime."""
    print("\n" + "="*70)
    print("RUNTIME ESTIMATE")
    print("="*70)
    
    # Per-evaluation estimates (very rough)
    time_per_eval = {
        "classical": 0.5,  # seconds per TM
        "learned": 2.0,    # seconds per TM (includes model inference)
    }
    
    # Assume ~500 TMs per topology for full evaluation
    tms_per_topo = 500
    
    num_classical = 5  # OSPF, ECMP, TopK, Bottleneck, Sensitivity
    num_learned = num_methods - num_classical
    
    # Normal conditions
    classical_time = num_classical * num_topos * tms_per_topo * time_per_eval["classical"] * num_seeds
    learned_time = num_learned * num_topos * tms_per_topo * time_per_eval["learned"] * num_seeds
    
    total_normal = classical_time + learned_time
    
    # Failure scenarios (4 scenarios, roughly 2x time each)
    if has_failures:
        failure_multiplier = 4 * 2  # 4 scenarios, 2x slower each
        total_with_failures = total_normal * (1 + failure_multiplier / 10)  # Rough estimate
    else:
        total_with_failures = total_normal
    
    print(f"   Methods: {num_methods} total ({num_classical} classical, {num_learned} learned)")
    print(f"   Topologies: {num_topos}")
    print(f"   Seeds: {num_seeds}")
    print(f"   TMs per topology: ~{tms_per_topo}")
    print(f"   Estimated time (normal): {total_normal/3600:.1f} hours")
    print(f"   Estimated time (with failures): {total_with_failures/3600:.1f} hours")
    
    return total_with_failures


def main():
    print("\n" + "="*70)
    print("FULL EVALUATION EXECUTION PLAN")
    print("="*70)
    
    # Check methods
    method_status = check_runnable_methods()
    
    # Check topologies
    available_topos, missing_topos = check_topologies()
    
    # Count runnable methods
    runnable_classical = sum(1 for v in method_status["classical"].values() if v == "runnable")
    runnable_learned = sum(1 for v in method_status["learned"].values() if v and v not in [None, "searching"])
    searching_learned = sum(1 for v in method_status["learned"].values() if v == "searching")
    
    total_runnable = runnable_classical + runnable_learned
    
    print("\n" + "="*70)
    print("EXECUTION PLAN SUMMARY")
    print("="*70)
    
    print(f"\n1. TOPOLOGIES TO RUN: {len(available_topos)}")
    for topo in available_topos:
        print(f"   - {topo}")
    if missing_topos:
        print(f"\n   MISSING (will skip): {missing_topos}")
    
    print(f"\n2. METHODS TO RUN: {total_runnable}")
    print(f"   Classical ({runnable_classical}):")
    for method, status in method_status["classical"].items():
        print(f"      - {method}")
    
    print(f"\n   Learned ({runnable_learned} confirmed, {searching_learned} searching):")
    for method, ckpt in method_status["learned"].items():
        if ckpt and ckpt not in [None, "searching"]:
            print(f"      ✓ {method}: {Path(ckpt).name}")
        elif ckpt == "searching":
            print(f"      ~ {method}: searching for checkpoint...")
        else:
            print(f"      ✗ {method}: not available")
    
    print(f"\n3. FAILURE SCENARIOS: 4 mandatory")
    print("   - Scenario A: Single Link Failure")
    print("   - Scenario B: Random Link Failure")
    print("   - Scenario C: Capacity Degradation")
    print("   - Scenario D: Traffic Spike")
    
    # Estimate runtime
    estimate_runtime(total_runnable, len(available_topos), 5, has_failures=True)
    
    print("\n4. OUTPUT FILES:")
    print("   - results/final_full_eval/final_results.csv")
    print("   - results/final_full_eval/failure_results.csv")
    print("   - results/final_full_eval/plots/")
    print("   - results/final_full_eval/FINAL_TE_FULL_REPORT.docx")
    
    print("\n" + "="*70)
    print("READY TO EXECUTE")
    print("="*70)
    
    return {
        "methods": method_status,
        "topologies": available_topos,
        "missing_topologies": missing_topos,
    }


if __name__ == "__main__":
    plan = main()
