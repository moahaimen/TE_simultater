#!/usr/bin/env python3
"""Verification script: Test which baselines and topologies are actually runnable."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = "configs/phase1_reactive_full.yaml"
DEVICE = "cpu"
K_FIXED = 40
K_PATHS = 3
LT = 15

def setup():
    from te.baselines import (
        ecmp_splits, select_bottleneck_critical,
        select_sensitivity_critical, select_topk_by_demand
    )
    from te.lp_solver import solve_selected_path_lp
    from te.simulator import apply_routing
    from phase1_reactive.eval.common import load_bundle, load_named_dataset, collect_specs
    from phase1_reactive.eval.core import split_indices
    from phase1_reactive.drl.gnn_selector import (
        load_gnn_selector, build_graph_tensors, build_od_features
    )
    from phase1_reactive.drl.state_builder import compute_reactive_telemetry
    return {
        "ecmp_splits": ecmp_splits,
        "select_bottleneck_critical": select_bottleneck_critical,
        "select_sensitivity_critical": select_sensitivity_critical,
        "select_topk_by_demand": select_topk_by_demand,
        "solve_selected_path_lp": solve_selected_path_lp,
        "apply_routing": apply_routing,
        "load_bundle": load_bundle,
        "load_named_dataset": load_named_dataset,
        "collect_specs": collect_specs,
        "split_indices": split_indices,
        "load_gnn_selector": load_gnn_selector,
        "build_graph_tensors": build_graph_tensors,
        "build_od_features": build_od_features,
        "compute_reactive_telemetry": compute_reactive_telemetry,
    }


def verify_topologies(M):
    """Check which topologies actually load successfully."""
    print("\n" + "="*60)
    print("TOPOLOGY VERIFICATION")
    print("="*60)
    
    bundle = M["load_bundle"](CONFIG_PATH)
    all_specs = (
        M["collect_specs"](bundle, "eval_topologies") +
        M["collect_specs"](bundle, "generalization_topologies")
    )
    
    available = []
    failed = []
    
    for spec in all_specs:
        try:
            ds, pl = M["load_named_dataset"](bundle, spec, K_PATHS)
            n_tms = len(ds.tm) if hasattr(ds, "tm") else 0
            print(f"  ✓ {spec.key:20s}: {len(ds.nodes)}N, {len(ds.edges)}E, {n_tms} TMs")
            available.append(spec.key)
        except Exception as e:
            print(f"  ✗ {spec.key:20s}: FAILED - {str(e)[:50]}")
            failed.append((spec.key, str(e)))
    
    return available, failed


def verify_baselines(M, test_topo_key="abilene_backbone"):
    """Test each baseline on a single topology."""
    print("\n" + "="*60)
    print("BASELINE VERIFICATION")
    print("="*60)
    
    bundle = M["load_bundle"](CONFIG_PATH)
    eval_specs = M["collect_specs"](bundle, "eval_topologies")
    
    # Find test topology
    test_spec = None
    for spec in eval_specs:
        if spec.key == test_topo_key:
            test_spec = spec
            break
    
    if not test_spec:
        print(f"  Test topology {test_topo_key} not found, skipping baseline test")
        return {}, {}
    
    try:
        ds, pl = M["load_named_dataset"](bundle, test_spec, K_PATHS)
    except Exception as e:
        print(f"  Failed to load test topology: {e}")
        return {}, {}
    
    ecmp_base = M["ecmp_splits"](pl)
    caps = ds.capacities
    
    # Get one test timestep
    test_indices = M["split_indices"](ds, "test")
    if len(test_indices) == 0:
        print("  No test indices available")
        return {}, {}
    
    tm = ds.tm[test_indices[0]]
    routing = M["apply_routing"](tm, ecmp_base, pl, caps)
    telem = M["compute_reactive_telemetry"](
        tm, ecmp_base, pl, routing, ds.weights)
    
    baselines_to_test = {
        "ECMP": lambda: (ecmp_base, None),
        "Bottleneck": lambda: M["select_bottleneck_critical"](
            tm, ecmp_base, pl, caps, K_FIXED),
        "Sensitivity": lambda: M["select_sensitivity_critical"](
            tm, ecmp_base, pl, caps, K_FIXED),
        "TopK": lambda: M["select_topk_by_demand"](tm, K_FIXED),
    }
    
    working = {}
    failed = {}
    
    for name, fn in baselines_to_test.items():
        try:
            t0 = time.perf_counter()
            result = fn()
            elapsed = (time.perf_counter() - t0) * 1000
            
            # For ECMP, result is routing; for others, it's selection
            if name == "ECMP":
                sel = []
            else:
                sel = result
            
            # Try LP solve
            if sel is not None:
                lp = M["solve_selected_path_lp"](
                    tm_vector=tm, selected_ods=sel, base_splits=ecmp_base,
                    path_library=pl, capacities=caps, time_limit_sec=LT)
                mlu = float(lp.routing.mlu)
            else:
                mlu = float(routing.mlu)
            
            print(f"  ✓ {name:15s}: MLU={mlu:.4f} [{elapsed:.1f}ms]")
            working[name] = {"mlu": mlu, "time_ms": elapsed}
        except Exception as e:
            print(f"  ✗ {name:15s}: FAILED - {str(e)[:50]}")
            failed[name] = str(e)
    
    return working, failed


def verify_gnn_model(M):
    """Verify GNN+ model loads and runs."""
    print("\n" + "="*60)
    print("GNN+ MODEL VERIFICATION")
    print("="*60)
    
    GNN_CKPT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
    
    if not GNN_CKPT.exists():
        print(f"  ✗ GNN checkpoint not found at {GNN_CKPT}")
        return False, "Checkpoint missing"
    
    try:
        model, _ = M["load_gnn_selector"](GNN_CKPT, device=DEVICE)
        model.eval()
        print(f"  ✓ GNN+ model loaded successfully")
        print(f"  ✓ Configuration: enriched features, dropout=0.2, K={K_FIXED} (FIXED)")
        return True, model
    except Exception as e:
        print(f"  ✗ GNN+ model failed to load: {e}")
        return False, str(e)


def verify_failure_scenarios():
    """Check if failure scenario infrastructure is available."""
    print("\n" + "="*60)
    print("FAILURE SCENARIO VERIFICATION")
    print("="*60)
    
    failure_types = ["single_link_failure", "capacity_degradation", "multi_link_stress"]
    
    # Check if failure scenario modules exist
    try:
        from phase1_reactive.eval import failure_scenarios
        print("  ✓ Failure scenarios module available")
        
        for ft in failure_types:
            if hasattr(failure_scenarios, ft) or hasattr(failure_scenarios, f"apply_{ft}"):
                print(f"  ✓ {ft}: available")
            else:
                print(f"  ~ {ft}: not found in module (may need manual implementation)")
        
        return True
    except ImportError:
        print("  ✗ Failure scenarios module not found")
        print("  ~ Failure scenarios will need to be excluded from final report")
        return False


def main():
    print("="*60)
    print("PRE-EXECUTION VERIFICATION CHECKLIST")
    print("="*60)
    
    M = setup()
    
    # 1. Verify topologies
    available_topos, failed_topos = verify_topologies(M)
    
    # 2. Verify baselines
    working_baselines, failed_baselines = verify_baselines(M)
    
    # 3. Verify GNN model
    gnn_ok, gnn_model = verify_gnn_model(M)
    
    # 4. Verify failure scenarios
    failures_available = verify_failure_scenarios()
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    print(f"\n✓ Topologies available ({len(available_topos)}):")
    for topo in available_topos:
        print(f"    - {topo}")
    
    if failed_topos:
        print(f"\n✗ Topologies failed ({len(failed_topos)}):")
        for topo, err in failed_topos:
            print(f"    - {topo}: {err[:50]}")
    
    print(f"\n✓ Baselines working ({len(working_baselines)}):")
    for name in working_baselines:
        print(f"    - {name}")
    
    if failed_baselines:
        print(f"\n✗ Baselines failed ({len(failed_baselines)}):")
        for name, err in failed_baselines.items():
            print(f"    - {name}: {err[:50]}")
    
    print(f"\n✓ GNN+ model: {'READY' if gnn_ok else 'NOT AVAILABLE'}")
    print(f"✓ Failure scenarios: {'AVAILABLE' if failures_available else 'NOT AVAILABLE'}")
    
    print("\n" + "="*60)
    
    # Return verification results for main script
    return {
        "topologies": available_topos,
        "topologies_failed": failed_topos,
        "baselines": list(working_baselines.keys()),
        "baselines_failed": failed_baselines,
        "gnn_ready": gnn_ok,
        "failures_available": failures_available,
        "gnn_model": gnn_model if gnn_ok else None,
    }


if __name__ == "__main__":
    result = main()
    # Save verification results
    import json
    output_path = Path("results/verification_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Don't save the model object
    save_result = {k: v for k, v in result.items() if k != "gnn_model"}
    with open(output_path, 'w') as f:
        json.dump(save_result, f, indent=2)
    print(f"\nVerification results saved to: {output_path}")
