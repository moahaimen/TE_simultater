#!/usr/bin/env python3
"""Stable MLP MetaGate: disturbance penalty + expert switch penalty.

Extension of the baseline MetaGate evaluation. Does NOT modify any existing
results. All output goes to results/dynamic_metagate/stable/.

Two penalty mechanisms:
  1. Routing disturbance penalty: penalizes experts whose OD selection
     differs greatly from the previous timestep's actual selection.
  2. Expert switch penalty: penalizes changing to a different expert
     than the one used in the previous timestep.

Selection scoring per expert i at timestep t:
  score[i] = log(P_calibrated[i])
             - lambda_d * disturbance[i]
             - lambda_s * switch[i]

  where:
    disturbance[i] = |selected_ods_i ^ prev_selected_ods| / K_crit
    switch[i]      = 1 if i != prev_expert else 0

Parameter sweep:
  lambda_d in {0.05, 0.1, 0.2}
  lambda_s in {0.01, 0.05, 0.1}
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

CONFIG_PATH = "configs/phase1_reactive_full.yaml"
MAX_STEPS = 500
SEED = 42
K_CRIT = 40
LT = 20
DEVICE = "cpu"
OUTPUT_DIR = Path("results/dynamic_metagate/stable")
GNN_CHECKPOINT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
MODEL_PATH = Path("results/dynamic_metagate/models/metagate_unified.pt")

SELECTOR_NAMES = ["bottleneck", "topk", "sensitivity", "gnn"]
NUM_SELECTORS = len(SELECTOR_NAMES)
KNOWN_TOPOLOGIES = {"abilene", "geant", "cernet", "rocketfuel_ebone",
                    "rocketfuel_sprintlink", "rocketfuel_tiscali"}
UNSEEN_TOPOLOGIES = {"germany50", "topologyzoo_vtlwavenet2011"}

# Parameter sweep
LAMBDA_D_VALUES = [0.05, 0.1, 0.2]
LAMBDA_S_VALUES = [0.01, 0.05, 0.1]


def setup():
    """Import all modules (same as baseline)."""
    from te.baselines import (
        ecmp_splits, select_bottleneck_critical,
        select_sensitivity_critical, select_topk_by_demand,
    )
    from te.lp_solver import solve_selected_path_lp
    from te.simulator import apply_routing
    from phase1_reactive.eval.common import (
        load_bundle, load_named_dataset, collect_specs, max_steps_from_args,
    )
    from phase1_reactive.eval.core import split_indices
    from phase1_reactive.drl.dynamic_meta_gate import (
        DynamicMetaGate, MetaGateConfig, extract_features,
    )
    from phase1_reactive.drl.gnn_selector import (
        load_gnn_selector, build_graph_tensors, build_od_features,
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
        "max_steps_from_args": max_steps_from_args,
        "split_indices": split_indices,
        "DynamicMetaGate": DynamicMetaGate,
        "MetaGateConfig": MetaGateConfig,
        "extract_features": extract_features,
        "load_gnn_selector": load_gnn_selector,
        "build_graph_tensors": build_graph_tensors,
        "build_od_features": build_od_features,
        "compute_reactive_telemetry": compute_reactive_telemetry,
    }


def run_heuristic_selector(M, method, tm, ecmp_base, path_library, capacities, k_crit):
    if method == "topk":
        return M["select_topk_by_demand"](tm, k_crit)
    elif method == "bottleneck":
        return M["select_bottleneck_critical"](tm, ecmp_base, path_library, capacities, k_crit)
    elif method == "sensitivity":
        return M["select_sensitivity_critical"](tm, ecmp_base, path_library, capacities, k_crit)
    else:
        raise ValueError(f"Unknown heuristic: {method}")


def run_gnn_selector(M, tm, dataset, path_library, gnn_model, k_crit,
                     telemetry=None):
    import torch
    graph_data = M["build_graph_tensors"](dataset, telemetry=telemetry, device=DEVICE)
    od_data = M["build_od_features"](dataset, tm, path_library,
                                      telemetry=telemetry, device=DEVICE)
    active_mask = (np.asarray(tm, dtype=np.float64) > 1e-12).astype(np.float32)
    with torch.no_grad():
        selected, info = gnn_model.select_critical_flows(
            graph_data, od_data, active_mask=active_mask, k_crit_default=k_crit,
        )
    return selected, info


def run_all_experts(M, tm, ecmp_base, path_library, capacities, dataset,
                    gnn_model, k_crit, telemetry=None):
    selector_results = {}
    timing = {}

    for method in ["bottleneck", "topk", "sensitivity"]:
        t0 = time.time()
        sel = run_heuristic_selector(M, method, tm, ecmp_base, path_library,
                                     capacities, k_crit)
        timing[method] = (time.time() - t0) * 1000
        selector_results[method] = list(sel)

    t0 = time.time()
    gnn_sel, gnn_info = run_gnn_selector(M, tm, dataset, path_library, gnn_model, k_crit,
                                          telemetry=telemetry)
    timing["gnn"] = (time.time() - t0) * 1000
    selector_results["gnn"] = list(gnn_sel)

    return selector_results, gnn_info, timing


def compute_disturbance(current_ods, prev_ods, k_crit):
    """Compute routing disturbance between two OD selections.

    disturbance = |symmetric_difference| / k_crit
    Range: 0 (identical) to ~2 (completely different)
    """
    if prev_ods is None:
        return 0.0
    cur_set = set(current_ods)
    prev_set = set(prev_ods)
    sym_diff = len(cur_set ^ prev_set)
    return sym_diff / max(k_crit, 1)


def stable_select(probs, selector_results, prev_selected_ods, prev_expert_idx,
                  lambda_d, lambda_s, k_crit):
    """Select expert using stability-penalized scoring.

    score[i] = log(P[i]) - lambda_d * disturbance[i] - lambda_s * switch[i]

    Returns: (selected_idx, scores, disturbances, switch_flags)
    """
    scores = np.zeros(NUM_SELECTORS)
    disturbances = np.zeros(NUM_SELECTORS)
    switch_flags = np.zeros(NUM_SELECTORS)

    for i, name in enumerate(SELECTOR_NAMES):
        # Base score from calibrated MLP
        scores[i] = np.log(probs[i] + 1e-12)

        # Disturbance penalty
        expert_ods = selector_results[name]
        dist = compute_disturbance(expert_ods, prev_selected_ods, k_crit)
        disturbances[i] = dist
        scores[i] -= lambda_d * dist

        # Switch penalty
        if prev_expert_idx is not None and i != prev_expert_idx:
            switch_flags[i] = 1.0
            scores[i] -= lambda_s

    selected = int(np.argmax(scores))
    return selected, scores, disturbances, switch_flags


def evaluate_stable_on_topology(M, dataset, pl, gate, gnn_model, k_crit,
                                test_oracle_df, lambda_d, lambda_s):
    """Evaluate MetaGate with stability penalties on one topology."""
    ds_key = dataset.key
    test_indices = M["split_indices"](dataset, "test")
    num_nodes = len(dataset.nodes)
    num_edges = len(dataset.edges)

    ecmp_base = M["ecmp_splits"](pl)
    path_library = pl
    capacities = np.asarray(dataset.capacities, dtype=float)

    # ECMP baseline
    tm0 = np.asarray(dataset.tm[test_indices[0]], dtype=float)
    routing0 = M["apply_routing"](tm0, ecmp_base, path_library, capacities)
    ecmp_link_utils = routing0.link_utilizations if hasattr(routing0, "link_utilizations") else None

    # Oracle labels
    oracle_sub = test_oracle_df[test_oracle_df["dataset"] == ds_key]

    results = []
    prev_selected_ods = None
    prev_expert_idx = None

    for t_idx in test_indices:
        tm = np.asarray(dataset.tm[t_idx], dtype=float)
        t_total_start = time.time()

        # Run all 4 experts
        selector_results, gnn_info, expert_timing = run_all_experts(
            M, tm, ecmp_base, path_library, capacities, dataset,
            gnn_model, k_crit,
        )

        # Extract features
        t_feat_start = time.time()
        feats = M["extract_features"](
            tm, selector_results, num_nodes, num_edges, k_crit,
            gnn_info=gnn_info, ecmp_link_utils=ecmp_link_utils,
        )
        t_feat = (time.time() - t_feat_start) * 1000

        # Get calibrated MLP probabilities
        t_mlp_start = time.time()
        _, probs = gate.predict(feats)
        t_mlp = (time.time() - t_mlp_start) * 1000

        # Stability-penalized selection
        pred_class, scores, disturbances, switch_flags = stable_select(
            probs, selector_results, prev_selected_ods, prev_expert_idx,
            lambda_d, lambda_s, k_crit,
        )
        pred_name = SELECTOR_NAMES[pred_class]
        selected_ods = selector_results[pred_name]

        # Compute actual disturbance for the selected expert
        actual_disturbance = compute_disturbance(selected_ods, prev_selected_ods, k_crit)
        actual_switch = 1 if (prev_expert_idx is not None and pred_class != prev_expert_idx) else 0

        # LP solve
        t_lp_start = time.time()
        lp = M["solve_selected_path_lp"](
            tm, selected_ods, ecmp_base, path_library, capacities,
            time_limit_sec=LT,
        )
        routing = M["apply_routing"](tm, lp.splits, path_library, capacities)
        t_lp = (time.time() - t_lp_start) * 1000

        metagate_mlu = float(routing.mlu)

        # Run LP for all 4 experts (for oracle and comparison)
        expert_mlus = {}
        for name in SELECTOR_NAMES:
            try:
                elp = M["solve_selected_path_lp"](
                    tm, selector_results[name], ecmp_base, path_library,
                    capacities, time_limit_sec=LT,
                )
                erouting = M["apply_routing"](tm, elp.splits, path_library, capacities)
                expert_mlus[name] = float(erouting.mlu)
            except Exception:
                expert_mlus[name] = float("inf")

        oracle_name = min(expert_mlus, key=expert_mlus.get)
        oracle_mlu = expert_mlus[oracle_name]

        t_decision = sum(expert_timing.values()) + t_feat + t_mlp
        t_total = (time.time() - t_total_start) * 1000

        # Oracle from pre-computed labels
        oracle_row = oracle_sub[oracle_sub["timestep"] == t_idx]
        if len(oracle_row) > 0:
            oracle_name_label = oracle_row.iloc[0]["oracle_selector"]
            correct = (pred_name == oracle_name_label)
        else:
            oracle_name_label = oracle_name
            correct = (pred_name == oracle_name)

        results.append({
            "dataset": ds_key,
            "topology_type": "unseen" if ds_key in UNSEEN_TOPOLOGIES else "known",
            "timestep": t_idx,
            "lambda_d": lambda_d,
            "lambda_s": lambda_s,
            "metagate_selector": pred_name,
            "metagate_mlu": metagate_mlu,
            "oracle_selector": oracle_name_label,
            "oracle_mlu": oracle_mlu,
            "bn_mlu": expert_mlus["bottleneck"],
            "topk_mlu": expert_mlus["topk"],
            "sens_mlu": expert_mlus["sensitivity"],
            "gnn_mlu": expert_mlus["gnn"],
            "correct": correct,
            "disturbance": actual_disturbance,
            "expert_switch": actual_switch,
            "prob_bn": probs[0],
            "prob_topk": probs[1],
            "prob_sens": probs[2],
            "prob_gnn": probs[3],
            "t_bn_ms": expert_timing["bottleneck"],
            "t_topk_ms": expert_timing["topk"],
            "t_sens_ms": expert_timing["sensitivity"],
            "t_gnn_ms": expert_timing["gnn"],
            "t_features_ms": t_feat,
            "t_mlp_ms": t_mlp,
            "t_lp_ms": t_lp,
            "t_decision_ms": t_decision,
            "t_total_ms": t_total,
        })

        # Update state for next timestep
        prev_selected_ods = selected_ods
        prev_expert_idx = pred_class

    return results


def compute_features_and_oracle_for_topology(M, dataset, pl, indices, gnn_model, k_crit):
    """Compute features + oracle for calibration (same as baseline)."""
    ds_key = dataset.key
    num_nodes = len(dataset.nodes)
    num_edges = len(dataset.edges)
    ecmp_base = M["ecmp_splits"](pl)
    path_library = pl
    capacities = np.asarray(dataset.capacities, dtype=float)

    tm0 = np.asarray(dataset.tm[indices[0]], dtype=float)
    routing0 = M["apply_routing"](tm0, ecmp_base, path_library, capacities)
    ecmp_link_utils = routing0.link_utilizations if hasattr(routing0, "link_utilizations") else None

    all_X, all_y = [], []

    for t_idx in indices:
        tm = np.asarray(dataset.tm[t_idx], dtype=float)
        try:
            selector_results, gnn_info, _ = run_all_experts(
                M, tm, ecmp_base, path_library, capacities, dataset,
                gnn_model, k_crit,
            )
            feats = M["extract_features"](
                tm, selector_results, num_nodes, num_edges, k_crit,
                gnn_info=gnn_info, ecmp_link_utils=ecmp_link_utils,
            )

            mlus = {}
            for name in SELECTOR_NAMES:
                lp = M["solve_selected_path_lp"](
                    tm, selector_results[name], ecmp_base, path_library,
                    capacities, time_limit_sec=LT,
                )
                routing = M["apply_routing"](tm, lp.splits, path_library, capacities)
                mlus[name] = float(routing.mlu)

            oracle = min(range(NUM_SELECTORS), key=lambda i: mlus[SELECTOR_NAMES[i]])
            all_X.append(feats)
            all_y.append(oracle)
        except Exception:
            continue

    if not all_X:
        return np.array([]), np.array([])
    return np.array(all_X), np.array(all_y)


def load_test_oracle_4class(csv_path):
    """Load test oracle labels (same as baseline)."""
    df = pd.read_csv(csv_path)
    test_methods = ["bottleneck", "topk", "sensitivity", "gnn"]
    df = df[df["method"].isin(test_methods)]
    best = df.groupby(["dataset", "timestep"])["mlu"].idxmin()
    oracle = df.loc[best, ["dataset", "timestep", "method", "mlu"]].copy()
    oracle.rename(columns={"method": "oracle_selector", "mlu": "oracle_mlu"}, inplace=True)
    label_map = {name: i for i, name in enumerate(SELECTOR_NAMES)}
    oracle["oracle_label"] = oracle["oracle_selector"].map(label_map)
    return oracle


def main():
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STABLE METAGATE: Disturbance + Switch Penalty Extension")
    print("  This is a NEW experiment — existing results are NOT modified")
    print("  Output: results/dynamic_metagate/stable/")
    print("=" * 70)

    M = setup()

    # Load GNN model
    print(f"\nLoading GNN model from {GNN_CHECKPOINT}...")
    gnn_model, gnn_cfg = M["load_gnn_selector"](str(GNN_CHECKPOINT), device=DEVICE)

    # Load pre-trained MetaGate (from baseline)
    print(f"Loading pre-trained MetaGate from {MODEL_PATH}...")
    config = M["MetaGateConfig"](hidden_dim=128, dropout=0.3)
    gate = M["DynamicMetaGate"](config)
    gate.load(MODEL_PATH, feat_dim=49)

    # Load test oracle
    oracle_csv = Path("results/requirements_compliant_eval/all_results.csv")
    test_oracle_df = load_test_oracle_4class(oracle_csv)
    print(f"  {len(test_oracle_df)} test oracle labels")

    # Load all datasets
    bundle = M["load_bundle"](CONFIG_PATH)
    eval_specs = M["collect_specs"](bundle, "eval_topologies")
    gen_specs = M["collect_specs"](bundle, "generalization_topologies")

    known_datasets = []
    unseen_datasets = []
    for spec in eval_specs:
        try:
            dataset, pl = M["load_named_dataset"](bundle, spec, MAX_STEPS)
            known_datasets.append((dataset, pl))
        except Exception as e:
            print(f"  Skip {spec.key}: {e}")
    for spec in gen_specs:
        try:
            dataset, pl = M["load_named_dataset"](bundle, spec, MAX_STEPS)
            unseen_datasets.append((dataset, pl))
        except Exception as e:
            print(f"  Skip {spec.key}: {e}")

    all_datasets = known_datasets + unseen_datasets
    print(f"  Known: {len(known_datasets)}, Unseen: {len(unseen_datasets)}")

    # Parameter sweep
    param_combos = list(itertools.product(LAMBDA_D_VALUES, LAMBDA_S_VALUES))
    print(f"\n  Parameter sweep: {len(param_combos)} combinations")
    for ld, ls in param_combos:
        print(f"    lambda_d={ld}, lambda_s={ls}")

    all_results = []

    for combo_idx, (lambda_d, lambda_s) in enumerate(param_combos):
        print(f"\n{'='*70}")
        print(f"  COMBO {combo_idx+1}/{len(param_combos)}: lambda_d={lambda_d}, lambda_s={lambda_s}")
        print(f"{'='*70}")

        for dataset, pl in all_datasets:
            ds_key = dataset.key
            topo_type = "UNSEEN" if ds_key in UNSEEN_TOPOLOGIES else "known"
            print(f"\n  [{topo_type}] {ds_key}...")

            # Calibrate (same as baseline — 10 val TMs)
            val_indices = M["split_indices"](dataset, "val")
            calib_indices = val_indices[:10]
            try:
                _, calib_labels = compute_features_and_oracle_for_topology(
                    M, dataset, pl, calib_indices, gnn_model, K_CRIT,
                )
                if len(calib_labels) > 0:
                    win_counts = np.bincount(calib_labels, minlength=NUM_SELECTORS)
                    gate.calibrate(win_counts, smoothing=1.0, strength=5.0)
                else:
                    gate.clear_calibration()
            except Exception:
                gate.clear_calibration()

            # Evaluate with stability penalties
            try:
                results = evaluate_stable_on_topology(
                    M, dataset, pl, gate, gnn_model, K_CRIT,
                    test_oracle_df, lambda_d, lambda_s,
                )
                all_results.extend(results)

                if results:
                    df = pd.DataFrame(results)
                    acc = df["correct"].mean()
                    mg_mlu = df["metagate_mlu"].mean()
                    mean_dist = df["disturbance"].mean()
                    total_switches = df["expert_switch"].sum()
                    sel_counts = df["metagate_selector"].value_counts()

                    print(f"    Accuracy: {acc:.1%}")
                    print(f"    MLU: {mg_mlu:.4f}")
                    print(f"    Mean disturbance: {mean_dist:.4f}")
                    print(f"    Expert switches: {total_switches}/{len(df)}")
                    for name in SELECTOR_NAMES:
                        c = sel_counts.get(name, 0)
                        print(f"      {name}: {c}/{len(df)} ({100*c/len(df):.0f}%)")

            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()

    # ================================================================
    # Save results
    # ================================================================
    if not all_results:
        print("\nNo results generated!")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "stable_metagate_results.csv", index=False)
    print(f"\nSaved {len(results_df)} rows to {OUTPUT_DIR / 'stable_metagate_results.csv'}")

    # Summary per (lambda_d, lambda_s, dataset)
    summary = results_df.groupby(["lambda_d", "lambda_s", "dataset"]).agg(
        topology_type=("topology_type", "first"),
        accuracy=("correct", "mean"),
        metagate_mlu=("metagate_mlu", "mean"),
        oracle_mlu=("oracle_mlu", "mean"),
        mean_disturbance=("disturbance", "mean"),
        total_switches=("expert_switch", "sum"),
        n_timesteps=("timestep", "count"),
        t_decision_ms=("t_decision_ms", "mean"),
        t_total_ms=("t_total_ms", "mean"),
    ).reset_index()
    summary["oracle_gap_pct"] = (
        (summary["metagate_mlu"] - summary["oracle_mlu"]) / summary["oracle_mlu"] * 100
    )
    summary["switch_rate"] = summary["total_switches"] / summary["n_timesteps"]
    summary.to_csv(OUTPUT_DIR / "stable_metagate_summary.csv", index=False)

    # Best combo comparison
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP RESULTS")
    print("=" * 70)

    combo_summary = results_df.groupby(["lambda_d", "lambda_s"]).agg(
        mean_mlu=("metagate_mlu", "mean"),
        mean_disturbance=("disturbance", "mean"),
        total_switches=("expert_switch", "sum"),
        accuracy=("correct", "mean"),
        n=("timestep", "count"),
    ).reset_index()
    combo_summary["switch_rate"] = combo_summary["total_switches"] / combo_summary["n"]

    print(f"\n{'lambda_d':>8} {'lambda_s':>8} {'MLU':>12} {'Disturbance':>12} "
          f"{'Switches':>10} {'Switch%':>8} {'Accuracy':>8}")
    print("-" * 80)
    for _, r in combo_summary.iterrows():
        print(f"{r['lambda_d']:>8.2f} {r['lambda_s']:>8.2f} {r['mean_mlu']:>12.4f} "
              f"{r['mean_disturbance']:>12.4f} {int(r['total_switches']):>10} "
              f"{r['switch_rate']:>7.1%} {r['accuracy']:>7.1%}")

    combo_summary.to_csv(OUTPUT_DIR / "parameter_sweep_summary.csv", index=False)
    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
