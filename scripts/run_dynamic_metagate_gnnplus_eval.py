#!/usr/bin/env python3
"""Evaluate the 4-expert MLP MetaGate with GNN+ as the learned expert.

This creates a fresh output bundle under results/dynamic_metagate_gnnplus/
without touching the validated original MetaGate+GNN bundle.

Architecture:
  - MLP Meta-Gate selects among {Bottleneck, TopK, Sensitivity, GNN+}
  - ONE unified gate trained on pooled data from 6 known topologies
  - Germany50 and VtlWavenet2011 are unseen at gate-training time
  - Per-component end-to-end timing: BN+TopK+Sens+GNN++features+MLP+LP

Pipeline per timestep:
  1. Run all 4 experts (BN, TopK, Sens, GNN+) -> 4 OD selections
  2. Extract features from TM + expert outputs + GNN+ diagnostics
  3. MLP classifier predicts best expert
  4. Run predicted expert's selection through LP -> MLU

Oracle labels:
  For each split TM, all 4 experts are evaluated with LP.
  Oracle = argmin(BN_MLU, TopK_MLU, Sens_MLU, GNNPLUS_MLU).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

CONFIG_PATH = "configs/phase1_reactive_full.yaml"
MAX_STEPS = 500
SEED = 42
K_CRIT = 40
LT = 20
DEVICE = "cpu"
N_CALIB = 10
SOFT_LABEL_TEMPERATURE = 0.05
REGRET_LOSS_WEIGHT = 0.25
REGRET_CLIP = 10.0

OUTPUT_DIR = Path(os.environ.get("METAGATE_GNNPLUS_OUTPUT_DIR", "results/dynamic_metagate_gnnplus"))
MODEL_DIR = OUTPUT_DIR / "models"
GNNPLUS_CHECKPOINT = Path("results/gnn_plus_retrained_fixedk40/gnn_plus_fixed_k40.pt")
GNNPLUS_SOURCE_CHECKPOINT = Path("results/gnn_plus/stage1_regularization/training_d02/final.pt")

SELECTOR_NAMES = ["bottleneck", "topk", "sensitivity", "gnnplus"]
NUM_SELECTORS = len(SELECTOR_NAMES)
KNOWN_TOPOLOGIES = {
    "abilene",
    "geant",
    "cernet",
    "rocketfuel_ebone",
    "rocketfuel_sprintlink",
    "rocketfuel_tiscali",
}
UNSEEN_TOPOLOGIES = {"germany50", "topologyzoo_vtlwavenet2011"}


def setup():
    from te.baselines import (
        ecmp_splits,
        select_bottleneck_critical,
        select_sensitivity_critical,
        select_topk_by_demand,
    )
    from te.lp_solver import solve_selected_path_lp
    from te.simulator import apply_routing
    from phase1_reactive.eval.common import (
        load_bundle,
        load_named_dataset,
        collect_specs,
        max_steps_from_args,
    )
    from phase1_reactive.eval.core import split_indices
    from phase1_reactive.drl.dynamic_meta_gate import (
        DynamicMetaGate,
        MetaGateConfig,
        extract_features,
    )
    from phase1_reactive.drl.gnn_plus_selector import (
        load_gnn_plus,
        build_graph_tensors_plus,
        build_od_features_plus,
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
        "load_gnn_plus": load_gnn_plus,
        "build_graph_tensors_plus": build_graph_tensors_plus,
        "build_od_features_plus": build_od_features_plus,
        "compute_reactive_telemetry": compute_reactive_telemetry,
    }


def run_heuristic_selector(M, method, tm, ecmp_base, path_library, capacities, k_crit):
    if method == "topk":
        return M["select_topk_by_demand"](tm, k_crit)
    if method == "bottleneck":
        return M["select_bottleneck_critical"](tm, ecmp_base, path_library, capacities, k_crit)
    if method == "sensitivity":
        return M["select_sensitivity_critical"](tm, ecmp_base, path_library, capacities, k_crit)
    raise ValueError(f"Unknown heuristic: {method}")


def run_gnnplus_selector(M, tm, dataset, path_library, gnnplus_model, k_crit, telemetry=None):
    graph_data = M["build_graph_tensors_plus"](
        dataset,
        tm_vector=tm,
        path_library=path_library,
        telemetry=telemetry,
        device=DEVICE,
    )
    od_data = M["build_od_features_plus"](
        dataset,
        tm,
        path_library,
        telemetry=telemetry,
        device=DEVICE,
    )
    active_mask = (np.asarray(tm, dtype=np.float64) > 1e-12).astype(np.float32)
    with torch.no_grad():
        selected, info = gnnplus_model.select_critical_flows(
            graph_data,
            od_data,
            active_mask=active_mask,
            k_crit_default=k_crit,
            path_library=path_library,
            telemetry=telemetry,
            force_default_k=True,
        )
    return selected, info


def run_all_experts(M, tm, ecmp_base, path_library, capacities, dataset, gnnplus_model, k_crit, telemetry=None):
    timing = {}

    t0 = time.perf_counter()
    bn_sel = run_heuristic_selector(M, "bottleneck", tm, ecmp_base, path_library, capacities, k_crit)
    timing["bottleneck"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    topk_sel = run_heuristic_selector(M, "topk", tm, ecmp_base, path_library, capacities, k_crit)
    timing["topk"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    sens_sel = run_heuristic_selector(M, "sensitivity", tm, ecmp_base, path_library, capacities, k_crit)
    timing["sensitivity"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    gnnplus_sel, gnnplus_info = run_gnnplus_selector(
        M,
        tm,
        dataset,
        path_library,
        gnnplus_model,
        k_crit,
        telemetry=telemetry,
    )
    timing["gnnplus"] = (time.perf_counter() - t0) * 1000

    selector_results = {
        "bottleneck": bn_sel,
        "topk": topk_sel,
        "sensitivity": sens_sel,
        "gnnplus": gnnplus_sel,
    }
    return selector_results, gnnplus_info, timing


def alias_selector_results_for_features(selector_results: dict[str, list[int]]) -> dict[str, list[int]]:
    aliased = dict(selector_results)
    aliased["gnn"] = list(selector_results["gnnplus"])
    return aliased


def compute_expert_mlus(M, tm, selector_results, ecmp_base, path_library, capacities) -> dict[str, float]:
    mlus = {}
    for name in SELECTOR_NAMES:
        try:
            lp = M["solve_selected_path_lp"](
                tm,
                selector_results[name],
                ecmp_base,
                path_library,
                capacities,
                time_limit_sec=LT,
            )
            routing = M["apply_routing"](tm, lp.splits, path_library, capacities)
            mlus[name] = float(routing.mlu)
        except Exception:
            mlus[name] = float("inf")
    return mlus


def build_soft_targets_and_regret_costs(
    mlu_records: list[dict[str, float]],
    temperature: float = SOFT_LABEL_TEMPERATURE,
    regret_clip: float = REGRET_CLIP,
) -> tuple[np.ndarray, np.ndarray]:
    soft_targets = []
    regret_rows = []
    temp = max(float(temperature), 1e-6)

    for mlus in mlu_records:
        values = np.array([float(mlus[name]) for name in SELECTOR_NAMES], dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            regrets = np.ones(len(SELECTOR_NAMES), dtype=np.float32)
            targets = np.ones(len(SELECTOR_NAMES), dtype=np.float32) / len(SELECTOR_NAMES)
        else:
            finite_values = values[finite]
            best = float(np.min(finite_values))
            denom = max(abs(best), 1e-6)
            regrets = np.full(len(SELECTOR_NAMES), regret_clip, dtype=np.float64)
            regrets[finite] = np.clip((values[finite] - best) / denom, 0.0, regret_clip)
            logits = -regrets / temp
            logits[~finite] = -1e9
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            targets = probs / max(np.sum(probs), 1e-12)
        soft_targets.append(np.asarray(targets, dtype=np.float32))
        regret_rows.append(np.asarray(regrets, dtype=np.float32))

    return np.stack(soft_targets), np.stack(regret_rows)


def compute_features_and_oracle_for_topology(M, dataset, path_library, timesteps, gnnplus_model, k_crit):
    ecmp_base = M["ecmp_splits"](path_library)
    capacities = np.asarray(dataset.capacities, dtype=float)
    num_nodes = len(dataset.nodes)
    num_edges = len(dataset.edges)
    weights = np.asarray(dataset.weights, dtype=float)

    features_list = []
    labels_list = []
    valid_timesteps = []
    mlu_records = []

    for t_idx in timesteps:
        tm = np.asarray(dataset.tm[t_idx], dtype=float)
        if np.max(tm) < 1e-12:
            continue

        routing_ecmp = M["apply_routing"](tm, ecmp_base, path_library, capacities)
        telemetry = M["compute_reactive_telemetry"](tm, ecmp_base, path_library, routing_ecmp, weights)
        ecmp_link_utils = (
            np.array(routing_ecmp.utilization, dtype=np.float32)
            if hasattr(routing_ecmp, "utilization")
            else None
        )

        selector_results, gnnplus_info, _ = run_all_experts(
            M,
            tm,
            ecmp_base,
            path_library,
            capacities,
            dataset,
            gnnplus_model,
            k_crit,
            telemetry=telemetry,
        )

        feats = M["extract_features"](
            tm,
            alias_selector_results_for_features(selector_results),
            num_nodes,
            num_edges,
            k_crit,
            gnn_info=gnnplus_info,
            ecmp_link_utils=ecmp_link_utils,
        )
        mlus = compute_expert_mlus(M, tm, selector_results, ecmp_base, path_library, capacities)
        best_name = min(mlus, key=mlus.get)
        best_label = SELECTOR_NAMES.index(best_name)

        features_list.append(feats)
        labels_list.append(best_label)
        valid_timesteps.append(int(t_idx))
        mlu_records.append(mlus)

    if features_list:
        soft_targets, regret_costs = build_soft_targets_and_regret_costs(mlu_records)
        return (
            np.stack(features_list),
            np.array(labels_list, dtype=np.int64),
            valid_timesteps,
            mlu_records,
            soft_targets,
            regret_costs,
        )
    return (
        np.zeros((0, 49)),
        np.array([], dtype=np.int64),
        [],
        [],
        np.zeros((0, NUM_SELECTORS), dtype=np.float32),
        np.zeros((0, NUM_SELECTORS), dtype=np.float32),
    )


def build_oracle_rows(dataset_key: str, timesteps: list[int], labels: np.ndarray, mlu_records: list[dict[str, float]]):
    rows = []
    for t_idx, label, mlus in zip(timesteps, labels.tolist(), mlu_records):
        oracle_selector = SELECTOR_NAMES[int(label)]
        rows.append(
            {
                "dataset": dataset_key,
                "timestep": int(t_idx),
                "oracle_selector": oracle_selector,
                "oracle_mlu": float(mlus[oracle_selector]),
                "bottleneck": float(mlus["bottleneck"]),
                "topk": float(mlus["topk"]),
                "sensitivity": float(mlus["sensitivity"]),
                "gnnplus": float(mlus["gnnplus"]),
            }
        )
    return rows


def evaluate_on_topology(M, dataset, path_library, gate, gnnplus_model, k_crit, oracle_df: pd.DataFrame):
    ds_key = dataset.key
    test_indices = M["split_indices"](dataset, "test")
    ecmp_base = M["ecmp_splits"](path_library)
    capacities = np.asarray(dataset.capacities, dtype=float)
    num_nodes = len(dataset.nodes)
    num_edges = len(dataset.edges)
    weights = np.asarray(dataset.weights, dtype=float)

    oracle_by_ts = oracle_df.set_index("timestep") if not oracle_df.empty else pd.DataFrame()
    results = []
    decisions = []

    for t_idx in test_indices:
        tm = np.asarray(dataset.tm[t_idx], dtype=float)
        if np.max(tm) < 1e-12:
            continue

        t_total_start = time.perf_counter()

        routing_ecmp = M["apply_routing"](tm, ecmp_base, path_library, capacities)
        telemetry = M["compute_reactive_telemetry"](tm, ecmp_base, path_library, routing_ecmp, weights)
        ecmp_link_utils = (
            np.array(routing_ecmp.utilization, dtype=np.float32)
            if hasattr(routing_ecmp, "utilization")
            else None
        )

        selector_results, gnnplus_info, expert_timing = run_all_experts(
            M,
            tm,
            ecmp_base,
            path_library,
            capacities,
            dataset,
            gnnplus_model,
            k_crit,
            telemetry=telemetry,
        )

        t_feat_start = time.perf_counter()
        feats = M["extract_features"](
            tm,
            alias_selector_results_for_features(selector_results),
            num_nodes,
            num_edges,
            k_crit,
            gnn_info=gnnplus_info,
            ecmp_link_utils=ecmp_link_utils,
        )
        t_feat_ms = (time.perf_counter() - t_feat_start) * 1000

        t_mlp_start = time.perf_counter()
        pred_class, probs = gate.predict(feats)
        t_mlp_ms = (time.perf_counter() - t_mlp_start) * 1000
        pred_name = SELECTOR_NAMES[pred_class]
        confidence = float(probs[pred_class])

        t_lp_start = time.perf_counter()
        selected_ods = selector_results[pred_name]
        lp = M["solve_selected_path_lp"](
            tm,
            selected_ods,
            ecmp_base,
            path_library,
            capacities,
            time_limit_sec=LT,
        )
        routing = M["apply_routing"](tm, lp.splits, path_library, capacities)
        t_lp_ms = (time.perf_counter() - t_lp_start) * 1000

        t_total_ms = (time.perf_counter() - t_total_start) * 1000
        metagate_mlu = float(routing.mlu)
        t_decision_ms = sum(expert_timing.values()) + t_feat_ms + t_mlp_ms

        oracle_selector = "unknown"
        oracle_mlu = metagate_mlu
        bn_mlu = topk_mlu = sens_mlu = gnnplus_mlu = np.nan
        if not oracle_by_ts.empty and t_idx in oracle_by_ts.index:
            row = oracle_by_ts.loc[t_idx]
            oracle_selector = str(row["oracle_selector"])
            oracle_mlu = float(row["oracle_mlu"])
            bn_mlu = float(row.get("bottleneck", np.nan))
            topk_mlu = float(row.get("topk", np.nan))
            sens_mlu = float(row.get("sensitivity", np.nan))
            gnnplus_mlu = float(row.get("gnnplus", np.nan))

        topo_type = "unseen" if ds_key in UNSEEN_TOPOLOGIES else "known"
        results.append(
            {
                "dataset": ds_key,
                "topology_type": topo_type,
                "timestep": int(t_idx),
                "metagate_selector": pred_name,
                "metagate_confidence": confidence,
                "metagate_mlu": metagate_mlu,
                "oracle_selector": oracle_selector,
                "oracle_mlu": oracle_mlu,
                "bn_mlu": bn_mlu,
                "topk_mlu": topk_mlu,
                "sens_mlu": sens_mlu,
                "gnnplus_mlu": gnnplus_mlu,
                "correct": 1 if pred_name == oracle_selector else 0,
                "t_bn_ms": expert_timing["bottleneck"],
                "t_topk_ms": expert_timing["topk"],
                "t_sens_ms": expert_timing["sensitivity"],
                "t_gnnplus_ms": expert_timing["gnnplus"],
                "t_features_ms": t_feat_ms,
                "t_mlp_ms": t_mlp_ms,
                "t_lp_ms": t_lp_ms,
                "t_decision_ms": t_decision_ms,
                "t_total_ms": t_total_ms,
            }
        )
        decisions.append(
            {
                "dataset": ds_key,
                "topology_type": topo_type,
                "timestep": int(t_idx),
                "predicted": pred_name,
                "oracle": oracle_selector,
                "confidence": confidence,
                "prob_bn": float(probs[0]),
                "prob_topk": float(probs[1]),
                "prob_sens": float(probs[2]),
                "prob_gnnplus": float(probs[3]),
            }
        )

    return results, decisions


def main():
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  4-EXPERT MLP METAGATE EVALUATION (GNN+ expert version)")
    print("  Experts: {Bottleneck, TopK, Sensitivity, GNN+}")
    print("  Unified gate trained on 6 known topologies")
    print("  Per-topology calibration: 10 val TMs -> Bayesian prior fusion")
    print("  Unseen test: Germany50, VtlWavenet2011")
    print("=" * 72)

    M = setup()

    print(f"\nLoading GNN+ model from {GNNPLUS_CHECKPOINT}...")
    if not GNNPLUS_CHECKPOINT.exists():
        print(f"  ERROR: GNN+ checkpoint not found at {GNNPLUS_CHECKPOINT}")
        sys.exit(1)
    gnnplus_model, gnnplus_cfg = M["load_gnn_plus"](str(GNNPLUS_CHECKPOINT), device=DEVICE)
    print("  GNN+ model loaded successfully")

    bundle = M["load_bundle"](CONFIG_PATH)
    eval_specs = M["collect_specs"](bundle, "eval_topologies")
    gen_specs = M["collect_specs"](bundle, "generalization_topologies")
    max_steps = M["max_steps_from_args"](bundle, MAX_STEPS)

    known_datasets = []
    unseen_datasets = []
    for spec in eval_specs:
        try:
            dataset, pl = M["load_named_dataset"](bundle, spec, max_steps)
            known_datasets.append((dataset, pl))
        except Exception as e:
            print(f"  Skip {spec.key}: {e}")
    for spec in gen_specs:
        try:
            dataset, pl = M["load_named_dataset"](bundle, spec, max_steps)
            unseen_datasets.append((dataset, pl))
        except Exception as e:
            print(f"  Skip {spec.key}: {e}")

    print(f"  Known topologies: {len(known_datasets)}, Unseen: {len(unseen_datasets)}")

    all_train_X = []
    all_train_y = []
    all_train_soft = []
    all_train_regret = []
    all_val_X = []
    all_val_y = []
    all_val_soft = []
    all_val_regret = []
    train_distribution_rows = []

    print("\n" + "=" * 72)
    print("  PHASE 1: Computing 4-class oracle on KNOWN topologies (train+val)")
    print("=" * 72)
    for dataset, pl in known_datasets:
        ds_key = dataset.key
        train_indices = M["split_indices"](dataset, "train")
        val_indices = M["split_indices"](dataset, "val")

        print(f"\n  {ds_key}: computing oracle for {len(train_indices)} train TMs...")
        t0 = time.time()
        train_X, train_y, train_ts, train_mlus, train_soft, train_regret = compute_features_and_oracle_for_topology(
            M,
            dataset,
            pl,
            train_indices,
            gnnplus_model,
            K_CRIT,
        )
        elapsed = time.time() - t0
        print(f"    Train: {len(train_X)} samples in {elapsed:.1f}s")

        if len(train_X) > 0:
            counts = {}
            for idx, name in enumerate(SELECTOR_NAMES):
                c = int(np.sum(train_y == idx))
                counts[name] = c
                print(f"      {name}: {c}/{len(train_y)} ({100 * c / len(train_y):.0f}%)")
            train_distribution_rows.append(
                {
                    "dataset": ds_key,
                    "samples": int(len(train_y)),
                    "bottleneck": counts["bottleneck"],
                    "topk": counts["topk"],
                    "sensitivity": counts["sensitivity"],
                    "gnnplus": counts["gnnplus"],
                }
            )
            all_train_X.append(train_X)
            all_train_y.append(train_y)
            all_train_soft.append(train_soft)
            all_train_regret.append(train_regret)

        print(f"    Computing val oracle for {len(val_indices)} TMs...")
        val_X, val_y, val_ts, val_mlus, val_soft, val_regret = compute_features_and_oracle_for_topology(
            M,
            dataset,
            pl,
            val_indices,
            gnnplus_model,
            K_CRIT,
        )
        print(f"    Val: {len(val_X)} samples")
        if len(val_X) > 0:
            all_val_X.append(val_X)
            all_val_y.append(val_y)
            all_val_soft.append(val_soft)
            all_val_regret.append(val_regret)

    pooled_train_X = np.concatenate(all_train_X, axis=0)
    pooled_train_y = np.concatenate(all_train_y, axis=0)
    pooled_train_soft = np.concatenate(all_train_soft, axis=0)
    pooled_train_regret = np.concatenate(all_train_regret, axis=0)
    pooled_val_X = np.concatenate(all_val_X, axis=0) if all_val_X else None
    pooled_val_y = np.concatenate(all_val_y, axis=0) if all_val_y else None
    pooled_val_soft = np.concatenate(all_val_soft, axis=0) if all_val_soft else None
    pooled_val_regret = np.concatenate(all_val_regret, axis=0) if all_val_regret else None

    print(f"\n  Pooled training set: {len(pooled_train_X)} samples from {len(known_datasets)} topologies")
    print(f"  Pooled validation set: {len(pooled_val_X) if pooled_val_X is not None else 0} samples")
    overall_train_counts = {}
    print("  Overall oracle distribution:")
    for idx, name in enumerate(SELECTOR_NAMES):
        c = int(np.sum(pooled_train_y == idx))
        overall_train_counts[name] = c
        print(f"    {name}: {c}/{len(pooled_train_y)} ({100 * c / len(pooled_train_y):.0f}%)")

    print("\n" + "=" * 72)
    print("  PHASE 2: Training UNIFIED MLP MetaGate (GNN+ expert version)")
    print("=" * 72)

    config = M["MetaGateConfig"](
        hidden_dim=128,
        dropout=0.3,
        learning_rate=5e-4,
        num_epochs=300,
        batch_size=64,
        use_soft_labels=True,
        soft_label_temperature=SOFT_LABEL_TEMPERATURE,
        regret_loss_weight=REGRET_LOSS_WEIGHT,
        regret_clip=REGRET_CLIP,
    )
    gate = M["DynamicMetaGate"](config)
    train_acc, val_acc = gate.train(
        pooled_train_X,
        pooled_train_y,
        pooled_val_X,
        pooled_val_y,
        soft_targets=pooled_train_soft,
        regret_costs=pooled_train_regret,
        val_soft_targets=pooled_val_soft,
        val_regret_costs=pooled_val_regret,
    )
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Val accuracy:   {val_acc:.3f}")

    gate.save(MODEL_DIR / "metagate_gnnplus_unified.pt")
    print(f"  Model saved to {MODEL_DIR / 'metagate_gnnplus_unified.pt'}")

    print("\n" + "=" * 72)
    print(f"  PHASE 3: Calibration ({N_CALIB} val TMs) + Evaluation on ALL 8 topologies")
    print("=" * 72)

    all_results = []
    all_decisions = []
    test_oracle_rows = []
    calibration_rows = []

    for dataset, pl in known_datasets + unseen_datasets:
        ds_key = dataset.key
        topo_type = "UNSEEN" if ds_key in UNSEEN_TOPOLOGIES else "known"
        print(f"\n  [{topo_type}] {ds_key}...")

        val_indices = M["split_indices"](dataset, "val")
        calib_indices = val_indices[:N_CALIB]
        print(f"    Calibrating on {len(calib_indices)} val TMs...")
        try:
            _, calib_labels, _, _, _, _ = compute_features_and_oracle_for_topology(
                M,
                dataset,
                pl,
                calib_indices,
                gnnplus_model,
                K_CRIT,
            )
            if len(calib_labels) > 0:
                win_counts = np.bincount(calib_labels, minlength=NUM_SELECTORS)
                gate.calibrate(win_counts, smoothing=1.0, strength=5.0)
                prior = gate._calibration_prior
                calibration_rows.append(
                    {
                        "dataset": ds_key,
                        "topology_type": "unseen" if ds_key in UNSEEN_TOPOLOGIES else "known",
                        "bottleneck_prior": float(prior[0]),
                        "topk_prior": float(prior[1]),
                        "sensitivity_prior": float(prior[2]),
                        "gnnplus_prior": float(prior[3]),
                    }
                )
                print(
                    f"    Calibration prior: BN={prior[0]:.2f} TopK={prior[1]:.2f} "
                    f"Sens={prior[2]:.2f} GNN+={prior[3]:.2f}"
                )
            else:
                gate.clear_calibration()
                calibration_rows.append(
                    {
                        "dataset": ds_key,
                        "topology_type": "unseen" if ds_key in UNSEEN_TOPOLOGIES else "known",
                        "bottleneck_prior": np.nan,
                        "topk_prior": np.nan,
                        "sensitivity_prior": np.nan,
                        "gnnplus_prior": np.nan,
                    }
                )
                print("    No valid calibration TMs, using raw MLP predictions")
        except Exception as e:
            print(f"    Calibration failed: {e}, using raw MLP predictions")
            gate.clear_calibration()

        test_indices = M["split_indices"](dataset, "test")
        print(f"    Computing test oracle for {len(test_indices)} TMs...")
        _, test_y, test_ts, test_mlus, _, _ = compute_features_and_oracle_for_topology(
            M,
            dataset,
            pl,
            test_indices,
            gnnplus_model,
            K_CRIT,
        )
        oracle_rows = build_oracle_rows(ds_key, test_ts, test_y, test_mlus)
        test_oracle_rows.extend(oracle_rows)
        oracle_df = pd.DataFrame(oracle_rows)
        if not oracle_df.empty:
            print(f"    Test oracle distribution: {oracle_df['oracle_selector'].value_counts().to_dict()}")

        try:
            results, decisions = evaluate_on_topology(
                M,
                dataset,
                pl,
                gate,
                gnnplus_model,
                K_CRIT,
                oracle_df,
            )
            all_results.extend(results)
            all_decisions.extend(decisions)

            if results:
                df = pd.DataFrame(results)
                acc = df["correct"].mean()
                mg_mlu = df["metagate_mlu"].mean()
                sel_counts = df["metagate_selector"].value_counts()

                print(f"    Accuracy: {acc:.1%} ({int(df['correct'].sum())}/{len(df)})")
                print(f"    MetaGate MLU: {mg_mlu:.6f}")
                print("    Selector distribution:")
                for name in SELECTOR_NAMES:
                    c = int(sel_counts.get(name, 0))
                    print(f"      {name}: {c}/{len(df)} ({100 * c / len(df):.0f}%)")

                changes = int((df["metagate_selector"].values[1:] != df["metagate_selector"].values[:-1]).sum())
                print(f"    Selector switches: {changes}/{max(len(df) - 1, 1)}")

                print(
                    f"    Timing (mean): BN={df['t_bn_ms'].mean():.1f}ms "
                    f"TopK={df['t_topk_ms'].mean():.1f}ms "
                    f"Sens={df['t_sens_ms'].mean():.1f}ms "
                    f"GNN+={df['t_gnnplus_ms'].mean():.1f}ms "
                    f"Feat={df['t_features_ms'].mean():.2f}ms "
                    f"MLP={df['t_mlp_ms'].mean():.3f}ms "
                    f"LP={df['t_lp_ms'].mean():.1f}ms "
                    f"Total={df['t_total_ms'].mean():.1f}ms"
                )
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback

            traceback.print_exc()

    if not all_results:
        print("\nNo results generated!")
        return

    results_df = pd.DataFrame(all_results)
    decisions_df = pd.DataFrame(all_decisions)
    oracle_df = pd.DataFrame(test_oracle_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    train_dist_df = pd.DataFrame(train_distribution_rows)

    results_df.to_csv(OUTPUT_DIR / "metagate_results.csv", index=False)
    decisions_df.to_csv(OUTPUT_DIR / "metagate_decisions.csv", index=False)
    oracle_df.to_csv(OUTPUT_DIR / "test_oracle.csv", index=False)
    calibration_df.to_csv(OUTPUT_DIR / "calibration_priors.csv", index=False)
    train_dist_df.to_csv(OUTPUT_DIR / "train_oracle_distribution.csv", index=False)

    summary = (
        results_df.groupby("dataset")
        .agg(
            topology_type=("topology_type", "first"),
            accuracy=("correct", "mean"),
            metagate_mlu=("metagate_mlu", "mean"),
            oracle_mlu=("oracle_mlu", "mean"),
            bn_mlu=("bn_mlu", "mean"),
            topk_mlu=("topk_mlu", "mean"),
            sens_mlu=("sens_mlu", "mean"),
            gnnplus_mlu=("gnnplus_mlu", "mean"),
            t_decision_ms=("t_decision_ms", "mean"),
            t_lp_ms=("t_lp_ms", "mean"),
            t_total_ms=("t_total_ms", "mean"),
            n_timesteps=("timestep", "count"),
        )
        .reset_index()
    )
    summary["best_forced_mlu"] = summary[["bn_mlu", "topk_mlu", "sens_mlu", "gnnplus_mlu"]].min(axis=1)
    summary["metagate_vs_oracle_gap_pct"] = (
        (summary["metagate_mlu"] - summary["oracle_mlu"]) / summary["oracle_mlu"] * 100.0
    )
    summary.to_csv(OUTPUT_DIR / "metagate_summary.csv", index=False)

    timing_summary = (
        results_df.groupby("dataset")
        .agg(
            t_bn_ms=("t_bn_ms", "mean"),
            t_topk_ms=("t_topk_ms", "mean"),
            t_sens_ms=("t_sens_ms", "mean"),
            t_gnnplus_ms=("t_gnnplus_ms", "mean"),
            t_features_ms=("t_features_ms", "mean"),
            t_mlp_ms=("t_mlp_ms", "mean"),
            t_lp_ms=("t_lp_ms", "mean"),
            t_decision_ms=("t_decision_ms", "mean"),
            t_total_ms=("t_total_ms", "mean"),
        )
        .reset_index()
    )
    timing_summary.to_csv(OUTPUT_DIR / "metagate_timing.csv", index=False)

    total_acc = float(results_df["correct"].mean())
    known_acc = float(results_df[results_df["topology_type"] == "known"]["correct"].mean())
    unseen_df = results_df[results_df["topology_type"] == "unseen"]
    unseen_acc = float(unseen_df["correct"].mean()) if not unseen_df.empty else 0.0

    training_summary = {
        "generated_at": datetime.now().isoformat(),
        "selector_names": SELECTOR_NAMES,
        "gnnplus_checkpoint": str(GNNPLUS_CHECKPOINT),
        "gnnplus_source_checkpoint": str(GNNPLUS_SOURCE_CHECKPOINT),
        "gnnplus_config": {
            "node_dim": int(gnnplus_cfg.node_dim),
            "edge_dim": int(gnnplus_cfg.edge_dim),
            "od_dim": int(gnnplus_cfg.od_dim),
            "hidden_dim": int(gnnplus_cfg.hidden_dim),
            "dropout": float(gnnplus_cfg.dropout),
            "learn_k_crit": bool(gnnplus_cfg.learn_k_crit),
            "k_crit_min": int(gnnplus_cfg.k_crit_min),
            "k_crit_max": int(gnnplus_cfg.k_crit_max),
        },
        "metagate_config": {
            "hidden_dim": int(config.hidden_dim),
            "dropout": float(config.dropout),
            "learning_rate": float(config.learning_rate),
            "num_epochs": int(config.num_epochs),
            "batch_size": int(config.batch_size),
            "use_soft_labels": bool(config.use_soft_labels),
            "soft_label_temperature": float(config.soft_label_temperature),
            "regret_loss_weight": float(config.regret_loss_weight),
            "regret_clip": float(config.regret_clip),
            "feature_clip": float(config.feature_clip),
        },
        "pooled_train_samples": int(len(pooled_train_X)),
        "pooled_val_samples": int(len(pooled_val_X) if pooled_val_X is not None else 0),
        "train_class_counts": overall_train_counts,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "overall_test_accuracy": total_acc,
        "known_test_accuracy": known_acc,
        "unseen_test_accuracy": unseen_acc,
        "calibration_size": int(N_CALIB),
    }
    (OUTPUT_DIR / "training_summary.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("  FINAL SUMMARY: 4-EXPERT MLP METAGATE WITH GNN+")
    print("=" * 72)
    print(
        f"\n{'Topology':<28} {'Type':<7} {'Acc':>6} {'MG_MLU':>12} {'Oracle':>12} "
        f"{'BestForced':>12} {'vs_Oracle':>10} {'Decision':>8} {'Total':>8}"
    )
    print("-" * 114)
    for _, row in summary.iterrows():
        print(
            f"{row['dataset']:<28} {row['topology_type']:<7} {row['accuracy']:>5.1%} "
            f"{row['metagate_mlu']:>12.6f} {row['oracle_mlu']:>12.6f} "
            f"{row['best_forced_mlu']:>12.6f} {row['metagate_vs_oracle_gap_pct']:>+9.2f}% "
            f"{row['t_decision_ms']:>7.1f}ms {row['t_total_ms']:>7.1f}ms"
        )

    print(f"\n  Overall accuracy:     {total_acc:.1%}")
    print(f"  Known topo accuracy:  {known_acc:.1%}")
    print(f"  Unseen topo accuracy: {unseen_acc:.1%}")
    print(f"\n  Mean decision time:   {results_df['t_decision_ms'].mean():.1f}ms")
    print(f"  Mean LP time:         {results_df['t_lp_ms'].mean():.1f}ms")
    print(f"  Mean total time:      {results_df['t_total_ms'].mean():.1f}ms")
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
