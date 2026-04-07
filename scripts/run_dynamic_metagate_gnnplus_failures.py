#!/usr/bin/env python3
"""Evaluate the trained MetaGate+GNN+ model under failure scenarios."""

from __future__ import annotations

import json
import os
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from phase1_reactive.drl.dynamic_meta_gate import DynamicMetaGate, MetaGateConfig

CONFIG_PATH = "configs/phase1_reactive_full.yaml"
MAX_STEPS = 500
K_CRIT = 40
DEVICE = "cpu"
N_CALIB = 10
FEAT_DIM = 49
DISABLE_CALIBRATION = os.environ.get("METAGATE_GNNPLUS_DISABLE_CALIBRATION", "0") == "1"

OUTPUT_DIR = Path(
    os.environ.get(
        "METAGATE_GNNPLUS_OUTPUT_DIR",
        str(PROJECT_ROOT / "results" / "dynamic_metagate_gnnplus"),
    )
).resolve()
MODEL_PATH = OUTPUT_DIR / "models" / "metagate_gnnplus_unified.pt"
FAILURE_RESULTS_CSV = OUTPUT_DIR / "metagate_failure_results.csv"
FAILURE_SUMMARY_CSV = OUTPUT_DIR / "metagate_failure_summary.csv"
FAILURE_DECISIONS_CSV = OUTPUT_DIR / "metagate_failure_decisions.csv"
FAILURE_CALIB_CSV = OUTPUT_DIR / "metagate_failure_calibration.csv"
FAILURE_AUDIT_JSON = OUTPUT_DIR / "metagate_failure_summary.json"
FAILURE_SDN_METRICS_CSV = OUTPUT_DIR / "metagate_failure_sdn_metrics.csv"

FAILURE_SCENARIOS = [
    "single_link_failure",
    "random_link_failure_1",
    "random_link_failure_2",
    "capacity_degradation_50",
    "traffic_spike_2x",
]


def load_normal_module():
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "run_dynamic_metagate_gnnplus_eval.py"
    spec = importlib.util.spec_from_file_location("run_dynamic_metagate_gnnplus_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_normal_module()
SELECTOR_NAMES = list(BASE.SELECTOR_NAMES)
KNOWN_TOPOLOGIES = set(BASE.KNOWN_TOPOLOGIES)
UNSEEN_TOPOLOGIES = set(BASE.UNSEEN_TOPOLOGIES)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


def apply_failure_inputs(M, dataset, path_library, tm_vector, scenario: str, timestep: int):
    capacities = np.asarray(dataset.capacities, dtype=float)
    weights = np.asarray(dataset.weights, dtype=float)
    ecmp_base = M["ecmp_splits"](path_library)

    normal_routing = M["apply_routing"](tm_vector, ecmp_base, path_library, capacities)
    pre_failure_mlu = float(normal_routing.mlu)
    failure_mask = np.ones_like(capacities, dtype=float)
    failed_links = []

    if scenario == "single_link_failure":
        util = np.asarray(normal_routing.utilization, dtype=float)
        fail_idx = int(np.argmax(util))
        failure_mask[fail_idx] = 0.0
        failed_links = [fail_idx]
    elif scenario == "random_link_failure_1":
        rng = np.random.default_rng(stable_seed(dataset.key, scenario, timestep))
        fail_idx = int(rng.integers(0, len(capacities)))
        failure_mask[fail_idx] = 0.0
        failed_links = [fail_idx]
    elif scenario == "random_link_failure_2":
        rng = np.random.default_rng(stable_seed(dataset.key, scenario, timestep))
        count = min(2, len(capacities))
        failed_links = sorted(rng.choice(len(capacities), size=count, replace=False).astype(int).tolist())
        for idx in failed_links:
            failure_mask[idx] = 0.0
    elif scenario == "capacity_degradation_50":
        util = np.asarray(normal_routing.utilization, dtype=float)
        degraded = np.where(util > 0.5)[0].astype(int).tolist()
        failed_links = degraded
        for idx in degraded:
            failure_mask[idx] = 0.5
    elif scenario == "traffic_spike_2x":
        pass
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    if scenario == "traffic_spike_2x":
        effective_tm = np.asarray(tm_vector, dtype=float).copy()
        top_demands = np.argsort(effective_tm)[-min(K_CRIT, len(effective_tm)) :]
        effective_tm[top_demands] *= 2.0
        effective_caps = capacities.copy()
    else:
        effective_tm = np.asarray(tm_vector, dtype=float)
        effective_caps = capacities * failure_mask

    post_failure_routing = M["apply_routing"](effective_tm, ecmp_base, path_library, effective_caps)
    return {
        "effective_tm": effective_tm,
        "effective_caps": effective_caps,
        "weights": weights,
        "ecmp_base": ecmp_base,
        "pre_failure_mlu": pre_failure_mlu,
        "post_failure_ecmp_mlu": float(post_failure_routing.mlu),
        "failure_mask": failure_mask,
        "failed_links": failed_links,
    }


def evaluate_failure_timestep(M, dataset, path_library, timestep: int, scenario: str, gnnplus_model):
    tm_vector = np.asarray(dataset.tm[timestep], dtype=float)
    if np.max(tm_vector) < 1e-12:
        return None

    failure_ctx = apply_failure_inputs(M, dataset, path_library, tm_vector, scenario, timestep)
    effective_tm = failure_ctx["effective_tm"]
    effective_caps = failure_ctx["effective_caps"]
    ecmp_base = failure_ctx["ecmp_base"]

    routing_ecmp = M["apply_routing"](effective_tm, ecmp_base, path_library, effective_caps)
    telemetry = M["compute_reactive_telemetry"](
        effective_tm,
        ecmp_base,
        path_library,
        routing_ecmp,
        failure_ctx["weights"],
    )
    ecmp_link_utils = (
        np.asarray(routing_ecmp.utilization, dtype=np.float32)
        if hasattr(routing_ecmp, "utilization")
        else None
    )

    expert_start = time.perf_counter()
    selector_results, gnnplus_info, expert_timing = BASE.run_all_experts(
        M,
        effective_tm,
        ecmp_base,
        path_library,
        effective_caps,
        dataset,
        gnnplus_model,
        K_CRIT,
        telemetry=telemetry,
    )
    expert_total_ms = (time.perf_counter() - expert_start) * 1000

    feat_start = time.perf_counter()
    features = M["extract_features"](
        effective_tm,
        BASE.alias_selector_results_for_features(selector_results),
        len(dataset.nodes),
        len(dataset.edges),
        K_CRIT,
        gnn_info=gnnplus_info,
        ecmp_link_utils=ecmp_link_utils,
    )
    feat_ms = (time.perf_counter() - feat_start) * 1000

    expert_mlus = {}
    expert_lp_ms = {}
    expert_splits = {}
    expert_routings = {}
    for name in SELECTOR_NAMES:
        lp_start = time.perf_counter()
        try:
            lp = M["solve_selected_path_lp"](
                effective_tm,
                selector_results[name],
                ecmp_base,
                path_library,
                effective_caps,
                time_limit_sec=BASE.LT,
            )
            routing = M["apply_routing"](effective_tm, lp.splits, path_library, effective_caps)
            expert_mlus[name] = float(routing.mlu)
            expert_splits[name] = [np.asarray(s, dtype=float).copy() for s in lp.splits]
            expert_routings[name] = routing
        except Exception:
            expert_mlus[name] = float("inf")
            expert_splits[name] = [np.asarray(s, dtype=float).copy() for s in ecmp_base]
            expert_routings[name] = M["apply_routing"](effective_tm, ecmp_base, path_library, effective_caps)
        expert_lp_ms[name] = (time.perf_counter() - lp_start) * 1000

    return {
        "features": features,
        "expert_mlus": expert_mlus,
        "expert_lp_ms": expert_lp_ms,
        "expert_splits": expert_splits,
        "expert_routings": expert_routings,
        "expert_timing": expert_timing,
        "expert_total_ms": expert_total_ms,
        "failure_ctx": failure_ctx,
        "selector_results": selector_results,
        "gnnplus_info": gnnplus_info,
        "feat_ms": feat_ms,
    }


def calibrate_for_failure(M, gate, dataset, path_library, scenario: str, gnnplus_model):
    if DISABLE_CALIBRATION:
        gate.clear_calibration()
        return np.zeros(len(SELECTOR_NAMES), dtype=np.int64), 0
    val_indices = M["split_indices"](dataset, "val")[:N_CALIB]
    counts = np.zeros(len(SELECTOR_NAMES), dtype=np.int64)
    valid = 0
    for timestep in val_indices:
        step = evaluate_failure_timestep(M, dataset, path_library, int(timestep), scenario, gnnplus_model)
        if step is None:
            continue
        oracle_name = min(step["expert_mlus"], key=step["expert_mlus"].get)
        counts[SELECTOR_NAMES.index(oracle_name)] += 1
        valid += 1
    if valid > 0:
        gate.calibrate(counts, smoothing=1.0, strength=5.0)
    else:
        gate.clear_calibration()
    return counts, valid


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained MetaGate model: {MODEL_PATH}")

    M = BASE.setup()
    gnnplus_model, _ = M["load_gnn_plus"](str(BASE.GNNPLUS_CHECKPOINT), device=DEVICE)

    gate = DynamicMetaGate(MetaGateConfig(hidden_dim=128, dropout=0.3, learning_rate=5e-4, num_epochs=300, batch_size=64))
    gate.load(MODEL_PATH, feat_dim=FEAT_DIM)

    bundle = M["load_bundle"](CONFIG_PATH)
    eval_specs = M["collect_specs"](bundle, "eval_topologies")
    gen_specs = M["collect_specs"](bundle, "generalization_topologies")
    max_steps = M["max_steps_from_args"](bundle, MAX_STEPS)

    datasets = []
    for spec in eval_specs + gen_specs:
        try:
            dataset, pl = M["load_named_dataset"](bundle, spec, max_steps)
            datasets.append((dataset, pl))
        except Exception as exc:
            print(f"Skip {spec.key}: {exc}")

    results_rows = []
    decisions_rows = []
    calibration_rows = []

    print("=" * 72)
    print("  METAGATE + GNN+ FAILURE EVALUATION")
    print("  Scenarios:", ", ".join(FAILURE_SCENARIOS))
    print("=" * 72)

    for dataset, path_library in datasets:
        topo_type = "unseen" if dataset.key in UNSEEN_TOPOLOGIES else "known"
        test_indices = M["split_indices"](dataset, "test")
        topo_mapping = M["SDNTopologyMapping"].from_mininet(dataset.nodes, dataset.edges, dataset.od_pairs)
        for scenario in FAILURE_SCENARIOS:
            print(f"\n[{topo_type}] {dataset.key} / {scenario}")
            counts, valid = calibrate_for_failure(M, gate, dataset, path_library, scenario, gnnplus_model)
            prior = getattr(gate, "_calibration_prior", None)
            current_splits = [np.asarray(s, dtype=float).copy() for s in M["ecmp_splits"](path_library)]
            current_groups, _ = M["build_ecmp_baseline_rules"](path_library, topo_mapping, dataset.edges)
            prev_latency_by_od = None
            calibration_rows.append(
                {
                    "dataset": dataset.key,
                    "topology_type": topo_type,
                    "scenario": scenario,
                    "valid_calibration_samples": int(valid),
                    "bottleneck_prior": float(prior[0]) if prior is not None else np.nan,
                    "topk_prior": float(prior[1]) if prior is not None else np.nan,
                    "sensitivity_prior": float(prior[2]) if prior is not None else np.nan,
                    "gnnplus_prior": float(prior[3]) if prior is not None else np.nan,
                    "oracle_bottleneck": int(counts[0]),
                    "oracle_topk": int(counts[1]),
                    "oracle_sensitivity": int(counts[2]),
                    "oracle_gnnplus": int(counts[3]),
                }
            )
            if DISABLE_CALIBRATION:
                print("  Pure zero-shot mode: calibration disabled")
            elif prior is not None:
                print(
                    f"  Calibration prior: BN={prior[0]:.2f} TopK={prior[1]:.2f} "
                    f"Sens={prior[2]:.2f} GNN+={prior[3]:.2f}"
                )

            for timestep in test_indices:
                step = evaluate_failure_timestep(M, dataset, path_library, int(timestep), scenario, gnnplus_model)
                if step is None:
                    continue

                mlp_start = time.perf_counter()
                pred_class, probs = gate.predict(step["features"])
                mlp_ms = (time.perf_counter() - mlp_start) * 1000
                pred_name = SELECTOR_NAMES[pred_class]
                oracle_name = min(step["expert_mlus"], key=step["expert_mlus"].get)
                pred_mlu = float(step["expert_mlus"][pred_name])
                oracle_mlu = float(step["expert_mlus"][oracle_name])
                decision_ms = (
                    float(step["expert_total_ms"])
                    + float(step["feat_ms"])
                    + float(mlp_ms)
                )
                total_ms = decision_ms + float(step["expert_lp_ms"][pred_name])
                selected_splits = step["expert_splits"][pred_name]
                selected_ods = step["selector_results"][pred_name]
                selected_routing = step["expert_routings"][pred_name]
                ctx = step["failure_ctx"]
                rule_start = time.perf_counter()
                new_groups, _ = M["splits_to_openflow_rules"](
                    selected_splits,
                    selected_ods,
                    path_library,
                    topo_mapping,
                    dataset.edges,
                )
                changed_groups = M["compute_rule_diff"](current_groups, new_groups)
                flow_table_updates = int(len(changed_groups))
                rule_install_delay_ms = (time.perf_counter() - rule_start) * 1000
                telemetry_selected = M["compute_reactive_telemetry"](
                    ctx["effective_tm"],
                    selected_splits,
                    path_library,
                    selected_routing,
                    ctx["weights"],
                    prev_latency_by_od=prev_latency_by_od,
                )
                disturbance = float(M["compute_disturbance"](current_splits, selected_splits, ctx["effective_tm"]))
                failure_recovery_ms = float(total_ms + rule_install_delay_ms)

                results_rows.append(
                    {
                        "dataset": dataset.key,
                        "topology_type": topo_type,
                        "scenario": scenario,
                        "timestep": int(timestep),
                        "metagate_selector": pred_name,
                        "metagate_confidence": float(probs[pred_class]),
                        "metagate_mlu": pred_mlu,
                        "oracle_selector": oracle_name,
                        "oracle_mlu": oracle_mlu,
                        "bn_mlu": float(step["expert_mlus"]["bottleneck"]),
                        "topk_mlu": float(step["expert_mlus"]["topk"]),
                        "sens_mlu": float(step["expert_mlus"]["sensitivity"]),
                        "gnnplus_mlu": float(step["expert_mlus"]["gnnplus"]),
                        "correct": 1 if pred_name == oracle_name else 0,
                        "pre_failure_mlu": float(ctx["pre_failure_mlu"]),
                        "post_failure_ecmp_mlu": float(ctx["post_failure_ecmp_mlu"]),
                        "failed_links": int(len(ctx["failed_links"])),
                        "throughput": float(telemetry_selected.throughput),
                        "mean_latency": float(telemetry_selected.mean_latency),
                        "p95_latency": float(telemetry_selected.p95_latency),
                        "packet_loss": float(telemetry_selected.packet_loss),
                        "jitter": float(telemetry_selected.jitter),
                        "disturbance": disturbance,
                        "flow_table_updates": flow_table_updates,
                        "rule_install_delay_ms": float(rule_install_delay_ms),
                        "failure_recovery_ms": failure_recovery_ms,
                        "t_bn_ms": float(step["expert_timing"]["bottleneck"]),
                        "t_topk_ms": float(step["expert_timing"]["topk"]),
                        "t_sens_ms": float(step["expert_timing"]["sensitivity"]),
                        "t_gnnplus_ms": float(step["expert_timing"]["gnnplus"]),
                        "t_features_ms": float(step["feat_ms"]),
                        "t_mlp_ms": float(mlp_ms),
                        "t_selected_lp_ms": float(step["expert_lp_ms"][pred_name]),
                        "t_decision_ms": float(decision_ms),
                        "t_total_ms": float(total_ms),
                    }
                )
                decisions_rows.append(
                    {
                        "dataset": dataset.key,
                        "topology_type": topo_type,
                        "scenario": scenario,
                        "timestep": int(timestep),
                        "predicted": pred_name,
                        "oracle": oracle_name,
                        "confidence": float(probs[pred_class]),
                        "prob_bn": float(probs[0]),
                        "prob_topk": float(probs[1]),
                        "prob_sens": float(probs[2]),
                        "prob_gnnplus": float(probs[3]),
                    }
                )
                current_splits = selected_splits
                current_groups = new_groups
                prev_latency_by_od = np.asarray(telemetry_selected.latency_by_od, dtype=float)

            sub = pd.DataFrame([r for r in results_rows if r["dataset"] == dataset.key and r["scenario"] == scenario])
            if not sub.empty:
                counts = sub["metagate_selector"].value_counts()
                print(
                    f"  Accuracy={sub['correct'].mean():.1%}, "
                    f"GNN+%={100.0 * counts.get('gnnplus', 0) / len(sub):.0f}%, "
                    f"Gap={((sub['metagate_mlu'].mean() - sub['oracle_mlu'].mean()) / sub['oracle_mlu'].mean()) * 100.0:+.2f}%"
                )

    results_df = pd.DataFrame(results_rows)
    decisions_df = pd.DataFrame(decisions_rows)
    calibration_df = pd.DataFrame(calibration_rows)

    summary_df = (
        results_df.groupby(["dataset", "topology_type", "scenario"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            metagate_mlu=("metagate_mlu", "mean"),
            oracle_mlu=("oracle_mlu", "mean"),
            bn_mlu=("bn_mlu", "mean"),
            topk_mlu=("topk_mlu", "mean"),
            sens_mlu=("sens_mlu", "mean"),
            gnnplus_mlu=("gnnplus_mlu", "mean"),
            pre_failure_mlu=("pre_failure_mlu", "mean"),
            post_failure_ecmp_mlu=("post_failure_ecmp_mlu", "mean"),
            throughput=("throughput", "mean"),
            mean_latency=("mean_latency", "mean"),
            p95_latency=("p95_latency", "mean"),
            packet_loss=("packet_loss", "mean"),
            jitter=("jitter", "mean"),
            mean_disturbance=("disturbance", "mean"),
            flow_table_updates=("flow_table_updates", "mean"),
            rule_install_delay_ms=("rule_install_delay_ms", "mean"),
            failure_recovery_ms=("failure_recovery_ms", "mean"),
            t_decision_ms=("t_decision_ms", "mean"),
            t_total_ms=("t_total_ms", "mean"),
            n_timesteps=("timestep", "count"),
        )
    )
    summary_df["metagate_vs_oracle_gap_pct"] = (
        (summary_df["metagate_mlu"] - summary_df["oracle_mlu"]) / summary_df["oracle_mlu"] * 100.0
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(FAILURE_RESULTS_CSV, index=False)
    summary_df.to_csv(FAILURE_SUMMARY_CSV, index=False)
    decisions_df.to_csv(FAILURE_DECISIONS_CSV, index=False)
    calibration_df.to_csv(FAILURE_CALIB_CSV, index=False)
    (
        summary_df[[
            "dataset",
            "topology_type",
            "scenario",
            "metagate_mlu",
            "throughput",
            "mean_disturbance",
            "mean_latency",
            "p95_latency",
            "packet_loss",
            "jitter",
            "flow_table_updates",
            "rule_install_delay_ms",
            "failure_recovery_ms",
            "t_decision_ms",
            "t_total_ms",
        ]]
        .rename(
            columns={
                "dataset": "topology",
                "topology_type": "status",
                "metagate_mlu": "mean_mlu",
                "mean_latency": "mean_latency_au",
                "p95_latency": "p95_latency_au",
                "jitter": "jitter_au",
                "t_decision_ms": "decision_time_ms",
            }
        )
        .assign(status=lambda df: df["status"].str.lower())
        .to_csv(FAILURE_SDN_METRICS_CSV, index=False)
    )

    overall = {
        "scenarios": FAILURE_SCENARIOS,
        "rows": int(len(results_df)),
        "datasets": sorted(results_df["dataset"].astype(str).unique().tolist()),
        "overall_accuracy": float(results_df["correct"].mean()),
    }
    FAILURE_AUDIT_JSON.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(f"\nSaved failure results to {FAILURE_RESULTS_CSV}")


if __name__ == "__main__":
    main()
