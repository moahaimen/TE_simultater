#!/usr/bin/env python3
"""
GNN+ eval-only diagnostic — no retraining, no hyperparameter changes.

Produces two bundles:
  Task 1: results/gnnplus_failuregate_evalonly_c03/
    - Base checkpoint: results/gnn_plus_retrained_fixedk40/gnn_plus_fixed_k40.pt
    - Failure-history gate: ON (current source behaviour)

  Task 2: results/gnnplus_step1to5_failgate_gateoff_eval/
    - Base checkpoint: results/gnnplus_step1to5_failgate/training/gnn_plus_improved_fixedk40.pt
    - Failure-history gate: OFF (prev_selected/prev_disturbance passed through even during failures)

Both bundles record:
  - packet_sdn_summary.csv      (normal scenarios, all methods)
  - packet_sdn_failure.csv      (failure scenarios, all methods)
  - packet_sdn_sdn_metrics.csv
  - experiment_audit.md
  - selector_latency.csv        (selector-only ms, separate from whole-cycle decision_time_ms)

Usage
-----
  # Task 1 — gate on, base checkpoint
  python scripts/run_gnnplus_evalonly_diagnostic.py --mode gate-on

  # Task 2 — gate off, step1to5 checkpoint
  python scripts/run_gnnplus_evalonly_diagnostic.py --mode gate-off
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent.parent / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from phase1_reactive.drl.gnn_plus_selector import (
    GNNPlusFlowSelector,
    build_graph_tensors_plus,
    build_od_features_plus,
    load_gnn_plus,
)
from phase1_reactive.routing.path_cache import assert_selected_ods_have_paths, surviving_od_mask
from te.baselines import ecmp_splits, select_bottleneck_critical
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.simulator import apply_routing

DEVICE = "cpu"
K_CRIT = 40
LP_TIME_LIMIT = 20
NUM_RUNS = 3
SEED = 42
FEATURE_VARIANT = "section7_temporal"

KNOWN_TOPOLOGIES = ["abilene", "cernet", "geant", "ebone", "sprintlink", "tiscali"]
UNSEEN_TOPOLOGIES = ["germany50", "vtlwavenet2011"]
ALL_TOPOLOGIES = KNOWN_TOPOLOGIES + UNSEEN_TOPOLOGIES
CORE_METHODS = ["ecmp", "bottleneck", "gnn", "gnnplus"]

FAILURE_SCENARIOS = [
    "single_link_failure",
    "random_link_failure_1",
    "random_link_failure_2",
    "capacity_degradation_50",
    "traffic_spike_2x",
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_runner():
    runner_path = PROJECT_ROOT / "scripts" / "run_gnnplus_packet_sdn_full.py"
    spec = importlib.util.spec_from_file_location("_runner_full", runner_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_runner_full"] = mod
    spec.loader.exec_module(mod)
    return mod


def _has_active_failure(failure_mask) -> bool:
    if failure_mask is None:
        return False
    return bool(np.any(np.asarray(failure_mask, dtype=np.float64) > 0.5))


def gnnplus_select(
    model: GNNPlusFlowSelector,
    *,
    dataset,
    path_library,
    tm_vector,
    telemetry,
    prev_tm,
    prev_util,
    prev_selected_indicator,
    prev_disturbance: float,
    continuity_bonus: float,
    k_crit: int,
    failure_mask=None,
    gate_active: bool = True,
) -> tuple[list[int], dict, float]:
    """Select critical flows; returns (selected, info, selector_ms).

    gate_active=True  — zeros prev_selected/prev_disturbance/continuity_bonus during failures
    gate_active=False — passes prev values through regardless of failure status
    """
    active_failure = _has_active_failure(failure_mask)

    if gate_active and active_failure:
        eff_psi = None
        eff_pd = 0.0
        eff_cb = 0.0
    else:
        eff_psi = prev_selected_indicator
        eff_pd = float(prev_disturbance)
        eff_cb = float(continuity_bonus)

    t0 = time.perf_counter()
    graph_data = build_graph_tensors_plus(
        dataset,
        tm_vector=tm_vector,
        path_library=path_library,
        telemetry=telemetry,
        prev_util=prev_util,
        prev_tm=prev_tm,
        prev_selected_indicator=eff_psi,
        prev_disturbance=eff_pd,
        failure_mask=failure_mask,
        feature_variant=FEATURE_VARIANT,
        device=DEVICE,
    )
    od_data = build_od_features_plus(
        dataset,
        tm_vector,
        path_library,
        telemetry=telemetry,
        prev_tm=prev_tm,
        prev_util=prev_util,
        prev_selected_indicator=eff_psi,
        prev_disturbance=eff_pd,
        failure_mask=failure_mask,
        feature_variant=FEATURE_VARIANT,
        device=DEVICE,
    )
    active_mask = (
        (np.asarray(tm_vector, dtype=np.float64) > 1e-12) & surviving_od_mask(path_library)
    ).astype(np.float32)
    selected, info = model.select_critical_flows(
        graph_data=graph_data,
        od_data=od_data,
        active_mask=active_mask,
        k_crit_default=k_crit,
        force_default_k=True,
        prev_selected_indicator=eff_psi,
        continuity_bonus=eff_cb,
    )
    selector_ms = (time.perf_counter() - t0) * 1000.0
    assert_selected_ods_have_paths(path_library, selected, context=f"{dataset.key}:gnnplus_diag")
    return selected, info, selector_ms


def eval_normal(runner, topo_key: str, gnn_cache: dict, gnnplus_model, gate_active: bool, continuity_bonus: float) -> tuple[list[dict], list[dict]]:
    """Run normal scenario eval. Returns (sdn_rows, selector_latency_rows)."""
    dataset, path_library = runner.load_dataset(topo_key)
    capacities = np.asarray(dataset.capacities, dtype=float)
    weights = np.asarray(dataset.weights, dtype=float)
    ecmp_base = ecmp_splits(path_library)
    topo_mapping = runner.SDNTopologyMapping.from_mininet(dataset.nodes, dataset.edges, dataset.od_pairs)
    test_indices = list(range(int(dataset.split["test_start"]), dataset.tm.shape[0]))

    gnn_model = gnn_cache.get(topo_key)
    if gnn_model is None:
        gnn_model = runner.load_gnn_model(dataset, path_library)
        gnn_cache[topo_key] = gnn_model

    sdn_rows, lat_rows = [], []
    for method in CORE_METHODS:
        acc = defaultdict(list)
        sel_times = []
        for _ in range(NUM_RUNS):
            current_splits = [s.copy() for s in ecmp_base]
            current_groups, _ = runner.build_ecmp_baseline_rules(path_library, topo_mapping, dataset.edges)
            prev_latency = None
            psi = np.zeros(len(dataset.od_pairs), dtype=np.float32)
            prev_dist = 0.0
            prev_tm = None
            prev_util = None
            for t_idx in test_indices:
                tm_vec = dataset.tm[t_idx]
                if method == "gnnplus":
                    t_total = time.perf_counter()
                    routing_pre = apply_routing(tm_vec, current_splits, path_library, capacities)
                    pre_tel = runner.compute_telemetry(
                        tm_vector=tm_vec, splits=current_splits, path_library=path_library,
                        routing=routing_pre, weights=weights, prev_latency_by_od=prev_latency,
                    )
                    selected, _, sel_ms = gnnplus_select(
                        gnnplus_model, dataset=dataset, path_library=path_library,
                        tm_vector=tm_vec, telemetry=pre_tel, prev_tm=prev_tm,
                        prev_util=prev_util, prev_selected_indicator=psi,
                        prev_disturbance=prev_dist, continuity_bonus=continuity_bonus,
                        k_crit=K_CRIT, failure_mask=None, gate_active=gate_active,
                    )
                    sel_times.append(sel_ms)
                    lp = runner.solve_selected_path_lp_safe(
                        tm_vector=tm_vec, selected_ods=selected, base_splits=ecmp_base,
                        path_library=path_library, capacities=capacities,
                        time_limit_sec=LP_TIME_LIMIT, context=f"{topo_key}:gnnplus:normal",
                    )
                    new_splits = [s.copy() for s in lp.splits]
                    decision_ms = (time.perf_counter() - t_total) * 1000.0
                    new_groups, _ = runner.splits_to_openflow_rules(
                        new_splits, selected, path_library, topo_mapping, dataset.edges)
                    changed = runner.compute_rule_diff(current_groups, new_groups)
                    t_rule = time.perf_counter()
                    rule_delay = (time.perf_counter() - t_rule) * 1000.0
                    routing_post = apply_routing(tm_vec, new_splits, path_library, capacities)
                    tel_post = runner.compute_telemetry(
                        tm_vector=tm_vec, splits=new_splits, path_library=path_library,
                        routing=routing_post, weights=weights, prev_latency_by_od=prev_latency,
                    )
                    dist = compute_disturbance(
                        current_splits, new_splits, np.asarray(tm_vec, dtype=float))
                    acc["post_mlus"].append(float(routing_post.mlu))
                    acc["disturbances"].append(float(dist))
                    acc["throughputs"].append(float(tel_post.throughput))
                    acc["latencies"].append(float(tel_post.mean_latency))
                    acc["p95_latencies"].append(float(tel_post.p95_latency))
                    acc["packet_losses"].append(float(tel_post.packet_loss))
                    acc["jitters"].append(float(tel_post.jitter))
                    acc["decision_times"].append(decision_ms)
                    acc["flow_updates"].append(float(len(changed)))
                    acc["rule_delays"].append(rule_delay)
                    prev_latency = tel_post.latency_by_od
                    psi_arr = np.zeros(len(dataset.od_pairs), dtype=np.float32)
                    for od in selected:
                        if od < len(psi_arr):
                            psi_arr[int(od)] = 1.0
                    psi = psi_arr
                    prev_dist = float(dist)
                    prev_tm = np.asarray(tm_vec, dtype=float)
                    prev_util = np.asarray(tel_post.utilization, dtype=float)
                    current_splits = new_splits
                    current_groups = new_groups
                else:
                    result, current_splits, current_groups, prev_latency = runner.run_sdn_cycle(
                        tm_vector=tm_vec, method=method, dataset=dataset, path_library=path_library,
                        ecmp_base=ecmp_base, current_splits=current_splits, current_groups=current_groups,
                        topo_mapping=topo_mapping, capacities=capacities, weights=weights,
                        gnn_model=gnn_model, gnnplus_model=None, prev_latency_by_od=prev_latency,
                    )
                    acc["post_mlus"].append(result.post_mlu)
                    acc["disturbances"].append(result.disturbance)
                    acc["throughputs"].append(result.throughput)
                    acc["latencies"].append(result.mean_latency)
                    acc["p95_latencies"].append(result.p95_latency)
                    acc["packet_losses"].append(result.packet_loss)
                    acc["jitters"].append(result.jitter)
                    acc["decision_times"].append(result.decision_time_ms)
                    acc["flow_updates"].append(result.flow_table_updates)
                    acc["rule_delays"].append(result.rule_install_delay_ms)

        sdn_rows.append({
            "topology": topo_key, "status": "known" if topo_key in KNOWN_TOPOLOGIES else "unseen",
            "method": method, "scenario": "normal",
            "nodes": len(dataset.nodes), "edges": len(dataset.edges),
            "mean_mlu": float(np.mean(acc["post_mlus"])),
            "mean_disturbance": float(np.mean(acc["disturbances"])),
            "throughput": float(np.mean(acc["throughputs"])),
            "mean_latency_au": float(np.mean(acc["latencies"])),
            "p95_latency_au": float(np.mean(acc["p95_latencies"])),
            "packet_loss": float(np.mean(acc["packet_losses"])),
            "jitter_au": float(np.mean(acc["jitters"])),
            "decision_time_ms": float(np.mean(acc["decision_times"])),
            "flow_table_updates": float(np.mean(acc["flow_updates"])),
            "rule_install_delay_ms": float(np.mean(acc["rule_delays"])),
        })
        if sel_times:
            lat_rows.append({
                "topology": topo_key, "scenario": "normal", "method": "gnnplus",
                "gate_active": gate_active,
                "mean_selector_ms": float(np.mean(sel_times)),
                "p95_selector_ms": float(np.percentile(sel_times, 95)),
                "n_measurements": len(sel_times),
                "note": "selector-only: feature_build+model_forward+sort (excludes LP, telemetry, rule_diff)",
            })
        print(f"[normal] {topo_key} {method} mlu={sdn_rows[-1]['mean_mlu']:.4f}", flush=True)
    return sdn_rows, lat_rows


def eval_failures(runner, topo_key: str, gnn_cache: dict, gnnplus_model, gate_active: bool, continuity_bonus: float) -> tuple[list[dict], list[dict]]:
    """Run failure scenario eval. Returns (failure_rows, selector_latency_rows)."""
    dataset, path_library = runner.load_dataset(topo_key)
    capacities = np.asarray(dataset.capacities, dtype=float)
    weights = np.asarray(dataset.weights, dtype=float)
    ecmp_base = ecmp_splits(path_library)
    gnn_model = gnn_cache.get(topo_key)
    if gnn_model is None:
        gnn_model = runner.load_gnn_model(dataset, path_library)
        gnn_cache[topo_key] = gnn_model
    test_indices = list(range(int(dataset.split["test_start"]), dataset.tm.shape[0]))
    sample_indices = test_indices[:: max(1, len(test_indices) // 10)]

    fail_rows, lat_rows = [], []
    for scenario in FAILURE_SCENARIOS:
        for method in CORE_METHODS:
            acc = defaultdict(list)
            sel_times = []
            for t_idx in sample_indices:
                tm_vec = dataset.tm[t_idx]
                if method == "gnnplus":
                    fs = runner._build_failure_execution_state(
                        scenario=scenario, tm_vector=tm_vec, dataset=dataset,
                        path_library=path_library, capacities=capacities, weights=weights,
                        normal_routing=apply_routing(tm_vec, ecmp_base, path_library, capacities),
                    )
                    e_tm = fs["effective_tm"]
                    e_caps = fs["effective_caps"]
                    e_pl = fs["effective_path_library"]
                    e_ds = fs["effective_dataset"]
                    e_ecmp = fs["effective_ecmp"]
                    fmask = fs["failure_mask"]
                    e_w = fs["effective_weights"]
                    pre_routing = apply_routing(e_tm, e_ecmp, e_pl, e_caps)
                    pre_failure_mlu = float(apply_routing(tm_vec, ecmp_base, path_library, capacities).mlu)
                    tel = runner.compute_telemetry(
                        tm_vector=e_tm, splits=e_ecmp, path_library=e_pl,
                        routing=pre_routing, weights=e_w, prev_latency_by_od=None,
                    )
                    t_start = time.perf_counter()
                    selected, _, sel_ms = gnnplus_select(
                        gnnplus_model, dataset=e_ds, path_library=e_pl,
                        tm_vector=e_tm, telemetry=tel, prev_tm=None, prev_util=None,
                        prev_selected_indicator=np.zeros(len(e_ds.od_pairs), dtype=np.float32),
                        prev_disturbance=0.0, continuity_bonus=continuity_bonus,
                        k_crit=K_CRIT, failure_mask=fmask, gate_active=gate_active,
                    )
                    sel_times.append(sel_ms)
                    lp = runner.solve_selected_path_lp_safe(
                        tm_vector=e_tm, selected_ods=selected, base_splits=e_ecmp,
                        path_library=e_pl, capacities=e_caps, time_limit_sec=LP_TIME_LIMIT,
                        context=f"{topo_key}:{scenario}:gnnplus",
                    )
                    recovery_ms = (time.perf_counter() - t_start) * 1000.0
                    post_mlu = float(apply_routing(e_tm, [s.copy() for s in lp.splits], e_pl, e_caps).mlu)
                    acc["recovery_times"].append(recovery_ms)
                    acc["pre_mlus"].append(pre_failure_mlu)
                    acc["post_mlus"].append(post_mlu)
                else:
                    rec_ms, pre_mlu, post_mlu, _ = runner.run_failure_scenario(
                        scenario=scenario, tm_vector=tm_vec, method=method, dataset=dataset,
                        path_library=path_library, ecmp_base=ecmp_base, capacities=capacities,
                        weights=weights, topo_mapping=None, gnn_model=gnn_model, gnnplus_model=None,
                    )
                    acc["recovery_times"].append(rec_ms)
                    acc["pre_mlus"].append(pre_mlu)
                    acc["post_mlus"].append(post_mlu)

            fail_rows.append({
                "topology": topo_key, "status": "known" if topo_key in KNOWN_TOPOLOGIES else "unseen",
                "method": method, "scenario": scenario,
                "nodes": len(dataset.nodes), "edges": len(dataset.edges),
                "mean_mlu": float(np.mean(acc["post_mlus"])),
                "pre_failure_mlu": float(np.mean(acc["pre_mlus"])),
                "failure_recovery_ms": float(np.mean(acc["recovery_times"])),
            })
            if sel_times:
                lat_rows.append({
                    "topology": topo_key, "scenario": scenario, "method": "gnnplus",
                    "gate_active": gate_active,
                    "mean_selector_ms": float(np.mean(sel_times)),
                    "p95_selector_ms": float(np.percentile(sel_times, 95)),
                    "n_measurements": len(sel_times),
                    "note": "selector-only ms (excludes LP solve)",
                })
            print(f"[failure] {topo_key} {scenario} {method} mlu={fail_rows[-1]['mean_mlu']:.4f}", flush=True)
    return fail_rows, lat_rows


def prepare_sdn_metrics(summary_df: pd.DataFrame, failure_df: pd.DataFrame) -> pd.DataFrame:
    recovery = (
        failure_df.groupby(["topology", "method"], as_index=False)["failure_recovery_ms"]
        .mean()
        .rename(columns={"failure_recovery_ms": "avg_failure_recovery_ms"})
    )
    metrics = summary_df.merge(recovery, on=["topology", "method"], how="left")
    return metrics


def write_audit(out_dir: Path, checkpoint_path: Path, gate_active: bool,
                summary_df: pd.DataFrame, failure_df: pd.DataFrame, selector_df: pd.DataFrame) -> None:
    gate_str = "active (zeros prev_selected/prev_disturbance during failures)" if gate_active else "DISABLED (prev_selected/prev_disturbance passed through during failures)"
    lines = [
        "# GNN+ Eval-Only Diagnostic Audit",
        "",
        f"- Checkpoint: `{checkpoint_path.relative_to(PROJECT_ROOT)}`",
        f"- Failure-history gate: {gate_str}",
        f"- Feature variant: {FEATURE_VARIANT}",
        f"- Fixed K: {K_CRIT}",
        f"- No retraining performed",
        f"- Zero-shot topologies (unseen): {UNSEEN_TOPOLOGIES}",
        f"- Normal rows: {len(summary_df)}",
        f"- Failure rows: {len(failure_df)}",
        f"- Selector latency rows: {len(selector_df)}",
        f"- Methods evaluated: {CORE_METHODS}",
        f"- Topologies: {ALL_TOPOLOGIES}",
        "",
        "## Latency metric definitions",
        "- `decision_time_ms` in packet_sdn_summary.csv: whole-cycle (telemetry + selection + LP solve + rule_diff)",
        "- `mean_selector_ms` in selector_latency.csv: GNN+ only (feature_build + model_forward + argsort), excludes LP",
        "",
        "## Zero-shot guards",
        "- bayesian_calibration_used: false",
        "- few_shot_adaptation_used: false",
        "- per_topology_tuning_used: false",
        "- metagate_used: false",
    ]
    (out_dir / "experiment_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gate-on", "gate-off"], required=True,
                        help="gate-on: base checkpoint + gate active; gate-off: step1to5 checkpoint + gate disabled")
    args = parser.parse_args()

    gate_active = (args.mode == "gate-on")

    if gate_active:
        checkpoint_path = PROJECT_ROOT / "results" / "gnn_plus_retrained_fixedk40" / "gnn_plus_fixed_k40.pt"
        out_dir = PROJECT_ROOT / "results" / "gnnplus_failuregate_evalonly_c03"
        continuity_bonus = 0.05  # default pre-step5 value (matches empty bundle manifest)
        tag = "gnnplus_failuregate_evalonly_c03"
    else:
        checkpoint_path = PROJECT_ROOT / "results" / "gnnplus_step1to5_failgate" / "training" / "gnn_plus_improved_fixedk40.pt"
        out_dir = PROJECT_ROOT / "results" / "gnnplus_step1to5_failgate_gateoff_eval"
        continuity_bonus = 0.02  # matching step1to5_failgate training
        tag = "gnnplus_step1to5_failgate_gateoff_eval"

    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    seed_all(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison").mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "training").mkdir(exist_ok=True)

    print(f"[diag] mode={args.mode} gate_active={gate_active}", flush=True)
    print(f"[diag] checkpoint={checkpoint_path}", flush=True)
    print(f"[diag] out_dir={out_dir}", flush=True)

    print("[diag] loading model...", flush=True)
    model, cfg = load_gnn_plus(str(checkpoint_path), device=DEVICE)
    model.cfg.feature_variant = FEATURE_VARIANT
    model.cfg.learn_k_crit = False
    model.cfg.k_crit_min = K_CRIT
    model.cfg.k_crit_max = K_CRIT
    model.eval()

    print("[diag] loading runner...", flush=True)
    runner = load_runner()

    gnn_cache: dict = {}
    normal_rows: list[dict] = []
    normal_lat: list[dict] = []
    failure_rows: list[dict] = []
    failure_lat: list[dict] = []

    for topo in ALL_TOPOLOGIES:
        print(f"\n[diag] === {topo} (normal) ===", flush=True)
        nr, nl = eval_normal(runner, topo, gnn_cache, model, gate_active, continuity_bonus)
        normal_rows.extend(nr)
        normal_lat.extend(nl)

    for topo in ALL_TOPOLOGIES:
        print(f"\n[diag] === {topo} (failures) ===", flush=True)
        fr, fl = eval_failures(runner, topo, gnn_cache, model, gate_active, continuity_bonus)
        failure_rows.extend(fr)
        failure_lat.extend(fl)

    summary_df = pd.DataFrame(normal_rows)
    failure_df = pd.DataFrame(failure_rows)
    selector_df = pd.DataFrame(normal_lat + failure_lat)
    metrics_df = prepare_sdn_metrics(summary_df, failure_df)

    summary_df.to_csv(out_dir / "packet_sdn_summary.csv", index=False)
    failure_df.to_csv(out_dir / "packet_sdn_failure.csv", index=False)
    metrics_df.to_csv(out_dir / "packet_sdn_sdn_metrics.csv", index=False)
    selector_df.to_csv(out_dir / "selector_latency.csv", index=False)

    manifest = {
        "base_objective": "zero-shot generalization",
        "methods": CORE_METHODS,
        "known_topologies": KNOWN_TOPOLOGIES,
        "unseen_topologies": UNSEEN_TOPOLOGIES,
        "evaluation_topologies": ALL_TOPOLOGIES,
        "zero_shot_guards": {
            "bayesian_calibration_used": False,
            "few_shot_adaptation_used": False,
            "per_topology_tuning_used": False,
            "metagate_used": False,
        },
        "fixed_k": K_CRIT,
        "feature_variant": FEATURE_VARIANT,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "failure_history_gate": "active" if gate_active else "disabled",
        "continuity_bonus": continuity_bonus,
        "evaluation_mode": "eval_only_no_retrain",
        "latency_fix": "failure-path Python edge scan instead of per-path NumPy mask slicing",
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    write_audit(out_dir, checkpoint_path, gate_active, summary_df, failure_df, selector_df)

    print(f"\n[diag] DONE: {out_dir}", flush=True)
    print(f"[diag] Normal rows: {len(summary_df)}, Failure rows: {len(failure_df)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
