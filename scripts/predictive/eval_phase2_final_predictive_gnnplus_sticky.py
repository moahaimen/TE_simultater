"""Phase 2 FINAL: Predictive GNN+ Sticky Traffic Engineering.

This is the final main method that the professor asked for. It composes:

  Predictive Layer (GRU link-util forecast + heuristic failure-risk)
        ↓
  Predictive Feature Builder
        ↓
  Phase 1 GNN+ Selector  ←  re-uses gnnplus_select_stateful internals
        ↓
  Score-level fusion (alpha*GNN + beta*bn_pred + gamma*hotspot
                      + delta*failure + eta*alt_gain + rho*demand_growth)
        ↓
  Top-K  re-selection
        ↓
  Sticky Post-Filter        ←  Phase 1 _sticky_compose_selection unchanged
        ↓
  LP Split Optimization     ←  Phase 1 solve_selected_path_lp_safe unchanged
        ↓
  Do-No-Harm Gate           ←  Phase 1 apply_do_no_harm_gate (the exact same fn)
        ↓
  SDN Simulator

We re-use the production Phase 1 modules wherever possible. The GNN+
checkpoint, the MoE gate, the sticky filter, and the do-no-harm gate
are all unchanged. We only add a fusion layer between the GNN+ selector
output and the do-no-harm gate input.

When weights = FusionWeights.phase1_only(), Phase 2 Final reduces
exactly to Phase 1 GNN+ Sticky.

Usage:
    python scripts/predictive/eval_phase2_final_predictive_gnnplus_sticky.py \\
        --topologies abilene,cernet \\
        --scenarios normal,spike_2x,flash_crowd \\
        --k 10,20,40 --delay 0,1,2 \\
        --weight_profile balanced
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "predictive"))

from phase2_6_common import (  # noqa: E402
    DATA_ROOT, summarize_mlu, recovery_time, make_scenario_fn,
)
from phase2_final_fusion import (  # noqa: E402
    FusionWeights, compute_predictive_features, predictive_rerank_top_k,
    derive_failure_risk_from_predicted_util, compute_effective_capacities,
)

# ── Pull production Phase 1 modules ────────────────────────────────────
from run_gnnplus_improved_fixedk40_experiment import (  # noqa: E402
    apply_do_no_harm_gate, compute_candidate_pool, candidate_indices_from_od_data,
    load_final_checkpoint_summary, K_CRIT, INFER_CANDIDATE_POOL_SIZE,
    FEATURE_VARIANT, DEVICE, LP_TIME_LIMIT, do_no_harm_threshold_for_topology,
)
from phase1_reactive.drl.gnn_plus_selector import (  # noqa: E402
    build_graph_tensors_plus, build_od_features_plus,
)
from phase1_reactive.routing.path_cache import surviving_od_mask  # noqa: E402
from scripts.run_gnnplus_packet_sdn_full import (  # noqa: E402
    load_dataset, solve_selected_path_lp_safe,
)
from te.baselines import ecmp_splits  # noqa: E402
from te.disturbance import compute_disturbance  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "results" / "phase2_final"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

ALL_TOPOLOGIES = [
    "abilene", "cernet", "ebone", "geant",
    "sprintlink", "tiscali", "germany50", "vtlwavenet2011",
]


# ─────────────────────────────────────────────────────────────────────
# Lightweight runner adapter (we don't need the full SDNRunner; LP fn is enough)
# ─────────────────────────────────────────────────────────────────────
class _MinimalRunner:
    """Just exposes solve_selected_path_lp_safe to apply_do_no_harm_gate."""

    @staticmethod
    def solve_selected_path_lp_safe(*args, **kwargs):
        return solve_selected_path_lp_safe(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────
# GNN+ score-only — replicates gnnplus_select_stateful but exposes scores
# ─────────────────────────────────────────────────────────────────────
def gnnplus_score_only(
    model, *,
    dataset, path_library, capacities, tm_vector,
    telemetry, prev_tm, prev_util, prev_selected_indicator,
    prev_disturbance: float, k_crit: int,
    failure_mask: np.ndarray | None = None,
    gate_temperature: float = 1.0,
):
    """Return raw GNN+ scores + candidate pool for downstream fusion.

    This is the exact preamble of gnnplus_select_stateful, stopping just
    before continuity_select. We need the scores tensor to do score-level
    fusion before final top-K.
    """
    candidate_indices = compute_candidate_pool(
        tm_vector=np.asarray(tm_vector, dtype=np.float64),
        path_library=path_library,
        capacities=np.asarray(capacities, dtype=float),
        k_crit=int(k_crit),
        candidate_limit=INFER_CANDIDATE_POOL_SIZE,
    )
    graph_data = build_graph_tensors_plus(
        dataset, tm_vector=tm_vector, path_library=path_library,
        telemetry=telemetry, prev_util=prev_util, prev_tm=prev_tm,
        prev_selected_indicator=prev_selected_indicator,
        prev_disturbance=prev_disturbance,
        failure_mask=failure_mask, feature_variant=FEATURE_VARIANT, device=DEVICE,
    )
    graph_data["gate_temperature"] = float(gate_temperature)
    od_data = build_od_features_plus(
        dataset, tm_vector, path_library,
        telemetry=telemetry, prev_tm=prev_tm, prev_util=prev_util,
        prev_selected_indicator=prev_selected_indicator,
        prev_disturbance=prev_disturbance,
        failure_mask=failure_mask, candidate_od_indices=candidate_indices,
        feature_variant=FEATURE_VARIANT, device=DEVICE,
    )
    with torch.no_grad():
        scores, _, info = model(graph_data, od_data)
    candidate_np = candidate_indices_from_od_data(od_data)
    return {
        "scores": scores.detach().cpu().numpy(),
        "candidate_indices": candidate_np,
        "info": {k: v for k, v in info.items() if not str(k).startswith("_")},
    }


# ─────────────────────────────────────────────────────────────────────
# Lightweight telemetry stub (we don't have a real SDN runner)
# ─────────────────────────────────────────────────────────────────────
class _Telemetry:
    """Minimal telemetry stub. The GNN+ feature builder reads:
       - .utilization (required)
       - .link_delay  (graph_data builder uses it)
       - .failure_mask (optional)
    For .link_delay we use a vector of zeros (no delay info) since this
    eval script does not run the SDN simulator -- the GNN+ MoE gate uses
    delay only as a soft feature, not as a hard signal.
    """

    def __init__(self, util: np.ndarray, num_links: int, failure_mask=None):
        self.utilization = np.asarray(util, dtype=np.float64)[:num_links]
        self.link_delay = np.zeros(int(num_links), dtype=np.float64)
        self.failure_mask = failure_mask


def compute_link_util_under_splits(
    *, tm_vector, splits, path_library, capacities, edges_count,
) -> np.ndarray:
    """Per-link utilization given splits and TM."""
    link_load = np.zeros(edges_count, dtype=np.float64)
    n_od = len(tm_vector)
    for od in range(n_od):
        if tm_vector[od] <= 1e-12:
            continue
        s = splits[od] if od < len(splits) else None
        if s is None or not hasattr(s, "__len__"):
            continue
        s = np.asarray(s, dtype=np.float64)
        paths = path_library.edge_idx_paths_by_od[od]
        if not paths:
            continue
        path_count = min(len(paths), s.size)
        for p in range(path_count):
            share = float(s[p]) if p < s.size else 0.0
            if share <= 0.0:
                continue
            for eidx in paths[p]:
                if 0 <= int(eidx) < edges_count:
                    link_load[int(eidx)] += float(tm_vector[od]) * share
    return link_load / np.maximum(capacities, 1e-9)


# ─────────────────────────────────────────────────────────────────────
# Main per-topology eval
# ─────────────────────────────────────────────────────────────────────
def evaluate_cell(
    *, topo: str, scenario: str, k: int, delay: int, weight_profile: str,
    methods: Sequence[str] = ("phase1", "current_apg", "predictive", "oracle"),
    max_steps: int | None = None,
    lp_time_limit: int = 5,
    prediction_horizon: int = 1,
    mpc_lambda: float = 0.0,
) -> dict:
    """Run the Phase 2 Final pipeline for one (topo, scenario, K, delay) cell.

    Methods compared:
      - phase1:       FusionWeights.phase1_only() — recovers Phase 1 GNN+ Sticky
      - current_apg:  GNN+ + alt-path-gain features computed from CURRENT util
      - predictive:   GNN+ + features from GRU-predicted util
      - oracle:       GNN+ + features from actual util[t+d] (upper bound)
    """
    print(f"\n[{topo}/{scenario}/K={k}/d={delay}] starting Phase 2 Final ...", flush=True)
    t_start = time.time()

    # 1) Load data + checkpoints
    dataset, path_library = load_dataset(topo)
    util_npz = np.load(DATA_ROOT / topo / "link_util_series.npz")
    util_series = util_npz["util"].astype(np.float64)
    pred_npz = np.load(DATA_ROOT / topo / "predicted_util_test.npz")
    pred_series = pred_npz["predicted_util"]   # (num_steps, max_horizon, num_links)
    test_start_idx = int(pred_npz["test_start_idx"])
    max_horizon = int(pred_npz["max_horizon"])
    split = json.loads((DATA_ROOT / topo / "split_indices.json").read_text())

    num_steps = util_series.shape[0]
    num_links = util_series.shape[1]
    num_od = len(dataset.od_pairs)
    capacities = np.asarray(dataset.capacities, dtype=np.float64)
    base_splits = ecmp_splits(path_library)

    test_start = test_start_idx
    test_end = num_steps - 1 - delay
    if max_steps is not None:
        test_end = min(test_end, test_start + int(max_steps))
    if test_end - test_start < 5:
        return {"topo": topo, "scenario": scenario, "k": k, "delay": delay,
                "error": "test window too small"}

    # 2) Apply scenario perturbation, anchored to test_start
    scenario_fn = make_scenario_fn(scenario, test_start=test_start)
    perturbed_tm = scenario_fn(dataset.tm.copy())

    # 3) Load GNN+ checkpoint
    gnnplus_model, _ = load_final_checkpoint_summary()
    gnnplus_model.eval()

    # 4) Set sticky env (Phase 1)
    import os
    os.environ.setdefault("GNNPLUS_STICKY_EPS", "0.005")
    runner = _MinimalRunner()

    # 5) Walk through cycles for each method
    method_weights = {
        "phase1": FusionWeights.phase1_only(),
        "current_apg": FusionWeights.balanced(),       # uses current util features
        "predictive": (FusionWeights.balanced() if weight_profile == "balanced"
                       else FusionWeights.conservative() if weight_profile == "conservative"
                       else FusionWeights.proactive()),
        "oracle": FusionWeights.balanced(),
    }
    out_per_method = {}

    for method in methods:
        weights = method_weights[method]
        prev_splits = None
        prev_util_obs = None
        prev_tm_obs = None
        prev_selected_indicator = np.zeros(num_od, dtype=np.float32)
        prev_disturbance = 0.0
        guard_cache = {}
        guard_cooldown = 0

        mlu_list = []
        splits_list = []
        score_ms_list = []
        lp_ms_list = []
        sticky_applied_list = []

        for t in range(test_start, test_end + 1):
            t_decision = time.time()
            tm_now = perturbed_tm[t]
            scale = float(tm_now.sum()) / max(float(dataset.tm[t].sum()), 1e-9)
            current_util = util_series[t] * scale

            telemetry = _Telemetry(current_util, num_links)

            # 5a) Choose the predicted-util horizon to use for feature building
            if method == "phase1":
                # Phase 1 uses no prediction features, but we still need to
                # populate the pred_horizon for compute_predictive_features.
                # With FusionWeights.phase1_only, the predictive features are
                # multiplied by 0, so their values don't matter.
                pred_horizon = current_util[None, :]
            elif method == "current_apg":
                # Use current util as the "prediction" — exposes the gain
                # purely from the alt-path-gain selector with no real
                # forecasting.
                pred_horizon = current_util[None, :]
            elif method == "predictive":
                # Use the GRU forecast over `prediction_horizon` cycles ahead.
                # This is Path 1: multi-step prediction so the predicted state
                # diverges meaningfully from current state.
                h = max(1, min(prediction_horizon, max_horizon))
                pred_horizon = pred_series[t, :h, :]
                # Apply scale to predictions to keep them consistent with
                # the perturbed traffic regime (the GRU was trained on
                # un-perturbed util).
                pred_horizon = pred_horizon * scale
            elif method == "oracle":
                # Cheats: actual util over the SAME horizon used by predictive
                h_steps = max(1, prediction_horizon)
                future_idx = [min(t + 1 + i, num_steps - 1) for i in range(h_steps)]
                oracle_util_series = []
                for fi in future_idx:
                    s_fi = float(perturbed_tm[fi].sum()) / max(float(dataset.tm[fi].sum()), 1e-9)
                    oracle_util_series.append(util_series[fi] * s_fi)
                pred_horizon = np.stack(oracle_util_series, axis=0)
            else:
                pred_horizon = current_util[None, :]

            # 5b) Get GNN+ raw scores on candidate pool
            try:
                gnn_pkg = gnnplus_score_only(
                    gnnplus_model,
                    dataset=dataset, path_library=path_library, capacities=capacities,
                    tm_vector=tm_now, telemetry=telemetry,
                    prev_tm=prev_tm_obs, prev_util=prev_util_obs,
                    prev_selected_indicator=prev_selected_indicator,
                    prev_disturbance=prev_disturbance, k_crit=k,
                )
            except Exception as exc:
                print(f"[{topo}/{scenario}/{method}] GNN+ failed at t={t}: {exc}", flush=True)
                continue

            scores = gnn_pkg["scores"]
            candidate_indices = gnn_pkg["candidate_indices"]
            if candidate_indices is None or len(candidate_indices) == 0:
                continue

            # 5c) Predictive features
            failure_risk = derive_failure_risk_from_predicted_util(pred_horizon)
            pf = compute_predictive_features(
                candidate_indices=candidate_indices,
                tm_now=tm_now, tm_prev=prev_tm_obs,
                predicted_util_horizon=pred_horizon,
                failure_risk_per_link=failure_risk,
                path_library=path_library,
            )

            # 5d) Score-level fusion + top-K
            tm_arr = np.asarray(tm_now, dtype=np.float64)
            active_mask_full = (tm_arr > 1e-12) & surviving_od_mask(path_library)
            selected_ods, _ = predictive_rerank_top_k(
                gnn_scores=scores, candidate_indices=candidate_indices,
                active_mask_full=active_mask_full, num_od=num_od,
                predictive_features=pf, k_crit=k, weights=weights,
            )
            if not selected_ods:
                continue

            # 5d-bis) Path 2 — Predictive MPC at the LP level.
            # Phase 1 always uses raw capacities. Other methods use effective
            # capacities derived from their respective predicted-util signal.
            # When mpc_lambda == 0 these are bit-identical to raw capacities.
            if method == "phase1":
                effective_capacities = capacities
            else:
                effective_capacities = compute_effective_capacities(
                    capacities, pred_horizon, mpc_lambda=mpc_lambda,
                )

            # 5e) Phase 1 sticky + LP + DNH gate (unchanged production code)
            t_lp = time.time()
            prev_selected_ods_list = [int(i) for i in np.where(prev_selected_indicator > 0.5)[0].tolist()]
            try:
                selected_ods, lp_result, gate_info, guard_cache, guard_cooldown = apply_do_no_harm_gate(
                    runner,
                    tm_vector=tm_arr, selected_ods=selected_ods,
                    base_splits=base_splits,
                    warm_start_splits=prev_splits, path_library=path_library,
                    capacities=effective_capacities, k_crit=k,
                    context=f"{topo}:phase2_final:{method}:t{t}",
                    topology_key=str(dataset.key),
                    guard_bottleneck_selected=list(candidate_indices[:min(k, len(candidate_indices))].tolist()),
                    guard_cache=guard_cache, step_index=t,
                    guard_fallback_cooldown=guard_cooldown,
                    prev_selected_ods=prev_selected_ods_list,
                )
                lp_ms = (time.time() - t_lp) * 1000.0
                new_splits = [s.copy() for s in lp_result.splits]
                sticky_applied_list.append(int(bool(gate_info.get("sticky_applied", False))))
            except Exception as exc:
                print(f"[{topo}/{scenario}/{method}] DNH gate failed at t={t}: {exc}", flush=True)
                continue

            # 5f) Evaluate on actual TM[t+d]
            tm_eval = perturbed_tm[t + delay]
            util_eval = compute_link_util_under_splits(
                tm_vector=tm_eval, splits=new_splits, path_library=path_library,
                capacities=capacities, edges_count=num_links,
            )
            mlu_list.append(float(util_eval.max()))
            splits_list.append(new_splits)
            score_ms_list.append((time.time() - t_decision) * 1000.0 - lp_ms)
            lp_ms_list.append(lp_ms)

            # Update state for next cycle
            new_indicator = np.zeros(num_od, dtype=np.float32)
            for od in selected_ods:
                if 0 <= od < num_od:
                    new_indicator[od] = 1.0
            try:
                prev_disturbance = float(compute_disturbance(prev_splits if prev_splits is not None else new_splits, new_splits, tm_arr))
            except Exception:
                prev_disturbance = 0.0
            prev_splits = new_splits
            prev_util_obs = util_eval
            prev_tm_obs = tm_now
            prev_selected_indicator = new_indicator

        # Aggregate metrics
        if not mlu_list:
            out_per_method[method] = {"error": "no successful cycles"}
            continue
        mlu_arr = np.asarray(mlu_list, dtype=np.float64)
        m = summarize_mlu(mlu_arr)
        # disturbance series
        disturb_vals = []
        for i in range(1, len(splits_list)):
            try:
                d = compute_disturbance(splits_list[i - 1], splits_list[i], perturbed_tm[test_start + i])
                disturb_vals.append(float(d))
            except Exception:
                disturb_vals.append(0.0)
        peak_idx = int(mlu_arr.argmax())
        baseline_mlu = float(mlu_arr[: max(1, peak_idx)].mean()) if peak_idx > 0 else float(mlu_arr.mean())
        out_per_method[method] = {
            **m,
            "mean_disturbance": float(np.mean(disturb_vals)) if disturb_vals else 0.0,
            "mean_score_ms": float(np.mean(score_ms_list)),
            "mean_lp_ms": float(np.mean(lp_ms_list)),
            "mean_decision_ms": float(np.mean(score_ms_list) + np.mean(lp_ms_list)),
            "sticky_applied_rate": float(np.mean(sticky_applied_list)) if sticky_applied_list else 0.0,
            "n_steps": len(mlu_list),
            "recovery_time": recovery_time(mlu_arr, peak_idx, baseline_mlu),
        }

    out = {
        "topo": topo, "scenario": scenario, "k": k, "delay": delay,
        "weight_profile": weight_profile,
        "wall_seconds": round(time.time() - t_start, 1),
        **out_per_method,
    }

    # Compute deltas vs phase1
    if "mean" in out.get("phase1", {}):
        ref = out["phase1"]
        for m in ["current_apg", "predictive", "oracle"]:
            if "mean" in out.get(m, {}):
                cur = out[m]
                cur["delta_mean_pct"] = (cur["mean"] - ref["mean"]) / max(ref["mean"], 1e-9) * 100.0
                cur["delta_p95_pct"] = (cur["p95"] - ref["p95"]) / max(ref["p95"], 1e-9) * 100.0
                cur["delta_peak_pct"] = (cur["peak"] - ref["peak"]) / max(ref["peak"], 1e-9) * 100.0
                cur["delta_disturb_pct"] = (cur["mean_disturbance"] - ref["mean_disturbance"]) / max(ref["mean_disturbance"], 1e-9) * 100.0

    # Compute deltas of predictive vs current_apg (the key ablation)
    if "mean" in out.get("current_apg", {}):
        ref = out["current_apg"]
        for m in ["predictive", "oracle"]:
            if "mean" in out.get(m, {}):
                cur = out[m]
                cur["abl_delta_mean_vs_currapg_pct"] = (cur["mean"] - ref["mean"]) / max(ref["mean"], 1e-9) * 100.0
                cur["abl_delta_p95_vs_currapg_pct"] = (cur["p95"] - ref["p95"]) / max(ref["p95"], 1e-9) * 100.0
                cur["abl_delta_peak_vs_currapg_pct"] = (cur["peak"] - ref["peak"]) / max(ref["peak"], 1e-9) * 100.0

    print(f"[{topo}/{scenario}/K={k}/d={delay}] DONE in {out['wall_seconds']}s. " + "  ".join(
        f"{m}={out.get(m, {}).get('mean', float('nan')):.4f}"
        for m in ["phase1", "current_apg", "predictive", "oracle"]
    ), flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topologies", default=",".join(ALL_TOPOLOGIES))
    parser.add_argument("--scenarios", default="normal,spike_2x,ramp_up,flash_crowd")
    parser.add_argument("--k", default="10,20,40")
    parser.add_argument("--delay", default="0,1,2")
    parser.add_argument("--weight_profile", default="balanced",
                        choices=["conservative", "balanced", "proactive"])
    parser.add_argument("--methods", default="phase1,current_apg,predictive,oracle")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--lp_time_limit", type=int, default=5)
    parser.add_argument("--prediction_horizon", type=int, default=1,
                        help="Look-ahead horizon for the GRU forecast (Path 1: try 5)")
    parser.add_argument("--mpc_lambda", type=float, default=0.0,
                        help="Path 2 -- LP capacity penalty for predicted-hot links")
    parser.add_argument("--out_tag", default="phase2_final")
    args = parser.parse_args()

    topos = [t.strip() for t in args.topologies.split(",") if t.strip()]
    scens = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    ks = [int(x) for x in args.k.split(",")]
    delays = [int(x) for x in args.delay.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]

    cells = []
    for t in topos:
        for s in scens:
            for k in ks:
                for d in delays:
                    cells.append((t, s, k, d))
    print(f"[main] {len(cells)} cells (methods: {methods})", flush=True)

    all_rows = []
    for topo, scen, k, d in cells:
        try:
            r = evaluate_cell(
                topo=topo, scenario=scen, k=k, delay=d,
                weight_profile=args.weight_profile,
                methods=methods, max_steps=args.max_steps,
                lp_time_limit=args.lp_time_limit,
                prediction_horizon=args.prediction_horizon,
                mpc_lambda=args.mpc_lambda,
            )
            all_rows.append(r)
        except Exception as exc:
            print(f"[{topo}/{scen}/K={k}/d={d}] FAILED: {exc}", flush=True)
            all_rows.append({"topo": topo, "scenario": scen, "k": k, "delay": d, "error": str(exc)})

    out_json = OUT_ROOT / f"{args.out_tag}_full.json"
    out_json.write_text(json.dumps(all_rows, indent=2, default=str) + "\n")
    print(f"\nWrote {out_json}")

    # CSV
    csv_path = OUT_ROOT / f"{args.out_tag}_routing_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "topology", "scenario", "k", "delay", "method",
            "mean_mlu", "p95_mlu", "peak_mlu", "overload_0p7", "overload_0p9",
            "mean_disturb", "mean_decision_ms", "sticky_applied_rate", "recovery_time",
            "delta_mean_pct", "delta_p95_pct", "delta_peak_pct", "delta_disturb_pct",
            "abl_delta_mean_vs_currapg_pct", "abl_delta_p95_vs_currapg_pct",
            "abl_delta_peak_vs_currapg_pct",
            "n_steps",
        ])
        for r in all_rows:
            if "error" in r:
                continue
            for m in ["phase1", "current_apg", "predictive", "oracle"]:
                d_cell = r.get(m, {})
                if not d_cell or "error" in d_cell:
                    continue
                w.writerow([
                    r["topo"], r["scenario"], r["k"], r["delay"], m,
                    f"{d_cell.get('mean', 0):.6f}", f"{d_cell.get('p95', 0):.6f}",
                    f"{d_cell.get('peak', 0):.6f}",
                    d_cell.get("overload_0p7", 0), d_cell.get("overload_0p9", 0),
                    f"{d_cell.get('mean_disturbance', 0):.6f}",
                    f"{d_cell.get('mean_decision_ms', 0):.2f}",
                    f"{d_cell.get('sticky_applied_rate', 0):.3f}",
                    d_cell.get("recovery_time", -1),
                    f"{d_cell.get('delta_mean_pct', 0):+.3f}",
                    f"{d_cell.get('delta_p95_pct', 0):+.3f}",
                    f"{d_cell.get('delta_peak_pct', 0):+.3f}",
                    f"{d_cell.get('delta_disturb_pct', 0):+.3f}",
                    f"{d_cell.get('abl_delta_mean_vs_currapg_pct', 0):+.3f}",
                    f"{d_cell.get('abl_delta_p95_vs_currapg_pct', 0):+.3f}",
                    f"{d_cell.get('abl_delta_peak_vs_currapg_pct', 0):+.3f}",
                    d_cell.get("n_steps", 0),
                ])
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
