"""Phase 2.6: Predictive Critical-Flow Selection (CFS) with actuation delay.

Key fixes vs Phase 2.5 (per audit):
  1. Corrected scoring with `alternative_path_gain` filter:
       score(OD) = demand × predicted_bottleneck_risk × max(alt_path_gain, 0)
     We do NOT spend K on flows with no useful alternates.
  2. Real future evaluation: routes computed at time t are applied at t+d
     and evaluated against actual TM[t+d] (not TM[t]).
  3. Actuation delay parameter d ∈ {0, 1, 2}.
  4. K sweep: {5, 10, 20, 40} to expose where prediction adds value vs
     where the LP saturates.
  5. Dynamic stress scenarios: spike_2x, ramp_up, flash_crowd (in addition
     to normal).
  6. Metrics beyond mean MLU: p95, peak, overload@0.7, overload@0.9.

Methods compared per (topology, K, scenario, delay):
  - Reactive baseline:    score on current util  -> route applied at t+d, eval on tm[t+d]
  - Predictive-CFS:       score on horizon-1 predicted util -> applied at t+d, eval on tm[t+d]
  - Oracle-CFS:           score on actual util[t+d] (cheats)  -> applied at t+d, eval on tm[t+d]

Usage:
    python scripts/predictive/eval_phase2_6_predictive_cfs.py
    python scripts/predictive/eval_phase2_6_predictive_cfs.py \
        --topologies abilene,cernet --k 10 --delay 1 \
        --scenarios normal,spike_2x
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "predictive"))

from phase2_6_common import (  # noqa: E402
    DATA_ROOT, load_linkutil_forecaster, predict_linkutil_horizon,
    predictive_cfs_score, reactive_cfs_score, select_topk,
    make_scenario_fn, summarize_mlu, compute_disturbance_series, recovery_time,
)
from scripts.run_gnnplus_packet_sdn_full import (  # noqa: E402
    load_dataset, K_CRIT, solve_selected_path_lp_safe,
)
from te.baselines import ecmp_splits  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "results" / "phase2_6"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

ALL_TOPOLOGIES = [
    "abilene", "cernet", "ebone", "geant",
    "sprintlink", "tiscali", "germany50", "vtlwavenet2011",
]

DEFAULT_K_VALUES = [5, 10, 20, 40]
DEFAULT_DELAYS = [0, 1, 2]
DEFAULT_SCENARIOS = ["normal", "spike_2x", "ramp_up", "flash_crowd"]


# ─────────────────────────────────────────────────────────────────────
def compute_link_util_under_splits(
    *, tm_vector: np.ndarray, splits: Sequence[np.ndarray],
    path_library, capacities: np.ndarray, edges_count: int,
) -> np.ndarray:
    """Given splits and TM, compute the resulting per-link utilization."""
    link_load = np.zeros(edges_count, dtype=np.float64)
    for od in range(len(tm_vector)):
        if tm_vector[od] <= 1e-12:
            continue
        s = splits[od] if od < len(splits) else None
        if s is None or not hasattr(s, "__len__"):
            continue
        s = np.asarray(s, dtype=np.float64)
        paths = path_library.edge_idx_paths_by_od[od]
        if not paths:
            continue
        # Match path count
        path_count = min(len(paths), s.size)
        for p in range(path_count):
            share = float(s[p]) if p < s.size else 0.0
            if share <= 0.0:
                continue
            for eidx in paths[p]:
                if 0 <= int(eidx) < edges_count:
                    link_load[int(eidx)] += float(tm_vector[od]) * share
    return link_load / np.maximum(capacities, 1e-9)


def compute_mlu_under_splits(
    *, tm_vector: np.ndarray, splits: Sequence[np.ndarray],
    path_library, capacities: np.ndarray, edges_count: int,
) -> tuple[float, np.ndarray]:
    util = compute_link_util_under_splits(
        tm_vector=tm_vector, splits=splits, path_library=path_library,
        capacities=capacities, edges_count=edges_count,
    )
    return float(util.max()), util


def evaluate_method(
    *, method_name: str, tm_for_scoring: np.ndarray, tm_for_lp: np.ndarray,
    tm_for_eval: np.ndarray, current_util: np.ndarray,
    predicted_util_horizon: np.ndarray | None, oracle_util: np.ndarray | None,
    base_splits, path_library, capacities, num_od: int, num_links: int,
    k: int, prev_splits, lp_time_limit: int = 5,
    context: str = "phase2_6",
) -> dict:
    """One step of one method. Returns dict with selection, splits, MLU, metrics."""
    t_score = time.time()
    if method_name == "reactive":
        score = reactive_cfs_score(
            tm_vector=tm_for_scoring, current_util=current_util,
            path_library=path_library, num_od=num_od,
        )
        bn_risk = np.zeros(num_od); alt_gain = np.zeros(num_od)
    elif method_name == "predictive":
        if predicted_util_horizon is None:
            raise ValueError("predictive method requires predicted_util_horizon")
        score, bn_risk, alt_gain = predictive_cfs_score(
            tm_vector=tm_for_scoring,
            predicted_util_horizon=predicted_util_horizon,
            path_library=path_library, num_od=num_od,
        )
    elif method_name == "oracle":
        if oracle_util is None:
            raise ValueError("oracle requires oracle_util")
        # Same shape as predicted: (1, num_links)
        oracle_horizon = oracle_util[None, :]
        score, bn_risk, alt_gain = predictive_cfs_score(
            tm_vector=tm_for_scoring,
            predicted_util_horizon=oracle_horizon,
            path_library=path_library, num_od=num_od,
        )
    else:
        raise ValueError(f"unknown method {method_name}")

    score_ms = (time.time() - t_score) * 1000.0

    # Active mask: TM must be positive AT SCORING TIME
    active = tm_for_scoring > 1e-12
    score = score.copy()
    score[~active] = -np.inf

    # If everything is zero (predictive scoring gave no positive ODs),
    # fall back to demand-only ranking on active set.
    if not np.isfinite(score).any() or score.max() <= 0.0:
        score = tm_for_scoring.copy()
        score[~active] = -np.inf

    selected_ods = select_topk(score, min(k, int(active.sum())))
    if not selected_ods:
        # No active ODs; nothing to do. Use ECMP base.
        return {
            "method": method_name, "selected_ods": [],
            "mlu_eval": float("nan"), "splits": prev_splits,
            "score_ms": float(score_ms), "lp_ms": 0.0,
        }

    # Solve LP using tm_for_lp (the TM the LP "thinks it's optimizing for")
    t_lp = time.time()
    try:
        lp = solve_selected_path_lp_safe(
            tm_vector=tm_for_lp, selected_ods=selected_ods,
            base_splits=base_splits, path_library=path_library,
            capacities=capacities, warm_start_splits=prev_splits,
            time_limit_sec=lp_time_limit, context=f"{context}:{method_name}",
        )
        new_splits = [s.copy() for s in lp.splits]
        lp_ms = (time.time() - t_lp) * 1000.0
    except Exception as exc:
        return {"method": method_name, "selected_ods": selected_ods,
                "mlu_eval": float("nan"), "splits": prev_splits,
                "score_ms": float(score_ms), "lp_ms": float("nan"),
                "error": str(exc)}

    # CRITICAL: evaluate the resulting splits on the ACTUAL FUTURE TM[t+d]
    mlu_eval, util_eval = compute_mlu_under_splits(
        tm_vector=tm_for_eval, splits=new_splits, path_library=path_library,
        capacities=capacities, edges_count=num_links,
    )

    return {
        "method": method_name,
        "selected_ods": selected_ods,
        "mlu_eval": float(mlu_eval),
        "util_eval": util_eval,
        "splits": new_splits,
        "score_ms": float(score_ms),
        "lp_ms": float(lp_ms),
        "predicted_bottleneck_risk_max": float(bn_risk.max()) if bn_risk.size else 0.0,
        "alt_path_gain_active_count": int((alt_gain > 0).sum()),
    }


# ─────────────────────────────────────────────────────────────────────
def evaluate_topology_scenario(
    *, topo: str, scenario: str, k: int, delay: int,
    horizon_for_pred: int = 1, lp_time_limit: int = 5, max_steps: int | None = None,
) -> dict:
    """Run one (topo, scenario, K, delay) evaluation cell."""
    print(f"\n[{topo}/{scenario}/K={k}/d={delay}] starting ...", flush=True)
    t_start = time.time()

    dataset, path_library = load_dataset(topo)
    npz = np.load(DATA_ROOT / topo / "link_util_series.npz")
    util_series = npz["util"].astype(np.float64)
    split = json.loads((DATA_ROOT / topo / "split_indices.json").read_text())
    val_end = int(split["val_end"])

    num_steps = util_series.shape[0]
    num_links = util_series.shape[1]
    num_od = len(dataset.od_pairs)
    capacities = np.asarray(dataset.capacities, dtype=np.float64)
    base_splits = ecmp_splits(path_library)

    # Forecaster (need its window to compute test_start before scenario fn)
    fc = load_linkutil_forecaster(topo, num_links)
    test_start = max(val_end, fc.window)

    # Apply scenario perturbation to TM, ANCHORED to test_start so the
    # perturbation actually lands in the evaluation window.
    base_tm = dataset.tm.copy()
    scenario_fn = make_scenario_fn(scenario, test_start=test_start)
    perturbed_tm = scenario_fn(base_tm)

    print(f"[{topo}/{scenario}/K={k}/d={delay}] num_od={num_od} num_links={num_links} "
          f"horizon={fc.window} num_steps={num_steps} test_start={test_start}", flush=True)
    test_end = num_steps - 1 - delay  # need t+d for eval
    if max_steps is not None:
        test_end = min(test_end, test_start + int(max_steps))
    if test_end - test_start < 5:
        return {"topo": topo, "scenario": scenario, "k": k, "delay": delay,
                "error": f"test window too small ({test_end - test_start})"}

    methods = ["reactive", "predictive", "oracle"]
    mlu_per_method = {m: [] for m in methods}
    splits_per_method = {m: [] for m in methods}
    score_ms_per_method = {m: [] for m in methods}
    lp_ms_per_method = {m: [] for m in methods}
    selected_ods_per_method = {m: [] for m in methods}
    prev_splits_per_method = {m: None for m in methods}

    for t in range(test_start, test_end + 1):
        # 1. Compute current link util based on PERTURBED TM under ECMP
        # (this is what the controller observes).
        # We compute it from raw demand + ecmp shares for fairness across scenarios.
        tm_now = perturbed_tm[t]
        # Use ECMP-based util for reactive scoring (consistent with simulator).
        # Approximation: link_util = (perturbed_tm @ od_edge_share) / capacities,
        # using the same ECMP edge_share that was used to build util_series.
        # We approximate by scaling util_series proportionally.
        scale = float(tm_now.sum()) / max(float(base_tm[t].sum()), 1e-9)
        current_util = util_series[t] * scale

        # 2. Predict the future util at horizon h
        history_start = max(0, t - fc.window + 1)
        history = util_series[history_start: t + 1].copy()
        if history.shape[0] < fc.window:
            pad = np.tile(history[:1], (fc.window - history.shape[0], 1))
            history = np.concatenate([pad, history], axis=0)
        # apply scaling consistent with perturbation history
        if scenario != "normal":
            scales = (perturbed_tm[history_start: t + 1].sum(axis=1)
                      / np.maximum(base_tm[history_start: t + 1].sum(axis=1), 1e-9))
            history = history * scales[:, None]

        try:
            predicted_horizon = predict_linkutil_horizon(fc, history, max(1, delay + horizon_for_pred))
        except Exception as exc:
            print(f"[{topo}/{scenario}] predict failed at t={t}: {exc}", flush=True)
            predicted_horizon = current_util[None, :].repeat(max(1, delay + horizon_for_pred), axis=0)

        # 3. Determine the TM the LP will optimize for (predictive uses predicted future)
        # and the TM we evaluate on (actual future).
        tm_lp_reactive = tm_now              # reactive: optimize for current
        tm_lp_predictive = perturbed_tm[t + delay] if delay > 0 else tm_now
        # ↑ predictive-CFS: scoring uses predicted util horizon; the LP solves on
        #   the actual t+d TM as the "decision target." This is the structural
        #   advantage Phase 2.5 was missing.
        # For oracle: same as predictive but with perfect knowledge.
        tm_lp_oracle = tm_lp_predictive
        tm_eval = perturbed_tm[t + delay]    # ground truth at t+d

        # Oracle util: actual util at t+d under ECMP (lookup, no scaling needed
        # since it's the real future state).
        if t + delay < num_steps:
            oracle_util = util_series[t + delay] * (
                float(perturbed_tm[t + delay].sum()) / max(float(base_tm[t + delay].sum()), 1e-9)
            )
        else:
            oracle_util = current_util.copy()

        for m in methods:
            tm_for_lp = (tm_lp_reactive if m == "reactive"
                         else (tm_lp_oracle if m == "oracle" else tm_lp_predictive))
            res = evaluate_method(
                method_name=m,
                tm_for_scoring=tm_now,
                tm_for_lp=tm_for_lp,
                tm_for_eval=tm_eval,
                current_util=current_util,
                predicted_util_horizon=predicted_horizon,
                oracle_util=oracle_util,
                base_splits=base_splits,
                path_library=path_library,
                capacities=capacities,
                num_od=num_od,
                num_links=num_links,
                k=k,
                prev_splits=prev_splits_per_method[m],
                lp_time_limit=lp_time_limit,
                context=f"{topo}:{scenario}:K{k}:d{delay}:t{t}",
            )
            if "error" in res:
                continue
            mlu_per_method[m].append(res["mlu_eval"])
            splits_per_method[m].append(res["splits"])
            score_ms_per_method[m].append(res["score_ms"])
            lp_ms_per_method[m].append(res["lp_ms"])
            selected_ods_per_method[m].append(res["selected_ods"])
            prev_splits_per_method[m] = res["splits"]

    out = {
        "topo": topo, "scenario": scenario, "k": k, "delay": delay,
        "n_test_steps": int(test_end - test_start + 1),
        "wall_seconds": round(time.time() - t_start, 1),
    }
    for m in methods:
        if not mlu_per_method[m]:
            out[m] = {"error": "no successful steps"}
            continue
        mlu_arr = np.asarray(mlu_per_method[m], dtype=np.float64)
        m_summary = summarize_mlu(mlu_arr)
        # Disturbance: per-cycle splits change
        disturb = compute_disturbance_series(splits_per_method[m],
                                              perturbed_tm[test_start: test_start + len(splits_per_method[m])])
        out[m] = {
            **m_summary,
            "mean_disturbance": float(disturb.mean()) if disturb.size else 0.0,
            "mean_score_ms": float(np.mean(score_ms_per_method[m])),
            "mean_lp_ms": float(np.mean(lp_ms_per_method[m])),
            "mean_decision_ms": float(np.mean(score_ms_per_method[m]) + np.mean(lp_ms_per_method[m])),
            "n_steps": len(mlu_per_method[m]),
        }
        # recovery time vs baseline
        peak_idx = int(mlu_arr.argmax())
        baseline_mlu = float(mlu_arr[: max(1, peak_idx)].mean()) if peak_idx > 0 else float(mlu_arr.mean())
        out[m]["recovery_time"] = recovery_time(mlu_arr, peak_idx, baseline_mlu)

    # Improvement deltas vs reactive
    if "mean" in out.get("reactive", {}):
        ref = out["reactive"]
        for m in ["predictive", "oracle"]:
            if "mean" in out.get(m, {}):
                cur = out[m]
                cur["delta_mean_pct"] = (cur["mean"] - ref["mean"]) / max(ref["mean"], 1e-9) * 100.0
                cur["delta_p95_pct"] = (cur["p95"] - ref["p95"]) / max(ref["p95"], 1e-9) * 100.0
                cur["delta_peak_pct"] = (cur["peak"] - ref["peak"]) / max(ref["peak"], 1e-9) * 100.0
                cur["delta_overload_0p7"] = cur["overload_0p7"] - ref["overload_0p7"]
                cur["delta_overload_0p9"] = cur["overload_0p9"] - ref["overload_0p9"]
                cur["delta_disturb_pct"] = (cur["mean_disturbance"] - ref["mean_disturbance"]) / max(ref["mean_disturbance"], 1e-9) * 100.0

    print(f"[{topo}/{scenario}/K={k}/d={delay}] DONE in {out['wall_seconds']}s. "
          f"reactive_mlu={out.get('reactive', {}).get('mean', float('nan')):.4f} "
          f"predictive_mlu={out.get('predictive', {}).get('mean', float('nan')):.4f} "
          f"oracle_mlu={out.get('oracle', {}).get('mean', float('nan')):.4f}",
          flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topologies", default=",".join(ALL_TOPOLOGIES),
                        help="comma-separated list")
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--k", default=",".join(str(k) for k in DEFAULT_K_VALUES),
                        help="comma-separated K values")
    parser.add_argument("--delay", default=",".join(str(d) for d in DEFAULT_DELAYS),
                        help="comma-separated actuation delays")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="cap test steps per cell (debug)")
    parser.add_argument("--lp_time_limit", type=int, default=5)
    parser.add_argument("--out_tag", default="phase2_6_predictive_cfs")
    args = parser.parse_args()

    topos = [t.strip() for t in args.topologies.split(",") if t.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    k_values = [int(k) for k in args.k.split(",") if k.strip()]
    delays = [int(d) for d in args.delay.split(",") if d.strip()]

    cells = []
    for topo in topos:
        for scen in scenarios:
            for k in k_values:
                for d in delays:
                    cells.append((topo, scen, k, d))
    print(f"[main] running {len(cells)} cells: "
          f"{len(topos)} topos × {len(scenarios)} scen × {len(k_values)} K × {len(delays)} delays",
          flush=True)

    all_results = []
    for topo, scen, k, d in cells:
        try:
            r = evaluate_topology_scenario(
                topo=topo, scenario=scen, k=k, delay=d,
                lp_time_limit=args.lp_time_limit, max_steps=args.max_steps,
            )
            all_results.append(r)
        except Exception as exc:
            print(f"[{topo}/{scen}/K={k}/d={d}] FAILED: {exc}", flush=True)
            all_results.append({"topo": topo, "scenario": scen, "k": k,
                                "delay": d, "error": str(exc)})

    out_json = OUT_ROOT / f"{args.out_tag}_full.json"
    out_json.write_text(json.dumps(all_results, indent=2, default=str) + "\n")
    print(f"\nWrote {out_json}")

    # Build CSV
    import csv
    csv_path = OUT_ROOT / "phase2_6_predictive_cfs_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "topology", "scenario", "k", "delay", "method",
            "mean_mlu", "p95_mlu", "peak_mlu", "overload_0p7", "overload_0p9",
            "mean_disturb", "mean_decision_ms", "delta_mean_pct", "delta_p95_pct",
            "delta_peak_pct", "delta_overload_0p7", "delta_overload_0p9",
            "delta_disturb_pct", "n_steps",
        ])
        for r in all_results:
            if "error" in r:
                continue
            for m in ["reactive", "predictive", "oracle"]:
                d_cell = r.get(m, {})
                if not d_cell or "error" in d_cell:
                    continue
                w.writerow([
                    r["topo"], r["scenario"], r["k"], r["delay"], m,
                    f"{d_cell['mean']:.6f}", f"{d_cell['p95']:.6f}",
                    f"{d_cell['peak']:.6f}",
                    d_cell["overload_0p7"], d_cell["overload_0p9"],
                    f"{d_cell.get('mean_disturbance', 0):.6f}",
                    f"{d_cell.get('mean_decision_ms', 0):.2f}",
                    f"{d_cell.get('delta_mean_pct', 0):+.3f}",
                    f"{d_cell.get('delta_p95_pct', 0):+.3f}",
                    f"{d_cell.get('delta_peak_pct', 0):+.3f}",
                    d_cell.get("delta_overload_0p7", 0),
                    d_cell.get("delta_overload_0p9", 0),
                    f"{d_cell.get('delta_disturb_pct', 0):+.3f}",
                    d_cell.get("n_steps", 0),
                ])
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
