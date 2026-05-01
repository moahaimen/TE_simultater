"""Phase 2.5: Inference-only test of "does the link-util forecaster help routing?"

We compare three OD-selection strategies, all using the same K-budget LP and
the same path library. No retraining, no GNN+ model loading.

  1. Reactive bottleneck (current pipeline's baseline):
       score[OD] = tm[OD] * max(current_link_util on OD's primary path)
       Pick top-K ODs.
  2. Predictive bottleneck (Phase 2.5):
       score[OD] = tm[OD] * max(GRU-predicted next-cycle util on OD's paths)
       Pick top-K ODs.
  3. Oracle bottleneck (upper bound):
       score[OD] = tm[OD] * max(actual next-cycle util on OD's paths)
       This is an upper bound on what any forecaster could achieve --
       cheats by seeing t+1 truth.

For each test cycle t we solve the LP on each method's selection and compare
the resulting MLU. Disturbance is measured between consecutive cycles
within the same method.

Pre-registered Phase 2.5 verdict (relative to Reactive baseline):
  - Predictive must reduce mean MLU on >=4 of 8 topologies, AND
  - Oracle must show >=2x more improvement than Predictive (otherwise the
    forecaster is already capturing most of the available signal).

If Predictive does NOT reduce MLU but Oracle does -> forecaster accuracy
is the limiting factor (try better model later).
If neither reduces MLU -> link util at t+1 is not a useful routing signal
(stop pursuing this direction).

Usage:
    python scripts/predictive/eval_phase2_5_predictive_routing.py
    python scripts/predictive/eval_phase2_5_predictive_routing.py --topology abilene
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "predictive"))

from train_gru_forecaster import GRUForecaster  # noqa: E402

# Reuse the same dataset/path-library loader the production runner uses.
from scripts.run_gnnplus_packet_sdn_full import (  # noqa: E402
    load_dataset, K_CRIT, solve_selected_path_lp_safe,
)
from te.baselines import ecmp_splits  # noqa: E402
from te.disturbance import compute_disturbance  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data" / "forecasting"
GRU_ROOT = PROJECT_ROOT / "results" / "predictive_phase2_gru_linkutil"
OUT_ROOT = PROJECT_ROOT / "results" / "predictive_phase2_5"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

ALL_TOPOLOGIES = [
    "abilene", "cernet", "ebone", "geant",
    "sprintlink", "tiscali", "germany50", "vtlwavenet2011",
]


def load_link_util_series(topo: str) -> tuple[np.ndarray, dict]:
    npz = np.load(DATA_ROOT / topo / "link_util_series.npz")
    util = npz["util"].astype(np.float64)
    split = json.loads((DATA_ROOT / topo / "split_indices.json").read_text())
    return util, split


def load_gru_for_topology(topo: str, num_links: int) -> tuple[GRUForecaster, np.ndarray, np.ndarray, int]:
    ckpt_path = GRU_ROOT / topo / "gru_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No GRU checkpoint at {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    feat_mean = payload["feat_mean"]
    feat_std = payload["feat_std"]
    model = GRUForecaster(num_od=num_links, hidden=cfg["hidden"], layers=cfg["layers"], dropout=0.0)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, feat_mean, feat_std, int(cfg["window"])


def predict_next_util(
    model: GRUForecaster, feat_mean: np.ndarray, feat_std: np.ndarray,
    window: int, util_history: np.ndarray,
) -> np.ndarray:
    """util_history: (window, num_links) raw util. Returns predicted next-cycle util in real-space.

    feat_mean / feat_std were saved as shape (1, 1, num_links) from training.
    """
    util_log = np.log1p(util_history)            # (window, num_links)
    fm = feat_mean.squeeze(axis=(0, 1))          # (num_links,)
    fs = feat_std.squeeze(axis=(0, 1))           # (num_links,)
    x_norm = (util_log - fm) / fs                # (window, num_links)
    x = torch.from_numpy(x_norm).float().unsqueeze(0)   # (1, window, num_links) — 3D as GRU expects
    with torch.no_grad():
        out_norm = model(x).cpu().numpy().squeeze(0)    # (num_links,)
    out_log = out_norm * fs + fm                 # (num_links,)
    return np.expm1(out_log)


def compute_od_path_max_util(util: np.ndarray, path_library, num_od: int) -> np.ndarray:
    """Per OD, max link util across its primary (cheapest) path. Returns (num_od,)."""
    out = np.zeros(num_od, dtype=np.float64)
    for od in range(num_od):
        paths = path_library.edge_idx_paths_by_od[od]
        if not paths:
            continue
        # Use first (cheapest) path as the primary; same as bottleneck baseline.
        primary = paths[0]
        if not primary:
            continue
        edge_idx = np.asarray(primary, dtype=np.int64)
        out[od] = float(util[edge_idx].max()) if edge_idx.size else 0.0
    return out


def select_topk_by_score(score: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[int]:
    if mask is not None:
        score = score.copy()
        score[~mask] = -np.inf
    if k >= len(score):
        return list(np.argsort(-score).tolist())
    idx = np.argpartition(-score, k)[:k]
    return list(idx[np.argsort(-score[idx])].tolist())


def solve_lp_safe(*, tm_vector, selected_ods, base_splits, path_library,
                  capacities, warm_start_splits, time_limit_sec=5, context="phase2_5"):
    return solve_selected_path_lp_safe(
        tm_vector=tm_vector,
        selected_ods=selected_ods,
        base_splits=base_splits,
        path_library=path_library,
        capacities=capacities,
        warm_start_splits=warm_start_splits,
        time_limit_sec=time_limit_sec,
        context=context,
    )


def evaluate_topology(topo: str) -> dict:
    print(f"\n[{topo}] starting Phase 2.5 evaluation ...", flush=True)
    t0 = time.time()
    dataset, path_library = load_dataset(topo)
    util_series, split = load_link_util_series(topo)
    num_steps = util_series.shape[0]
    num_links = util_series.shape[1]
    num_od = len(dataset.od_pairs)
    train_end = int(split["train_end"])
    val_end = int(split["val_end"])

    model, feat_mean, feat_std, window = load_gru_for_topology(topo, num_links)
    print(f"[{topo}] loaded GRU window={window} num_links={num_links} num_od={num_od}", flush=True)

    capacities = np.asarray(dataset.capacities, dtype=np.float64)
    base_splits = ecmp_splits(path_library)

    # Walk through the test split. At each step t we have actual util[t] and need to
    # predict util[t+1]. We can score reactive on util[t], predictive on predicted_util[t+1],
    # and oracle on actual util[t+1] (if available in test window).
    test_start = max(val_end, window)  # need at least `window` history before the first test step
    test_end = num_steps - 1  # need t+1 for oracle

    if test_end - test_start < 5:
        return {"topology": topo, "error": f"test window too small ({test_end - test_start})"}

    methods = ["reactive", "predictive", "oracle"]
    mlu_per_method: dict[str, list[float]] = {m: [] for m in methods}
    disturbance_per_method: dict[str, list[float]] = {m: [] for m in methods}
    selection_per_method: dict[str, list[list[int]]] = {m: [] for m in methods}
    decision_ms_per_method: dict[str, list[float]] = {m: [] for m in methods}

    prev_splits_per_method: dict[str, list[np.ndarray] | None] = {m: None for m in methods}

    for t in range(test_start, test_end + 1):
        tm_vector = dataset.tm[t]
        actual_util_t = util_series[t]
        actual_util_tp1 = util_series[t + 1] if t + 1 < num_steps else actual_util_t

        # Predicted util at t+1 from the GRU on history [t-window+1 .. t]
        history = util_series[t - window + 1: t + 1]
        if history.shape[0] < window:
            # First few steps may not have full history; pad by repeating earliest
            pad = np.tile(history[:1], (window - history.shape[0], 1))
            history = np.concatenate([pad, history], axis=0)
        try:
            predicted_util_tp1 = predict_next_util(model, feat_mean, feat_std, window, history)
        except Exception as exc:
            print(f"[{topo}] GRU predict failed at t={t}: {exc}", flush=True)
            predicted_util_tp1 = actual_util_t.copy()

        # Per-method scoring
        scores = {
            "reactive": tm_vector * compute_od_path_max_util(actual_util_t, path_library, num_od),
            "predictive": tm_vector * compute_od_path_max_util(predicted_util_tp1, path_library, num_od),
            "oracle": tm_vector * compute_od_path_max_util(actual_util_tp1, path_library, num_od),
        }

        # Active-OD mask (TM > 0)
        active = tm_vector > 1e-12

        for m in methods:
            ods = select_topk_by_score(scores[m], K_CRIT, active)
            if not ods:
                # fallback: take top-K by demand
                ods = select_topk_by_score(tm_vector, K_CRIT, active)
            t_dec_start = time.time()
            try:
                lp = solve_lp_safe(
                    tm_vector=tm_vector, selected_ods=ods,
                    base_splits=base_splits, path_library=path_library,
                    capacities=capacities,
                    warm_start_splits=prev_splits_per_method[m],
                    context=f"{topo}:{m}:t={t}",
                )
                dec_ms = (time.time() - t_dec_start) * 1000.0
                mlu = float(lp.routing.mlu)
                splits = [s.copy() for s in lp.splits]
            except Exception as exc:
                print(f"[{topo}] LP failed for {m} at t={t}: {exc}", flush=True)
                continue

            # Disturbance vs previous cycle for this method
            disturb = 0.0
            if compute_disturbance is not None and prev_splits_per_method[m] is not None:
                try:
                    disturb = float(compute_disturbance(prev_splits_per_method[m], splits, tm_vector))
                except Exception:
                    pass

            mlu_per_method[m].append(mlu)
            disturbance_per_method[m].append(disturb)
            selection_per_method[m].append(ods)
            decision_ms_per_method[m].append(dec_ms)
            prev_splits_per_method[m] = splits

    # Aggregate
    summary = {
        "topology": topo,
        "num_test_steps": int(test_end - test_start + 1),
        "wall_seconds": round(time.time() - t0, 1),
    }
    for m in methods:
        if mlu_per_method[m]:
            summary[m] = {
                "mean_mlu": float(np.mean(mlu_per_method[m])),
                "mean_disturbance": float(np.mean(disturbance_per_method[m])),
                "mean_decision_ms": float(np.mean(decision_ms_per_method[m])),
                "n": len(mlu_per_method[m]),
            }
        else:
            summary[m] = {"error": "no successful LP solves"}

    # Improvements relative to reactive
    if "reactive" in summary and "mean_mlu" in summary["reactive"]:
        ref_mlu = summary["reactive"]["mean_mlu"]
        ref_dist = summary["reactive"]["mean_disturbance"]
        for m in ["predictive", "oracle"]:
            if "mean_mlu" in summary.get(m, {}):
                summary[m]["mlu_rel_pct"] = (summary[m]["mean_mlu"] - ref_mlu) / max(ref_mlu, 1e-9) * 100.0
                summary[m]["disturb_rel_pct"] = (summary[m]["mean_disturbance"] - ref_dist) / max(ref_dist, 1e-9) * 100.0

    out_dir = OUT_ROOT / topo
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{topo}] DONE. reactive_mlu={summary['reactive'].get('mean_mlu', float('nan')):.4f} "
          f"predictive_mlu={summary['predictive'].get('mean_mlu', float('nan')):.4f} "
          f"oracle_mlu={summary['oracle'].get('mean_mlu', float('nan')):.4f}",
          flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="all")
    args = parser.parse_args()
    topos = ALL_TOPOLOGIES if args.topology == "all" else [args.topology]

    all_results = []
    for topo in topos:
        try:
            r = evaluate_topology(topo)
            all_results.append(r)
        except Exception as exc:
            print(f"[{topo}] FAILED: {exc}", flush=True)
            all_results.append({"topology": topo, "error": str(exc)})

    summary_path = OUT_ROOT / "phase2_5_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2) + "\n")

    md = ["# Phase 2.5 — Predictive routing (inference-only)\n"]
    md.append("> Compares Reactive (current util) vs Predictive (GRU forecast) "
              "vs Oracle (cheats with t+1 actual) bottleneck-style OD selection.\n")
    md.append("| Topology | reactive MLU | predictive MLU | oracle MLU | predictive Δ% | oracle Δ% |")
    md.append("|---|---:|---:|---:|---:|---:|")
    pred_wins = 0
    oracle_better = 0
    for r in all_results:
        if "error" in r:
            md.append(f"| {r['topology']} | err | — | — | — | — |")
            continue
        rx = r.get("reactive", {})
        px = r.get("predictive", {})
        ox = r.get("oracle", {})
        if "mean_mlu" not in rx:
            md.append(f"| {r['topology']} | — | — | — | — | — |")
            continue
        p_rel = px.get("mlu_rel_pct", float("nan"))
        o_rel = ox.get("mlu_rel_pct", float("nan"))
        if p_rel <= -0.1: pred_wins += 1
        if o_rel < p_rel - 0.5: oracle_better += 1
        md.append(
            f"| {r['topology']} | {rx['mean_mlu']:.4f} | "
            f"{px.get('mean_mlu', float('nan')):.4f} | "
            f"{ox.get('mean_mlu', float('nan')):.4f} | "
            f"{p_rel:+.2f}% | {o_rel:+.2f}% |"
        )
    md.append("")
    md.append(f"Predictive reduces MLU on: {pred_wins} / 8 topologies")
    md.append(f"Oracle is materially better than Predictive on: {oracle_better} / 8 topologies")
    if pred_wins >= 4:
        md.append("\n**VERDICT: PASS** — predictive routing reduces MLU on at least 4/8.")
    else:
        if oracle_better >= 4:
            md.append("\n**VERDICT: FAIL (forecaster-limited)** — oracle would help, "
                      "but the GRU forecaster's accuracy is insufficient. Either improve "
                      "the forecaster or accept this as the upper bound.")
        else:
            md.append("\n**VERDICT: FAIL (signal-limited)** — even the oracle does not "
                      "materially improve MLU, meaning predicted-util is not a useful "
                      "routing signal beyond the current state. Stop pursuing.")
    md_path = OUT_ROOT / "phase2_5_summary.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"\nWrote {md_path}")
    print()
    for line in md[-12:]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
