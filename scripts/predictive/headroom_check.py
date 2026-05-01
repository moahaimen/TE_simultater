"""Phase 2.1: Headroom check + honest baselines.

For each topology:
  1. Compute per-OD cycle-to-cycle relative change distribution
     (how much room is there to predict).
  2. Score three baselines on the test split:
       - Last-value:  pred[t+1] = actual[t]
       - Mean:        pred[t+1] = mean over training window
       - EWMA(0.3):   pred[t+1] = 0.3 * actual[t] + 0.7 * pred[t]
  3. Report MAPE, MAE, R^2 per topology.
  4. Write a Markdown summary with the verdict on whether to proceed
     to a deep model (Phase 2.2 onwards).

Verdict rule (pre-registered):
  - median per-OD relative change < 5% on >=5 of 8 topologies => abandon Phase 2
  - 5%-30%                                                    => proceed to GRU (Phase 2.2)
  - >30%                                                      => proceed to Transformer (Phase 2.3)

Usage:
    python scripts/predictive/headroom_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "forecasting"
OUT_ROOT = PROJECT_ROOT / "results" / "predictive_phase2_headroom"

ALL_TOPOLOGIES = [
    "abilene", "cernet", "ebone", "geant",
    "sprintlink", "tiscali", "germany50", "vtlwavenet2011",
]


def load_tm(topo: str) -> tuple[np.ndarray, dict]:
    npz = np.load(DATA_ROOT / topo / "tm_series.npz")
    tm = npz["tm"]
    split = json.loads((DATA_ROOT / topo / "split_indices.json").read_text())
    return tm, split


def relative_change(tm: np.ndarray) -> np.ndarray:
    """Per-OD per-cycle |Δ| / max(prev, 1e-9). Drops the first row."""
    prev = tm[:-1]
    nxt = tm[1:]
    eps = 1e-9
    nz_mask = prev > eps
    rel = np.zeros_like(prev, dtype=np.float64)
    rel[nz_mask] = np.abs(nxt[nz_mask] - prev[nz_mask]) / np.maximum(prev[nz_mask], eps)
    rel[~nz_mask] = 0.0
    return rel


def mape(actual: np.ndarray, pred: np.ndarray, eps: float = 1e-9) -> float:
    """Mean Absolute Percentage Error, safe-divided. Returns %."""
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    denom = np.maximum(np.abs(actual), eps)
    err = np.abs(actual - pred) / denom
    nz = np.abs(actual) > eps
    if not nz.any():
        return float("nan")
    return float(err[nz].mean() * 100.0)


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(np.asarray(actual) - np.asarray(pred)).mean())


def r2(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def score_baselines(tm: np.ndarray, split: dict, alpha: float = 0.3) -> dict:
    """Run last-value / mean / EWMA on the test split.

    Forecasting setup:
      - Use train portion to fit the mean and to seed EWMA.
      - On test portion: at each step t in test, predict tm[t] from history.
    """
    train_end = int(split.get("train_end", int(tm.shape[0] * 0.7)))
    val_end = int(split.get("val_end", int(tm.shape[0] * 0.85)))
    num_steps = tm.shape[0]
    test_start = val_end
    if test_start >= num_steps - 1:
        return {"error": "test split too small"}

    train = tm[:train_end]
    test_actual = tm[test_start + 1:]      # what we want to predict
    test_history_t = tm[test_start: -1]    # the values at time t for last-value baseline

    # Last-value
    pred_last = test_history_t.copy()

    # Mean (one fixed value, broadcast)
    train_mean = train.mean(axis=0, keepdims=True)
    pred_mean = np.broadcast_to(train_mean, test_actual.shape).copy()

    # EWMA: warm-start the smoother on the entire history up to test_start, then roll forward.
    smoother = train_mean[0].copy()
    for t in range(train_end, test_start + 1):
        smoother = alpha * tm[t] + (1.0 - alpha) * smoother
    pred_ewma = np.zeros_like(test_actual)
    state = smoother.copy()
    for i in range(test_actual.shape[0]):
        pred_ewma[i] = state
        state = alpha * test_actual[i] + (1.0 - alpha) * state

    return {
        "n_test_steps": int(test_actual.shape[0]),
        "n_od": int(tm.shape[1]),
        "last_value": {
            "mape_pct": mape(test_actual, pred_last),
            "mae": mae(test_actual, pred_last),
            "r2": r2(test_actual, pred_last),
        },
        "mean": {
            "mape_pct": mape(test_actual, pred_mean),
            "mae": mae(test_actual, pred_mean),
            "r2": r2(test_actual, pred_mean),
        },
        "ewma_0p3": {
            "mape_pct": mape(test_actual, pred_ewma),
            "mae": mae(test_actual, pred_ewma),
            "r2": r2(test_actual, pred_ewma),
        },
    }


def headroom_stats(tm: np.ndarray) -> dict:
    rel = relative_change(tm)
    nz = rel[(rel > 0) & np.isfinite(rel)]
    if nz.size == 0:
        return {"median_rel_change_pct": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0, "n_nonzero": 0}
    return {
        "median_rel_change_pct": float(np.median(nz) * 100.0),
        "p25": float(np.percentile(nz, 25) * 100.0),
        "p75": float(np.percentile(nz, 75) * 100.0),
        "p95": float(np.percentile(nz, 95) * 100.0),
        "n_nonzero": int(nz.size),
    }


def classify_headroom(median_pct: float) -> str:
    if median_pct < 5.0:
        return "ABANDON (no signal)"
    if median_pct < 30.0:
        return "PROCEED (GRU sufficient)"
    return "PROCEED (Transformer justified)"


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for topo in ALL_TOPOLOGIES:
        try:
            tm, split = load_tm(topo)
        except FileNotFoundError as exc:
            print(f"[skip] {topo}: {exc}")
            continue
        head = headroom_stats(tm)
        scores = score_baselines(tm, split)
        rows.append({
            "topology": topo,
            "headroom": head,
            "baselines": scores,
            "verdict": classify_headroom(head["median_rel_change_pct"]),
        })

    out_json = OUT_ROOT / "phase2_headroom.json"
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_json}")

    # ── Markdown summary ─────────────────────────────────────────────
    lines = ["# Phase 2.1 — Headroom & baseline summary\n"]
    lines.append("> Pre-registered verdict thresholds:")
    lines.append("> - median per-OD relative change < 5% on ≥5 of 8 topologies → ABANDON")
    lines.append("> - 5–30% → proceed to GRU (Phase 2.2)")
    lines.append("> - >30% → proceed to Transformer (Phase 2.3)\n")

    lines.append("## Cycle-to-cycle change (per-OD relative |Δ|)\n")
    lines.append("| Topology | median % | p25 % | p75 % | p95 % | verdict |")
    lines.append("|---|---:|---:|---:|---:|:---|")
    abandon_count = 0
    proceed_gru_count = 0
    proceed_xfmr_count = 0
    for r in rows:
        h = r["headroom"]
        v = r["verdict"]
        if "ABANDON" in v: abandon_count += 1
        elif "Transformer" in v: proceed_xfmr_count += 1
        else: proceed_gru_count += 1
        lines.append(
            f"| {r['topology']} | {h['median_rel_change_pct']:.2f} | "
            f"{h['p25']:.2f} | {h['p75']:.2f} | {h['p95']:.2f} | {v} |"
        )
    lines.append("")
    lines.append(f"Abandon: {abandon_count} / 8")
    lines.append(f"Proceed-GRU: {proceed_gru_count} / 8")
    lines.append(f"Proceed-Transformer: {proceed_xfmr_count} / 8\n")

    lines.append("## Baseline MAPE (%) on the test split\n")
    lines.append("| Topology | last-value | mean | EWMA(0.3) | best | best-method |")
    lines.append("|---|---:|---:|---:|---:|:---|")
    for r in rows:
        b = r["baselines"]
        if "error" in b:
            lines.append(f"| {r['topology']} | — | — | — | — | error: {b['error']} |")
            continue
        lv = b["last_value"]["mape_pct"]
        mn = b["mean"]["mape_pct"]
        ew = b["ewma_0p3"]["mape_pct"]
        best_name, best_val = min(
            [("last-value", lv), ("mean", mn), ("EWMA(0.3)", ew)],
            key=lambda kv: kv[1] if not np.isnan(kv[1]) else float("inf"),
        )
        lines.append(
            f"| {r['topology']} | {lv:.2f} | {mn:.2f} | {ew:.2f} | "
            f"**{best_val:.2f}** | {best_name} |"
        )
    lines.append("")

    # Final aggregate verdict
    lines.append("## Aggregate verdict\n")
    if abandon_count >= 5:
        lines.append("**ABANDON Phase 2** — most topologies have <5% per-OD cycle-to-cycle "
                     "change. A deep forecaster cannot improve over last-value enough to "
                     "matter downstream. Recommend redirecting to a different research lever.")
    elif proceed_xfmr_count >= 1:
        lines.append("**PROCEED to Phase 2.3 (Transformer)** — at least one topology has "
                     ">30% median per-OD change, justifying a higher-capacity model. "
                     "Run Phase 2.2 (GRU) first as a fair-comparison baseline.")
    else:
        lines.append("**PROCEED to Phase 2.2 (GRU)** — most topologies have 5–30% median "
                     "per-OD change. A small recurrent model is the right starting point. "
                     "GRU must beat last-value by ≥10% MAPE on ≥6 of 8 topologies.")

    out_md = OUT_ROOT / "phase2_headroom.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")

    # Echo the summary table
    print()
    for line in lines[-10:]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
