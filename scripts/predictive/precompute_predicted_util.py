"""Phase 2 Final: precompute predicted link utilization for every test timestep
of every topology, using the existing per-topology GRU checkpoints.

Output:
  data/forecasting/<topo>/predicted_util_test.npz with arrays:
    - 'predicted_util': shape (num_test_steps, num_links) — t+1 prediction
    - 'test_start_idx': int (where the test window begins in the original tm series)
    - 'window': int (history window the GRU expects)

This file is loaded once at the start of a Phase 2 Final eval run and used
to look up predicted util by absolute timestep without repeatedly calling
the GRU online.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "predictive"))

from phase2_6_common import (  # noqa: E402
    DATA_ROOT, load_linkutil_forecaster, predict_linkutil_horizon,
)

ALL_TOPOLOGIES = [
    "abilene", "cernet", "ebone", "geant",
    "sprintlink", "tiscali", "germany50", "vtlwavenet2011",
]


def precompute_for_topology(topo: str, *, max_horizon: int = 3) -> dict:
    """For each test-window timestep t, predict util at t+1 .. t+max_horizon.

    Saves to data/forecasting/<topo>/predicted_util_test.npz.

    The saved array has shape (num_steps, max_horizon, num_links). Index 0
    of dim 1 is the t+1 prediction, index 1 is t+2, etc.
    """
    npz = np.load(DATA_ROOT / topo / "link_util_series.npz")
    util_series = npz["util"].astype(np.float64)
    split = json.loads((DATA_ROOT / topo / "split_indices.json").read_text())
    num_steps = util_series.shape[0]
    num_links = util_series.shape[1]
    val_end = int(split["val_end"])

    fc = load_linkutil_forecaster(topo, num_links)
    test_start = max(val_end, fc.window)

    print(f"[{topo}] num_steps={num_steps} num_links={num_links} "
          f"window={fc.window} test_start={test_start}", flush=True)

    # We predict for every t in [test_start, num_steps), where for each t we
    # produce predictions of t+1 .. t+max_horizon.
    pred = np.zeros((num_steps, max_horizon, num_links), dtype=np.float64)
    for t in range(test_start, num_steps):
        history_start = max(0, t - fc.window + 1)
        history = util_series[history_start: t + 1]
        if history.shape[0] < fc.window:
            pad = np.tile(history[:1], (fc.window - history.shape[0], 1))
            history = np.concatenate([pad, history], axis=0)
        try:
            ph = predict_linkutil_horizon(fc, history, max_horizon)
            pred[t] = ph
        except Exception as exc:
            print(f"[{topo}] predict failed at t={t}: {exc}", flush=True)
            # Fallback to last-value
            pred[t] = util_series[t][None, :].repeat(max_horizon, axis=0)

    out_path = DATA_ROOT / topo / "predicted_util_test.npz"
    np.savez_compressed(
        out_path,
        predicted_util=pred,
        test_start_idx=int(test_start),
        window=int(fc.window),
        max_horizon=int(max_horizon),
    )
    print(f"[{topo}] wrote {out_path}", flush=True)
    return {
        "topo": topo, "test_start": test_start, "max_horizon": max_horizon,
        "num_links": num_links, "num_predictions": int(num_steps - test_start),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="all")
    parser.add_argument("--max_horizon", type=int, default=3)
    args = parser.parse_args()
    topos = ALL_TOPOLOGIES if args.topology == "all" else [args.topology]

    summaries = []
    for topo in topos:
        try:
            s = precompute_for_topology(topo, max_horizon=args.max_horizon)
            summaries.append(s)
        except Exception as exc:
            print(f"[{topo}] FAILED: {exc}", flush=True)
            summaries.append({"topo": topo, "error": str(exc)})
    print(f"\n[done] {len(summaries)} topologies processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
