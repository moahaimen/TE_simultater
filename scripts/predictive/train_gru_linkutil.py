"""Phase 2.2b: GRU forecaster — link utilization (not OD demand).

The OD-level GRU run failed the pre-registered bar because:
  - Most OD pairs are near-stationary; last-value is already near-optimal.
  - High-volatility topologies (germany50: 2450 ODs / 288 steps) are
    data-starved for any deep model in OD space.

Link utilization is what the professor actually asked us to forecast.
It has 5-50x fewer dimensions per topology (30-192 edges vs 132-8372 ODs)
for the same number of timesteps, so the data-to-parameter ratio is
much friendlier.

Same residual GRU architecture, same train/val/test split, same loss.
Pre-registered verdict identical: GRU must beat last-value by >=10%
MAPE on >=6 of 8 topologies.

Usage:
    python scripts/predictive/train_gru_linkutil.py
    python scripts/predictive/train_gru_linkutil.py --topology abilene
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "predictive"))

# Reuse model + helpers from train_gru_forecaster
from train_gru_forecaster import GRUForecaster, mape, mae, r2, build_windows  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data" / "forecasting"
OUT_ROOT = PROJECT_ROOT / "results" / "predictive_phase2_gru_linkutil"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

ALL_TOPOLOGIES = [
    "abilene", "cernet", "ebone", "geant",
    "sprintlink", "tiscali", "germany50", "vtlwavenet2011",
]


def load_link_util(topo: str) -> tuple[np.ndarray, dict]:
    npz = np.load(DATA_ROOT / topo / "link_util_series.npz")
    util = npz["util"].astype(np.float64)
    split = json.loads((DATA_ROOT / topo / "split_indices.json").read_text())
    return util, split


def train_one_topology(topo: str, *, window: int = 12, hidden: int = 64,
                       layers: int = 2, dropout: float = 0.1, batch_size: int = 32,
                       lr: float = 1e-3, weight_decay: float = 1e-5, epochs: int = 80,
                       patience: int = 15, device: str = "cpu", seed: int = 1234) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)

    util, split = load_link_util(topo)
    num_steps, num_links = util.shape
    train_end = int(split["train_end"])
    val_end = int(split["val_end"])

    print(f"[{topo}] num_steps={num_steps} num_links={num_links} "
          f"train={train_end} val={val_end - train_end} test={num_steps - val_end}", flush=True)

    # log1p handles wide range, preserves zeros
    util_log = np.log1p(util)
    X_all, y_all = build_windows(util_log, window)
    target_t = np.arange(window, num_steps)

    train_mask = target_t < train_end
    val_mask = (target_t >= train_end) & (target_t < val_end)
    test_mask = target_t >= val_end

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    print(f"[{topo}] windows: train={len(X_train)} val={len(X_val)} test={len(X_test)}", flush=True)
    if len(X_train) < batch_size or len(X_val) < 1 or len(X_test) < 1:
        return {"topology": topo, "error": "not enough windows"}

    feat_mean = X_train.mean(axis=(0, 1), keepdims=True)
    feat_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    norm = lambda x: (x - feat_mean) / feat_std
    norm_y = lambda y: (y - feat_mean[:, 0, :]) / feat_std[:, 0, :]
    denorm_y = lambda y_n: y_n * feat_std[:, 0, :] + feat_mean[:, 0, :]

    Xt_n = torch.from_numpy(norm(X_train)).float()
    yt_n = torch.from_numpy(norm_y(y_train)).float()
    Xv_n = torch.from_numpy(norm(X_val)).float()
    Xs_n = torch.from_numpy(norm(X_test)).float()

    train_loader = DataLoader(TensorDataset(Xt_n, yt_n), batch_size=batch_size, shuffle=True)

    model = GRUForecaster(num_od=num_links, hidden=hidden, layers=layers, dropout=dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
    bad_epochs = 0
    train_log = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optim.zero_grad()
            pred = model(xb.to(device))
            loss = loss_fn(pred, yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv_n.to(device)).cpu().numpy()
        val_pred_real = np.expm1(denorm_y(val_pred))
        val_actual_real = np.expm1(y_val)
        val_mape = mape(val_actual_real, val_pred_real)
        train_log.append({"epoch": epoch, "train_loss": train_loss, "val_mape_pct": val_mape})

        if val_mape < best_val - 1e-3:
            best_val = val_mape
            best_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{topo}] epoch={epoch:3d} train_loss={train_loss:.4f} "
                  f"val_mape={val_mape:.2f}% best={best_val:.2f}%", flush=True)
        if bad_epochs >= patience:
            print(f"[{topo}] early stop at epoch {epoch}", flush=True)
            break
    train_seconds = time.time() - t0

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(Xs_n.to(device)).cpu().numpy()
    test_pred_real = np.expm1(denorm_y(test_pred))
    test_actual_real = np.expm1(y_test)

    last_value_pred_real = np.expm1(X_test[:, -1, :])

    gru_mape = mape(test_actual_real, test_pred_real)
    last_mape = mape(test_actual_real, last_value_pred_real)
    rel_imp = (last_mape - gru_mape) / max(last_mape, 1e-9) * 100.0

    # Hotspot F1: util > 0.7
    hotspot_thresh = 0.7
    actual_hotspots = (test_actual_real > hotspot_thresh).astype(int)
    gru_hotspots = (test_pred_real > hotspot_thresh).astype(int)
    last_hotspots = (last_value_pred_real > hotspot_thresh).astype(int)

    def f1(actual, pred):
        tp = ((actual == 1) & (pred == 1)).sum()
        fp = ((actual == 0) & (pred == 1)).sum()
        fn = ((actual == 1) & (pred == 0)).sum()
        p = tp / max(tp + fp, 1)
        r_v = tp / max(tp + fn, 1)
        return float(2 * p * r_v / max(p + r_v, 1e-9))

    out_dir = OUT_ROOT / topo
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "feat_mean": feat_mean, "feat_std": feat_std,
                "config": {"window": window, "hidden": hidden, "layers": layers,
                           "dropout": dropout, "num_links": num_links}},
               out_dir / "gru_checkpoint.pt")

    metrics = {
        "topology": topo,
        "num_links": int(num_links),
        "n_test": int(len(y_test)),
        "train_seconds": round(train_seconds, 1),
        "best_val_mape_pct": round(best_val, 3),
        "gru": {
            "mape_pct": round(gru_mape, 3),
            "mae": float(mae(test_actual_real, test_pred_real)),
            "r2": round(r2(test_actual_real, test_pred_real), 4),
            "hotspot_f1": round(f1(actual_hotspots, gru_hotspots), 4),
        },
        "last_value": {
            "mape_pct": round(last_mape, 3),
            "mae": float(mae(test_actual_real, last_value_pred_real)),
            "r2": round(r2(test_actual_real, last_value_pred_real), 4),
            "hotspot_f1": round(f1(actual_hotspots, last_hotspots), 4),
        },
        "rel_improvement_pct": round(rel_imp, 2),
        "config": {"window": window, "hidden": hidden, "layers": layers,
                   "epochs_run": len(train_log), "lr": lr, "batch_size": batch_size},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2) + "\n")

    print(f"[{topo}] DONE. GRU MAPE={gru_mape:.2f}% LV MAPE={last_mape:.2f}% "
          f"impr={rel_imp:+.2f}% F1(GRU)={metrics['gru']['hotspot_f1']:.3f} "
          f"F1(LV)={metrics['last_value']['hotspot_f1']:.3f} in {train_seconds:.0f}s", flush=True)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="all")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    topos = ALL_TOPOLOGIES if args.topology == "all" else [args.topology]
    all_metrics = []
    for topo in topos:
        try:
            m = train_one_topology(topo, window=args.window, hidden=args.hidden,
                                    layers=args.layers, dropout=args.dropout,
                                    batch_size=args.batch_size, lr=args.lr,
                                    epochs=args.epochs, patience=args.patience,
                                    device=args.device)
            all_metrics.append(m)
        except Exception as exc:
            print(f"[{topo}] FAILED: {exc}", flush=True)
            all_metrics.append({"topology": topo, "error": str(exc)})

    summary_path = OUT_ROOT / "phase2_gru_linkutil_summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2) + "\n")

    md = ["# Phase 2.2b — GRU forecaster on LINK UTILIZATION\n"]
    md.append("> Pre-registered verdict: GRU must beat last-value by ≥10% MAPE on ≥6/8 topologies.\n")
    md.append("| Topology | num_links | GRU MAPE % | LV MAPE % | Δ rel % | GRU R² | F1(GRU) | F1(LV) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    big_win = 0
    for m in all_metrics:
        if "error" in m:
            md.append(f"| {m['topology']} | err | — | — | — | — | — | — |")
            continue
        rel = m["rel_improvement_pct"]
        if rel >= 10: big_win += 1
        md.append(
            f"| {m['topology']} | {m['num_links']} | {m['gru']['mape_pct']:.2f} | "
            f"{m['last_value']['mape_pct']:.2f} | {rel:+.2f} | {m['gru']['r2']:.3f} | "
            f"{m['gru']['hotspot_f1']:.3f} | {m['last_value']['hotspot_f1']:.3f} |"
        )
    md.append("")
    md.append(f"GRU beats LV by ≥10% MAPE: {big_win} / 8 topologies")
    if big_win >= 6:
        md.append("\n**VERDICT: PASS** — link-utilization GRU clears ≥10% threshold on ≥6/8.")
    else:
        md.append(f"\n**VERDICT: FAIL** — only {big_win}/8 met the ≥10% threshold.")
    summary_md = OUT_ROOT / "phase2_gru_linkutil_summary.md"
    summary_md.write_text("\n".join(md) + "\n")
    print(f"\nWrote {summary_md}")
    print()
    for line in md[-10:]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
