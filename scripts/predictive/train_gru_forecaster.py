"""Phase 2.2: GRU forecaster (per-topology, multivariate over OD pairs).

For each topology:
  - Input window: last W cycles of OD demands (log1p-transformed)
  - Output: next 1 cycle prediction (log1p-transformed)
  - Loss: MSE in log1p-space (handles wide dynamic range)
  - Train on train split, early-stop on val split, evaluate on test split
  - Save: model checkpoint + per-topology metrics JSON

Pre-registered verdict (Phase 2.2):
  GRU must beat last-value baseline by >=10% MAPE on >=6 of 8 topologies.

Usage:
    python scripts/predictive/train_gru_forecaster.py
    python scripts/predictive/train_gru_forecaster.py --topology abilene
    python scripts/predictive/train_gru_forecaster.py --topology abilene --epochs 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "forecasting"
OUT_ROOT = PROJECT_ROOT / "results" / "predictive_phase2_gru"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

ALL_TOPOLOGIES = [
    "abilene", "cernet", "ebone", "geant",
    "sprintlink", "tiscali", "germany50", "vtlwavenet2011",
]


# ── Model ────────────────────────────────────────────────────────────
class GRUForecaster(nn.Module):
    """Residual GRU forecaster.

    Predicts a *delta* from last-value, then adds last-value back. With the
    residual head zero-initialized, the model output starts at last-value
    (i.e. matches the strong naive baseline out of the box) and only learns
    the correction. This is critical when most ODs barely change between
    cycles -- a from-scratch absolute predictor underperforms last-value.
    """

    def __init__(self, num_od: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_od = num_od
        self.gru = nn.GRU(
            input_size=num_od,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, num_od)
        # Zero-init the residual head so initial prediction == last-value.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, num_od) — already normalized
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]              # (B, hidden)
        delta = self.head(last_hidden)           # (B, num_od) — starts at zero
        last_value = x[:, -1, :]                 # (B, num_od) — the input at time t
        return last_value + delta                # prediction == last_value + learned delta


# ── Data ─────────────────────────────────────────────────────────────
def load_tm(topo: str) -> tuple[np.ndarray, dict]:
    npz = np.load(DATA_ROOT / topo / "tm_series.npz")
    tm = npz["tm"].astype(np.float64)
    split = json.loads((DATA_ROOT / topo / "split_indices.json").read_text())
    return tm, split


def build_windows(tm_log: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return X (n, window, num_od) and y (n, num_od) sliding windows.

    Window length = window. y[i] = tm_log[i + window].
    """
    n = tm_log.shape[0] - window
    if n <= 0:
        return np.empty((0, window, tm_log.shape[1])), np.empty((0, tm_log.shape[1]))
    X = np.stack([tm_log[i:i + window] for i in range(n)], axis=0)
    y = np.stack([tm_log[i + window]   for i in range(n)], axis=0)
    return X, y


# ── Metrics ──────────────────────────────────────────────────────────
def mape(actual: np.ndarray, pred: np.ndarray, eps: float = 1e-9) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    nz = np.abs(actual) > eps
    if not nz.any():
        return float("nan")
    err = np.abs(actual[nz] - pred[nz]) / np.maximum(np.abs(actual[nz]), eps)
    return float(err.mean() * 100.0)


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


# ── Train / eval ─────────────────────────────────────────────────────
def train_one_topology(
    topo: str,
    *,
    window: int = 12,
    hidden: int = 128,
    layers: int = 2,
    dropout: float = 0.1,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 60,
    patience: int = 10,
    device: str = "cpu",
    seed: int = 1234,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tm, split = load_tm(topo)
    num_steps, num_od = tm.shape
    train_end = int(split["train_end"])
    val_end = int(split["val_end"])

    print(f"[{topo}] num_steps={num_steps} num_od={num_od} train={train_end} "
          f"val={val_end - train_end} test={num_steps - val_end}", flush=True)

    # log1p transform handles wide dynamic range without losing zeros.
    tm_log = np.log1p(tm)

    # Train set must include the window of context BEFORE train_end
    # so that y reaches up to train_end - 1. We slide windows over the
    # full series and split the windows by their TARGET time index.
    X_all, y_all = build_windows(tm_log, window)
    target_t = np.arange(window, num_steps)
    train_mask = target_t < train_end
    val_mask = (target_t >= train_end) & (target_t < val_end)
    test_mask = target_t >= val_end

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    print(f"[{topo}] windows: train={len(X_train)} val={len(X_val)} test={len(X_test)}",
          flush=True)
    if len(X_train) < batch_size or len(X_val) < 1 or len(X_test) < 1:
        return {"topology": topo, "error": "not enough windows for one of train/val/test"}

    # Per-feature standardization on train set ONLY.
    feat_mean = X_train.mean(axis=(0, 1), keepdims=True)
    feat_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6

    def norm(x: np.ndarray) -> np.ndarray:
        return (x - feat_mean) / feat_std

    def norm_y(y: np.ndarray) -> np.ndarray:
        return (y - feat_mean[:, 0, :]) / feat_std[:, 0, :]

    def denorm_y(y_norm: np.ndarray) -> np.ndarray:
        return y_norm * feat_std[:, 0, :] + feat_mean[:, 0, :]

    Xt_n = torch.from_numpy(norm(X_train)).float()
    yt_n = torch.from_numpy(norm_y(y_train)).float()
    Xv_n = torch.from_numpy(norm(X_val)).float()
    yv_n = torch.from_numpy(norm_y(y_val)).float()
    Xs_n = torch.from_numpy(norm(X_test)).float()
    ys_n = torch.from_numpy(norm_y(y_test)).float()

    train_loader = DataLoader(TensorDataset(Xt_n, yt_n), batch_size=batch_size, shuffle=True)

    model = GRUForecaster(num_od=num_od, hidden=hidden, layers=layers, dropout=dropout).to(device)
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
            xb = xb.to(device); yb = yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{topo}] epoch={epoch:3d} train_loss={train_loss:.4f} "
                  f"val_mape={val_mape:.2f}% best={best_val:.2f}%", flush=True)
        if bad_epochs >= patience:
            print(f"[{topo}] early stop at epoch {epoch}", flush=True)
            break
    train_seconds = time.time() - t0

    # Restore best and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(Xs_n.to(device)).cpu().numpy()
    test_pred_real = np.expm1(denorm_y(test_pred))
    test_actual_real = np.expm1(y_test)

    # Compare against last-value on the same windows
    last_value_pred = X_test[:, -1, :]                      # log1p
    last_value_pred_real = np.expm1(last_value_pred)

    gru_mape = mape(test_actual_real, test_pred_real)
    last_mape = mape(test_actual_real, last_value_pred_real)
    gru_mae = mae(test_actual_real, test_pred_real)
    last_mae = mae(test_actual_real, last_value_pred_real)
    gru_r2 = r2(test_actual_real, test_pred_real)
    last_r2 = r2(test_actual_real, last_value_pred_real)

    rel_improvement = (last_mape - gru_mape) / max(last_mape, 1e-9) * 100.0

    out_dir = OUT_ROOT / topo
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "config": {
            "window": window, "hidden": hidden, "layers": layers,
            "dropout": dropout, "num_od": num_od,
        },
    }, out_dir / "gru_checkpoint.pt")

    metrics = {
        "topology": topo,
        "num_od": int(num_od),
        "n_test": int(len(y_test)),
        "train_seconds": round(train_seconds, 1),
        "best_val_mape_pct": round(best_val, 3),
        "gru": {
            "mape_pct": round(gru_mape, 3),
            "mae": float(gru_mae),
            "r2": round(gru_r2, 4),
        },
        "last_value": {
            "mape_pct": round(last_mape, 3),
            "mae": float(last_mae),
            "r2": round(last_r2, 4),
        },
        "rel_improvement_pct": round(rel_improvement, 2),
        "config": {
            "window": window, "hidden": hidden, "layers": layers,
            "dropout": dropout, "epochs_run": len(train_log), "lr": lr,
            "batch_size": batch_size,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2) + "\n")

    print(f"[{topo}] DONE. GRU MAPE={gru_mape:.2f}% LV MAPE={last_mape:.2f}% "
          f"improvement={rel_improvement:+.2f}% in {train_seconds:.0f}s",
          flush=True)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="all")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    topos = ALL_TOPOLOGIES if args.topology == "all" else [args.topology]
    all_metrics = []
    for topo in topos:
        try:
            m = train_one_topology(
                topo,
                window=args.window, hidden=args.hidden, layers=args.layers,
                dropout=args.dropout, batch_size=args.batch_size, lr=args.lr,
                epochs=args.epochs, patience=args.patience, device=args.device,
            )
            all_metrics.append(m)
        except Exception as exc:
            print(f"[{topo}] FAILED: {exc}", flush=True)
            all_metrics.append({"topology": topo, "error": str(exc)})

    # Aggregate report
    summary_path = OUT_ROOT / "phase2_gru_summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2) + "\n")

    md = ["# Phase 2.2 — GRU forecaster summary\n"]
    md.append("> Pre-registered verdict: GRU must beat last-value by ≥10% MAPE on ≥6/8 topologies.\n")
    md.append("| Topology | num_od | GRU MAPE % | LV MAPE % | Δ (rel %) | R² (GRU) | sec |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    win_count = 0
    big_win_count = 0
    for m in all_metrics:
        if "error" in m:
            md.append(f"| {m['topology']} | err | — | — | — | — | — |")
            continue
        rel = m["rel_improvement_pct"]
        if rel >= 0: win_count += 1
        if rel >= 10: big_win_count += 1
        md.append(
            f"| {m['topology']} | {m['num_od']} | {m['gru']['mape_pct']:.2f} | "
            f"{m['last_value']['mape_pct']:.2f} | {rel:+.2f} | "
            f"{m['gru']['r2']:.3f} | {m['train_seconds']} |"
        )
    md.append("")
    md.append(f"GRU wins on MAPE: {win_count} / 8 topologies")
    md.append(f"GRU beats LV by ≥10% MAPE: {big_win_count} / 8 topologies")
    if big_win_count >= 6:
        md.append("\n**VERDICT: PASS** — GRU clears the pre-registered ≥10% threshold on ≥6/8.")
    else:
        md.append(f"\n**VERDICT: FAIL** — only {big_win_count}/8 met the ≥10% threshold.")
    summary_md = OUT_ROOT / "phase2_gru_summary.md"
    summary_md.write_text("\n".join(md) + "\n")
    print(f"\nWrote {summary_md}")
    print()
    for line in md[-15:]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
