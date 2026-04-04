#!/usr/bin/env python3
"""Stage 1, Step 1.5a: Regularization pass for features-only GNN.

Tests dropout=0.2 and dropout=0.3 with full enriched features, fixed K=40.
Same training pipeline as original (supervised + REINFORCE).

Goal: fix unseen topology regression from Step 1 while keeping known gains.

Output: results/gnn_plus/stage1_regularization/
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ---------- constants ----------
CONFIG_PATH = "configs/phase1_reactive_full.yaml"
SEED = 42
K_CRIT = 40
LT = 20
DEVICE = "cpu"

GNN_ORIG_CKPT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
OUTPUT_ROOT = Path("results/gnn_plus/stage1_regularization")

DROPOUT_VALUES = [0.2, 0.3]

# Match original training exactly
SUP_LR = 5e-4
SUP_WD = 1e-5
SUP_EPOCHS = 30
SUP_PATIENCE = 8
SUP_MARGIN = 0.1
MAX_TRAIN_PER_TOPO = 40
MAX_VAL_PER_TOPO = 20

RL_LR = 1e-4
RL_EPOCHS = 10
RL_PATIENCE = 4
RL_EMA = 0.9

MAX_TEST_STEPS = 75

KNOWN_TOPOS = {"abilene", "geant", "cernet", "rocketfuel_ebone",
               "rocketfuel_sprintlink", "rocketfuel_tiscali"}
UNSEEN_TOPOS = {"germany50", "topologyzoo_vtlwavenet2011"}


def setup():
    from te.baselines import (ecmp_splits, select_bottleneck_critical,
                               select_sensitivity_critical, select_topk_by_demand)
    from te.lp_solver import solve_selected_path_lp
    from te.simulator import apply_routing
    from phase1_reactive.eval.common import load_bundle, load_named_dataset, collect_specs
    from phase1_reactive.eval.core import split_indices
    from phase1_reactive.drl.gnn_selector import (
        load_gnn_selector, build_graph_tensors, build_od_features)
    from phase1_reactive.drl.gnn_plus_selector import (
        GNNPlusConfig, GNNPlusFlowSelector,
        build_graph_tensors_plus, build_od_features_plus,
        save_gnn_plus, load_gnn_plus)
    from phase1_reactive.drl.state_builder import compute_reactive_telemetry
    from phase1_reactive.drl.gnn_training import _ranking_loss
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
        "split_indices": split_indices,
        "load_gnn_selector": load_gnn_selector,
        "build_graph_tensors": build_graph_tensors,
        "build_od_features": build_od_features,
        "GNNPlusConfig": GNNPlusConfig,
        "GNNPlusFlowSelector": GNNPlusFlowSelector,
        "build_graph_tensors_plus": build_graph_tensors_plus,
        "build_od_features_plus": build_od_features_plus,
        "save_gnn_plus": save_gnn_plus,
        "load_gnn_plus": load_gnn_plus,
        "compute_reactive_telemetry": compute_reactive_telemetry,
        "_ranking_loss": _ranking_loss,
    }


def load_all_topologies(M):
    bundle = M["load_bundle"](CONFIG_PATH)
    eval_specs = M["collect_specs"](bundle, "eval_topologies")
    gen_specs = M["collect_specs"](bundle, "generalization_topologies")
    datasets = {}
    for spec in eval_specs + gen_specs:
        try:
            ds, pl = M["load_named_dataset"](bundle, spec, 500)
        except Exception as e:
            print(f"  Skip {spec.key}: {e}"); continue
        ds.path_library = pl
        datasets[ds.key] = ds
        label = "known" if ds.key in KNOWN_TOPOS else "unseen"
        n_tms = len(ds.tm) if hasattr(ds, "tm") else 0
        print(f"  [{label}] {ds.key}: {len(ds.nodes)}N, {len(ds.edges)}E, {n_tms} TMs")
    return datasets


def collect_oracle(M, tm, ecmp_base, path_library, capacities):
    best_mlu, best_sel, best_m = float("inf"), [], "topk"
    for name, fn in [
        ("topk", lambda: M["select_topk_by_demand"](tm, K_CRIT)),
        ("bottleneck", lambda: M["select_bottleneck_critical"](
            tm, ecmp_base, path_library, capacities, K_CRIT)),
        ("sensitivity", lambda: M["select_sensitivity_critical"](
            tm, ecmp_base, path_library, capacities, K_CRIT)),
    ]:
        try:
            sel = fn()
            lp = M["solve_selected_path_lp"](tm_vector=tm, selected_ods=sel,
                base_splits=ecmp_base, path_library=path_library,
                capacities=capacities, time_limit_sec=LT)
            mlu = float(lp.routing.mlu)
            if np.isfinite(mlu) and mlu < best_mlu:
                best_mlu, best_sel, best_m = mlu, sel, name
        except Exception:
            continue
    return best_sel, best_mlu, best_m


def collect_samples(M, datasets, split, max_per, topo_filter):
    rng = np.random.default_rng(SEED)
    samples = []
    for key, ds in datasets.items():
        if key not in topo_filter:
            continue
        pl = ds.path_library
        ecmp_base = M["ecmp_splits"](pl)
        caps = np.asarray(ds.capacities, dtype=float)
        indices = M["split_indices"](ds, split)
        if len(indices) > max_per:
            indices = sorted(rng.choice(indices, size=max_per, replace=False).tolist())
        count = 0
        prev_tm = None
        for t_idx in indices:
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12:
                prev_tm = tm; continue
            routing = M["apply_routing"](tm, ecmp_base, pl, caps)
            telem = M["compute_reactive_telemetry"](
                tm, ecmp_base, pl, routing, np.asarray(ds.weights, dtype=float))
            o_sel, o_mlu, o_m = collect_oracle(M, tm, ecmp_base, pl, caps)
            if not o_sel:
                prev_tm = tm; continue
            samples.append({"dataset": ds, "path_library": pl,
                "tm_vector": tm, "prev_tm": prev_tm, "telemetry": telem,
                "oracle_selected": o_sel, "oracle_mlu": o_mlu,
                "k_crit": K_CRIT, "capacities": caps})
            count += 1; prev_tm = tm
        print(f"    {key}: {count} {split}")
    return samples


# ---------- Training ----------

def train_one_config(M, dropout, train_samples, val_samples):
    tag = f"d{int(dropout*10):02d}"
    out_dir = OUTPUT_ROOT / f"training_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = M["GNNPlusConfig"]()
    cfg.dropout = dropout
    cfg.learn_k_crit = False
    cfg.k_crit_min = 40
    cfg.k_crit_max = 40

    model = M["GNNPlusFlowSelector"](cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SUP_LR, weight_decay=SUP_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUP_EPOCHS)
    rng = np.random.default_rng(SEED)
    logs = []
    best_vl = float("inf"); best_ep = 0; stale = 0
    t0 = time.perf_counter()

    print(f"\n[Supervised dropout={dropout}]", flush=True)
    for epoch in range(1, SUP_EPOCHS + 1):
        ep0 = time.perf_counter()
        model.train()
        ep_losses = []
        for si in rng.permutation(len(train_samples)):
            s = train_samples[si]
            gd = M["build_graph_tensors_plus"](s["dataset"], tm_vector=s["tm_vector"],
                path_library=s["path_library"], telemetry=s["telemetry"], device=DEVICE)
            od = M["build_od_features_plus"](s["dataset"], s["tm_vector"], s["path_library"],
                telemetry=s["telemetry"], prev_tm=s["prev_tm"], device=DEVICE)
            scores, _, _ = model(gd, od)
            om = torch.zeros(scores.size(0), device=DEVICE)
            for oid in s["oracle_selected"]:
                if oid < scores.size(0): om[oid] = 1.0
            loss = M["_ranking_loss"](scores, om, margin=SUP_MARGIN)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); ep_losses.append(float(loss.item()))
        scheduler.step()
        tl = float(np.mean(ep_losses)) if ep_losses else 0.0

        model.eval()
        vls, vos = [], []
        with torch.no_grad():
            for s in val_samples:
                gd = M["build_graph_tensors_plus"](s["dataset"], tm_vector=s["tm_vector"],
                    path_library=s["path_library"], telemetry=s["telemetry"], device=DEVICE)
                od = M["build_od_features_plus"](s["dataset"], s["tm_vector"], s["path_library"],
                    telemetry=s["telemetry"], prev_tm=s["prev_tm"], device=DEVICE)
                scores, _, _ = model(gd, od)
                om = torch.zeros(scores.size(0), device=DEVICE)
                for oid in s["oracle_selected"]:
                    if oid < scores.size(0): om[oid] = 1.0
                vls.append(float(M["_ranking_loss"](scores, om, margin=SUP_MARGIN).item()))
                sn = scores.cpu().numpy()
                ai = np.where(s["tm_vector"] > 0)[0]
                if ai.size > 0:
                    take = min(K_CRIT, ai.size)
                    top = np.argsort(-sn[ai])[:take]
                    ps = set(ai[top].tolist()); os_ = set(s["oracle_selected"])
                    vos.append(len(ps & os_) / max(len(ps | os_), 1))
        vl = float(np.mean(vls)) if vls else 0.0
        vo = float(np.mean(vos)) if vos else 0.0
        et = time.perf_counter() - ep0
        logs.append({"stage": "supervised", "epoch": epoch, "train_loss": tl,
            "val_loss": vl, "val_overlap": vo, "alpha": float(model.alpha.item()),
            "epoch_time_sec": et})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}: tl={tl:.4f} vl={vl:.4f} ov={vo:.3f} a={model.alpha.item():.3f} [{et:.1f}s]", flush=True)
        if vl + 1e-6 < best_vl:
            best_vl = vl; best_ep = epoch; stale = 0
            M["save_gnn_plus"](model, out_dir / "supervised.pt",
                extra_meta={"best_epoch": best_ep, "stage": "supervised", "dropout": dropout})
        else:
            stale += 1
        if stale >= SUP_PATIENCE:
            print(f"  Early stop at {epoch}"); break
    sup_time = time.perf_counter() - t0
    print(f"  Supervised done: {sup_time:.0f}s, best_ep={best_ep}", flush=True)
    pd.DataFrame(logs).to_csv(out_dir / "supervised_log.csv", index=False)

    # Reload best supervised
    model, _ = M["load_gnn_plus"](out_dir / "supervised.pt", device=DEVICE)

    # REINFORCE
    optimizer = torch.optim.Adam(model.parameters(), lr=RL_LR)
    rng = np.random.default_rng(SEED)
    baseline_reward = None; best_vm = float("inf"); best_re = 0; stale = 0
    rl_logs = []; t0 = time.perf_counter()
    print(f"\n[REINFORCE dropout={dropout}]", flush=True)

    for epoch in range(1, RL_EPOCHS + 1):
        ep0 = time.perf_counter()
        model.train()
        ep_mlu = []
        for si in rng.permutation(len(train_samples)):
            s = train_samples[si]
            gd = M["build_graph_tensors_plus"](s["dataset"], tm_vector=s["tm_vector"],
                path_library=s["path_library"], telemetry=s["telemetry"], device=DEVICE)
            od = M["build_od_features_plus"](s["dataset"], s["tm_vector"], s["path_library"],
                telemetry=s["telemetry"], prev_tm=s["prev_tm"], device=DEVICE)
            scores, _, _ = model(gd, od)
            active = s["tm_vector"] > 0; ai = np.where(active)[0]
            if ai.size == 0: continue
            as_ = scores[torch.tensor(ai, dtype=torch.long, device=DEVICE)]
            lp_ = F.log_softmax(as_, dim=0)
            take = min(K_CRIT, ai.size)
            _, tl_ = torch.topk(as_, take)
            sel = [int(ai[i]) for i in tl_.cpu().numpy()]
            slp = lp_[tl_].sum()
            ecmp = M["ecmp_splits"](s["path_library"])
            try:
                lp = M["solve_selected_path_lp"](tm_vector=s["tm_vector"], selected_ods=sel,
                    base_splits=ecmp, path_library=s["path_library"],
                    capacities=s["capacities"], time_limit_sec=10)
                mlu = float(lp.routing.mlu)
            except Exception: continue
            if not np.isfinite(mlu): continue
            ep_mlu.append(mlu)
            reward = -mlu
            if baseline_reward is None: baseline_reward = reward
            else: baseline_reward = RL_EMA * baseline_reward + (1 - RL_EMA) * reward
            adv = reward - baseline_reward
            loss = -adv * slp
            om = torch.zeros(scores.size(0), device=DEVICE)
            for oid in s["oracle_selected"]:
                if oid < scores.size(0): om[oid] = 1.0
            rl_ = M["_ranking_loss"](scores, om, margin=0.05)
            total = loss + 0.3 * rl_
            optimizer.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        mm = float(np.mean(ep_mlu)) if ep_mlu else float("inf")

        model.eval()
        vms = []
        with torch.no_grad():
            for s in val_samples:
                gd = M["build_graph_tensors_plus"](s["dataset"], tm_vector=s["tm_vector"],
                    path_library=s["path_library"], telemetry=s["telemetry"], device=DEVICE)
                od = M["build_od_features_plus"](s["dataset"], s["tm_vector"], s["path_library"],
                    telemetry=s["telemetry"], prev_tm=s["prev_tm"], device=DEVICE)
                scores, _, _ = model(gd, od)
                active = s["tm_vector"] > 0; ai = np.where(active)[0]
                if ai.size == 0: continue
                as_ = scores[torch.tensor(ai, dtype=torch.long, device=DEVICE)]
                take = min(K_CRIT, ai.size)
                _, tl_ = torch.topk(as_, take)
                sel = [int(ai[i]) for i in tl_.cpu().numpy()]
                ecmp = M["ecmp_splits"](s["path_library"])
                try:
                    lp = M["solve_selected_path_lp"](tm_vector=s["tm_vector"], selected_ods=sel,
                        base_splits=ecmp, path_library=s["path_library"],
                        capacities=s["capacities"], time_limit_sec=10)
                    vms.append(float(lp.routing.mlu))
                except Exception: pass
        vm = float(np.mean(vms)) if vms else float("inf")
        et = time.perf_counter() - ep0
        rl_logs.append({"stage": "reinforce", "epoch": epoch, "train_mlu": mm,
            "val_mlu": vm, "alpha": float(model.alpha.item()), "epoch_time_sec": et})
        print(f"  RL Ep {epoch}: train_mlu={mm:.2f} val_mlu={vm:.2f} a={model.alpha.item():.3f} [{et:.0f}s]", flush=True)
        if vm + 1e-6 < best_vm:
            best_vm = vm; best_re = epoch; stale = 0
            M["save_gnn_plus"](model, out_dir / "final.pt",
                extra_meta={"best_epoch": best_re, "stage": "reinforce", "dropout": dropout})
        else:
            stale += 1
        if stale >= RL_PATIENCE: print(f"  RL early stop at {epoch}"); break

    rl_time = time.perf_counter() - t0
    pd.DataFrame(rl_logs).to_csv(out_dir / "reinforce_log.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "dropout": dropout, "sup_time": sup_time, "sup_best_epoch": best_ep,
        "rl_time": rl_time, "rl_best_epoch": best_re, "rl_best_val_mlu": best_vm,
        "total_time": sup_time + rl_time,
    }, indent=2) + "\n", encoding="utf-8")

    # Reload best
    ckpt = out_dir / "final.pt"
    if not ckpt.exists(): ckpt = out_dir / "supervised.pt"
    model, _ = M["load_gnn_plus"](ckpt, device=DEVICE)
    return model, sup_time + rl_time


# ---------- Evaluation ----------

def evaluate_all(M, datasets, models_dict):
    """Evaluate original GNN + all dropout variants."""
    eval_dir = OUTPUT_ROOT / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    gnn_orig, _ = M["load_gnn_selector"](GNN_ORIG_CKPT, device=DEVICE)
    gnn_orig.eval()
    for m in models_dict.values():
        m.eval()

    rows = []
    eval_order = sorted(KNOWN_TOPOS & set(datasets.keys())) + \
                 sorted(UNSEEN_TOPOS & set(datasets.keys()))

    for topo_key in eval_order:
        if topo_key not in datasets: continue
        ds = datasets[topo_key]
        pl = ds.path_library; ecmp_base = M["ecmp_splits"](pl)
        caps = np.asarray(ds.capacities, dtype=float)
        is_unseen = topo_key in UNSEEN_TOPOS

        test_idx = M["split_indices"](ds, "test")[:MAX_TEST_STEPS]
        label = "UNSEEN" if is_unseen else "known"
        print(f"\n[Eval] {topo_key} ({label}): {len(test_idx)} steps", flush=True)

        prev_sels = {"orig": None}
        for tag in models_dict: prev_sels[tag] = None
        prev_tm = None

        for step_i, t_idx in enumerate(test_idx):
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12: prev_tm = tm; continue
            routing = M["apply_routing"](tm, ecmp_base, pl, caps)
            telem = M["compute_reactive_telemetry"](
                tm, ecmp_base, pl, routing, np.asarray(ds.weights, dtype=float))
            amask = (np.asarray(tm, dtype=np.float64) > 1e-12).astype(np.float32)
            o_sel, o_mlu, _ = collect_oracle(M, tm, ecmp_base, pl, caps)

            row = {"topology": topo_key, "is_unseen": is_unseen,
                   "step": step_i, "tm_idx": t_idx, "oracle_mlu": o_mlu}

            # Original GNN
            t0 = time.perf_counter()
            gd = M["build_graph_tensors"](ds, telemetry=telem, device=DEVICE)
            od = M["build_od_features"](ds, tm, pl, telemetry=telem, device=DEVICE)
            with torch.no_grad():
                sel, info = gnn_orig.select_critical_flows(
                    gd, od, active_mask=amask, k_crit_default=K_CRIT, force_default_k=True)
            t_orig = (time.perf_counter() - t0) * 1000
            try:
                lp = M["solve_selected_path_lp"](tm_vector=tm, selected_ods=sel,
                    base_splits=ecmp_base, path_library=pl, capacities=caps, time_limit_sec=LT)
                mlu = float(lp.routing.mlu)
            except Exception: mlu = float("inf")
            row["mlu_orig"] = mlu
            row["time_ms_orig"] = t_orig
            row["dist_orig"] = _dist(sel, prev_sels["orig"], K_CRIT)
            row["pr_orig"] = (mlu - o_mlu) / (o_mlu + 1e-12)
            prev_sels["orig"] = sel

            # Each dropout variant
            for tag, model in models_dict.items():
                t0 = time.perf_counter()
                gd = M["build_graph_tensors_plus"](ds, tm_vector=tm, path_library=pl,
                    telemetry=telem, device=DEVICE)
                od = M["build_od_features_plus"](ds, tm, pl, telemetry=telem,
                    prev_tm=prev_tm, device=DEVICE)
                with torch.no_grad():
                    sel, info = model.select_critical_flows(
                        gd, od, active_mask=amask, k_crit_default=K_CRIT, force_default_k=True)
                t_m = (time.perf_counter() - t0) * 1000
                try:
                    lp = M["solve_selected_path_lp"](tm_vector=tm, selected_ods=sel,
                        base_splits=ecmp_base, path_library=pl, capacities=caps, time_limit_sec=LT)
                    mlu_m = float(lp.routing.mlu)
                except Exception: mlu_m = float("inf")
                row[f"mlu_{tag}"] = mlu_m
                row[f"time_ms_{tag}"] = t_m
                row[f"dist_{tag}"] = _dist(sel, prev_sels[tag], K_CRIT)
                row[f"pr_{tag}"] = (mlu_m - o_mlu) / (o_mlu + 1e-12)
                prev_sels[tag] = sel

            rows.append(row)
            prev_tm = tm
            if (step_i + 1) % 25 == 0:
                parts = [f"orig={row['mlu_orig']:.4f}"]
                for tag in models_dict:
                    parts.append(f"{tag}={row[f'mlu_{tag}']:.4f}")
                print(f"    step {step_i+1}: {' '.join(parts)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(eval_dir / "regularization_per_timestep.csv", index=False)

    # Summary
    tags = list(models_dict.keys())
    summary_rows = []
    for topo_key in eval_order:
        sub = df[df["topology"] == topo_key]
        if sub.empty: continue
        summary_rows.append(_summary(sub, topo_key, topo_key in UNSEEN_TOPOS, tags))

    for label, mask in [("KNOWN_AGG", ~df["is_unseen"]), ("UNSEEN_AGG", df["is_unseen"]),
                         ("TOTAL_AGG", pd.Series(True, index=df.index))]:
        sub = df[mask]
        if not sub.empty:
            summary_rows.append(_summary(sub, label, "UNSEEN" in label, tags))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(eval_dir / "regularization_summary.csv", index=False)
    print(f"\n[Eval] Saved {len(df)} rows + summary")
    return df, summary_df


def _dist(cur, prev, k):
    if prev is None: return 0.0
    return len(set(cur) ^ set(prev)) / max(k, 1)


def _summary(sub, label, is_unseen, tags):
    row = {"topology": label, "is_unseen": is_unseen, "num_steps": len(sub),
           "mean_mlu_orig": sub["mlu_orig"].mean(),
           "mean_dist_orig": sub["dist_orig"].mean(),
           "mean_time_ms_orig": sub["time_ms_orig"].mean(),
           "mean_pr_orig": sub["pr_orig"].mean()}
    for tag in tags:
        row[f"mean_mlu_{tag}"] = sub[f"mlu_{tag}"].mean()
        row[f"mean_dist_{tag}"] = sub[f"dist_{tag}"].mean()
        row[f"mean_time_ms_{tag}"] = sub[f"time_ms_{tag}"].mean()
        row[f"mean_pr_{tag}"] = sub[f"pr_{tag}"].mean()
        row[f"mlu_improve_{tag}_pct"] = (
            (sub["mlu_orig"].mean() - sub[f"mlu_{tag}"].mean())
            / (sub["mlu_orig"].mean() + 1e-12) * 100)
        row[f"wins_{tag}"] = int((sub[f"mlu_{tag}"] < sub["mlu_orig"]).sum())
        row[f"wins_orig_vs_{tag}"] = int((sub["mlu_orig"] < sub[f"mlu_{tag}"]).sum())
    return row


# ---------- Plotting ----------

def generate_plots(df, summary_df, tags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = OUTPUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    topos = sorted(df["topology"].unique())
    known = [t for t in topos if t in KNOWN_TOPOS]
    unseen = [t for t in topos if t in UNSEEN_TOPOS]
    colors = {"orig": "tab:blue", "d02": "tab:orange", "d03": "tab:red"}
    labels = {"orig": "Original GNN", "d02": "Feat+dropout=0.2", "d03": "Feat+dropout=0.3"}

    # Plot 1: MLU CDF known
    if known:
        n = len(known)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
        for i, topo in enumerate(known):
            ax = axes[0, i]; sub = df[df["topology"] == topo]
            for key in ["orig"] + tags:
                col = f"mlu_{key}"
                vals = np.sort(sub[col].values)
                cdf = np.arange(1, len(vals)+1)/len(vals)
                ax.plot(vals, cdf, label=labels.get(key, key), color=colors.get(key, "gray"), lw=2)
            ax.set_title(topo, fontweight="bold"); ax.set_xlabel("MLU"); ax.set_ylabel("CDF")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        fig.suptitle("MLU CDF — Known Topologies (Regularization)", fontweight="bold")
        fig.tight_layout(); fig.savefig(plot_dir / "mlu_cdf_known.png", dpi=150); plt.close(fig)

    # Plot 2: MLU CDF unseen
    if unseen:
        fig, axes = plt.subplots(1, len(unseen), figsize=(4.5*len(unseen), 4), squeeze=False)
        for i, topo in enumerate(unseen):
            ax = axes[0, i]; sub = df[df["topology"] == topo]
            for key in ["orig"] + tags:
                col = f"mlu_{key}"
                vals = np.sort(sub[col].values)
                cdf = np.arange(1, len(vals)+1)/len(vals)
                ax.plot(vals, cdf, label=labels.get(key, key), color=colors.get(key, "gray"), lw=2)
            ax.set_title(topo, fontweight="bold"); ax.set_xlabel("MLU"); ax.set_ylabel("CDF")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        fig.suptitle("MLU CDF — Unseen Topologies (Regularization)", fontweight="bold")
        fig.tight_layout(); fig.savefig(plot_dir / "mlu_cdf_unseen.png", dpi=150); plt.close(fig)

    # Plot 3: Bar chart aggregate
    agg_rows = summary_df[summary_df["topology"].isin(["KNOWN_AGG", "UNSEEN_AGG", "TOTAL_AGG"])]
    if not agg_rows.empty:
        agg_labels = agg_rows["topology"].tolist()
        x = np.arange(len(agg_labels))
        w = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))
        for j, key in enumerate(["orig"] + tags):
            col = f"mean_mlu_{key}" if key != "orig" else "mean_mlu_orig"
            vals = agg_rows[col].tolist()
            bars = ax.bar(x + (j - 1) * w, vals, w, label=labels.get(key, key),
                         color=colors.get(key, "gray"), alpha=0.8)
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                       f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(agg_labels)
        ax.set_ylabel("Mean MLU"); ax.set_title("Aggregate MLU Comparison", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(plot_dir / "aggregate_mlu_bar.png", dpi=150); plt.close(fig)

    # Plot 4: Disturbance comparison
    if unseen:
        fig, axes = plt.subplots(1, len(unseen), figsize=(4.5*len(unseen), 4), squeeze=False)
        for i, topo in enumerate(unseen):
            ax = axes[0, i]; sub = df[df["topology"] == topo]
            for key in ["orig"] + tags:
                col = f"dist_{key}"
                vals = np.sort(sub[col].values)
                cdf = np.arange(1, len(vals)+1)/len(vals)
                ax.plot(vals, cdf, label=labels.get(key, key), color=colors.get(key, "gray"), lw=2)
            ax.set_title(topo, fontweight="bold"); ax.set_xlabel("Disturbance"); ax.set_ylabel("CDF")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        fig.suptitle("Disturbance CDF — Unseen Topologies", fontweight="bold")
        fig.tight_layout(); fig.savefig(plot_dir / "disturbance_unseen.png", dpi=150); plt.close(fig)

    print(f"[Plots] Saved to {plot_dir}")


# ---------- Main ----------

if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 1, STEP 1.5a: REGULARIZATION PASS")
    print("=" * 70)

    M = setup()
    print("\n[1] Loading topologies...", flush=True)
    datasets = load_all_topologies(M)

    print("\n[2] Collecting samples (6 known)...", flush=True)
    train_samples = collect_samples(M, datasets, "train", MAX_TRAIN_PER_TOPO, KNOWN_TOPOS)
    val_samples = collect_samples(M, datasets, "val", MAX_VAL_PER_TOPO, KNOWN_TOPOS)
    print(f"  Total: {len(train_samples)} train, {len(val_samples)} val")

    models = {}
    for dropout in DROPOUT_VALUES:
        tag = f"d{int(dropout*10):02d}"
        print(f"\n[3] Training dropout={dropout}...", flush=True)
        model, t_time = train_one_config(M, dropout, train_samples, val_samples)
        models[tag] = model
        print(f"  Total training time: {t_time:.0f}s")

    print("\n[4] Evaluating all configs...", flush=True)
    df, summary_df = evaluate_all(M, datasets, models)

    print("\n[5] Generating plots...", flush=True)
    generate_plots(df, summary_df, list(models.keys()))

    print("\n" + "=" * 70)
    print("STAGE 1 REGULARIZATION RESULTS")
    print("=" * 70)
    # Print key columns only
    display_cols = ["topology", "num_steps", "mean_mlu_orig"]
    for tag in models:
        display_cols += [f"mean_mlu_{tag}", f"mlu_improve_{tag}_pct", f"wins_{tag}"]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 70)
