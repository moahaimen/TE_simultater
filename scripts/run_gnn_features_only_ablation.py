#!/usr/bin/env python3
"""Step 1 Ablation: Features-only GNN vs Original GNN.

Tests whether enriched input features ALONE improve the GNN expert,
with dynamic K disabled (fixed K=40) and identical training pipeline.

Training pipeline (identical to original):
  Stage 1: Supervised oracle-aligned training (30 epochs, patience=8)
  Stage 2: REINFORCE LP-in-the-loop fine-tuning (10 epochs, patience=4)

Comparison:
  1. Original GNN: original features, fixed K=40, original checkpoint
  2. Features-only GNN: enriched features, fixed K=40, newly trained here

Topologies:
  Training: all 6 known (abilene, geant, cernet, ebone, sprintlink, tiscali)
  Evaluation: all 6 known + 2 unseen (germany50, vtlwavenet)

Output: results/gnn_plus/step1_features_only/
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
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ---------- constants (match original exactly) ----------
CONFIG_PATH = "configs/phase1_reactive_full.yaml"
SEED = 42
K_CRIT = 40
LT = 20
DEVICE = "cpu"

GNN_ORIG_CKPT = Path("results/phase1_reactive/gnn_selector/train/gnn_selector/gnn_selector.pt")
OUTPUT_ROOT = Path("results/gnn_plus/step1_features_only")
TRAIN_DIR = OUTPUT_ROOT / "training"
EVAL_DIR = OUTPUT_ROOT / "eval"
PLOT_DIR = OUTPUT_ROOT / "plots"

# Match original training config exactly
SUPERVISED_LR = 5e-4
SUPERVISED_WD = 1e-5
SUPERVISED_EPOCHS = 30
SUPERVISED_PATIENCE = 8
SUPERVISED_MARGIN = 0.1
MAX_TRAIN_PER_TOPO = 40
MAX_VAL_PER_TOPO = 20

REINFORCE_LR = 1e-4
REINFORCE_EPOCHS = 10
REINFORCE_PATIENCE = 4
REINFORCE_EMA = 0.9

MAX_TEST_STEPS = 75

KNOWN_TOPOS = {"abilene", "geant", "cernet", "rocketfuel_ebone",
               "rocketfuel_sprintlink", "rocketfuel_tiscali"}
UNSEEN_TOPOS = {"germany50", "topologyzoo_vtlwavenet2011"}
ALL_TOPOS = KNOWN_TOPOS | UNSEEN_TOPOS


# ---------- setup ----------

def setup():
    from te.baselines import (
        ecmp_splits, select_bottleneck_critical,
        select_sensitivity_critical, select_topk_by_demand,
    )
    from te.lp_solver import solve_selected_path_lp
    from te.simulator import apply_routing
    from phase1_reactive.eval.common import (
        load_bundle, load_named_dataset, collect_specs,
    )
    from phase1_reactive.eval.core import split_indices
    from phase1_reactive.drl.gnn_selector import (
        load_gnn_selector, build_graph_tensors, build_od_features,
        GNNSelectorConfig, GNNFlowSelector, save_gnn_selector,
    )
    from phase1_reactive.drl.gnn_plus_selector import (
        GNNPlusConfig, GNNPlusFlowSelector,
        build_graph_tensors_plus, build_od_features_plus,
        save_gnn_plus, load_gnn_plus,
    )
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
        "GNNSelectorConfig": GNNSelectorConfig,
        "GNNFlowSelector": GNNFlowSelector,
        "save_gnn_selector": save_gnn_selector,
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
            print(f"  Skip {spec.key}: {e}")
            continue
        ds.path_library = pl
        datasets[ds.key] = ds
        n_tms = len(ds.tm) if hasattr(ds, "tm") else 0
        label = "known" if ds.key in KNOWN_TOPOS else "unseen"
        print(f"  [{label}] {ds.key}: {len(ds.nodes)}N, {len(ds.edges)}E, "
              f"{len(ds.od_pairs)} ODs, {n_tms} TMs")
    return datasets


# ---------- Oracle label collection (identical to original) ----------

def collect_oracle_labels(M, dataset, path_library, tm, ecmp_base, capacities, k_crit):
    selectors = {
        "topk": lambda: M["select_topk_by_demand"](tm, k_crit),
        "bottleneck": lambda: M["select_bottleneck_critical"](
            tm, ecmp_base, path_library, capacities, k_crit),
        "sensitivity": lambda: M["select_sensitivity_critical"](
            tm, ecmp_base, path_library, capacities, k_crit),
    }
    best_mlu = float("inf")
    best_selected = []
    best_method = "topk"
    for name, fn in selectors.items():
        try:
            sel = fn()
            lp = M["solve_selected_path_lp"](
                tm_vector=tm, selected_ods=sel, base_splits=ecmp_base,
                path_library=path_library, capacities=capacities,
                time_limit_sec=LT,
            )
            mlu = float(lp.routing.mlu)
            if np.isfinite(mlu) and mlu < best_mlu:
                best_mlu = mlu
                best_selected = sel
                best_method = name
        except Exception:
            continue
    return best_selected, best_mlu, best_method


# ---------- Sample collection ----------

def collect_samples(M, datasets, split, max_per_topo, topo_filter=None):
    rng = np.random.default_rng(SEED)
    samples = []
    for key, ds in datasets.items():
        if topo_filter and key not in topo_filter:
            continue
        pl = ds.path_library
        ecmp_base = M["ecmp_splits"](pl)
        caps = np.asarray(ds.capacities, dtype=float)
        indices = M["split_indices"](ds, split)
        if len(indices) > max_per_topo:
            indices = sorted(rng.choice(indices, size=max_per_topo, replace=False).tolist())
        count = 0
        prev_tm = None
        for t_idx in indices:
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12:
                prev_tm = tm
                continue
            routing = M["apply_routing"](tm, ecmp_base, pl, caps)
            telemetry = M["compute_reactive_telemetry"](
                tm, ecmp_base, pl, routing, np.asarray(ds.weights, dtype=float),
            )
            oracle_sel, oracle_mlu, oracle_method = collect_oracle_labels(
                M, ds, pl, tm, ecmp_base, caps, K_CRIT,
            )
            if not oracle_sel:
                prev_tm = tm
                continue
            samples.append({
                "dataset": ds, "path_library": pl,
                "tm_vector": tm, "prev_tm": prev_tm,
                "telemetry": telemetry,
                "oracle_selected": oracle_sel, "oracle_mlu": oracle_mlu,
                "k_crit": K_CRIT, "capacities": caps,
            })
            count += 1
            prev_tm = tm
        print(f"    {key}: {count} {split} samples")
    return samples


# ---------- Stage 1: Supervised training ----------

def train_supervised(M, train_samples, val_samples):
    """Supervised oracle-aligned training — identical logic to gnn_training.py."""
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # Features-only config: enriched dims, but K learning disabled
    cfg = M["GNNPlusConfig"]()
    cfg.learn_k_crit = False  # NO dynamic K
    cfg.k_crit_min = 40
    cfg.k_crit_max = 40

    model = M["GNNPlusFlowSelector"](cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SUPERVISED_LR, weight_decay=SUPERVISED_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUPERVISED_EPOCHS)

    rng = np.random.default_rng(SEED)
    logs = []
    best_val_loss = float("inf")
    best_epoch = 0
    stale = 0
    t_start = time.perf_counter()

    for epoch in range(1, SUPERVISED_EPOCHS + 1):
        ep_start = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_samples))
        ep_losses = []

        for si in order:
            s = train_samples[si]
            graph_data = M["build_graph_tensors_plus"](
                s["dataset"], tm_vector=s["tm_vector"],
                path_library=s["path_library"],
                telemetry=s["telemetry"], prev_util=None, device=DEVICE,
            )
            od_data = M["build_od_features_plus"](
                s["dataset"], s["tm_vector"], s["path_library"],
                telemetry=s["telemetry"], prev_tm=s["prev_tm"], device=DEVICE,
            )

            scores, k_pred, info = model(graph_data, od_data)

            num_od = scores.size(0)
            oracle_mask = torch.zeros(num_od, device=DEVICE)
            for oid in s["oracle_selected"]:
                if oid < num_od:
                    oracle_mask[oid] = 1.0

            loss = M["_ranking_loss"](scores, oracle_mask, margin=SUPERVISED_MARGIN)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(float(loss.item()))

        scheduler.step()
        train_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        # Validation
        model.eval()
        val_losses = []
        val_overlaps = []
        with torch.no_grad():
            for s in val_samples:
                graph_data = M["build_graph_tensors_plus"](
                    s["dataset"], tm_vector=s["tm_vector"],
                    path_library=s["path_library"],
                    telemetry=s["telemetry"], prev_util=None, device=DEVICE,
                )
                od_data = M["build_od_features_plus"](
                    s["dataset"], s["tm_vector"], s["path_library"],
                    telemetry=s["telemetry"], prev_tm=s["prev_tm"], device=DEVICE,
                )
                scores, _, _ = model(graph_data, od_data)

                num_od = scores.size(0)
                oracle_mask = torch.zeros(num_od, device=DEVICE)
                for oid in s["oracle_selected"]:
                    if oid < num_od:
                        oracle_mask[oid] = 1.0
                vloss = M["_ranking_loss"](scores, oracle_mask, margin=SUPERVISED_MARGIN)
                val_losses.append(float(vloss.item()))

                # Selection overlap
                scores_np = scores.cpu().numpy()
                active = s["tm_vector"] > 0
                active_idx = np.where(active)[0]
                if active_idx.size > 0:
                    take = min(K_CRIT, active_idx.size)
                    active_scores = scores_np[active_idx]
                    top_local = np.argsort(-active_scores)[:take]
                    pred_set = set(active_idx[top_local].tolist())
                    oracle_set = set(s["oracle_selected"])
                    overlap = len(pred_set & oracle_set) / max(len(pred_set | oracle_set), 1)
                    val_overlaps.append(overlap)

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_overlap = float(np.mean(val_overlaps)) if val_overlaps else 0.0
        ep_time = time.perf_counter() - ep_start

        logs.append({
            "stage": "supervised", "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "val_overlap": val_overlap,
            "alpha": float(model.alpha.item()),
            "lr": float(scheduler.get_last_lr()[0]),
            "epoch_time_sec": ep_time,
        })
        print(f"  Supervised {epoch:3d}: train={train_loss:.4f}  val={val_loss:.4f}  "
              f"overlap={val_overlap:.3f}  alpha={model.alpha.item():.3f}  [{ep_time:.1f}s]",
              flush=True)

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale = 0
            M["save_gnn_plus"](model, TRAIN_DIR / "features_only_supervised.pt", extra_meta={
                "best_epoch": best_epoch, "best_val_loss": best_val_loss,
                "stage": "supervised",
            })
        else:
            stale += 1
        if stale >= SUPERVISED_PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    sup_time = time.perf_counter() - t_start
    print(f"[Supervised] Done in {sup_time:.1f}s, best epoch={best_epoch}, "
          f"best val_loss={best_val_loss:.4f}", flush=True)

    # Save log
    pd.DataFrame(logs).to_csv(TRAIN_DIR / "supervised_log.csv", index=False)
    return model, cfg, logs, sup_time, best_epoch


# ---------- Stage 2: REINFORCE fine-tuning ----------

def train_reinforce(M, model, cfg, train_samples, val_samples):
    """REINFORCE LP-in-the-loop fine-tuning — identical logic to gnn_training.py."""
    optimizer = torch.optim.Adam(model.parameters(), lr=REINFORCE_LR)
    rng = np.random.default_rng(SEED)

    baseline_reward = None
    best_val_mlu = float("inf")
    best_epoch = 0
    stale = 0
    logs = []
    t_start = time.perf_counter()

    print(f"\n[REINFORCE] Fine-tuning on {len(train_samples)} samples, "
          f"max {REINFORCE_EPOCHS} epochs", flush=True)

    for epoch in range(1, REINFORCE_EPOCHS + 1):
        ep_start = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_samples))
        ep_rewards = []
        ep_mlu = []

        for si in order:
            s = train_samples[si]
            graph_data = M["build_graph_tensors_plus"](
                s["dataset"], tm_vector=s["tm_vector"],
                path_library=s["path_library"],
                telemetry=s["telemetry"], prev_util=None, device=DEVICE,
            )
            od_data = M["build_od_features_plus"](
                s["dataset"], s["tm_vector"], s["path_library"],
                telemetry=s["telemetry"], prev_tm=s["prev_tm"], device=DEVICE,
            )

            scores, _, _ = model(graph_data, od_data)
            k = s["k_crit"]

            active = s["tm_vector"] > 0
            active_idx = np.where(active)[0]
            if active_idx.size == 0:
                continue

            active_scores = scores[torch.tensor(active_idx, dtype=torch.long, device=DEVICE)]
            log_probs = F.log_softmax(active_scores, dim=0)

            take = min(k, active_idx.size)
            _, top_local = torch.topk(active_scores, take)
            selected_ods = [int(active_idx[i]) for i in top_local.cpu().numpy()]
            selected_log_prob = log_probs[top_local].sum()

            ecmp_base = M["ecmp_splits"](s["path_library"])
            try:
                lp = M["solve_selected_path_lp"](
                    tm_vector=s["tm_vector"], selected_ods=selected_ods,
                    base_splits=ecmp_base, path_library=s["path_library"],
                    capacities=s["capacities"], time_limit_sec=10,
                )
                mlu = float(lp.routing.mlu)
            except Exception:
                continue
            if not np.isfinite(mlu):
                continue

            reward = -mlu
            ep_rewards.append(reward)
            ep_mlu.append(mlu)

            if baseline_reward is None:
                baseline_reward = reward
            else:
                baseline_reward = REINFORCE_EMA * baseline_reward + (1 - REINFORCE_EMA) * reward

            advantage = reward - baseline_reward
            loss = -advantage * selected_log_prob

            # Ranking loss for oracle alignment (same as original: weight=0.3)
            oracle_mask = torch.zeros(scores.size(0), device=DEVICE)
            for oid in s["oracle_selected"]:
                if oid < scores.size(0):
                    oracle_mask[oid] = 1.0
            rank_loss = M["_ranking_loss"](scores, oracle_mask, margin=0.05)
            total_loss = loss + 0.3 * rank_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        mean_mlu = float(np.mean(ep_mlu)) if ep_mlu else float("inf")

        # Validation via LP
        model.eval()
        val_mlus = []
        with torch.no_grad():
            for s in val_samples:
                graph_data = M["build_graph_tensors_plus"](
                    s["dataset"], tm_vector=s["tm_vector"],
                    path_library=s["path_library"],
                    telemetry=s["telemetry"], prev_util=None, device=DEVICE,
                )
                od_data = M["build_od_features_plus"](
                    s["dataset"], s["tm_vector"], s["path_library"],
                    telemetry=s["telemetry"], prev_tm=s["prev_tm"], device=DEVICE,
                )
                scores, _, _ = model(graph_data, od_data)
                active = s["tm_vector"] > 0
                active_idx = np.where(active)[0]
                if active_idx.size == 0:
                    continue
                active_scores = scores[torch.tensor(active_idx, dtype=torch.long, device=DEVICE)]
                take = min(K_CRIT, active_idx.size)
                _, top_local = torch.topk(active_scores, take)
                selected_ods = [int(active_idx[i]) for i in top_local.cpu().numpy()]

                ecmp_base = M["ecmp_splits"](s["path_library"])
                try:
                    lp = M["solve_selected_path_lp"](
                        tm_vector=s["tm_vector"], selected_ods=selected_ods,
                        base_splits=ecmp_base, path_library=s["path_library"],
                        capacities=s["capacities"], time_limit_sec=10,
                    )
                    val_mlus.append(float(lp.routing.mlu))
                except Exception:
                    pass

        val_mlu = float(np.mean(val_mlus)) if val_mlus else float("inf")
        ep_time = time.perf_counter() - ep_start

        logs.append({
            "stage": "reinforce", "epoch": epoch,
            "train_mean_mlu": mean_mlu, "val_mean_mlu": val_mlu,
            "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "alpha": float(model.alpha.item()),
            "epoch_time_sec": ep_time,
        })
        print(f"  REINFORCE {epoch:3d}: train_mlu={mean_mlu:.4f}  val_mlu={val_mlu:.4f}  "
              f"alpha={model.alpha.item():.3f}  [{ep_time:.1f}s]", flush=True)

        if val_mlu + 1e-6 < best_val_mlu:
            best_val_mlu = val_mlu
            best_epoch = epoch
            stale = 0
            M["save_gnn_plus"](model, TRAIN_DIR / "features_only_final.pt", extra_meta={
                "best_epoch": best_epoch, "best_val_mlu": best_val_mlu,
                "stage": "reinforce",
            })
        else:
            stale += 1
        if stale >= REINFORCE_PATIENCE:
            print(f"  REINFORCE early stopping at epoch {epoch}")
            break

    rl_time = time.perf_counter() - t_start
    print(f"[REINFORCE] Done in {rl_time:.1f}s, best val_mlu={best_val_mlu:.4f}", flush=True)

    pd.DataFrame(logs).to_csv(TRAIN_DIR / "reinforce_log.csv", index=False)
    return model, logs, rl_time, best_epoch, best_val_mlu


# ---------- Evaluation ----------

def evaluate(M, datasets, feat_model):
    """Side-by-side evaluation: original GNN vs features-only GNN."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Eval] Loading original GNN...", flush=True)
    gnn_orig, _ = M["load_gnn_selector"](GNN_ORIG_CKPT, device=DEVICE)
    gnn_orig.eval()
    feat_model.eval()

    rows = []
    eval_order = sorted(KNOWN_TOPOS & set(datasets.keys())) + \
                 sorted(UNSEEN_TOPOS & set(datasets.keys()))

    for topo_key in eval_order:
        if topo_key not in datasets:
            continue
        ds = datasets[topo_key]
        pl = ds.path_library
        ecmp_base = M["ecmp_splits"](pl)
        caps = np.asarray(ds.capacities, dtype=float)
        is_unseen = topo_key in UNSEEN_TOPOS

        test_idx = M["split_indices"](ds, "test")
        if len(test_idx) > MAX_TEST_STEPS:
            test_idx = test_idx[:MAX_TEST_STEPS]

        label = "UNSEEN" if is_unseen else "known"
        print(f"\n[Eval] {topo_key} ({label}): {len(test_idx)} steps", flush=True)

        prev_sel_orig = None
        prev_sel_feat = None
        prev_tm = None

        for step_i, t_idx in enumerate(test_idx):
            tm = ds.tm[t_idx]
            if np.max(tm) < 1e-12:
                prev_tm = tm
                continue

            routing_ecmp = M["apply_routing"](tm, ecmp_base, pl, caps)
            telemetry = M["compute_reactive_telemetry"](
                tm, ecmp_base, pl, routing_ecmp,
                np.asarray(ds.weights, dtype=float),
            )
            active_mask = (np.asarray(tm, dtype=np.float64) > 1e-12).astype(np.float32)

            # Oracle
            oracle_sel, oracle_mlu, oracle_method = collect_oracle_labels(
                M, ds, pl, tm, ecmp_base, caps, K_CRIT,
            )

            # --- Original GNN ---
            t0 = time.perf_counter()
            gd_orig = M["build_graph_tensors"](ds, telemetry=telemetry, device=DEVICE)
            od_orig = M["build_od_features"](ds, tm, pl, telemetry=telemetry, device=DEVICE)
            with torch.no_grad():
                sel_orig, info_orig = gnn_orig.select_critical_flows(
                    gd_orig, od_orig, active_mask=active_mask,
                    k_crit_default=K_CRIT, force_default_k=True,
                )
            time_orig = (time.perf_counter() - t0) * 1000

            try:
                lp_orig = M["solve_selected_path_lp"](
                    tm_vector=tm, selected_ods=sel_orig, base_splits=ecmp_base,
                    path_library=pl, capacities=caps, time_limit_sec=LT,
                )
                mlu_orig = float(lp_orig.routing.mlu)
            except Exception:
                mlu_orig = float("inf")

            # --- Features-only GNN ---
            t0 = time.perf_counter()
            gd_feat = M["build_graph_tensors_plus"](
                ds, tm_vector=tm, path_library=pl,
                telemetry=telemetry, prev_util=None, device=DEVICE,
            )
            od_feat = M["build_od_features_plus"](
                ds, tm, pl, telemetry=telemetry, prev_tm=prev_tm, device=DEVICE,
            )
            with torch.no_grad():
                sel_feat, info_feat = feat_model.select_critical_flows(
                    gd_feat, od_feat, active_mask=active_mask,
                    k_crit_default=K_CRIT, force_default_k=True,  # FIXED K=40
                )
            time_feat = (time.perf_counter() - t0) * 1000

            try:
                lp_feat = M["solve_selected_path_lp"](
                    tm_vector=tm, selected_ods=sel_feat, base_splits=ecmp_base,
                    path_library=pl, capacities=caps, time_limit_sec=LT,
                )
                mlu_feat = float(lp_feat.routing.mlu)
            except Exception:
                mlu_feat = float("inf")

            # Disturbance
            dist_orig = _disturbance(sel_orig, prev_sel_orig, K_CRIT)
            dist_feat = _disturbance(sel_feat, prev_sel_feat, K_CRIT)

            # PR
            pr_orig = (mlu_orig - oracle_mlu) / (oracle_mlu + 1e-12) if oracle_mlu > 0 else 0.0
            pr_feat = (mlu_feat - oracle_mlu) / (oracle_mlu + 1e-12) if oracle_mlu > 0 else 0.0

            # Selection overlap between methods
            ov = len(set(sel_orig) & set(sel_feat)) / max(len(set(sel_orig) | set(sel_feat)), 1)

            rows.append({
                "topology": topo_key,
                "is_unseen": is_unseen,
                "step": step_i, "tm_idx": t_idx,
                "mlu_orig": mlu_orig, "mlu_feat": mlu_feat,
                "pr_orig": pr_orig, "pr_feat": pr_feat,
                "time_ms_orig": time_orig, "time_ms_feat": time_feat,
                "disturbance_orig": dist_orig, "disturbance_feat": dist_feat,
                "oracle_mlu": oracle_mlu, "oracle_method": oracle_method,
                "selection_overlap": ov,
                "alpha_orig": info_orig.get("alpha", 0),
                "alpha_feat": info_feat.get("alpha", 0),
                "confidence_orig": info_orig.get("confidence", 0),
                "confidence_feat": info_feat.get("confidence", 0),
                "w_bn_orig": info_orig.get("w_bottleneck", 0),
                "w_bn_feat": info_feat.get("w_bottleneck", 0),
            })

            prev_sel_orig = sel_orig
            prev_sel_feat = sel_feat
            prev_tm = tm

            if (step_i + 1) % 25 == 0:
                print(f"    step {step_i+1}/{len(test_idx)}: "
                      f"orig={mlu_orig:.4f} feat={mlu_feat:.4f} "
                      f"overlap={ov:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "features_only_per_timestep.csv", index=False)
    print(f"\n[Eval] Saved {len(df)} rows")

    # Summary
    summary_rows = []
    for topo_key in eval_order:
        sub = df[df["topology"] == topo_key]
        if sub.empty:
            continue
        summary_rows.append(_make_summary_row(sub, topo_key, topo_key in UNSEEN_TOPOS))

    # Known aggregate
    known_sub = df[~df["is_unseen"]]
    if not known_sub.empty:
        summary_rows.append(_make_summary_row(known_sub, "KNOWN_AGG", False))

    # Unseen aggregate
    unseen_sub = df[df["is_unseen"]]
    if not unseen_sub.empty:
        summary_rows.append(_make_summary_row(unseen_sub, "UNSEEN_AGG", True))

    # Total aggregate
    if not df.empty:
        summary_rows.append(_make_summary_row(df, "TOTAL_AGG", False))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(EVAL_DIR / "features_only_summary.csv", index=False)
    print(f"[Eval] Saved summary")
    return df, summary_df


def _disturbance(cur, prev, k):
    if prev is None:
        return 0.0
    return len(set(cur) ^ set(prev)) / max(k, 1)


def _make_summary_row(sub, label, is_unseen):
    return {
        "topology": label,
        "is_unseen": is_unseen,
        "num_steps": len(sub),
        "mean_mlu_orig": sub["mlu_orig"].mean(),
        "mean_mlu_feat": sub["mlu_feat"].mean(),
        "median_mlu_orig": sub["mlu_orig"].median(),
        "median_mlu_feat": sub["mlu_feat"].median(),
        "mean_pr_orig": sub["pr_orig"].mean(),
        "mean_pr_feat": sub["pr_feat"].mean(),
        "mean_time_ms_orig": sub["time_ms_orig"].mean(),
        "mean_time_ms_feat": sub["time_ms_feat"].mean(),
        "mean_dist_orig": sub["disturbance_orig"].mean(),
        "mean_dist_feat": sub["disturbance_feat"].mean(),
        "mean_selection_overlap": sub["selection_overlap"].mean(),
        "mean_oracle_mlu": sub["oracle_mlu"].mean(),
        "mlu_improvement_pct": (
            (sub["mlu_orig"].mean() - sub["mlu_feat"].mean())
            / (sub["mlu_orig"].mean() + 1e-12) * 100
        ),
        "feat_wins": int((sub["mlu_feat"] < sub["mlu_orig"]).sum()),
        "orig_wins": int((sub["mlu_orig"] < sub["mlu_feat"]).sum()),
        "ties": int((sub["mlu_orig"] == sub["mlu_feat"]).sum()),
    }


# ---------- Plotting ----------

def generate_plots(df, summary_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    topos = sorted(df["topology"].unique())
    known = [t for t in topos if t in KNOWN_TOPOS]
    unseen = [t for t in topos if t in UNSEEN_TOPOS]

    # Plot 1: MLU CDF per topology (known)
    n = len(known)
    if n > 0:
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
        for i, topo in enumerate(known):
            ax = axes[0, i]
            sub = df[df["topology"] == topo]
            for col, lab, c in [("mlu_orig", "Original GNN", "tab:blue"),
                                 ("mlu_feat", "Features-only GNN", "tab:green")]:
                vals = np.sort(sub[col].values)
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                ax.plot(vals, cdf, label=lab, color=c, linewidth=2)
            ax.set_title(topo, fontsize=12, fontweight="bold")
            ax.set_xlabel("MLU"); ax.set_ylabel("CDF")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        fig.suptitle("MLU CDF — Known Topologies", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "mlu_cdf_known.png", dpi=150)
        plt.close(fig)

    # Plot 2: MLU CDF unseen
    if unseen:
        fig, axes = plt.subplots(1, len(unseen), figsize=(4.5 * len(unseen), 4), squeeze=False)
        for i, topo in enumerate(unseen):
            ax = axes[0, i]
            sub = df[df["topology"] == topo]
            for col, lab, c in [("mlu_orig", "Original GNN", "tab:blue"),
                                 ("mlu_feat", "Features-only GNN", "tab:green")]:
                vals = np.sort(sub[col].values)
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                ax.plot(vals, cdf, label=lab, color=c, linewidth=2)
            ax.set_title(topo, fontsize=12, fontweight="bold")
            ax.set_xlabel("MLU"); ax.set_ylabel("CDF")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        fig.suptitle("MLU CDF — Unseen Topologies", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "mlu_cdf_unseen.png", dpi=150)
        plt.close(fig)

    # Plot 3: Bar chart — mean MLU all topos
    present = [r for _, r in summary_df.iterrows()
               if r["topology"] not in ("KNOWN_AGG", "UNSEEN_AGG", "TOTAL_AGG")]
    if present:
        labels = [r["topology"] for r in present]
        mlu_o = [r["mean_mlu_orig"] for r in present]
        mlu_f = [r["mean_mlu_feat"] for r in present]
        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
        b1 = ax.bar(x - w/2, mlu_o, w, label="Original GNN", color="tab:blue", alpha=0.8)
        b2 = ax.bar(x + w/2, mlu_f, w, label="Features-only GNN", color="tab:green", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
        ax.set_ylabel("Mean MLU"); ax.set_title("Mean MLU Comparison", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "mean_mlu_bar.png", dpi=150)
        plt.close(fig)

    # Plot 4: Disturbance comparison
    if len(topos) > 0:
        fig, axes = plt.subplots(1, min(len(topos), 4),
                                  figsize=(4.5 * min(len(topos), 4), 4), squeeze=False)
        for i, topo in enumerate(topos[:4]):
            ax = axes[0, i]
            sub = df[df["topology"] == topo]
            for col, lab, c in [("disturbance_orig", "Original GNN", "tab:blue"),
                                 ("disturbance_feat", "Features-only GNN", "tab:green")]:
                vals = np.sort(sub[col].values)
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                ax.plot(vals, cdf, label=lab, color=c, linewidth=2)
            ax.set_title(topo, fontsize=12, fontweight="bold")
            ax.set_xlabel("Disturbance"); ax.set_ylabel("CDF")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        fig.suptitle("Disturbance CDF", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "disturbance_cdf.png", dpi=150)
        plt.close(fig)

    print(f"[Plots] Saved to {PLOT_DIR}")


# ---------- Main ----------

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1 ABLATION: FEATURES-ONLY GNN vs ORIGINAL GNN")
    print("=" * 70)

    print("\n[1/6] Setup...", flush=True)
    M = setup()

    print("\n[2/6] Loading topologies...", flush=True)
    datasets = load_all_topologies(M)

    print("\n[3/6] Collecting training samples (6 known topologies)...", flush=True)
    train_samples = collect_samples(M, datasets, "train", MAX_TRAIN_PER_TOPO,
                                     topo_filter=KNOWN_TOPOS)
    val_samples = collect_samples(M, datasets, "val", MAX_VAL_PER_TOPO,
                                   topo_filter=KNOWN_TOPOS)
    print(f"  Total: {len(train_samples)} train, {len(val_samples)} val")

    print("\n[4/6] Stage 1: Supervised training...", flush=True)
    model, cfg, sup_logs, sup_time, sup_best = train_supervised(M, train_samples, val_samples)

    # Reload best supervised checkpoint before REINFORCE
    model, _ = M["load_gnn_plus"](TRAIN_DIR / "features_only_supervised.pt", device=DEVICE)

    print("\n[5/6] Stage 2: REINFORCE fine-tuning...", flush=True)
    model, rl_logs, rl_time, rl_best, rl_best_mlu = train_reinforce(
        M, model, cfg, train_samples, val_samples,
    )

    # Reload best REINFORCE checkpoint
    final_ckpt = TRAIN_DIR / "features_only_final.pt"
    if final_ckpt.exists():
        model, _ = M["load_gnn_plus"](final_ckpt, device=DEVICE)
    else:
        print("  WARNING: REINFORCE did not improve. Using supervised checkpoint.")
        model, _ = M["load_gnn_plus"](TRAIN_DIR / "features_only_supervised.pt", device=DEVICE)

    # Save training summary
    summary = {
        "supervised_time_sec": sup_time,
        "supervised_best_epoch": sup_best,
        "reinforce_time_sec": rl_time,
        "reinforce_best_epoch": rl_best,
        "reinforce_best_val_mlu": rl_best_mlu,
        "total_training_time_sec": sup_time + rl_time,
        "total_train_samples": len(train_samples),
        "total_val_samples": len(val_samples),
    }
    (TRAIN_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )

    print("\n[6/6] Evaluation...", flush=True)
    df, summary_df = evaluate(M, datasets, model)

    print("\n[Plots] Generating...", flush=True)
    generate_plots(df, summary_df)

    print("\n" + "=" * 70)
    print("STEP 1 ABLATION RESULTS")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nTraining: supervised={sup_time:.0f}s + reinforce={rl_time:.0f}s "
          f"= {sup_time + rl_time:.0f}s total")
    print(f"All outputs: {OUTPUT_ROOT}")
    print("=" * 70)
