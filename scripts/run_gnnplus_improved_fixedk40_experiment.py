#!/usr/bin/env python3
"""Train and evaluate an improved fixed-K40 GNN+ branch.

This runner turns the implemented section 3/4/5/7 code paths into an actual
checkpoint and a separate zero-shot evaluation bundle.

Scientific scope:
  - methods: ECMP, Bottleneck, Original GNN, GNN+
  - no MetaGate / Stable MetaGate
  - fixed K = 40
  - unseen evaluation is zero-shot

Improvement scope:
  - section 3: physical/stress-aware features
  - section 4: soft teacher + continuous criticality supervision
  - section 5: RL fine-tuning reward aligned with thesis metrics
  - section 7: disturbance-aware temporal features + continuity bonus
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent.parent / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from phase1_reactive.drl.gnn_plus_selector import (  # noqa: E402
    GNNPlusConfig,
    GNNPlusFlowSelector,
    build_graph_tensors_plus_section7,
    build_od_features_plus_section7,
    load_gnn_plus,
    save_gnn_plus,
)
from phase1_reactive.drl.gnn_training import (  # noqa: E402
    GNNReinforceConfig,
    _bottleneck_baseline_mlu,
    _collect_oracle_labels,
    _collect_soft_teacher_targets,
    _ranking_loss,
    _reference_model_mlu,
    _score_margin_regularizer,
    _soft_teacher_loss,
)
from phase1_reactive.drl.state_builder import compute_reactive_telemetry  # noqa: E402
from phase1_reactive.eval.core import split_indices  # noqa: E402
from phase1_reactive.routing.path_cache import assert_selected_ods_have_paths, surviving_od_mask  # noqa: E402
from te.baselines import clone_splits, ecmp_splits, select_bottleneck_critical  # noqa: E402
from te.disturbance import compute_disturbance  # noqa: E402
from te.lp_solver import solve_selected_path_lp  # noqa: E402
from te.simulator import apply_routing  # noqa: E402


SEED = 42
DEVICE = "cpu"
K_CRIT = 40
LP_TIME_LIMIT = 20
NUM_RUNS = 3
TRAIN_MAX_PER_TOPO = 30
VAL_MAX_PER_TOPO = 12
RL_MAX_TRAIN_SAMPLES = 120
RL_MAX_VAL_SAMPLES = 48
SUP_MAX_EPOCHS = 12
SUP_PATIENCE = 4
RL_MAX_EPOCHS = 4
RL_PATIENCE = 2
CONTINUITY_BONUS = 0.10

KNOWN_TOPOLOGIES = ["abilene", "cernet", "geant", "ebone", "sprintlink", "tiscali"]
UNSEEN_TOPOLOGIES = ["germany50", "vtlwavenet2011"]
ALL_TOPOLOGIES = KNOWN_TOPOLOGIES + UNSEEN_TOPOLOGIES
CORE_METHODS = ["ecmp", "bottleneck", "gnn", "gnnplus"]

TOPOLOGY_DISPLAY = {
    "abilene": "Abilene",
    "cernet": "CERNET",
    "geant": "GEANT",
    "ebone": "Ebone",
    "sprintlink": "Sprintlink",
    "tiscali": "Tiscali",
    "germany50": "Germany50",
    "vtlwavenet2011": "VtlWavenet2011",
}
METHOD_LABELS = {
    "ecmp": "ECMP",
    "bottleneck": "Bottleneck",
    "gnn": "Original GNN",
    "gnnplus": "GNN+",
}

OUTPUT_DIR = PROJECT_ROOT / "results" / "gnnplus_improved_fixedk40_experiment"
TRAIN_DIR = OUTPUT_DIR / "training"
PLOTS_DIR = OUTPUT_DIR / "plots"
COMPARISON_DIR = OUTPUT_DIR / "comparison"
REPORT_DOCX = OUTPUT_DIR / "GNNPLUS_IMPROVED_FIXEDK40_ZERO_SHOT_REPORT.docx"
AUDIT_MD = OUTPUT_DIR / "experiment_audit.md"

SUMMARY_CSV = OUTPUT_DIR / "packet_sdn_summary.csv"
FAILURE_CSV = OUTPUT_DIR / "packet_sdn_failure.csv"
SDN_METRICS_CSV = OUTPUT_DIR / "packet_sdn_sdn_metrics.csv"
SPLIT_MANIFEST_JSON = OUTPUT_DIR / "split_manifest.json"

SUP_CKPT = TRAIN_DIR / "gnn_plus_supervised_improved.pt"
FINAL_CKPT = TRAIN_DIR / "gnn_plus_improved_fixedk40.pt"
SUP_LOG_CSV = TRAIN_DIR / "supervised_train_log.csv"
RL_LOG_CSV = TRAIN_DIR / "reinforce_log.csv"
TRAINING_SUMMARY_JSON = TRAIN_DIR / "training_summary.json"

BASE_GNNPLUS_CKPT = PROJECT_ROOT / "results" / "gnn_plus_retrained_fixedk40" / "gnn_plus_fixed_k40.pt"
BASELINE_OUTPUT_DIR = PROJECT_ROOT / "results" / "professor_clean_gnnplus_zeroshot"
HELPER_PATH = PROJECT_ROOT / "scripts" / "build_gnnplus_packet_sdn_report_fixed.py"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_gnnplus_packet_sdn_full.py"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runner():
    return load_module(RUNNER_PATH, "run_gnnplus_packet_sdn_full_improved")


def load_helper():
    return load_module(HELPER_PATH, "build_gnnplus_packet_sdn_report_fixed_improved")


def selection_indicator(num_od: int, selected: list[int]) -> np.ndarray:
    out = np.zeros(int(num_od), dtype=np.float32)
    if selected:
        out[np.asarray(selected, dtype=int)] = 1.0
    return out


def topk_from_soft_target(soft_target: np.ndarray, k: int) -> list[int]:
    scores = np.asarray(soft_target, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return []
    positive = np.where(scores > 0.0)[0]
    if positive.size > 0:
        order = positive[np.argsort(-scores[positive], kind="mergesort")]
        return order[: min(int(k), int(order.size))].astype(int).tolist()
    order = np.argsort(-scores, kind="mergesort")
    return order[: min(int(k), int(order.size))].astype(int).tolist()


def build_section7_inputs(sample: dict, *, device: str):
    graph_data = build_graph_tensors_plus_section7(
        sample["dataset"],
        tm_vector=sample["tm_vector"],
        path_library=sample["path_library"],
        telemetry=sample["telemetry"],
        prev_util=sample["prev_util"],
        prev_tm=sample["prev_tm"],
        prev_selected_indicator=sample["prev_selected_indicator"],
        prev_disturbance=sample["prev_disturbance"],
        device=device,
    )
    od_data = build_od_features_plus_section7(
        sample["dataset"],
        sample["tm_vector"],
        sample["path_library"],
        telemetry=sample["telemetry"],
        prev_tm=sample["prev_tm"],
        prev_util=sample["prev_util"],
        prev_selected_indicator=sample["prev_selected_indicator"],
        prev_disturbance=sample["prev_disturbance"],
        device=device,
    )
    return graph_data, od_data


def continuity_select(
    scores: torch.Tensor,
    *,
    active_mask: np.ndarray,
    k: int,
    prev_selected_indicator: np.ndarray | None = None,
    continuity_bonus: float = 0.0,
):
    active = np.asarray(active_mask, dtype=bool)
    active_idx = np.where(active)[0]
    if active_idx.size == 0:
        return [], None, None
    take = min(int(k), int(active_idx.size))
    active_scores = scores[torch.tensor(active_idx, dtype=torch.long, device=scores.device)]
    ranking_scores = active_scores
    if continuity_bonus > 0.0 and prev_selected_indicator is not None:
        prev = np.asarray(prev_selected_indicator, dtype=np.float32).reshape(-1)
        if prev.size != scores.shape[0]:
            fixed = np.zeros(scores.shape[0], dtype=np.float32)
            fixed[: min(scores.shape[0], prev.size)] = prev[: min(scores.shape[0], prev.size)]
            prev = fixed
        prev_active = torch.tensor(prev[active_idx], dtype=active_scores.dtype, device=scores.device)
        score_min = torch.min(active_scores)
        score_span = torch.max(active_scores) - score_min
        if float(score_span.item()) > 1e-12:
            normalized = (active_scores - score_min) / (score_span + 1e-12)
        else:
            normalized = torch.zeros_like(active_scores)
        ranking_scores = normalized + float(continuity_bonus) * prev_active
    top_local = torch.topk(ranking_scores, take).indices
    selected = [int(active_idx[i]) for i in top_local.detach().cpu().numpy()]
    selected_log_prob = torch.log_softmax(ranking_scores, dim=0)[top_local].sum()
    return selected, ranking_scores, selected_log_prob


def collect_split_samples(
    runner,
    topo_keys: list[str],
    *,
    split_name: str,
    max_per_topology: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    datasets = {}
    samples = []

    for topo_key in topo_keys:
        dataset, path_library = runner.load_dataset(topo_key)
        datasets[topo_key] = (dataset, path_library)
        weights = np.asarray(dataset.weights, dtype=float)
        capacities = np.asarray(dataset.capacities, dtype=float)
        indices = split_indices(dataset, split_name)
        if len(indices) > max_per_topology:
            indices = sorted(rng.choice(indices, size=max_per_topology, replace=False).tolist())

        ecmp_base = ecmp_splits(path_library)
        prev_splits = clone_splits(ecmp_base)
        prev_selected = np.zeros(len(dataset.od_pairs), dtype=np.float32)
        prev_disturbance = 0.0
        prev_latency_by_od = None
        prev_tm = None
        prev_util = None

        print(f"[collect:{split_name}] {topo_key}: {len(indices)} candidate timesteps", flush=True)
        for t_idx in indices:
            tm_vector = np.asarray(dataset.tm[t_idx], dtype=float)
            if float(np.max(tm_vector)) < 1e-12:
                prev_tm = tm_vector
                continue

            routing = apply_routing(tm_vector, prev_splits, path_library, capacities)
            telemetry = compute_reactive_telemetry(
                tm_vector,
                prev_splits,
                path_library,
                routing,
                weights,
                prev_latency_by_od=prev_latency_by_od,
            )

            try:
                oracle_selected, oracle_mlu, oracle_method, _ = _collect_oracle_labels(
                    dataset,
                    path_library,
                    tm_vector,
                    ecmp_base,
                    capacities,
                    K_CRIT,
                    lp_time_limit_sec=LP_TIME_LIMIT,
                )
                soft_teacher, continuous_criticality = _collect_soft_teacher_targets(
                    dataset=dataset,
                    path_library=path_library,
                    tm_vector=tm_vector,
                    ecmp_base=ecmp_base,
                    capacities=capacities,
                    k_crit=K_CRIT,
                    lp_time_limit_sec=LP_TIME_LIMIT,
                    lp_teacher_weight=2.5,
                )
            except Exception:
                prev_tm = tm_vector
                continue

            if not oracle_selected:
                prev_tm = tm_vector
                continue

            samples.append(
                {
                    "topology": topo_key,
                    "dataset": dataset,
                    "path_library": path_library,
                    "tm_vector": tm_vector,
                    "prev_tm": None if prev_tm is None else np.asarray(prev_tm, dtype=float),
                    "telemetry": telemetry,
                    "oracle_selected": list(oracle_selected),
                    "oracle_mlu": float(oracle_mlu),
                    "oracle_method": str(oracle_method),
                    "soft_teacher": np.asarray(soft_teacher, dtype=np.float32),
                    "continuous_criticality": np.asarray(continuous_criticality, dtype=np.float32),
                    "k_crit": int(K_CRIT),
                    "capacities": capacities,
                    "weights": weights,
                    "prev_util": None if prev_util is None else np.asarray(prev_util, dtype=float),
                    "prev_selected_indicator": np.asarray(prev_selected, dtype=np.float32),
                    "prev_disturbance": float(prev_disturbance),
                    "prev_splits": clone_splits(prev_splits),
                }
            )

            teacher_selected = topk_from_soft_target(soft_teacher, K_CRIT) or list(oracle_selected)
            try:
                lp = solve_selected_path_lp(
                    tm_vector=tm_vector,
                    selected_ods=teacher_selected,
                    base_splits=ecmp_base,
                    path_library=path_library,
                    capacities=capacities,
                    time_limit_sec=LP_TIME_LIMIT,
                )
                disturbance = compute_disturbance(prev_splits, lp.splits, tm_vector)
                post_routing = apply_routing(tm_vector, lp.splits, path_library, capacities)
                post_telemetry = compute_reactive_telemetry(
                    tm_vector,
                    lp.splits,
                    path_library,
                    post_routing,
                    weights,
                    prev_latency_by_od=prev_latency_by_od,
                )
                prev_splits = clone_splits(lp.splits)
                prev_latency_by_od = post_telemetry.latency_by_od
                prev_util = np.asarray(post_telemetry.utilization, dtype=float)
                prev_disturbance = float(disturbance)
            except Exception:
                prev_util = np.asarray(telemetry.utilization, dtype=float)
                prev_disturbance = 0.0
            prev_selected = np.asarray(soft_teacher, dtype=np.float32)
            prev_tm = tm_vector

    return samples, datasets


def build_initial_gnnplus_model() -> GNNPlusFlowSelector:
    if BASE_GNNPLUS_CKPT.exists():
        model, _ = load_gnn_plus(BASE_GNNPLUS_CKPT, device=DEVICE)
        model.cfg.feature_variant = "section7_temporal"
        model.cfg.device = DEVICE
        model.cfg.learn_k_crit = False
        model.cfg.k_crit_min = K_CRIT
        model.cfg.k_crit_max = K_CRIT
        model.cfg.dropout = 0.2
        model.eval()
        return model.to(DEVICE)
    cfg = GNNPlusConfig(
        dropout=0.2,
        learn_k_crit=False,
        k_crit_min=K_CRIT,
        k_crit_max=K_CRIT,
        feature_variant="section7_temporal",
        device=DEVICE,
    )
    return GNNPlusFlowSelector(cfg).to(DEVICE)


def run_supervised_training(train_samples: list[dict], val_samples: list[dict]):
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    model = build_initial_gnnplus_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUP_MAX_EPOCHS)
    rng = np.random.default_rng(SEED)

    logs = []
    best_val_loss = float("inf")
    best_epoch = 0
    stale = 0
    start = time.perf_counter()

    print(f"[supervised] train={len(train_samples)} val={len(val_samples)}", flush=True)
    for epoch in range(1, SUP_MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_samples))
        epoch_losses = []

        for idx in order:
            sample = train_samples[int(idx)]
            graph_data, od_data = build_section7_inputs(sample, device=DEVICE)
            scores, _, _ = model(graph_data, od_data)

            num_od = scores.size(0)
            oracle_mask = torch.zeros(num_od, device=DEVICE)
            for od_idx in sample["oracle_selected"]:
                if 0 <= int(od_idx) < num_od:
                    oracle_mask[int(od_idx)] = 1.0

            loss = _ranking_loss(scores, oracle_mask, margin=0.1)
            loss = loss + _soft_teacher_loss(
                scores,
                soft_target=torch.tensor(sample["soft_teacher"], dtype=torch.float32, device=DEVICE),
                criticality=torch.tensor(sample["continuous_criticality"], dtype=torch.float32, device=DEVICE),
                soft_weight=0.35,
                criticality_weight=0.20,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        scheduler.step()
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        model.eval()
        val_losses = []
        val_overlap = []
        with torch.no_grad():
            for sample in val_samples:
                graph_data, od_data = build_section7_inputs(sample, device=DEVICE)
                scores, _, _ = model(graph_data, od_data)
                num_od = scores.size(0)
                oracle_mask = torch.zeros(num_od, device=DEVICE)
                for od_idx in sample["oracle_selected"]:
                    if 0 <= int(od_idx) < num_od:
                        oracle_mask[int(od_idx)] = 1.0
                vloss = _ranking_loss(scores, oracle_mask, margin=0.1)
                vloss = vloss + _soft_teacher_loss(
                    scores,
                    soft_target=torch.tensor(sample["soft_teacher"], dtype=torch.float32, device=DEVICE),
                    criticality=torch.tensor(sample["continuous_criticality"], dtype=torch.float32, device=DEVICE),
                    soft_weight=0.35,
                    criticality_weight=0.20,
                )
                val_losses.append(float(vloss.item()))

                selected, _, _ = continuity_select(
                    scores,
                    active_mask=(sample["tm_vector"] > 1e-12),
                    k=K_CRIT,
                    prev_selected_indicator=sample["prev_selected_indicator"],
                    continuity_bonus=CONTINUITY_BONUS,
                )
                pred_set = set(selected)
                oracle_set = set(sample["oracle_selected"])
                overlap = len(pred_set & oracle_set) / max(len(pred_set | oracle_set), 1)
                val_overlap.append(float(overlap))

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        overlap = float(np.mean(val_overlap)) if val_overlap else 0.0
        logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_selection_overlap": overlap,
                "epoch_time_sec": float(time.perf_counter() - epoch_start),
            }
        )
        print(
            f"[supervised] epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} overlap={overlap:.3f}",
            flush=True,
        )

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale = 0
            save_gnn_plus(
                model,
                SUP_CKPT,
                extra_meta={
                    "stage": "section3_4_7_supervised_finetune",
                    "base_checkpoint": str(BASE_GNNPLUS_CKPT.relative_to(PROJECT_ROOT)),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
            )
        else:
            stale += 1
        if stale >= SUP_PATIENCE:
            print(f"[supervised] early stop at epoch {epoch}", flush=True)
            break

    pd.DataFrame(logs).to_csv(SUP_LOG_CSV, index=False)
    model, _ = load_gnn_plus(SUP_CKPT, device=DEVICE)
    summary = {
        "mode": "supervised",
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "num_train_samples": int(len(train_samples)),
        "num_val_samples": int(len(val_samples)),
        "training_time_sec": float(time.perf_counter() - start),
        "continuity_bonus": float(CONTINUITY_BONUS),
        "feature_variant": "section7_temporal",
        "soft_teacher_weight": 0.35,
        "criticality_weight": 0.20,
    }
    return model, summary


def load_reference_model_cached(cache: dict, runner, sample: dict):
    topo_key = str(sample["topology"])
    if topo_key not in cache:
        cache[topo_key] = runner.load_gnn_model(sample["dataset"], sample["path_library"])
    return cache[topo_key]


def run_rl_finetune(model: GNNPlusFlowSelector, train_samples: list[dict], val_samples: list[dict], runner):
    rl_cfg = GNNReinforceConfig(
        lr=1e-4,
        max_epochs=RL_MAX_EPOCHS,
        patience=RL_PATIENCE,
        baseline_ema=0.9,
        w_reward_mlu=1.0,
        w_reward_improvement=0.8,
        w_reward_disturbance=0.25,
        w_reward_infeasible=2.0,
        w_reward_vs_bottleneck=0.35,
        w_reward_vs_reference=0.20,
        rank_loss_weight=0.25,
        score_margin_weight=0.05,
        infeasible_mlu_penalty=10.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(rl_cfg.lr))
    rng = np.random.default_rng(SEED)
    baseline_reward = None
    best_val_mlu = float("inf")
    best_epoch = 0
    stale = 0
    logs = []
    reference_cache = {}

    rl_train = list(train_samples[: min(len(train_samples), RL_MAX_TRAIN_SAMPLES)])
    rl_val = list(val_samples[: min(len(val_samples), RL_MAX_VAL_SAMPLES)])
    print(f"[reinforce] train={len(rl_train)} val={len(rl_val)}", flush=True)

    for epoch in range(1, rl_cfg.max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = rng.permutation(len(rl_train))
        epoch_rewards = []
        epoch_mlus = []
        epoch_disturbances = []
        epoch_improvements = []
        epoch_vs_bn = []
        epoch_vs_ref = []

        for idx in order:
            sample = rl_train[int(idx)]
            graph_data, od_data = build_section7_inputs(sample, device=DEVICE)
            scores, _, _ = model(graph_data, od_data)

            selected_ods, ranking_scores, selected_log_prob = continuity_select(
                scores,
                active_mask=(sample["tm_vector"] > 1e-12),
                k=K_CRIT,
                prev_selected_indicator=sample["prev_selected_indicator"],
                continuity_bonus=CONTINUITY_BONUS,
            )
            if ranking_scores is None or selected_log_prob is None:
                continue

            ecmp_base = ecmp_splits(sample["path_library"])
            ecmp_routing = apply_routing(sample["tm_vector"], ecmp_base, sample["path_library"], sample["capacities"])
            ecmp_mlu = float(ecmp_routing.mlu)

            feasible = True
            try:
                lp = solve_selected_path_lp(
                    tm_vector=sample["tm_vector"],
                    selected_ods=selected_ods,
                    base_splits=ecmp_base,
                    path_library=sample["path_library"],
                    capacities=sample["capacities"],
                    time_limit_sec=10,
                )
                routing = apply_routing(sample["tm_vector"], lp.splits, sample["path_library"], sample["capacities"])
                mlu = float(routing.mlu)
                if not math.isfinite(mlu):
                    feasible = False
                    mlu = float(rl_cfg.infeasible_mlu_penalty)
            except Exception:
                feasible = False
                mlu = float(rl_cfg.infeasible_mlu_penalty)
                lp = None

            if feasible and lp is not None:
                disturbance = float(compute_disturbance(sample["prev_splits"], lp.splits, sample["tm_vector"]))
            else:
                disturbance = 1.0
            improvement = (ecmp_mlu - mlu) / max(abs(ecmp_mlu), 1e-12)

            try:
                bottleneck_mlu = _bottleneck_baseline_mlu(
                    tm_vector=sample["tm_vector"],
                    ecmp_base=ecmp_base,
                    path_library=sample["path_library"],
                    capacities=sample["capacities"],
                    k_crit=K_CRIT,
                    time_limit_sec=10,
                )
            except Exception:
                bottleneck_mlu = None
            vs_bn = 0.0 if bottleneck_mlu is None else (float(bottleneck_mlu) - mlu) / max(abs(float(bottleneck_mlu)), 1e-12)

            ref_model = load_reference_model_cached(reference_cache, runner, sample)
            try:
                ref_mlu = _reference_model_mlu(
                    reference_model=ref_model,
                    dataset=sample["dataset"],
                    path_library=sample["path_library"],
                    tm_vector=sample["tm_vector"],
                    telemetry=sample["telemetry"],
                    ecmp_base=ecmp_base,
                    capacities=sample["capacities"],
                    k_crit=K_CRIT,
                    device=DEVICE,
                    time_limit_sec=10,
                )
            except Exception:
                ref_mlu = None
            vs_ref = 0.0 if ref_mlu is None else (float(ref_mlu) - mlu) / max(abs(float(ref_mlu)), 1e-12)

            reward = (
                -float(rl_cfg.w_reward_mlu) * float(mlu)
                + float(rl_cfg.w_reward_improvement) * float(improvement)
                - float(rl_cfg.w_reward_disturbance) * float(disturbance)
                - float(rl_cfg.w_reward_infeasible) * (0.0 if feasible else 1.0)
                + float(rl_cfg.w_reward_vs_bottleneck) * float(vs_bn)
                + float(rl_cfg.w_reward_vs_reference) * float(vs_ref)
            )
            baseline_reward = reward if baseline_reward is None else float(rl_cfg.baseline_ema) * baseline_reward + (1.0 - float(rl_cfg.baseline_ema)) * reward
            advantage = float(reward - baseline_reward)

            oracle_mask = torch.zeros(scores.size(0), device=DEVICE)
            for od_idx in sample["oracle_selected"]:
                if 0 <= int(od_idx) < scores.size(0):
                    oracle_mask[int(od_idx)] = 1.0

            loss = -selected_log_prob * torch.tensor(advantage, dtype=torch.float32, device=DEVICE)
            loss = loss + float(rl_cfg.rank_loss_weight) * _ranking_loss(scores, oracle_mask, margin=0.05)
            loss = loss + float(rl_cfg.score_margin_weight) * _score_margin_regularizer(
                scores,
                active_mask=(sample["tm_vector"] > 1e-12),
                k=K_CRIT,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_rewards.append(float(reward))
            epoch_mlus.append(float(mlu))
            epoch_disturbances.append(float(disturbance))
            epoch_improvements.append(float(improvement))
            epoch_vs_bn.append(float(vs_bn))
            epoch_vs_ref.append(float(vs_ref))

        model.eval()
        val_mlus = []
        with torch.no_grad():
            for sample in rl_val:
                graph_data, od_data = build_section7_inputs(sample, device=DEVICE)
                scores, _, _ = model(graph_data, od_data)
                selected_ods, _, _ = continuity_select(
                    scores,
                    active_mask=(sample["tm_vector"] > 1e-12),
                    k=K_CRIT,
                    prev_selected_indicator=sample["prev_selected_indicator"],
                    continuity_bonus=CONTINUITY_BONUS,
                )
                if not selected_ods:
                    continue
                try:
                    lp = solve_selected_path_lp(
                        tm_vector=sample["tm_vector"],
                        selected_ods=selected_ods,
                        base_splits=ecmp_splits(sample["path_library"]),
                        path_library=sample["path_library"],
                        capacities=sample["capacities"],
                        time_limit_sec=10,
                    )
                    routing = apply_routing(sample["tm_vector"], lp.splits, sample["path_library"], sample["capacities"])
                    val_mlus.append(float(routing.mlu))
                except Exception:
                    val_mlus.append(float(rl_cfg.infeasible_mlu_penalty))

        mean_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        mean_mlu = float(np.mean(epoch_mlus)) if epoch_mlus else float("inf")
        mean_val_mlu = float(np.mean(val_mlus)) if val_mlus else float("inf")
        logs.append(
            {
                "epoch": epoch,
                "train_mean_reward": mean_reward,
                "train_mean_mlu": mean_mlu,
                "train_mean_disturbance": float(np.mean(epoch_disturbances)) if epoch_disturbances else 0.0,
                "train_mean_improvement_vs_ecmp": float(np.mean(epoch_improvements)) if epoch_improvements else 0.0,
                "train_mean_vs_bottleneck": float(np.mean(epoch_vs_bn)) if epoch_vs_bn else 0.0,
                "train_mean_vs_reference": float(np.mean(epoch_vs_ref)) if epoch_vs_ref else 0.0,
                "val_mean_mlu": mean_val_mlu,
                "epoch_time_sec": float(time.perf_counter() - epoch_start),
            }
        )
        print(
            f"[reinforce] epoch={epoch:02d} reward={mean_reward:.4f} train_mlu={mean_mlu:.4f} val_mlu={mean_val_mlu:.4f}",
            flush=True,
        )

        if mean_val_mlu + 1e-6 < best_val_mlu:
            best_val_mlu = mean_val_mlu
            best_epoch = epoch
            stale = 0
            model.cfg.feature_variant = "section7_temporal"
            model.cfg.learn_k_crit = False
            model.cfg.k_crit_min = K_CRIT
            model.cfg.k_crit_max = K_CRIT
            save_gnn_plus(
                model,
                FINAL_CKPT,
                extra_meta={
                    "stage": "section3_4_5_7_reinforce_finetune",
                    "base_checkpoint": str(BASE_GNNPLUS_CKPT.relative_to(PROJECT_ROOT)),
                    "supervised_checkpoint": str(SUP_CKPT.relative_to(PROJECT_ROOT)),
                    "best_epoch": best_epoch,
                    "best_val_mlu": best_val_mlu,
                    "rl_config": asdict(rl_cfg),
                    "continuity_bonus": CONTINUITY_BONUS,
                },
            )
        else:
            stale += 1
        if stale >= rl_cfg.patience:
            print(f"[reinforce] early stop at epoch {epoch}", flush=True)
            break

    pd.DataFrame(logs).to_csv(RL_LOG_CSV, index=False)
    model, _ = load_gnn_plus(FINAL_CKPT, device=DEVICE)
    return model, {
        "mode": "reinforce",
        "best_epoch": int(best_epoch),
        "best_val_mlu": float(best_val_mlu),
        "continuity_bonus": float(CONTINUITY_BONUS),
        "rl_config": asdict(rl_cfg),
        "num_train_samples": int(len(rl_train)),
        "num_val_samples": int(len(rl_val)),
    }


def gnnplus_select_stateful(
    model,
    *,
    dataset,
    path_library,
    tm_vector: np.ndarray,
    telemetry,
    prev_tm,
    prev_util,
    prev_selected_indicator,
    prev_disturbance: float,
    k_crit: int,
    failure_mask=None,
):
    graph_data = build_graph_tensors_plus_section7(
        dataset,
        tm_vector=tm_vector,
        path_library=path_library,
        telemetry=telemetry,
        prev_util=prev_util,
        prev_tm=prev_tm,
        prev_selected_indicator=prev_selected_indicator,
        prev_disturbance=prev_disturbance,
        failure_mask=failure_mask,
        device=DEVICE,
    )
    od_data = build_od_features_plus_section7(
        dataset,
        tm_vector,
        path_library,
        telemetry=telemetry,
        prev_tm=prev_tm,
        prev_util=prev_util,
        prev_selected_indicator=prev_selected_indicator,
        prev_disturbance=prev_disturbance,
        device=DEVICE,
    )
    active_mask = ((np.asarray(tm_vector, dtype=np.float64) > 1e-12) & surviving_od_mask(path_library)).astype(np.float32)
    selected, info = model.select_critical_flows(
        graph_data=graph_data,
        od_data=od_data,
        active_mask=active_mask,
        k_crit_default=k_crit,
        force_default_k=True,
        prev_selected_indicator=prev_selected_indicator,
        continuity_bonus=CONTINUITY_BONUS,
    )
    assert_selected_ods_have_paths(path_library, selected, context=f"{dataset.key}:gnnplus_improved")
    return selected, info


def run_sdn_cycle_gnnplus_improved(
    runner,
    *,
    tm_vector,
    dataset,
    path_library,
    ecmp_base,
    current_splits,
    current_groups,
    topo_mapping,
    capacities,
    weights,
    gnnplus_model,
    prev_latency_by_od,
    gnnplus_state,
):
    t_total_start = time.perf_counter()
    routing_pre = apply_routing(tm_vector, current_splits, path_library, capacities)
    pre_mlu = float(routing_pre.mlu)
    pre_telemetry = runner.compute_telemetry(
        tm_vector=tm_vector,
        splits=current_splits,
        path_library=path_library,
        routing=routing_pre,
        weights=weights,
        prev_latency_by_od=prev_latency_by_od,
    )

    selected_ods, select_info = gnnplus_select_stateful(
        gnnplus_model,
        dataset=dataset,
        path_library=path_library,
        tm_vector=tm_vector,
        telemetry=pre_telemetry,
        prev_tm=gnnplus_state["prev_tm"],
        prev_util=gnnplus_state["prev_util"],
        prev_selected_indicator=gnnplus_state["prev_selected_indicator"],
        prev_disturbance=gnnplus_state["prev_disturbance"],
        k_crit=K_CRIT,
    )
    lp_result = runner.solve_selected_path_lp_safe(
        tm_vector=tm_vector,
        selected_ods=selected_ods,
        base_splits=ecmp_base,
        path_library=path_library,
        capacities=capacities,
        time_limit_sec=LP_TIME_LIMIT,
        context=f"{dataset.key}:gnnplus_improved:normal_cycle",
    )
    new_splits = [s.copy() for s in lp_result.splits]
    t_decision_end = time.perf_counter()
    decision_time_ms = (t_decision_end - t_total_start) * 1000.0

    t_rule_start = time.perf_counter()
    new_groups, _ = runner.splits_to_openflow_rules(new_splits, selected_ods, path_library, topo_mapping, dataset.edges)
    changed_groups = runner.compute_rule_diff(current_groups, new_groups)
    flow_table_updates = len(changed_groups)
    t_rule_end = time.perf_counter()
    rule_install_delay_ms = (t_rule_end - t_rule_start) * 1000.0

    routing_post = apply_routing(tm_vector, new_splits, path_library, capacities)
    post_mlu = float(routing_post.mlu)
    dist = compute_disturbance(current_splits, new_splits, tm_vector)
    telemetry_post = runner.compute_telemetry(
        tm_vector=tm_vector,
        splits=new_splits,
        path_library=path_library,
        routing=routing_post,
        weights=weights,
        prev_latency_by_od=prev_latency_by_od,
    )

    result = runner.SDNCycleResult(
        cycle=0,
        method="gnnplus",
        topology=dataset.key,
        pre_mlu=pre_mlu,
        post_mlu=post_mlu,
        disturbance=float(dist),
        throughput=telemetry_post.throughput,
        mean_latency=telemetry_post.mean_latency,
        p95_latency=telemetry_post.p95_latency,
        packet_loss=telemetry_post.packet_loss,
        jitter=telemetry_post.jitter,
        decision_time_ms=decision_time_ms,
        flow_table_updates=flow_table_updates,
        rule_install_delay_ms=rule_install_delay_ms,
    )
    next_state = {
        "prev_tm": np.asarray(tm_vector, dtype=float),
        "prev_util": np.asarray(telemetry_post.utilization, dtype=float),
        "prev_selected_indicator": selection_indicator(len(dataset.od_pairs), selected_ods),
        "prev_disturbance": float(dist),
        "select_info": dict(select_info),
    }
    return result, new_splits, new_groups, telemetry_post.latency_by_od, next_state


def run_failure_scenario_gnnplus_improved(
    runner,
    *,
    scenario,
    tm_vector,
    dataset,
    path_library,
    ecmp_base,
    capacities,
    weights,
    gnnplus_model,
):
    normal_routing = apply_routing(tm_vector, ecmp_base, path_library, capacities)
    pre_failure_mlu = float(normal_routing.mlu)
    failure_state = runner._build_failure_execution_state(
        scenario=scenario,
        tm_vector=tm_vector,
        dataset=dataset,
        path_library=path_library,
        capacities=capacities,
        weights=weights,
        normal_routing=normal_routing,
    )
    effective_tm = failure_state["effective_tm"]
    effective_caps = failure_state["effective_caps"]
    effective_path_library = failure_state["effective_path_library"]
    effective_dataset = failure_state["effective_dataset"]
    effective_ecmp = failure_state["effective_ecmp"]
    failure_mask = failure_state["failure_mask"]
    effective_weights = failure_state["effective_weights"]

    post_failure_routing = apply_routing(effective_tm, effective_ecmp, effective_path_library, effective_caps)
    pre_telemetry = runner.compute_telemetry(
        tm_vector=effective_tm,
        splits=effective_ecmp,
        path_library=effective_path_library,
        routing=post_failure_routing,
        weights=effective_weights,
        prev_latency_by_od=None,
    )

    t_start = time.perf_counter()
    selected, _ = gnnplus_select_stateful(
        gnnplus_model,
        dataset=effective_dataset,
        path_library=effective_path_library,
        tm_vector=effective_tm,
        telemetry=pre_telemetry,
        prev_tm=None,
        prev_util=None,
        prev_selected_indicator=np.zeros(len(effective_dataset.od_pairs), dtype=np.float32),
        prev_disturbance=0.0,
        k_crit=K_CRIT,
        failure_mask=failure_mask,
    )
    lp_result = runner.solve_selected_path_lp_safe(
        tm_vector=effective_tm,
        selected_ods=selected,
        base_splits=effective_ecmp,
        path_library=effective_path_library,
        capacities=effective_caps,
        time_limit_sec=LP_TIME_LIMIT,
        context=f"{dataset.key}:{scenario}:gnnplus_improved",
    )
    recovery_splits = [s.copy() for s in lp_result.splits]
    recovery_ms = (time.perf_counter() - t_start) * 1000.0
    post_routing = apply_routing(effective_tm, recovery_splits, effective_path_library, effective_caps)
    post_recovery_mlu = float(post_routing.mlu)
    return recovery_ms, pre_failure_mlu, post_recovery_mlu, failure_mask


def benchmark_topology_normal_improved(runner, topo_key: str, gnn_cache: dict, gnnplus_model) -> list[dict]:
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

    rows = []
    for method in CORE_METHODS:
        run_results = defaultdict(list)
        for _ in range(NUM_RUNS):
            current_splits = [s.copy() for s in ecmp_base]
            current_groups, _ = runner.build_ecmp_baseline_rules(path_library, topo_mapping, dataset.edges)
            prev_latency = None
            gnnplus_state = {
                "prev_tm": None,
                "prev_util": None,
                "prev_selected_indicator": np.zeros(len(dataset.od_pairs), dtype=np.float32),
                "prev_disturbance": 0.0,
            }
            for t_idx in test_indices:
                tm_vec = dataset.tm[t_idx]
                if method == "gnnplus":
                    result, current_splits, current_groups, prev_latency, gnnplus_state = run_sdn_cycle_gnnplus_improved(
                        runner,
                        tm_vector=tm_vec,
                        dataset=dataset,
                        path_library=path_library,
                        ecmp_base=ecmp_base,
                        current_splits=current_splits,
                        current_groups=current_groups,
                        topo_mapping=topo_mapping,
                        capacities=capacities,
                        weights=weights,
                        gnnplus_model=gnnplus_model,
                        prev_latency_by_od=prev_latency,
                        gnnplus_state=gnnplus_state,
                    )
                else:
                    result, current_splits, current_groups, prev_latency = runner.run_sdn_cycle(
                        tm_vector=tm_vec,
                        method=method,
                        dataset=dataset,
                        path_library=path_library,
                        ecmp_base=ecmp_base,
                        current_splits=current_splits,
                        current_groups=current_groups,
                        topo_mapping=topo_mapping,
                        capacities=capacities,
                        weights=weights,
                        gnn_model=gnn_model,
                        gnnplus_model=None,
                        prev_latency_by_od=prev_latency,
                    )
                run_results["post_mlus"].append(result.post_mlu)
                run_results["disturbances"].append(result.disturbance)
                run_results["throughputs"].append(result.throughput)
                run_results["latencies"].append(result.mean_latency)
                run_results["p95_latencies"].append(result.p95_latency)
                run_results["packet_losses"].append(result.packet_loss)
                run_results["jitters"].append(result.jitter)
                run_results["decision_times"].append(result.decision_time_ms)
                run_results["flow_updates"].append(result.flow_table_updates)
                run_results["rule_delays"].append(result.rule_install_delay_ms)

        row = {
            "topology": topo_key,
            "status": "known" if topo_key in KNOWN_TOPOLOGIES else "unseen",
            "method": method,
            "scenario": "normal",
            "nodes": len(dataset.nodes),
            "edges": len(dataset.edges),
            "mean_mlu": float(np.mean(run_results["post_mlus"])),
            "mean_disturbance": float(np.mean(run_results["disturbances"])),
            "throughput": float(np.mean(run_results["throughputs"])),
            "mean_latency_au": float(np.mean(run_results["latencies"])),
            "p95_latency_au": float(np.mean(run_results["p95_latencies"])),
            "packet_loss": float(np.mean(run_results["packet_losses"])),
            "jitter_au": float(np.mean(run_results["jitters"])),
            "decision_time_ms": float(np.mean(run_results["decision_times"])),
            "flow_table_updates": float(np.mean(run_results["flow_updates"])),
            "rule_install_delay_ms": float(np.mean(run_results["rule_delays"])),
        }
        rows.append(row)
        print(f"[eval:normal] {topo_key} {method} mean_mlu={row['mean_mlu']:.4f}", flush=True)
    return rows


def benchmark_topology_failures_improved(runner, topo_key: str, gnn_cache: dict, gnnplus_model) -> list[dict]:
    dataset, path_library = runner.load_dataset(topo_key)
    capacities = np.asarray(dataset.capacities, dtype=float)
    weights = np.asarray(dataset.weights, dtype=float)
    ecmp_base = ecmp_splits(path_library)
    gnn_model = gnn_cache.get(topo_key)
    if gnn_model is None:
        gnn_model = runner.load_gnn_model(dataset, path_library)
        gnn_cache[topo_key] = gnn_model
    test_indices = list(range(int(dataset.split["test_start"]), dataset.tm.shape[0]))

    rows = []
    for scenario in runner.FAILURE_SCENARIOS:
        if scenario == "normal":
            continue
        sample_indices = test_indices[:: max(1, len(test_indices) // 10)]
        for method in CORE_METHODS:
            run_results = defaultdict(list)
            for t_idx in sample_indices:
                tm_vec = dataset.tm[t_idx]
                if method == "gnnplus":
                    recovery_ms, pre_mlu, post_mlu, _ = run_failure_scenario_gnnplus_improved(
                        runner,
                        scenario=scenario,
                        tm_vector=tm_vec,
                        dataset=dataset,
                        path_library=path_library,
                        ecmp_base=ecmp_base,
                        capacities=capacities,
                        weights=weights,
                        gnnplus_model=gnnplus_model,
                    )
                else:
                    recovery_ms, pre_mlu, post_mlu, _ = runner.run_failure_scenario(
                        scenario=scenario,
                        tm_vector=tm_vec,
                        method=method,
                        dataset=dataset,
                        path_library=path_library,
                        ecmp_base=ecmp_base,
                        capacities=capacities,
                        weights=weights,
                        topo_mapping=None,
                        gnn_model=gnn_model,
                        gnnplus_model=None,
                    )
                run_results["recovery_times"].append(recovery_ms)
                run_results["pre_mlus"].append(pre_mlu)
                run_results["post_mlus"].append(post_mlu)

            row = {
                "topology": topo_key,
                "status": "known" if topo_key in KNOWN_TOPOLOGIES else "unseen",
                "method": method,
                "scenario": scenario,
                "nodes": len(dataset.nodes),
                "edges": len(dataset.edges),
                "mean_mlu": float(np.mean(run_results["post_mlus"])),
                "pre_failure_mlu": float(np.mean(run_results["pre_mlus"])),
                "failure_recovery_ms": float(np.mean(run_results["recovery_times"])),
            }
            rows.append(row)
            print(f"[eval:failure] {topo_key} {scenario} {method} mean_mlu={row['mean_mlu']:.4f}", flush=True)
    return rows


def prepare_sdn_metrics(summary_df: pd.DataFrame, failure_df: pd.DataFrame) -> pd.DataFrame:
    recovery = (
        failure_df.groupby(["topology", "method"], as_index=False)["failure_recovery_ms"]
        .mean()
        .rename(columns={"failure_recovery_ms": "avg_failure_recovery_ms"})
    )
    metrics = summary_df.merge(recovery, on=["topology", "method"], how="left")
    metrics["Method"] = metrics["method"].map(METHOD_LABELS)
    metrics["Topology"] = metrics["topology"].map(TOPOLOGY_DISPLAY)
    metrics["Status"] = metrics["status"].str.title()
    metrics = metrics.rename(
        columns={
            "mean_mlu": "Mean MLU",
            "throughput": "Throughput",
            "mean_disturbance": "Disturbance",
            "mean_latency_au": "Mean Delay",
            "p95_latency_au": "P95 Delay",
            "packet_loss": "Packet Loss",
            "jitter_au": "Jitter",
            "decision_time_ms": "Decision Time (ms)",
            "flow_table_updates": "Flow Table Updates",
            "rule_install_delay_ms": "Rule Install Delay (ms)",
            "avg_failure_recovery_ms": "Failure Recovery (ms)",
        }
    )
    cols = [
        "Method",
        "Topology",
        "Status",
        "Mean MLU",
        "Throughput",
        "Disturbance",
        "Mean Delay",
        "P95 Delay",
        "Packet Loss",
        "Jitter",
        "Decision Time (ms)",
        "Flow Table Updates",
        "Rule Install Delay (ms)",
        "Failure Recovery (ms)",
    ]
    return metrics[cols]


def build_split_manifest() -> dict:
    return {
        "base_objective": "zero-shot generalization",
        "methods": CORE_METHODS,
        "known_topologies": KNOWN_TOPOLOGIES,
        "unseen_topologies": UNSEEN_TOPOLOGIES,
        "train_topologies": KNOWN_TOPOLOGIES,
        "validation_topologies": KNOWN_TOPOLOGIES,
        "evaluation_topologies": ALL_TOPOLOGIES,
        "zero_shot_guards": {
            "bayesian_calibration_used": False,
            "few_shot_adaptation_used": False,
            "per_topology_tuning_used": False,
            "metagate_used": False,
            "stable_metagate_used": False,
        },
        "improvements_enabled": {
            "section3_physical_features": True,
            "section4_soft_teacher_targets": True,
            "section5_rl_reward_alignment": True,
            "section7_disturbance_aware_temporal_path": True,
        },
        "fixed_k": K_CRIT,
        "continuity_bonus": CONTINUITY_BONUS,
        "base_checkpoint": str(BASE_GNNPLUS_CKPT.relative_to(PROJECT_ROOT)),
    }


def build_comparison_tables(summary_df: pd.DataFrame, failure_df: pd.DataFrame) -> dict[str, Path]:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    old_summary = pd.read_csv(BASELINE_OUTPUT_DIR / "packet_sdn_summary.csv")
    old_failure = pd.read_csv(BASELINE_OUTPUT_DIR / "packet_sdn_failure.csv")

    old_new = old_summary[old_summary["method"] == "gnnplus"].merge(
        summary_df[summary_df["method"] == "gnnplus"],
        on=["topology", "status", "method", "scenario", "nodes", "edges"],
        suffixes=("_old", "_new"),
    )
    for metric in ["mean_mlu", "throughput", "mean_disturbance", "decision_time_ms"]:
        old_new[f"{metric}_delta"] = old_new[f"{metric}_new"] - old_new[f"{metric}_old"]
        old_new[f"{metric}_pct_delta"] = np.where(
            np.abs(old_new[f"{metric}_old"]) > 1e-12,
            (old_new[f"{metric}_new"] - old_new[f"{metric}_old"]) / np.abs(old_new[f"{metric}_old"]) * 100.0,
            0.0,
        )
    path1 = COMPARISON_DIR / "gnnplus_normal_old_vs_new.csv"
    old_new.to_csv(path1, index=False)

    old_new_failure = old_failure[old_failure["method"] == "gnnplus"].merge(
        failure_df[failure_df["method"] == "gnnplus"],
        on=["topology", "status", "method", "scenario", "nodes", "edges"],
        suffixes=("_old", "_new"),
    )
    for metric in ["mean_mlu", "failure_recovery_ms"]:
        old_new_failure[f"{metric}_delta"] = old_new_failure[f"{metric}_new"] - old_new_failure[f"{metric}_old"]
        old_new_failure[f"{metric}_pct_delta"] = np.where(
            np.abs(old_new_failure[f"{metric}_old"]) > 1e-12,
            (old_new_failure[f"{metric}_new"] - old_new_failure[f"{metric}_old"]) / np.abs(old_new_failure[f"{metric}_old"]) * 100.0,
            0.0,
        )
    path2 = COMPARISON_DIR / "gnnplus_failure_old_vs_new.csv"
    old_new_failure.to_csv(path2, index=False)

    ranking_rows = []
    for topo in ALL_TOPOLOGIES:
        topo_df = summary_df[summary_df["topology"] == topo].sort_values("mean_mlu")
        for rank, (_, row) in enumerate(topo_df.iterrows(), start=1):
            ranking_rows.append(
                {
                    "topology": topo,
                    "method": str(row["method"]),
                    "rank_by_mlu": int(rank),
                    "mean_mlu": float(row["mean_mlu"]),
                    "throughput": float(row["throughput"]),
                    "mean_disturbance": float(row["mean_disturbance"]),
                    "decision_time_ms": float(row["decision_time_ms"]),
                }
            )
    path3 = COMPARISON_DIR / "new_run_method_ranking.csv"
    pd.DataFrame(ranking_rows).to_csv(path3, index=False)

    overall = pd.DataFrame(
        [
            {
                "bundle": "baseline_clean_zeroshot",
                "gnnplus_mean_mlu": float(old_summary[old_summary["method"] == "gnnplus"]["mean_mlu"].mean()),
                "gnnplus_mean_throughput": float(old_summary[old_summary["method"] == "gnnplus"]["throughput"].mean()),
                "gnnplus_mean_disturbance": float(old_summary[old_summary["method"] == "gnnplus"]["mean_disturbance"].mean()),
                "gnnplus_mean_decision_time_ms": float(old_summary[old_summary["method"] == "gnnplus"]["decision_time_ms"].mean()),
            },
            {
                "bundle": "improved_fixedk40_experiment",
                "gnnplus_mean_mlu": float(summary_df[summary_df["method"] == "gnnplus"]["mean_mlu"].mean()),
                "gnnplus_mean_throughput": float(summary_df[summary_df["method"] == "gnnplus"]["throughput"].mean()),
                "gnnplus_mean_disturbance": float(summary_df[summary_df["method"] == "gnnplus"]["mean_disturbance"].mean()),
                "gnnplus_mean_decision_time_ms": float(summary_df[summary_df["method"] == "gnnplus"]["decision_time_ms"].mean()),
            },
        ]
    )
    path4 = COMPARISON_DIR / "overall_bundle_comparison.csv"
    overall.to_csv(path4, index=False)
    return {
        "normal_old_vs_new": path1,
        "failure_old_vs_new": path2,
        "method_ranking": path3,
        "overall": path4,
    }


def add_title_page(doc: Document) -> None:
    for _ in range(4):
        doc.add_paragraph()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Improved GNN+ Fixed-K40 Zero-Shot Report")
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(0x1A, 0x47, 0x80)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run(
        "Scope: ECMP, Bottleneck, Original GNN, GNN+\n"
        "Enhancements: Sections 3, 4, 5, and 7\n"
        "No MetaGate, No Stable MetaGate, No Bayesian Calibration"
    )
    run.font.size = Pt(11)

    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = note.add_run(
        "Important: packet-level SDN metrics are model-based analytical metrics, not live Mininet measurements."
    )
    run.bold = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
    doc.add_page_break()


def add_bullet(doc: Document, text: str) -> None:
    doc.add_paragraph(text, style="List Bullet")


def add_image(doc: Document, path: Path, caption: str, width: float = 6.2) -> None:
    if not path.exists():
        doc.add_paragraph(f"[Image missing: {path.name}]")
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def build_report(summary_df: pd.DataFrame, failure_df: pd.DataFrame, metrics_df: pd.DataFrame, training_summary: dict, split_manifest: dict):
    helper = load_helper()
    helper.PLOTS_DIR = PLOTS_DIR
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    helper.create_plots(summary_df, failure_df)

    normal_compare = pd.read_csv(COMPARISON_DIR / "gnnplus_normal_old_vs_new.csv")
    failure_compare = pd.read_csv(COMPARISON_DIR / "gnnplus_failure_old_vs_new.csv")
    overall_compare = pd.read_csv(COMPARISON_DIR / "overall_bundle_comparison.csv")

    doc = Document()
    helper.set_default_style(doc)
    add_title_page(doc)

    doc.add_heading("1. Scope", level=1)
    doc.add_paragraph(
        "This experiment converts the section 3/4/5/7 code changes into a new GNN+ checkpoint and a new clean zero-shot evaluation bundle. "
        "The method scope remains ECMP, Bottleneck, Original GNN, and GNN+."
    )
    add_bullet(doc, "Known training topologies remain the original six known topologies.")
    add_bullet(doc, "Germany50 and VtlWavenet2011 remain unseen zero-shot evaluation topologies.")
    add_bullet(doc, "No MetaGate, Stable MetaGate, Bayesian calibration, or per-topology adaptation is used.")

    doc.add_heading("2. Improvements Enabled", level=1)
    improvements = pd.DataFrame(
        [
            {"Section": "3", "Change": "Physical/stress-aware features", "Enabled": "Yes"},
            {"Section": "4", "Change": "Soft teacher targets + continuous criticality", "Enabled": "Yes"},
            {"Section": "5", "Change": "RL fine-tuning with MLU/improvement/disturbance reward", "Enabled": "Yes"},
            {"Section": "7", "Change": "Temporal disturbance-aware features + continuity bonus", "Enabled": "Yes"},
            {"Section": "6", "Change": f"Fixed K = {K_CRIT} main thesis branch", "Enabled": "Yes"},
        ]
    )
    helper.add_dataframe_table(doc, improvements, font_size=9)

    doc.add_heading("3. Training Summary", level=1)
    train_rows = pd.DataFrame(
        [
            {"Field": "Base checkpoint", "Value": training_summary["base_checkpoint"]},
            {"Field": "Final checkpoint", "Value": training_summary["final_checkpoint"]},
            {"Field": "Feature variant", "Value": training_summary["feature_variant"]},
            {"Field": "Continuity bonus", "Value": training_summary["continuity_bonus"]},
            {"Field": "Supervised train samples", "Value": training_summary["supervised"]["num_train_samples"]},
            {"Field": "Supervised val samples", "Value": training_summary["supervised"]["num_val_samples"]},
            {"Field": "RL train samples", "Value": training_summary["reinforce"]["num_train_samples"]},
            {"Field": "RL val samples", "Value": training_summary["reinforce"]["num_val_samples"]},
            {"Field": "Supervised best epoch", "Value": training_summary["supervised"]["best_epoch"]},
            {"Field": "RL best epoch", "Value": training_summary["reinforce"]["best_epoch"]},
        ]
    )
    helper.add_dataframe_table(doc, train_rows, font_size=9)

    doc.add_heading("4. Zero-Shot Protocol", level=1)
    protocol_rows = pd.DataFrame(
        [
            {"Item": "Train topologies", "Value": ", ".join(split_manifest["train_topologies"])},
            {"Item": "Unseen topologies", "Value": ", ".join(split_manifest["unseen_topologies"])},
            {"Item": "Fixed K", "Value": split_manifest["fixed_k"]},
            {"Item": "Bayesian calibration", "Value": "No"},
            {"Item": "Few-shot adaptation", "Value": "No"},
            {"Item": "MetaGate / Stable MetaGate", "Value": "No"},
        ]
    )
    helper.add_dataframe_table(doc, protocol_rows, font_size=9)

    doc.add_heading("5. Normal Results", level=1)
    add_image(doc, PLOTS_DIR / "mlu_comparison_normal.png", "Figure 1. Mean MLU comparison across the 8 topologies.")
    add_image(doc, PLOTS_DIR / "throughput_comparison_normal.png", "Figure 2. Throughput comparison across the 8 topologies.")
    add_image(doc, PLOTS_DIR / "disturbance_comparison.png", "Figure 3. Disturbance comparison across the 8 topologies.")
    add_image(doc, PLOTS_DIR / "decision_time_comparison.png", "Figure 4. Decision time comparison across the 8 topologies.")
    helper.add_dataframe_table(doc, metrics_df, font_size=8)

    doc.add_heading("6. Failure Results", level=1)
    add_image(doc, PLOTS_DIR / "failure_recovery_gnnplus.png", "Figure 5. GNN+ failure recovery time by scenario.")
    failure_table = failure_df[failure_df["method"] == "gnnplus"].copy()
    failure_table["Topology"] = failure_table["topology"].map(TOPOLOGY_DISPLAY)
    failure_table["Scenario"] = failure_table["scenario"]
    failure_table = failure_table[
        ["Topology", "Scenario", "mean_mlu", "pre_failure_mlu", "failure_recovery_ms"]
    ].rename(
        columns={
            "mean_mlu": "Post-Recovery MLU",
            "pre_failure_mlu": "Pre-Failure MLU",
            "failure_recovery_ms": "Recovery Time (ms)",
        }
    )
    helper.add_dataframe_table(doc, failure_table, font_size=8)

    doc.add_heading("7. Old vs Improved Comparison", level=1)
    doc.add_paragraph(
        "The tables below compare the previous clean zero-shot GNN+ bundle against this improved fixed-K40 run."
    )
    helper.add_dataframe_table(doc, overall_compare, font_size=9)
    helper.add_dataframe_table(
        doc,
        normal_compare[
            [
                "topology",
                "mean_mlu_old",
                "mean_mlu_new",
                "mean_mlu_pct_delta",
                "mean_disturbance_old",
                "mean_disturbance_new",
                "mean_disturbance_pct_delta",
                "decision_time_ms_old",
                "decision_time_ms_new",
                "decision_time_ms_pct_delta",
            ]
        ],
        font_size=8,
    )
    helper.add_dataframe_table(
        doc,
        failure_compare[
            [
                "topology",
                "scenario",
                "mean_mlu_old",
                "mean_mlu_new",
                "mean_mlu_pct_delta",
                "failure_recovery_ms_old",
                "failure_recovery_ms_new",
                "failure_recovery_ms_pct_delta",
            ]
        ],
        font_size=8,
    )

    doc.add_heading("8. Conclusions", level=1)
    doc.add_paragraph(
        "This bundle is the first one in this branch where the section 3/4/5/7 improvements are not only implemented in code but also carried through into a new checkpoint and a new zero-shot evaluation. "
        "It should be interpreted as the improved standalone GNN+ study, still separate from the MetaGate study."
    )
    add_bullet(doc, "The branch remains honest about scope: 4 methods only.")
    add_bullet(doc, "The zero-shot claim remains protected because unseen topologies are not adapted before inference.")
    add_bullet(doc, "The new checkpoint should be compared against the earlier clean zero-shot bundle, not mixed with MetaGate reports.")

    REPORT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(REPORT_DOCX))


def write_audit(training_summary: dict, summary_df: pd.DataFrame, failure_df: pd.DataFrame, comparison_paths: dict[str, Path]) -> None:
    lines = [
        "# Improved GNN+ Experiment Audit",
        "",
        f"- Final checkpoint: `{training_summary['final_checkpoint']}`",
        f"- Supervised log: `{SUP_LOG_CSV.relative_to(PROJECT_ROOT)}`",
        f"- RL log: `{RL_LOG_CSV.relative_to(PROJECT_ROOT)}`",
        f"- Normal rows: {len(summary_df)}",
        f"- Failure rows: {len(failure_df)}",
        f"- Methods in summary: {sorted(summary_df['method'].astype(str).unique().tolist())}",
        f"- Topologies in summary: {sorted(summary_df['topology'].astype(str).unique().tolist())}",
        f"- Comparison tables:",
    ]
    for key, path in comparison_paths.items():
        lines.append(f"  - {key}: `{path.relative_to(PROJECT_ROOT)}`")
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    seed_all(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    runner = load_runner()
    split_manifest = build_split_manifest()
    SPLIT_MANIFEST_JSON.write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")

    train_samples, _ = collect_split_samples(
        runner,
        KNOWN_TOPOLOGIES,
        split_name="train",
        max_per_topology=TRAIN_MAX_PER_TOPO,
        seed=SEED,
    )
    val_samples, _ = collect_split_samples(
        runner,
        KNOWN_TOPOLOGIES,
        split_name="val",
        max_per_topology=VAL_MAX_PER_TOPO,
        seed=SEED + 1,
    )

    supervised_model, supervised_summary = run_supervised_training(train_samples, val_samples)
    improved_model, reinforce_summary = run_rl_finetune(supervised_model, train_samples, val_samples, runner)

    gnn_cache = {}
    normal_rows = []
    for topo in ALL_TOPOLOGIES:
        normal_rows.extend(benchmark_topology_normal_improved(runner, topo, gnn_cache, improved_model))
    summary_df = pd.DataFrame(normal_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    failure_rows = []
    for topo in ALL_TOPOLOGIES:
        failure_rows.extend(benchmark_topology_failures_improved(runner, topo, gnn_cache, improved_model))
    failure_df = pd.DataFrame(failure_rows)
    failure_df.to_csv(FAILURE_CSV, index=False)

    metrics_df = prepare_sdn_metrics(summary_df, failure_df)
    metrics_df.to_csv(SDN_METRICS_CSV, index=False)

    training_summary = {
        "base_checkpoint": str(BASE_GNNPLUS_CKPT.relative_to(PROJECT_ROOT)),
        "final_checkpoint": str(FINAL_CKPT.relative_to(PROJECT_ROOT)),
        "feature_variant": "section7_temporal",
        "continuity_bonus": CONTINUITY_BONUS,
        "supervised": supervised_summary,
        "reinforce": reinforce_summary,
    }
    TRAINING_SUMMARY_JSON.write_text(json.dumps(training_summary, indent=2) + "\n", encoding="utf-8")

    comparison_paths = build_comparison_tables(summary_df, failure_df)
    build_report(summary_df, failure_df, metrics_df, training_summary, split_manifest)
    write_audit(training_summary, summary_df, failure_df, comparison_paths)

    print(f"[done] results: {OUTPUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
