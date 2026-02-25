#!/usr/bin/env python3
"""Run Phase-2 proactive TE (traffic prediction + proactive routing)."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from eval.plots import generate_plots_for_dataset
from phase2.predictors import BaseTMPredictor, PredictionMetrics, build_predictor, compute_prediction_metrics
from rl.policy import ODSelectorPolicy, build_od_features, deterministic_topk
from te.baselines import (
    clone_splits,
    ecmp_splits,
    ospf_splits,
    project_edge_flows_to_k_path_splits,
    select_bottleneck_critical,
    select_topk_by_demand,
)
from te.disturbance import compute_disturbance
from te.lp_solver import solve_full_mcf_min_mlu, solve_selected_path_lp
from te.scaling import apply_scale, compute_auto_scale_factor
from te.simulator import RoutingResult, apply_routing, build_paths, load_dataset

PHASE2_METHOD_DESCRIPTIONS = {
    "ospf": "Static OSPF baseline (decision does not use predictions)",
    "ecmp": "Static ECMP baseline (decision does not use predictions)",
    "topk_pred": "Proactive Top-K: select critical ODs by predicted demand, then LP",
    "bottleneck_pred": "Proactive bottleneck heuristic: select ODs by predicted bottleneck impact, then LP",
    "lp_optimal_pred": "Proactive full-MCF oracle on predicted TM (reference upper bound)",
    "rl_lp_pred": "Proactive RL selector on predicted TM + LP path-split optimization",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-2 proactive TE methods")
    parser.add_argument("--config", action="append", required=True, help="YAML config path (repeatable)")
    parser.add_argument("--output_dir", default="results/phase2", help="Output directory for CSV/plots/report")
    parser.add_argument(
        "--methods",
        default="ospf,ecmp,topk_pred,bottleneck_pred",
        help="Comma-separated method list",
    )
    parser.add_argument("--max_steps", type=int, default=None, help="Override max timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--k_paths", type=int, default=None, help="Override K-shortest paths")
    parser.add_argument("--k_crit", type=int, default=None, help="Override number of critical ODs")
    parser.add_argument("--lp_time_limit_sec", type=int, default=None, help="Override LP time limit")
    parser.add_argument(
        "--full_mcf_time_limit_sec",
        type=int,
        default=None,
        help="Override full-MCF LP time limit",
    )
    parser.add_argument(
        "--rl_checkpoint",
        default=None,
        help="Path to RL checkpoint for rl_lp_pred method",
    )
    parser.add_argument(
        "--predictor",
        default=None,
        help="Predictor name: naive_last | moving_avg | ar_ridge",
    )
    parser.add_argument("--predictor_window", type=int, default=None, help="Predictor lag window")
    parser.add_argument("--predictor_alpha", type=float, default=None, help="Ridge alpha for ar_ridge")
    parser.add_argument("--disable_auto_scale", action="store_true", help="Disable config scaling.auto_target")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_methods(methods_csv: str) -> List[str]:
    allowed = {"ospf", "ecmp", "topk_pred", "bottleneck_pred", "lp_optimal_pred", "rl_lp_pred"}
    methods = [item.strip() for item in methods_csv.split(",") if item.strip()]
    invalid = [item for item in methods if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported method(s): {invalid}. Allowed: {sorted(allowed)}")
    if not methods:
        raise ValueError("No methods specified")
    return methods


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_rl_policy(checkpoint_path: Path, device: torch.device) -> ODSelectorPolicy:
    payload = torch.load(checkpoint_path, map_location=device)
    input_dim = int(payload.get("input_dim", 3))
    hidden_dim = int(payload.get("hidden_dim", 64))
    policy = ODSelectorPolicy(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy


def _table_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def write_phase2_report(
    summary_df: pd.DataFrame,
    split_info: Dict[str, Dict[str, int]],
    scale_info: Dict[str, Dict[str, float]],
    predictor_info: Dict[str, Dict[str, object]],
    output_path: Path,
    run_meta: Dict[str, object],
) -> None:
    lines: list[str] = []
    lines.append("# Phase-2 Proactive TE Report")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- seed: `{run_meta.get('seed')}`")
    lines.append(f"- generated_at_utc: `{run_meta.get('generated_at_utc')}`")
    lines.append("")

    lines.append("## Methods")
    lines.append("")
    for method in sorted(set(summary_df["method"].tolist())):
        desc = PHASE2_METHOD_DESCRIPTIONS.get(method, "")
        lines.append(f"- `{method}`: {desc}")
    lines.append("")

    for dataset_key in sorted(summary_df["dataset"].unique()):
        ds_summary = summary_df[summary_df["dataset"] == dataset_key].copy()
        split = split_info.get(dataset_key, {})
        scale = scale_info.get(dataset_key, {})
        pred = predictor_info.get(dataset_key, {})

        lines.append(f"## Dataset: {dataset_key}")
        lines.append("")
        lines.append(
            "- split train/val/test: "
            f"{split.get('num_train', '?')}/{split.get('num_val', '?')}/{split.get('num_test', '?')}"
        )
        lines.append(f"- predictor: `{pred.get('name', '?')}`")
        lines.append(f"- predictor_window: `{pred.get('window', '?')}`")
        lines.append(f"- predictor_alpha: `{pred.get('alpha', '?')}`")
        lines.append(f"- scale_factor: `{scale.get('scale_factor', 1.0)}`")
        lines.append("")

        ordered = ds_summary.sort_values("mean_mlu", ascending=True)
        cols = [
            "method",
            "mean_mlu",
            "p95_mlu",
            "mean_disturbance",
            "p95_disturbance",
            "mean_pred_mae",
            "mean_pred_rmse",
            "mean_pred_smape",
        ]
        lines.append(_table_markdown(ordered, cols))
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_rollout(
    tm: np.ndarray,
    split: Dict[str, int],
    predictor: BaseTMPredictor,
) -> list[dict[str, object]]:
    required = max(1, predictor.required_history())
    start_decision = max(required - 1, 0)

    rows: list[dict[str, object]] = []
    for decision_t in range(start_decision, tm.shape[0] - 1):
        eval_t = decision_t + 1
        if eval_t < split["test_start"]:
            continue

        pred_tm = predictor.predict_next(tm[: decision_t + 1])
        actual_tm = tm[eval_t]
        metrics = compute_prediction_metrics(pred_tm, actual_tm)

        rows.append(
            {
                "decision_t": int(decision_t),
                "eval_t": int(eval_t),
                "pred_tm": pred_tm,
                "actual_tm": actual_tm,
                "pred_metrics": metrics,
            }
        )

    return rows


def run_predictive_method(
    method: str,
    dataset,
    path_library,
    rollout: list[dict[str, object]],
    ospf_base: Sequence[np.ndarray],
    ecmp_base: Sequence[np.ndarray],
    shortest_costs: np.ndarray,
    k_crit: int,
    lp_time_limit_sec: int,
    full_mcf_time_limit_sec: int,
    rl_policy: ODSelectorPolicy | None,
    device: torch.device,
) -> pd.DataFrame:
    prev_splits = None
    prev_selected = np.zeros(len(dataset.od_pairs), dtype=float)
    rows: list[dict[str, object]] = []

    for step_idx, step in enumerate(rollout):
        pred_tm = np.asarray(step["pred_tm"], dtype=float)
        actual_tm = np.asarray(step["actual_tm"], dtype=float)
        decision_t = int(step["decision_t"])
        eval_t = int(step["eval_t"])
        pred_m: PredictionMetrics = step["pred_metrics"]

        t0 = time.perf_counter()

        if method == "ospf":
            splits = clone_splits(ospf_base)
            routing = apply_routing(actual_tm, splits, path_library, dataset.capacities)
            status = "Static"

        elif method == "ecmp":
            splits = clone_splits(ecmp_base)
            routing = apply_routing(actual_tm, splits, path_library, dataset.capacities)
            status = "Static"

        elif method == "topk_pred":
            selected = select_topk_by_demand(pred_tm, k_crit=k_crit)
            lp = solve_selected_path_lp(
                tm_vector=pred_tm,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=dataset.capacities,
                time_limit_sec=lp_time_limit_sec,
            )
            splits = lp.splits
            routing = apply_routing(actual_tm, splits, path_library, dataset.capacities)
            status = lp.status

        elif method == "bottleneck_pred":
            selected = select_bottleneck_critical(
                tm_vector=pred_tm,
                ecmp_policy=ecmp_base,
                path_library=path_library,
                capacities=dataset.capacities,
                k_crit=k_crit,
            )
            lp = solve_selected_path_lp(
                tm_vector=pred_tm,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=dataset.capacities,
                time_limit_sec=lp_time_limit_sec,
            )
            splits = lp.splits
            routing = apply_routing(actual_tm, splits, path_library, dataset.capacities)
            status = lp.status

        elif method == "lp_optimal_pred":
            full = solve_full_mcf_min_mlu(
                tm_vector=pred_tm,
                od_pairs=dataset.od_pairs,
                nodes=dataset.nodes,
                edges=dataset.edges,
                capacities=dataset.capacities,
                time_limit_sec=full_mcf_time_limit_sec,
            )
            splits = project_edge_flows_to_k_path_splits(full.edge_flows_by_od, path_library)
            routing = apply_routing(actual_tm, splits, path_library, dataset.capacities)
            status = full.status

        elif method == "rl_lp_pred":
            if rl_policy is None:
                raise RuntimeError("Method rl_lp_pred requested but no RL checkpoint/policy was loaded.")

            features = build_od_features(pred_tm, shortest_costs, prev_selected).to(device)
            with torch.no_grad():
                scores = rl_policy(features).cpu()
            selected = deterministic_topk(scores, k=k_crit).tolist()

            lp = solve_selected_path_lp(
                tm_vector=pred_tm,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=dataset.capacities,
                time_limit_sec=lp_time_limit_sec,
            )
            splits = lp.splits
            routing = apply_routing(actual_tm, splits, path_library, dataset.capacities)
            status = lp.status

            prev_selected = np.zeros_like(prev_selected)
            prev_selected[selected] = 1.0

        else:
            raise ValueError(f"Unknown method: {method}")

        runtime_sec = time.perf_counter() - t0
        disturbance = compute_disturbance(prev_splits, splits, actual_tm)
        prev_splits = clone_splits(splits)

        rows.append(
            {
                "dataset": dataset.key,
                "method": method,
                "decision_t": decision_t,
                "eval_t": eval_t,
                "test_step": int(step_idx),
                "mlu": float(routing.mlu),
                "disturbance": float(disturbance),
                "mean_utilization": float(routing.mean_utilization),
                "solver_status": status,
                "runtime_sec": float(runtime_sec),
                "pred_mae": float(pred_m.mae),
                "pred_rmse": float(pred_m.rmse),
                "pred_smape": float(pred_m.smape),
            }
        )

    return pd.DataFrame(rows)


def summarize_method(timeseries: pd.DataFrame) -> pd.DataFrame:
    group = timeseries.groupby(["dataset", "method"], as_index=False)
    summary = group.agg(
        mean_mlu=("mlu", "mean"),
        p95_mlu=("mlu", lambda x: float(np.quantile(x, 0.95))),
        mean_disturbance=("disturbance", "mean"),
        p95_disturbance=("disturbance", lambda x: float(np.quantile(x, 0.95))),
        mean_runtime_sec=("runtime_sec", "mean"),
        mean_pred_mae=("pred_mae", "mean"),
        mean_pred_rmse=("pred_rmse", "mean"),
        mean_pred_smape=("pred_smape", "mean"),
        num_test_steps=("mlu", "count"),
    )
    return summary


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    rl_policy = None
    if "rl_lp_pred" in methods:
        if args.rl_checkpoint is None:
            raise RuntimeError("rl_lp_pred requested, but --rl_checkpoint was not provided.")
        checkpoint = Path(args.rl_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"RL checkpoint not found: {checkpoint}")
        rl_policy = load_rl_policy(checkpoint, device=device)

    all_timeseries = []
    all_summaries = []
    split_info: Dict[str, Dict[str, int]] = {}
    plot_paths: Dict[str, Dict[str, Path]] = {}
    scale_info: Dict[str, Dict[str, float]] = {}
    predictor_info: Dict[str, Dict[str, object]] = {}
    config_payload: Dict[str, object] = {}

    for config_path_raw in args.config:
        config_path = Path(config_path_raw)
        config = load_config(config_path)
        config_payload[str(config_path)] = config

        dataset = load_dataset(config, max_steps=args.max_steps)
        exp_cfg = config.get("experiment", {}) if isinstance(config.get("experiment"), dict) else {}
        phase2_cfg = exp_cfg.get("phase2", {}) if isinstance(exp_cfg.get("phase2"), dict) else {}

        k_paths = int(args.k_paths if args.k_paths is not None else exp_cfg.get("k_paths", 3))
        k_crit = int(args.k_crit if args.k_crit is not None else exp_cfg.get("k_crit", 20))
        lp_time_limit_sec = int(
            args.lp_time_limit_sec if args.lp_time_limit_sec is not None else exp_cfg.get("lp_time_limit_sec", 20)
        )
        full_mcf_time_limit_sec = int(
            args.full_mcf_time_limit_sec
            if args.full_mcf_time_limit_sec is not None
            else exp_cfg.get("full_mcf_time_limit_sec", 90)
        )

        predictor_name = str(args.predictor if args.predictor is not None else phase2_cfg.get("predictor", "ar_ridge"))
        predictor_window = int(
            args.predictor_window if args.predictor_window is not None else phase2_cfg.get("predictor_window", 6)
        )
        predictor_alpha = float(
            args.predictor_alpha if args.predictor_alpha is not None else phase2_cfg.get("predictor_alpha", 1e-2)
        )

        path_library = build_paths(dataset, k_paths=k_paths)
        ospf_base = ospf_splits(path_library)
        ecmp_base = ecmp_splits(path_library)

        tm_work = np.asarray(dataset.tm, dtype=float)
        scale_factor = 1.0
        baseline_probe_mean_mlu = float("nan")

        scaling_cfg = exp_cfg.get("scaling", {}) if isinstance(exp_cfg.get("scaling"), dict) else {}
        enable_auto_scale = bool(scaling_cfg.get("enable_auto_scale", False)) and not args.disable_auto_scale
        if enable_auto_scale:
            target_mlu = float(scaling_cfg.get("target_mlu_train", 1.0))
            probe_steps = int(scaling_cfg.get("scale_probe_steps", 200))
            scale_factor, probe = compute_auto_scale_factor(
                tm=tm_work,
                train_end=dataset.split["train_end"],
                path_library=path_library,
                capacities=dataset.capacities,
                target_mlu_train=target_mlu,
                scale_probe_steps=probe_steps,
            )
            baseline_probe_mean_mlu = float(probe.mean_mlu)
            tm_work = apply_scale(tm_work, scale_factor)

        predictor = build_predictor(predictor_name, window=predictor_window, alpha=predictor_alpha)
        predictor.fit(tm_work[: dataset.split["train_end"]])

        rollout = build_rollout(tm_work, dataset.split, predictor)
        if not rollout:
            raise RuntimeError(
                f"No phase2 rollout steps for dataset={dataset.key}. Increase max_steps or reduce predictor window."
            )

        shortest_costs = np.array(
            [min(costs) if costs else np.inf for costs in path_library.costs_by_od],
            dtype=float,
        )

        dataset_rows = []
        for method in methods:
            print(f"Running Phase-2 dataset={dataset.key} method={method}")
            method_df = run_predictive_method(
                method=method,
                dataset=dataset,
                path_library=path_library,
                rollout=rollout,
                ospf_base=ospf_base,
                ecmp_base=ecmp_base,
                shortest_costs=shortest_costs,
                k_crit=k_crit,
                lp_time_limit_sec=lp_time_limit_sec,
                full_mcf_time_limit_sec=full_mcf_time_limit_sec,
                rl_policy=rl_policy,
                device=device,
            )
            dataset_rows.append(method_df)

        dataset_ts = pd.concat(dataset_rows, ignore_index=True)
        dataset_summary = summarize_method(dataset_ts)

        pred_ts = pd.DataFrame(
            [
                {
                    "dataset": dataset.key,
                    "decision_t": int(step["decision_t"]),
                    "eval_t": int(step["eval_t"]),
                    "pred_mae": float(step["pred_metrics"].mae),
                    "pred_rmse": float(step["pred_metrics"].rmse),
                    "pred_smape": float(step["pred_metrics"].smape),
                }
                for step in rollout
            ]
        )

        dataset_out = output_dir / dataset.key
        dataset_out.mkdir(parents=True, exist_ok=True)
        dataset_ts.to_csv(dataset_out / "timeseries.csv", index=False)
        dataset_summary.to_csv(dataset_out / "summary.csv", index=False)
        pred_ts.to_csv(dataset_out / "prediction_timeseries.csv", index=False)

        plot_paths[dataset.key] = generate_plots_for_dataset(dataset_ts, dataset.key, dataset_out)

        split_info[dataset.key] = dict(dataset.split)
        scale_info[dataset.key] = {
            "scale_factor": float(scale_factor),
            "baseline_probe_mean_mlu": baseline_probe_mean_mlu,
        }
        predictor_info[dataset.key] = {
            "name": predictor_name,
            "window": predictor_window,
            "alpha": predictor_alpha,
        }

        all_timeseries.append(dataset_ts)
        all_summaries.append(dataset_summary)

    all_timeseries_df = pd.concat(all_timeseries, ignore_index=True)
    all_summary_df = pd.concat(all_summaries, ignore_index=True)

    all_timeseries_df.to_csv(output_dir / "timeseries_all.csv", index=False)
    all_summary_df.to_csv(output_dir / "summary_all.csv", index=False)

    run_meta = {
        "seed": args.seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methods": methods,
        "max_steps_override": args.max_steps,
        "config_paths": args.config,
        "split_info": split_info,
        "scale_info": scale_info,
        "predictor_info": predictor_info,
        "configs": config_payload,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    write_phase2_report(
        summary_df=all_summary_df,
        split_info=split_info,
        scale_info=scale_info,
        predictor_info=predictor_info,
        output_path=output_dir / "report.md",
        run_meta=run_meta,
    )

    print(f"Wrote summary: {output_dir / 'summary_all.csv'}")
    print(f"Wrote timeseries: {output_dir / 'timeseries_all.csv'}")
    print(f"Wrote report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
