#!/usr/bin/env python3
"""Compute Phase-2 complexity and timing report for copy-paste sharing."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from phase2.predictors import BaseTMPredictor, build_predictor
from te.baselines import clone_splits, ecmp_splits, select_bottleneck_critical, select_topk_by_demand
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.scaling import apply_scale, compute_auto_scale_factor
from te.simulator import apply_routing, build_paths, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-2 complexity and training-time reporter")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="YAML config path (repeatable). Default: configs/abilene.yaml + configs/geant.yaml",
    )
    parser.add_argument("--max_steps", type=int, default=500, help="Max timesteps to load")
    parser.add_argument("--predictor", default=None, help="Override predictor name")
    parser.add_argument("--predictor_window", type=int, default=None, help="Override predictor lag window")
    parser.add_argument("--predictor_alpha", type=float, default=None, help="Override predictor alpha")
    parser.add_argument("--fit_repeats", type=int, default=3, help="How many times to fit for stable timing")
    parser.add_argument("--predict_calls", type=int, default=500, help="How many predict calls to benchmark")
    parser.add_argument(
        "--bench_steps",
        type=int,
        default=30,
        help="How many proactive test steps to benchmark for topk_pred/bottleneck_pred",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_auto_scale", action="store_true")
    parser.add_argument("--output_dir", default="results/phase2_complexity")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_configs(args: argparse.Namespace) -> List[Path]:
    if args.config:
        return [Path(x) for x in args.config]
    return [Path("configs/abilene.yaml"), Path("configs/geant.yaml")]


def _phase2_params(cfg: Dict[str, Any], args: argparse.Namespace) -> tuple[str, int, float, int, int]:
    exp_cfg = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    phase2_cfg = exp_cfg.get("phase2", {}) if isinstance(exp_cfg.get("phase2"), dict) else {}

    predictor_name = str(args.predictor if args.predictor is not None else phase2_cfg.get("predictor", "ar_ridge"))
    predictor_window = int(
        args.predictor_window if args.predictor_window is not None else phase2_cfg.get("predictor_window", 6)
    )
    predictor_alpha = float(
        args.predictor_alpha if args.predictor_alpha is not None else phase2_cfg.get("predictor_alpha", 1e-2)
    )

    k_paths = int(exp_cfg.get("k_paths", 3))
    k_crit = int(exp_cfg.get("k_crit", 20))
    return predictor_name, predictor_window, predictor_alpha, k_paths, k_crit


def _apply_scaling_if_enabled(
    tm: np.ndarray,
    split: Dict[str, int],
    path_library,
    capacities: np.ndarray,
    cfg: Dict[str, Any],
    disable_auto_scale: bool,
) -> tuple[np.ndarray, float, float]:
    exp_cfg = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    scaling_cfg = exp_cfg.get("scaling", {}) if isinstance(exp_cfg.get("scaling"), dict) else {}

    enabled = bool(scaling_cfg.get("enable_auto_scale", False)) and not disable_auto_scale
    if not enabled:
        return np.asarray(tm, dtype=float), 1.0, float("nan")

    target_mlu = float(scaling_cfg.get("target_mlu_train", 1.0))
    probe_steps = int(scaling_cfg.get("scale_probe_steps", 200))

    factor, probe = compute_auto_scale_factor(
        tm=np.asarray(tm, dtype=float),
        train_end=int(split["train_end"]),
        path_library=path_library,
        capacities=np.asarray(capacities, dtype=float),
        target_mlu_train=target_mlu,
        scale_probe_steps=probe_steps,
    )
    return apply_scale(tm, factor), float(factor), float(probe.mean_mlu)


def predictor_timing(
    predictor_name: str,
    predictor_window: int,
    predictor_alpha: float,
    tm_train: np.ndarray,
    fit_repeats: int,
    predict_calls: int,
) -> tuple[BaseTMPredictor, Dict[str, float]]:
    fit_times: List[float] = []
    model: BaseTMPredictor | None = None

    repeats = max(1, int(fit_repeats))
    calls = max(1, int(predict_calls))

    for _ in range(repeats):
        model = build_predictor(predictor_name, window=predictor_window, alpha=predictor_alpha)
        t0 = time.perf_counter()
        model.fit(tm_train)
        fit_times.append(time.perf_counter() - t0)

    assert model is not None

    t0 = time.perf_counter()
    for _ in range(calls):
        _ = model.predict_next(tm_train)
    pred_total = time.perf_counter() - t0

    timing = {
        "fit_sec_mean": float(np.mean(fit_times)),
        "fit_sec_std": float(np.std(fit_times)),
        "predict_call_sec_mean": float(pred_total / calls),
        "fit_repeats": int(repeats),
        "predict_calls": int(calls),
    }
    return model, timing


def complexity_metrics(
    num_od: int,
    num_edges: int,
    train_steps: int,
    k_paths: int,
    k_crit: int,
    window: int,
    avg_path_len: float,
) -> Dict[str, float]:
    train_samples = max(0, int(train_steps) - int(window))

    predictor_train_ops = float(num_od) * (float(train_samples) * float(window * window) + float(window**3))
    predictor_infer_ops = float(num_od * window)

    lp_vars = float(k_crit * k_paths + 1)
    lp_constraints = float(k_crit + num_edges)
    lp_incidence_ops = float(k_crit * k_paths) * float(avg_path_len)

    return {
        "predictor_train_ops_proxy": predictor_train_ops,
        "predictor_infer_ops_proxy": predictor_infer_ops,
        "lp_vars": lp_vars,
        "lp_constraints": lp_constraints,
        "lp_incidence_ops_proxy": lp_incidence_ops,
    }


def avg_candidate_path_len(path_library) -> float:
    lens = []
    for od_paths in path_library.edge_idx_paths_by_od:
        for p in od_paths:
            lens.append(len(p))
    if not lens:
        return 0.0
    return float(np.mean(np.asarray(lens, dtype=float)))


def benchmark_proactive_method(
    method: str,
    tm: np.ndarray,
    split: Dict[str, int],
    predictor: BaseTMPredictor,
    path_library,
    capacities: np.ndarray,
    ecmp_base: Sequence[np.ndarray],
    k_crit: int,
    lp_time_limit_sec: int,
    bench_steps: int,
) -> Dict[str, float]:
    start_eval = max(int(split["test_start"]), max(1, predictor.required_history()))
    end_eval = min(int(tm.shape[0]), start_eval + max(1, int(bench_steps)))

    prev_splits = None
    total_predict = 0.0
    total_select = 0.0
    total_lp = 0.0
    total_route = 0.0
    total_all = 0.0
    mlus: List[float] = []
    disturbances: List[float] = []

    for eval_t in range(start_eval, end_eval):
        decision_t = eval_t - 1

        t0 = time.perf_counter()
        tp = time.perf_counter()
        pred_tm = predictor.predict_next(tm[: decision_t + 1])
        total_predict += time.perf_counter() - tp

        ts = time.perf_counter()
        if method == "topk_pred":
            selected = select_topk_by_demand(pred_tm, k_crit=k_crit)
        elif method == "bottleneck_pred":
            selected = select_bottleneck_critical(
                tm_vector=pred_tm,
                ecmp_policy=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                k_crit=k_crit,
            )
        else:
            raise ValueError(method)
        total_select += time.perf_counter() - ts

        tlp = time.perf_counter()
        lp = solve_selected_path_lp(
            tm_vector=pred_tm,
            selected_ods=selected,
            base_splits=ecmp_base,
            path_library=path_library,
            capacities=capacities,
            time_limit_sec=lp_time_limit_sec,
        )
        total_lp += time.perf_counter() - tlp

        tr = time.perf_counter()
        routing = apply_routing(tm[eval_t], lp.splits, path_library, capacities)
        total_route += time.perf_counter() - tr

        disturbance = compute_disturbance(prev_splits, lp.splits, tm[eval_t])
        prev_splits = clone_splits(lp.splits)

        total_all += time.perf_counter() - t0
        mlus.append(float(routing.mlu))
        disturbances.append(float(disturbance))

    n = max(1, end_eval - start_eval)
    return {
        "bench_steps": int(end_eval - start_eval),
        "mean_step_sec": float(total_all / n),
        "mean_predict_sec": float(total_predict / n),
        "mean_select_sec": float(total_select / n),
        "mean_lp_sec": float(total_lp / n),
        "mean_route_sec": float(total_route / n),
        "mean_mlu": float(np.mean(mlus)) if mlus else float("nan"),
        "mean_disturbance": float(np.mean(disturbances)) if disturbances else float("nan"),
    }


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_paths = resolve_configs(args)

    complexity_rows: List[Dict[str, Any]] = []
    benchmark_rows: List[Dict[str, Any]] = []

    for cfg_path in cfg_paths:
        cfg = load_yaml(cfg_path)
        dataset = load_dataset(cfg, max_steps=args.max_steps)

        predictor_name, predictor_window, predictor_alpha, k_paths, k_crit = _phase2_params(cfg, args)
        exp_cfg = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
        lp_time_limit_sec = int(exp_cfg.get("lp_time_limit_sec", 20))

        path_library = build_paths(dataset, k_paths=k_paths)
        ecmp_base = ecmp_splits(path_library)

        tm_scaled, scale_factor, baseline_probe_mean_mlu = _apply_scaling_if_enabled(
            tm=dataset.tm,
            split=dataset.split,
            path_library=path_library,
            capacities=dataset.capacities,
            cfg=cfg,
            disable_auto_scale=args.disable_auto_scale,
        )

        train_end = int(dataset.split["train_end"])
        tm_train = np.asarray(tm_scaled[:train_end], dtype=float)
        predictor, ptime = predictor_timing(
            predictor_name=predictor_name,
            predictor_window=predictor_window,
            predictor_alpha=predictor_alpha,
            tm_train=tm_train,
            fit_repeats=args.fit_repeats,
            predict_calls=args.predict_calls,
        )

        comp = complexity_metrics(
            num_od=len(dataset.od_pairs),
            num_edges=len(dataset.edges),
            train_steps=train_end,
            k_paths=k_paths,
            k_crit=k_crit,
            window=predictor_window,
            avg_path_len=avg_candidate_path_len(path_library),
        )

        complexity_rows.append(
            {
                "dataset": dataset.key,
                "num_steps": int(dataset.tm.shape[0]),
                "num_train": int(dataset.split["num_train"]),
                "num_val": int(dataset.split["num_val"]),
                "num_test": int(dataset.split["num_test"]),
                "num_od": int(len(dataset.od_pairs)),
                "num_edges": int(len(dataset.edges)),
                "k_paths": int(k_paths),
                "k_crit": int(k_crit),
                "predictor": predictor_name,
                "predictor_window": int(predictor_window),
                "predictor_alpha": float(predictor_alpha),
                "scale_factor": float(scale_factor),
                "baseline_probe_mean_mlu": float(baseline_probe_mean_mlu),
                **comp,
                **ptime,
            }
        )

        for method in ("topk_pred", "bottleneck_pred"):
            bench = benchmark_proactive_method(
                method=method,
                tm=np.asarray(tm_scaled, dtype=float),
                split=dataset.split,
                predictor=predictor,
                path_library=path_library,
                capacities=np.asarray(dataset.capacities, dtype=float),
                ecmp_base=ecmp_base,
                k_crit=k_crit,
                lp_time_limit_sec=lp_time_limit_sec,
                bench_steps=args.bench_steps,
            )
            benchmark_rows.append(
                {
                    "dataset": dataset.key,
                    "method": method,
                    **bench,
                }
            )

    complexity_df = pd.DataFrame(complexity_rows)
    benchmark_df = pd.DataFrame(benchmark_rows)

    complexity_csv = out_dir / "complexity_summary.csv"
    bench_csv = out_dir / "runtime_benchmark.csv"
    complexity_df.to_csv(complexity_csv, index=False)
    benchmark_df.to_csv(bench_csv, index=False)

    lines: List[str] = []
    lines.append("# Phase-2 Complexity and Timing Report")
    lines.append("")
    lines.append("## Complexity Summary")
    lines.append("")
    lines.append(_to_markdown(complexity_df))
    lines.append("")
    lines.append("## Runtime Benchmark (Proactive Methods)")
    lines.append("")
    lines.append(_to_markdown(benchmark_df))
    lines.append("")

    md_path = out_dir / "complexity_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "seed": args.seed,
        "configs": [str(p) for p in cfg_paths],
        "max_steps": args.max_steps,
        "fit_repeats": args.fit_repeats,
        "predict_calls": args.predict_calls,
        "bench_steps": args.bench_steps,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {complexity_csv}")
    print(f"Wrote: {bench_csv}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
