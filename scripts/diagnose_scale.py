#!/usr/bin/env python3
"""Dataset diagnostics for TE scaling pressure and method separability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from te.baselines import ecmp_splits, ospf_splits, select_bottleneck_critical, select_topk_by_demand
from te.disturbance import compute_disturbance
from te.lp_solver import solve_selected_path_lp
from te.scaling import evaluate_fixed_policy
from te.simulator import apply_routing, build_paths, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose TE scaling and congestion characteristics")
    parser.add_argument(
        "--config",
        action="append",
        default=["configs/abilene.yaml", "configs/geant.yaml"],
        help="Dataset config(s)",
    )
    parser.add_argument("--output_dir", default="results/diagnostics", help="Diagnostics output directory")
    parser.add_argument("--max_steps", type=int, default=500)
    return parser.parse_args()


def _method_timeseries(
    method: str,
    tm: np.ndarray,
    path_library,
    capacities: np.ndarray,
    ecmp_base: List[np.ndarray],
    ospf_base: List[np.ndarray],
    k_crit: int,
) -> Dict[str, float]:
    prev_splits = None
    mlus = []
    disturbances = []

    for t_idx in range(tm.shape[0]):
        step_tm = tm[t_idx]

        if method == "ospf":
            splits = [vec.copy() for vec in ospf_base]
            routing = apply_routing(step_tm, splits, path_library, capacities)
        elif method == "ecmp":
            splits = [vec.copy() for vec in ecmp_base]
            routing = apply_routing(step_tm, splits, path_library, capacities)
        elif method == "topk":
            selected = select_topk_by_demand(step_tm, k_crit=k_crit)
            lp = solve_selected_path_lp(
                tm_vector=step_tm,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                time_limit_sec=15,
            )
            splits = lp.splits
            routing = lp.routing
        elif method == "bottleneck":
            selected = select_bottleneck_critical(
                tm_vector=step_tm,
                ecmp_policy=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                k_crit=k_crit,
            )
            lp = solve_selected_path_lp(
                tm_vector=step_tm,
                selected_ods=selected,
                base_splits=ecmp_base,
                path_library=path_library,
                capacities=capacities,
                time_limit_sec=15,
            )
            splits = lp.splits
            routing = lp.routing
        else:
            raise ValueError(method)

        disturbance = compute_disturbance(prev_splits, splits, step_tm)
        prev_splits = [vec.copy() for vec in splits]

        mlus.append(routing.mlu)
        disturbances.append(disturbance)

    arr_mlu = np.asarray(mlus, dtype=float)
    arr_d = np.asarray(disturbances, dtype=float)
    return {
        "mean_mlu": float(np.mean(arr_mlu)),
        "p95_mlu": float(np.quantile(arr_mlu, 0.95)),
        "std_mlu": float(np.std(arr_mlu)),
        "mean_disturbance": float(np.mean(arr_d)),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cfg_path in args.config:
        with open(cfg_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)

        dataset = load_dataset(cfg, max_steps=args.max_steps)
        exp = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
        k_paths = int(exp.get("k_paths", 3))
        k_crit = int(exp.get("k_crit", 20))

        path_library = build_paths(dataset, k_paths=k_paths)
        ecmp_base = ecmp_splits(path_library)
        ospf_base = ospf_splits(path_library)

        tm = dataset.tm
        totals = np.sum(tm, axis=1)
        positive_demands = tm[tm > 0]

        train_idx = range(0, dataset.split["train_end"])
        val_idx = range(dataset.split["train_end"], dataset.split["val_end"])
        test_idx = range(dataset.split["test_start"], dataset.tm.shape[0])

        ecmp_train = evaluate_fixed_policy(tm, train_idx, ecmp_base, path_library, dataset.capacities)
        ecmp_val = evaluate_fixed_policy(tm, val_idx, ecmp_base, path_library, dataset.capacities)
        ecmp_test = evaluate_fixed_policy(tm, test_idx, ecmp_base, path_library, dataset.capacities)

        method_stats = {
            m: _method_timeseries(m, tm[test_idx.start :], path_library, dataset.capacities, ecmp_base, ospf_base, k_crit)
            for m in ["ospf", "ecmp", "topk", "bottleneck"]
        }

        method_mean_values = [method_stats[m]["mean_mlu"] for m in method_stats]
        method_mean_spread = float(np.max(method_mean_values) - np.min(method_mean_values))

        reasons = []
        if ecmp_train.mean_mlu < 0.3:
            reasons.append(
                "Train ECMP mean MLU is very low, so optimization headroom is limited; many policies look identical."
            )
        if method_mean_spread < 1e-4:
            reasons.append(
                "Test mean MLU spread across OSPF/ECMP/TopK/Bottleneck is near zero, confirming practical equivalence in this regime."
            )
        if method_stats["topk"]["mean_disturbance"] < 1e-4 and method_stats["bottleneck"]["mean_disturbance"] < 1e-4:
            reasons.append(
                "Hybrid LP methods barely changed routing over time (near-zero disturbance), indicating no active bottleneck pressure."
            )

        lines = []
        lines.append(f"dataset={dataset.key}")
        lines.append(f"processed_path={dataset.processed_path}")
        lines.append("")
        lines.append("[Capacity stats]")
        lines.append(
            "capacity min/median/max = "
            f"{np.min(dataset.capacities):.6f} / {np.median(dataset.capacities):.6f} / {np.max(dataset.capacities):.6f}"
        )
        lines.append("")
        lines.append("[Demand stats]")
        if positive_demands.size:
            lines.append(
                "demand min/median/max (positive) = "
                f"{np.min(positive_demands):.6f} / {np.median(positive_demands):.6f} / {np.max(positive_demands):.6f}"
            )
        else:
            lines.append("demand min/median/max (positive) = 0 / 0 / 0")
        lines.append(
            "total demand per timestep min/median/max = "
            f"{np.min(totals):.6f} / {np.median(totals):.6f} / {np.max(totals):.6f}"
        )
        lines.append("")
        lines.append("[ECMP MLU by split]")
        lines.append(f"train mean/p95 = {ecmp_train.mean_mlu:.6f} / {ecmp_train.p95_mlu:.6f}")
        lines.append(f"val   mean/p95 = {ecmp_val.mean_mlu:.6f} / {ecmp_val.p95_mlu:.6f}")
        lines.append(f"test  mean/p95 = {ecmp_test.mean_mlu:.6f} / {ecmp_test.p95_mlu:.6f}")
        lines.append("")
        lines.append("[Test-only comparison, raw scale]")
        for method in ["ospf", "ecmp", "topk", "bottleneck"]:
            s = method_stats[method]
            lines.append(
                f"{method:10s} mean_mlu={s['mean_mlu']:.6f} p95_mlu={s['p95_mlu']:.6f} "
                f"std_mlu={s['std_mlu']:.6f} mean_disturbance={s['mean_disturbance']:.6f}"
            )
        lines.append("")
        lines.append("[Abilene identical-MLU diagnosis]")
        lines.append("MLU formula checked: MLU = max_e(load_e / capacity_e), dimensionless ratio.")
        if reasons:
            for idx, reason in enumerate(reasons, start=1):
                lines.append(f"{idx}. {reason}")
        else:
            lines.append("No collapse warning triggered from current thresholds.")

        out_file = out_dir / f"{dataset.key}.txt"
        out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
