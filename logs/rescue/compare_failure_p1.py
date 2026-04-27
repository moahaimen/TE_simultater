"""Per-topology failure win/tie tally vs Bottleneck for Phase 1.

Usage:
    python logs/rescue/compare_failure_p1.py [--tag <tag>] [--baseline-tag <tag>]

Default --tag is the rescue bundle (sticky); --baseline-tag is Task A. Reports:
  - per (topology, scenario) win/tie/loss vs Bottleneck (using mean_mlu)
  - per-topology subtotal (out of 5)
  - grand total (out of 5 * num_topos)
  - delta vs the optional baseline if both bundles exist

Pre-registered target: 45/50 win-or-tie if 10 topologies are in the bundle, or
proportional 36/40 (90%) for the existing 8-topology setup.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FAILURE_ORDER = [
    "single_link_failure",
    "multiple_link_failure",
    "three_link_failure",
    "capacity_degradation_50",
    "traffic_spike_2x",
]

TOPO_ORDER = [
    "abilene", "cernet", "geant", "ebone",
    "sprintlink", "tiscali", "germany50", "vtlwavenet2011",
]


def load_pivot(path: Path) -> dict[tuple[str, str, str], float]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str, str], float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                out[(row["topology"], row["scenario"], row["method"])] = float(row["mean_mlu"])
            except (KeyError, ValueError):
                continue
    return out


def tally(pivot: dict, label: str) -> tuple[int, int]:
    total = 0
    wins = 0
    print(f"\n{label}")
    print("-" * 100)
    print(f"{'topology':<16} | " + " | ".join(s[:14].ljust(14) for s in FAILURE_ORDER) + " | wins")
    grand = 0
    for topo in TOPO_ORDER:
        row_marks = []
        topo_wins = 0
        for scen in FAILURE_ORDER:
            gnn = pivot.get((topo, scen, "gnnplus"))
            bn = pivot.get((topo, scen, "bottleneck"))
            if gnn is None or bn is None:
                row_marks.append("--")
                continue
            total += 1
            if gnn <= bn + 1e-9:
                row_marks.append("W")
                wins += 1
                topo_wins += 1
            else:
                row_marks.append(f"L({(gnn-bn)/bn*100:+.1f}%)")
        print(f"{topo:<16} | " + " | ".join(m.ljust(14) for m in row_marks) + f" | {topo_wins}/5")
        grand += topo_wins
    print(f"\n  TOTAL: {wins} win-or-tie / {total} cases  ({wins/total*100:.1f}%)" if total else "  no data")
    return wins, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="rescue_p1_sticky_005")
    parser.add_argument("--baseline-tag", default="gnnplus_8topo_stability_taskA")
    args = parser.parse_args()

    rescue = ROOT / "results" / args.tag / "packet_sdn_failure.csv"
    base = ROOT / "results" / args.baseline_tag / "packet_sdn_failure.csv"

    rescue_pivot = load_pivot(rescue)
    base_pivot = load_pivot(base)

    print("=" * 100)
    print(f"FAILURE WIN/TIE vs BOTTLENECK  (criterion: gnnplus_mlu <= bottleneck_mlu + 1e-9)")
    print("=" * 100)

    base_wins, base_total = tally(base_pivot, f"BASELINE: {args.baseline_tag}")
    rescue_wins, rescue_total = tally(rescue_pivot, f"RESCUE:   {args.tag}")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    if base_total:
        print(f"  baseline ({args.baseline_tag}): {base_wins}/{base_total}")
    if rescue_total:
        print(f"  rescue   ({args.tag}): {rescue_wins}/{rescue_total}")
    if base_total and rescue_total:
        delta = rescue_wins - base_wins
        sign = "+" if delta >= 0 else ""
        print(f"  delta:   {sign}{delta} cases")

    target_8 = 36   # 90% of 40
    target_10 = 45  # student's stated target if 10 topos
    if rescue_total == 40:
        verdict = "PASS" if rescue_wins >= target_8 else "FAIL"
        print(f"  vs target {target_8}/40 (90% of 8-topo set): {verdict}")
    elif rescue_total == 50:
        verdict = "PASS" if rescue_wins >= target_10 else "FAIL"
        print(f"  vs target {target_10}/50 (student's stated): {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
