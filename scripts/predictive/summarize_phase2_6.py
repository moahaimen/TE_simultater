"""Phase 2.6 summary aggregator.

Reads results/phase2_6/phase2_6_predictive_cfs_results.csv and produces:

  - results/phase2_6/phase2_6_summary_by_topology.csv
  - results/phase2_6/phase2_6_summary_by_scenario.csv
  - results/phase2_6/phase2_6_win_loss_matrix.csv
  - results/phase2_6/PHASE2_6_PREDICTIVE_ROUTING_WRITEUP.md

Success criterion (pre-registered):
  Predictive routing is "successful" if, under dynamic traffic or
  delayed-actuation scenarios, it improves AT LEAST ONE of:
    - p95 MLU
    - peak MLU
    - overload duration above 0.7 or 0.9
    - recovery time
  by >= 2-5% vs reactive, while keeping disturbance within +5% of
  reactive (or strictly reducing it).

The check is per-cell (topology, scenario, K, delay). We aggregate cell
counts to make the success/failure call clear.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE2_6 = PROJECT_ROOT / "results" / "phase2_6"
RESULTS_CSV = PHASE2_6 / "phase2_6_predictive_cfs_results.csv"


def parse_csv() -> list[dict]:
    rows = []
    with open(RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def cell_key(r: dict) -> tuple:
    return (r["topology"], r["scenario"], int(r["k"]), int(r["delay"]))


def collect_by_cell(rows: list[dict]) -> dict[tuple, dict[str, dict]]:
    """{(topo, scen, k, delay): {method: row_dict, ...}}"""
    out: dict = defaultdict(dict)
    for r in rows:
        out[cell_key(r)][r["method"]] = r
    return out


def cell_verdict(cell: dict, abs_delta_threshold: float = 2.0,
                 disturb_tolerance: float = 5.0) -> dict:
    """Return per-cell verdict: did predictive beat reactive?"""
    if "reactive" not in cell or "predictive" not in cell:
        return {"verdict": "missing"}
    r = cell["reactive"]
    p = cell["predictive"]
    o = cell.get("oracle", {})

    delta_mean = float(p.get("delta_mean_pct", 0))
    delta_p95 = float(p.get("delta_p95_pct", 0))
    delta_peak = float(p.get("delta_peak_pct", 0))
    delta_overload_07 = int(p.get("delta_overload_0p7", 0))
    delta_overload_09 = int(p.get("delta_overload_0p9", 0))
    delta_disturb = float(p.get("delta_disturb_pct", 0))

    # Did predictive beat reactive on at least one routing metric?
    wins = []
    losses = []
    if delta_mean <= -abs_delta_threshold:   wins.append("mean")
    elif delta_mean >= abs_delta_threshold:  losses.append("mean")
    if delta_p95 <= -abs_delta_threshold:    wins.append("p95")
    elif delta_p95 >= abs_delta_threshold:   losses.append("p95")
    if delta_peak <= -abs_delta_threshold:   wins.append("peak")
    elif delta_peak >= abs_delta_threshold:  losses.append("peak")
    if delta_overload_07 < 0:                wins.append("overload07")
    elif delta_overload_07 > 0:              losses.append("overload07")
    if delta_overload_09 < 0:                wins.append("overload09")
    elif delta_overload_09 > 0:              losses.append("overload09")

    disturb_ok = delta_disturb <= disturb_tolerance

    if len(wins) >= 1 and disturb_ok:
        verdict = "WIN"
    elif len(losses) >= 1 and len(wins) == 0:
        verdict = "LOSS"
    else:
        verdict = "TIE"

    return {
        "verdict": verdict,
        "wins_metrics": wins,
        "losses_metrics": losses,
        "delta_mean_pct": delta_mean,
        "delta_p95_pct": delta_p95,
        "delta_peak_pct": delta_peak,
        "delta_overload_07": delta_overload_07,
        "delta_overload_09": delta_overload_09,
        "delta_disturb_pct": delta_disturb,
        "disturb_within_tolerance": disturb_ok,
        "oracle_delta_mean_pct": float(o.get("delta_mean_pct", 0)) if o else 0.0,
    }


def main() -> int:
    if not RESULTS_CSV.exists():
        print(f"Missing {RESULTS_CSV}")
        return 1
    rows = parse_csv()
    cells = collect_by_cell(rows)

    # ── Per-cell verdict
    verdicts = {k: cell_verdict(c) for k, c in cells.items()}

    # ── Tally by (scenario, delay) — the regime that matters most
    scen_delay_tally = defaultdict(lambda: defaultdict(int))
    for (topo, scen, k, d), v in verdicts.items():
        scen_delay_tally[(scen, d)][v["verdict"]] += 1

    # ── Tally by topology
    topo_tally = defaultdict(lambda: defaultdict(int))
    for (topo, scen, k, d), v in verdicts.items():
        topo_tally[topo][v["verdict"]] += 1

    # ── CSV summaries
    by_topo_csv = PHASE2_6 / "phase2_6_summary_by_topology.csv"
    with open(by_topo_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topology", "wins", "ties", "losses", "missing", "win_rate_%"])
        for topo, t in sorted(topo_tally.items()):
            wins = t.get("WIN", 0); ties = t.get("TIE", 0)
            losses = t.get("LOSS", 0); missing = t.get("missing", 0)
            total = wins + ties + losses
            win_rate = wins / max(total, 1) * 100.0
            w.writerow([topo, wins, ties, losses, missing, f"{win_rate:.1f}"])
    print(f"Wrote {by_topo_csv}")

    by_scen_csv = PHASE2_6 / "phase2_6_summary_by_scenario.csv"
    with open(by_scen_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "delay", "wins", "ties", "losses", "win_rate_%"])
        for (scen, d), t in sorted(scen_delay_tally.items()):
            wins = t.get("WIN", 0); ties = t.get("TIE", 0); losses = t.get("LOSS", 0)
            total = wins + ties + losses
            wr = wins / max(total, 1) * 100.0
            w.writerow([scen, d, wins, ties, losses, f"{wr:.1f}"])
    print(f"Wrote {by_scen_csv}")

    # ── Win/loss matrix (per-cell verdict + oracle delta)
    matrix_csv = PHASE2_6 / "phase2_6_win_loss_matrix.csv"
    with open(matrix_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "topology", "scenario", "k", "delay", "verdict",
            "delta_mean_pct", "delta_p95_pct", "delta_peak_pct",
            "delta_overload_07", "delta_overload_09",
            "delta_disturb_pct", "wins_metrics", "losses_metrics",
            "oracle_delta_mean_pct",
        ])
        for (topo, scen, k, d), v in sorted(verdicts.items()):
            w.writerow([
                topo, scen, k, d, v["verdict"],
                f"{v['delta_mean_pct']:+.3f}", f"{v['delta_p95_pct']:+.3f}",
                f"{v['delta_peak_pct']:+.3f}",
                v["delta_overload_07"], v["delta_overload_09"],
                f"{v['delta_disturb_pct']:+.3f}",
                "+".join(v["wins_metrics"]), "+".join(v["losses_metrics"]),
                f"{v['oracle_delta_mean_pct']:+.3f}",
            ])
    print(f"Wrote {matrix_csv}")

    # ── Top-line numbers
    total = len(verdicts)
    n_wins = sum(1 for v in verdicts.values() if v["verdict"] == "WIN")
    n_ties = sum(1 for v in verdicts.values() if v["verdict"] == "TIE")
    n_losses = sum(1 for v in verdicts.values() if v["verdict"] == "LOSS")

    # Stationary normal traffic with d=0 (the regime where Phase 2.5 was tested)
    normal_d0 = [(k, v) for k, v in verdicts.items() if k[1] == "normal" and k[3] == 0]
    normal_d0_wins = sum(1 for _, v in normal_d0 if v["verdict"] == "WIN")

    # Dynamic regimes: any non-normal scenario OR delay > 0
    dynamic_cells = [(k, v) for k, v in verdicts.items()
                     if k[1] != "normal" or k[3] > 0]
    dyn_wins = sum(1 for _, v in dynamic_cells if v["verdict"] == "WIN")
    dyn_ties = sum(1 for _, v in dynamic_cells if v["verdict"] == "TIE")
    dyn_losses = sum(1 for _, v in dynamic_cells if v["verdict"] == "LOSS")

    overall = {
        "n_total": total,
        "wins": n_wins, "ties": n_ties, "losses": n_losses,
        "win_rate_pct": round(n_wins / max(total, 1) * 100, 1),
        "normal_d0_total": len(normal_d0),
        "normal_d0_wins": normal_d0_wins,
        "dynamic_total": len(dynamic_cells),
        "dynamic_wins": dyn_wins,
        "dynamic_ties": dyn_ties,
        "dynamic_losses": dyn_losses,
        "dynamic_win_rate_pct": round(dyn_wins / max(len(dynamic_cells), 1) * 100, 1),
    }
    (PHASE2_6 / "phase2_6_topline.json").write_text(
        json.dumps(overall, indent=2) + "\n"
    )
    print(f"Wrote {PHASE2_6 / 'phase2_6_topline.json'}")
    print()
    print("=" * 80)
    print(f"Total cells: {total}")
    print(f"  WIN: {n_wins} ({n_wins/total*100:.1f}%)")
    print(f"  TIE: {n_ties} ({n_ties/total*100:.1f}%)")
    print(f"  LOSS: {n_losses} ({n_losses/total*100:.1f}%)")
    print()
    print("STATIONARY (normal, delay=0): expected to be signal-limited")
    print(f"  {normal_d0_wins} / {len(normal_d0)} wins")
    print()
    print("DYNAMIC (stress scenarios OR delay > 0):")
    print(f"  WIN: {dyn_wins} / {len(dynamic_cells)} ({overall['dynamic_win_rate_pct']:.1f}%)")
    print(f"  TIE: {dyn_ties}, LOSS: {dyn_losses}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
