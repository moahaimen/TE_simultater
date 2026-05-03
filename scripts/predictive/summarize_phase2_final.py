"""Phase 2 Final summarizer.

Reads results/phase2_final/phase2_final_routing_results.csv and produces:
  - phase2_final_summary_by_topology.csv
  - phase2_final_summary_by_scenario.csv
  - phase2_final_ablation_prediction_vs_current.csv  (the key ablation)
  - phase2_final_win_loss_matrix.csv

Win definition: Phase 2 method beats Phase 1 GNN+ Sticky by ≥2% on at
least one of {mean MLU, p95 MLU, peak MLU} AND keeps disturbance within
+5% of phase1 (or strictly reduces it).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_DIR = PROJECT_ROOT / "results" / "phase2_final"
RESULTS_CSV = PHASE_DIR / "phase2_final_routing_results.csv"


def load_rows() -> list[dict]:
    rows = []
    with open(RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def by_cell(rows: list[dict]) -> dict[tuple, dict[str, dict]]:
    out: dict = defaultdict(dict)
    for r in rows:
        key = (r["topology"], r["scenario"], int(r["k"]), int(r["delay"]))
        out[key][r["method"]] = r
    return out


def cell_verdict_vs_phase1(cell: dict, *, abs_threshold: float = 2.0,
                           disturb_tol: float = 5.0) -> dict:
    """Compare each non-phase1 method to phase1.

    Returns dict with verdict per method.
    """
    if "phase1" not in cell:
        return {}
    out = {}
    for m in ["current_apg", "predictive", "oracle"]:
        if m not in cell:
            continue
        d_mean = float(cell[m].get("delta_mean_pct", 0))
        d_p95 = float(cell[m].get("delta_p95_pct", 0))
        d_peak = float(cell[m].get("delta_peak_pct", 0))
        d_disturb = float(cell[m].get("delta_disturb_pct", 0))

        wins = []
        losses = []
        if d_mean <= -abs_threshold: wins.append("mean")
        elif d_mean >= abs_threshold: losses.append("mean")
        if d_p95 <= -abs_threshold:  wins.append("p95")
        elif d_p95 >= abs_threshold: losses.append("p95")
        if d_peak <= -abs_threshold: wins.append("peak")
        elif d_peak >= abs_threshold: losses.append("peak")

        disturb_ok = d_disturb <= disturb_tol
        if wins and disturb_ok:
            verdict = "WIN"
        elif losses and not wins:
            verdict = "LOSS"
        else:
            verdict = "TIE"
        out[m] = {
            "verdict": verdict,
            "wins": wins, "losses": losses,
            "delta_mean_pct": d_mean, "delta_p95_pct": d_p95,
            "delta_peak_pct": d_peak, "delta_disturb_pct": d_disturb,
        }
    return out


def cell_predictive_vs_currapg(cell: dict, *, abs_threshold: float = 2.0) -> dict:
    """Key ablation: does predictive beat current_apg?"""
    if "current_apg" not in cell or "predictive" not in cell:
        return {}
    p = cell["predictive"]
    d_mean = float(p.get("abl_delta_mean_vs_currapg_pct", 0))
    d_p95 = float(p.get("abl_delta_p95_vs_currapg_pct", 0))
    d_peak = float(p.get("abl_delta_peak_vs_currapg_pct", 0))

    wins = []
    losses = []
    if d_mean <= -abs_threshold: wins.append("mean")
    elif d_mean >= abs_threshold: losses.append("mean")
    if d_p95 <= -abs_threshold: wins.append("p95")
    elif d_p95 >= abs_threshold: losses.append("p95")
    if d_peak <= -abs_threshold: wins.append("peak")
    elif d_peak >= abs_threshold: losses.append("peak")

    if wins:
        verdict = "WIN"
    elif losses:
        verdict = "LOSS"
    else:
        verdict = "TIE"
    return {
        "verdict": verdict, "wins": wins, "losses": losses,
        "delta_mean_pct": d_mean, "delta_p95_pct": d_p95, "delta_peak_pct": d_peak,
    }


def main() -> int:
    if not RESULTS_CSV.exists():
        print(f"Missing {RESULTS_CSV}")
        return 1
    rows = load_rows()
    cells = by_cell(rows)

    # ── Per-cell verdicts vs phase1
    verdicts_vs_phase1 = {k: cell_verdict_vs_phase1(c) for k, c in cells.items()}
    # Per-cell verdicts predictive vs current_apg (the key ablation)
    verdicts_abl = {k: cell_predictive_vs_currapg(c) for k, c in cells.items()}

    # ── Tally vs phase1 by (method, scenario, delay)
    tally_vs_phase1 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for k, vmap in verdicts_vs_phase1.items():
        topo, scen, kk, d = k
        for m, v in vmap.items():
            tally_vs_phase1[m][(scen, d)][v["verdict"]] += 1

    # ── Tally ablation by scenario × delay
    abl_tally = defaultdict(lambda: defaultdict(int))
    for k, v in verdicts_abl.items():
        topo, scen, kk, d = k
        if v:
            abl_tally[(scen, d)][v["verdict"]] += 1

    # ── Per-topology tally vs phase1 for predictive method
    topo_tally = defaultdict(lambda: defaultdict(int))
    for k, vmap in verdicts_vs_phase1.items():
        topo, _, _, _ = k
        if "predictive" in vmap:
            topo_tally[topo][vmap["predictive"]["verdict"]] += 1

    # ── CSV outputs
    by_topo_csv = PHASE_DIR / "phase2_final_summary_by_topology.csv"
    with open(by_topo_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topology", "wins_predictive", "ties", "losses",
                    "win_rate_%"])
        for topo, t in sorted(topo_tally.items()):
            wins = t.get("WIN", 0); ties = t.get("TIE", 0); losses = t.get("LOSS", 0)
            total = wins + ties + losses
            wr = wins / max(total, 1) * 100
            w.writerow([topo, wins, ties, losses, f"{wr:.1f}"])
    print(f"Wrote {by_topo_csv}")

    by_scen_csv = PHASE_DIR / "phase2_final_summary_by_scenario.csv"
    with open(by_scen_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "scenario", "delay", "wins", "ties", "losses",
                    "win_rate_%"])
        for m, m_tally in tally_vs_phase1.items():
            for (scen, d), t in sorted(m_tally.items()):
                wins = t.get("WIN", 0); ties = t.get("TIE", 0); losses = t.get("LOSS", 0)
                total = wins + ties + losses
                wr = wins / max(total, 1) * 100
                w.writerow([m, scen, d, wins, ties, losses, f"{wr:.1f}"])
    print(f"Wrote {by_scen_csv}")

    # ── Ablation CSV (predictive vs current_apg)
    abl_csv = PHASE_DIR / "phase2_final_ablation_prediction_vs_current.csv"
    with open(abl_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "delay", "wins_predictive_vs_currapg", "ties",
                    "losses", "win_rate_%"])
        for (scen, d), t in sorted(abl_tally.items()):
            wins = t.get("WIN", 0); ties = t.get("TIE", 0); losses = t.get("LOSS", 0)
            total = wins + ties + losses
            wr = wins / max(total, 1) * 100
            w.writerow([scen, d, wins, ties, losses, f"{wr:.1f}"])
    print(f"Wrote {abl_csv}")

    # ── Win/loss matrix
    matrix_csv = PHASE_DIR / "phase2_final_win_loss_matrix.csv"
    with open(matrix_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topology", "scenario", "k", "delay",
                    "phase1_mlu", "current_apg_mlu", "predictive_mlu", "oracle_mlu",
                    "predictive_vs_phase1", "predictive_vs_currapg",
                    "predictive_delta_mean_pct", "abl_delta_mean_vs_currapg_pct"])
        for k, c in sorted(cells.items()):
            topo, scen, kk, d = k
            row = [topo, scen, kk, d]
            for m in ["phase1", "current_apg", "predictive", "oracle"]:
                row.append(c.get(m, {}).get("mean_mlu", "-"))
            v = verdicts_vs_phase1.get(k, {}).get("predictive", {})
            v_abl = verdicts_abl.get(k, {})
            row.extend([v.get("verdict", "-"), v_abl.get("verdict", "-"),
                        f"{v.get('delta_mean_pct', 0):+.3f}",
                        f"{v_abl.get('delta_mean_pct', 0):+.3f}"])
            w.writerow(row)
    print(f"Wrote {matrix_csv}")

    # ── Top-line numbers
    n_total = len(verdicts_vs_phase1)
    pred_wins = sum(1 for v in verdicts_vs_phase1.values()
                    if v.get("predictive", {}).get("verdict") == "WIN")
    pred_ties = sum(1 for v in verdicts_vs_phase1.values()
                    if v.get("predictive", {}).get("verdict") == "TIE")
    pred_losses = sum(1 for v in verdicts_vs_phase1.values()
                      if v.get("predictive", {}).get("verdict") == "LOSS")
    curr_wins = sum(1 for v in verdicts_vs_phase1.values()
                    if v.get("current_apg", {}).get("verdict") == "WIN")
    abl_wins = sum(1 for v in verdicts_abl.values() if v.get("verdict") == "WIN")
    abl_ties = sum(1 for v in verdicts_abl.values() if v.get("verdict") == "TIE")
    abl_losses = sum(1 for v in verdicts_abl.values() if v.get("verdict") == "LOSS")
    abl_total = abl_wins + abl_ties + abl_losses

    overall = {
        "n_total": n_total,
        "predictive_vs_phase1": {"wins": pred_wins, "ties": pred_ties, "losses": pred_losses},
        "current_apg_vs_phase1_wins": curr_wins,
        "predictive_vs_current_apg": {"wins": abl_wins, "ties": abl_ties, "losses": abl_losses,
                                      "win_rate_%": round(abl_wins / max(abl_total, 1) * 100, 1)},
    }
    (PHASE_DIR / "phase2_final_topline.json").write_text(json.dumps(overall, indent=2) + "\n")
    print(f"Wrote {PHASE_DIR / 'phase2_final_topline.json'}")
    print()
    print("=" * 80)
    print(f"Total cells: {n_total}")
    print()
    print("Phase 2 method WINS vs Phase 1 GNN+ Sticky:")
    print(f"  Predictive:      {pred_wins} W / {pred_ties} T / {pred_losses} L")
    print(f"  Current_apg:     {curr_wins} W (selector-only baseline)")
    print()
    print("KEY ABLATION — does PREDICTION add value over CURRENT-STATE selection?")
    print(f"  Predictive vs Current_apg: {abl_wins} W / {abl_ties} T / {abl_losses} L "
          f"({overall['predictive_vs_current_apg']['win_rate_%']}%)")
    if abl_wins >= 0.5 * abl_total:
        print("  -> Prediction adds clear value")
    elif abl_wins >= 0.2 * abl_total:
        print("  -> Prediction adds partial value (mostly under specific regimes)")
    else:
        print("  -> Prediction adds little value over current-state selection;")
        print("     the gain is from the corrected selector (alt-path-gain).")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
