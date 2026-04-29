"""Compare Phase 1 bundle against Task A baseline on the disturbance bar.

Usage:
    python logs/rescue/compare_p1.py [--rescue-tag TAG]

Phase 1 hypothesis:
    Inference-only knobs (sticky-selection post-filter, continuity-bonus
    env override, disturbance-aware do-no-harm tiebreak) close the
    disturbance gap to FlexEntry on at least 3 of the 5 topologies where
    Task A currently loses (cernet, ebone, sprintlink, tiscali, germany50)
    without regressing MLU above guardrails.

Verdict rule (pre-registered, stricter MLU guardrails than Check #5
because this is a disturbance lever):
  - PASS  if max |seen MLU rel%| <= 0.5
          AND max |unseen MLU rel%| <= 1.0
          AND beats FlexEntry on disturbance for >= 3 of
              {cernet, ebone, sprintlink, tiscali, germany50}
          AND keeps disturbance wins on {abilene, geant, vtlwavenet2011}.
  - FAIL  otherwise (MLU busted or disturbance bar not met).

FlexEntry disturbance reference (from
  results/requirements_compliant_eval/table_external_baselines.csv):

    abilene        0.091
    cernet         0.018
    ebone          0.039
    geant          0.134
    sprintlink     0.013
    tiscali        0.021
    germany50      0.114
    vtlwavenet2011 0.007
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SEEN = {"abilene", "cernet", "ebone", "geant", "sprintlink", "tiscali"}
UNSEEN = {"germany50", "vtlwavenet2011"}

FLEXENTRY_DISTURB = {
    "abilene": 0.091,
    "cernet": 0.018,
    "ebone": 0.039,
    "geant": 0.134,
    "sprintlink": 0.013,
    "tiscali": 0.021,
    "germany50": 0.114,
    "vtlwavenet2011": 0.007,
}

LOSING_TOPOS = {"cernet", "ebone", "sprintlink", "tiscali", "germany50"}
WINNING_TOPOS = {"abilene", "geant", "vtlwavenet2011"}

METRICS = ["mean_mlu", "mean_disturbance", "decision_time_ms", "do_no_harm_fallback_rate"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[(df["method"] == "gnnplus") & (df["scenario"] == "normal")].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rescue-tag",
        default="rescue_p1_sticky_combined",
        help="Subdir under results/ holding the Phase 1 bundle",
    )
    parser.add_argument(
        "--baseline-tag",
        default="gnnplus_8topo_stability_taskA",
        help="Subdir under results/ holding the Task A baseline bundle",
    )
    args = parser.parse_args()

    base_path = ROOT / "results" / args.baseline_tag / "packet_sdn_summary.csv"
    resc_path = ROOT / "results" / args.rescue_tag / "packet_sdn_summary.csv"

    if not base_path.exists():
        print(f"MISSING baseline: {base_path}")
        return 2
    if not resc_path.exists():
        print(f"MISSING rescue: {resc_path}")
        return 2

    b = load(base_path).set_index("topology")
    r = load(resc_path).set_index("topology")

    shared = sorted(set(b.index) & set(r.index))
    rows = []
    for topo in shared:
        row = {"topology": topo, "status": b.loc[topo, "status"]}
        for m in METRICS:
            base_v = float(b.loc[topo, m])
            rescue_v = float(r.loc[topo, m])
            delta = rescue_v - base_v
            rel = (delta / base_v) if abs(base_v) > 1e-12 else float("nan")
            row[f"{m}__base"] = base_v
            row[f"{m}__rescue"] = rescue_v
            row[f"{m}__delta"] = delta
            row[f"{m}__rel_pct"] = rel * 100.0 if rel == rel else float("nan")
        row["flexentry_disturb"] = FLEXENTRY_DISTURB.get(topo, float("nan"))
        row["beats_flexentry"] = (
            row["mean_disturbance__rescue"] <= row["flexentry_disturb"]
            if row["flexentry_disturb"] == row["flexentry_disturb"]
            else False
        )
        rows.append(row)
    out = pd.DataFrame(rows)

    print("=" * 96)
    print("PER-TOPOLOGY DELTAS  (rescue - baseline) / baseline  +  disturbance vs FlexEntry")
    print("=" * 96)
    for _, r_ in out.iterrows():
        tag = r_["status"]
        print(f"\n[{tag:>6}] {r_['topology']}")
        for m in METRICS:
            base = r_[f"{m}__base"]
            resc = r_[f"{m}__rescue"]
            rel = r_[f"{m}__rel_pct"]
            print(f"    {m:>28}: {base:12.6f} -> {resc:12.6f}   ({rel:+.3f}%)")
        fe = r_["flexentry_disturb"]
        beats = r_["beats_flexentry"]
        mark = "BEATS" if beats else "misses"
        disturb_rescue = r_["mean_disturbance__rescue"]
        print(
            f"    {'disturbance vs FlexEntry':>28}: "
            f"{disturb_rescue:12.6f} vs {fe:12.6f}   ({mark})"
        )

    seen_rows = out[out["status"] == "known"]
    unseen_rows = out[out["status"] == "unseen"]

    # Only flag MLU *regressions* (positive = worse); improvements are fine.
    seen_mlu_rel_abs_max = seen_rows["mean_mlu__rel_pct"].clip(lower=0).max() if len(seen_rows) else 0.0
    unseen_mlu_rel_abs_max = unseen_rows["mean_mlu__rel_pct"].clip(lower=0).max() if len(unseen_rows) else 0.0

    losing_wins = {
        topo: bool(out.loc[out["topology"] == topo, "beats_flexentry"].iloc[0])
        for topo in LOSING_TOPOS
        if topo in out["topology"].values
    }
    winning_keeps = {
        topo: bool(out.loc[out["topology"] == topo, "beats_flexentry"].iloc[0])
        for topo in WINNING_TOPOS
        if topo in out["topology"].values
    }
    losing_win_count = sum(int(v) for v in losing_wins.values())

    print("\n" + "=" * 96)
    print("VERDICT INPUTS")
    print("=" * 96)
    print(f"  max |seen MLU rel%|    = {seen_mlu_rel_abs_max:+.3f}%   (guardrail: <=0.5%)")
    print(f"  max |unseen MLU rel%|  = {unseen_mlu_rel_abs_max:+.3f}%   (guardrail: <=1.0%)")
    print(f"  FlexEntry wins on previously-losing topos: {losing_win_count} of 5")
    for topo in sorted(LOSING_TOPOS):
        mark = "BEATS" if losing_wins.get(topo, False) else "misses"
        print(f"      {topo:<18} -> {mark}")
    print(f"  Keeps disturbance win on previously-winning topos:")
    for topo in sorted(WINNING_TOPOS):
        mark = "BEATS" if winning_keeps.get(topo, False) else "REGRESSED"
        print(f"      {topo:<18} -> {mark}")

    print("\n" + "=" * 96)
    if seen_mlu_rel_abs_max > 0.5:
        verdict = "FAIL"
        reason = f"Seen MLU moved {seen_mlu_rel_abs_max:.3f}% (>0.5% guardrail)."
    elif unseen_mlu_rel_abs_max > 1.0:
        verdict = "FAIL"
        reason = f"Unseen MLU moved {unseen_mlu_rel_abs_max:.3f}% (>1.0% guardrail)."
    elif any(not v for v in winning_keeps.values()):
        lost = [t for t, v in winning_keeps.items() if not v]
        verdict = "FAIL"
        reason = f"Disturbance regressed on previously-winning topo(s): {lost}"
    elif losing_win_count >= 3:
        verdict = "PASS"
        reason = f"Beats FlexEntry on {losing_win_count} of 5 previously-losing topos; MLU guardrails held."
    else:
        verdict = "FAIL"
        reason = f"Only {losing_win_count} of 5 previously-losing topos cross FlexEntry bar; target is >=3."
    print(f"VERDICT: {verdict}")
    print(f"REASON : {reason}")
    print("=" * 96)

    out_path = ROOT / "results" / args.rescue_tag / "delta_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nDelta table written to: {out_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
