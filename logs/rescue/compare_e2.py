"""Compare Check #2 rescue bundle against Task A baseline.

Usage:
    python logs/rescue/compare_e2.py

Check #2 hypothesis:
    Dropping BOTTLENECK_MOE_FLOOR from 0.10 -> 0.00 removes the hard
    10% bottleneck-weight cap from GNN+'s gate, letting GNN+ commit to
    its own selection when it disagrees with bottleneck.

Reads packet_sdn_summary.csv from baseline and rescue bundles, computes
per-topology deltas for the GNN+ method only, and emits PASS/FAIL/NULL.

Verdict rule (pre-registered, different from Check #3 because this is an
MLU-quality lever, not a latency lever):
  - PASS  if max |seen MLU rel%| <= 1.0  AND  max |unseen MLU rel%| <= 2.0
          AND mean overall MLU rel% <= -0.30 (at least 0.3% average MLU
          improvement across all topologies).
  - NULL  if all |rel%| < 0.3 across every metric and topology.
  - FAIL  otherwise.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "gnnplus_8topo_stability_taskA" / "packet_sdn_summary.csv"
RESCUE = ROOT / "results" / "rescue_e2_moe_floor_0" / "packet_sdn_summary.csv"

SEEN = {"abilene", "cernet", "ebone", "geant", "sprintlink", "tiscali"}
UNSEEN = {"germany50", "vtlwavenet2011"}

METRICS = ["mean_mlu", "mean_disturbance", "decision_time_ms", "do_no_harm_fallback_rate"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[(df["method"] == "gnnplus") & (df["scenario"] == "normal")].copy()


def main() -> int:
    if not BASE.exists():
        print(f"MISSING baseline: {BASE}")
        return 2
    if not RESCUE.exists():
        print(f"MISSING rescue: {RESCUE}")
        return 2

    b = load(BASE).set_index("topology")
    r = load(RESCUE).set_index("topology")

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
        rows.append(row)
    out = pd.DataFrame(rows)

    print("=" * 80)
    print("PER-TOPOLOGY DELTAS  (rescue - baseline) / baseline")
    print("=" * 80)
    for _, r_ in out.iterrows():
        tag = r_["status"]
        print(f"\n[{tag:>6}] {r_['topology']}")
        for m in METRICS:
            base = r_[f"{m}__base"]
            resc = r_[f"{m}__rescue"]
            rel = r_[f"{m}__rel_pct"]
            arrow = "->"
            print(f"    {m:>28}: {base:12.6f} {arrow} {resc:12.6f}   ({rel:+.3f}%)")

    # verdict
    seen_rows = out[out["status"] == "known"]
    unseen_rows = out[out["status"] == "unseen"]

    seen_mlu_rel_abs_max = seen_rows["mean_mlu__rel_pct"].abs().max() if len(seen_rows) else 0.0
    unseen_mlu_rel_abs_max = unseen_rows["mean_mlu__rel_pct"].abs().max() if len(unseen_rows) else 0.0
    overall_mlu_rel_mean = out["mean_mlu__rel_pct"].mean()

    seen_mlu_rel_mean = seen_rows["mean_mlu__rel_pct"].mean() if len(seen_rows) else 0.0
    unseen_mlu_rel_mean = unseen_rows["mean_mlu__rel_pct"].mean() if len(unseen_rows) else 0.0

    print("\n" + "=" * 80)
    print("VERDICT INPUTS")
    print("=" * 80)
    print(f"  max |seen MLU rel%|   = {seen_mlu_rel_abs_max:+.3f}%   (guardrail: <=1.0%)")
    print(f"  max |unseen MLU rel%| = {unseen_mlu_rel_abs_max:+.3f}%   (guardrail: <=2.0%)")
    print(f"  mean seen MLU rel%    = {seen_mlu_rel_mean:+.3f}%")
    print(f"  mean unseen MLU rel%  = {unseen_mlu_rel_mean:+.3f}%")
    print(f"  mean overall MLU rel% = {overall_mlu_rel_mean:+.3f}%   (target:   <=-0.30%)")

    # NULL: everything tiny
    all_deltas = out[[f"{m}__rel_pct" for m in METRICS]].abs().to_numpy()
    all_tiny = (all_deltas < 0.3).all() if all_deltas.size else False

    print("\n" + "=" * 80)
    if all_tiny:
        verdict = "NULL"
        reason = "All relative deltas <0.3%; change is within noise."
    elif seen_mlu_rel_abs_max > 1.0:
        verdict = "FAIL"
        reason = f"Seen MLU moved {seen_mlu_rel_abs_max:.2f}% (>1% guardrail)."
    elif unseen_mlu_rel_abs_max > 2.0:
        verdict = "FAIL"
        reason = f"Unseen MLU moved {unseen_mlu_rel_abs_max:.2f}% (>2% guardrail)."
    elif overall_mlu_rel_mean <= -0.30:
        verdict = "PASS"
        reason = f"Mean overall MLU improved by {-overall_mlu_rel_mean:.2f}% with guardrails held."
    else:
        verdict = "FAIL"
        reason = f"Mean overall MLU moved {overall_mlu_rel_mean:+.2f}% (<0.30% improvement target)."
    print(f"VERDICT: {verdict}")
    print(f"REASON : {reason}")
    print("=" * 80)

    out.to_csv(ROOT / "results" / "rescue_e2_moe_floor_0" / "delta_table.csv", index=False)
    return 0 if verdict == "PASS" else (1 if verdict == "FAIL" else 3)


if __name__ == "__main__":
    raise SystemExit(main())
