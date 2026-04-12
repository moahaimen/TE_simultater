#!/usr/bin/env python3
"""Build a compact Bottleneck-vs-GNN+ failure comparison table for seminar slides."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "results" / "professor_clean_gnnplus_zeroshot" / "seminar_failure_comparison" / "failure_old_vs_new_full.csv"
OUT_DIR = PROJECT_ROOT / "results" / "professor_clean_gnnplus_zeroshot" / "seminar_failure_comparison"

PICKS = [
    ("Abilene", "Random Link Failure (2)"),
    ("Sprintlink", "Random Link Failure (1)"),
    ("Germany50", "Single Link Failure"),
    ("Germany50", "Random Link Failure (2)"),
    ("VtlWavenet2011", "Random Link Failure (2)"),
]


def fmt_num(v: float) -> str:
    if abs(v) >= 1e4 or (0 < abs(v) < 1e-2):
        return f"{v:.2e}"
    return f"{v:.4f}".rstrip("0").rstrip(".")


def fmt_ms(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:.1f}"
    return f"{v:.2f}"


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topo, scenario in PICKS:
        sub = df[(df["Topology"] == topo) & (df["Scenario"] == scenario) & (df["Method"].isin(["Bottleneck", "GNN+"]))].copy()
        if len(sub) != 2:
            continue
        bn = sub[sub["Method"] == "Bottleneck"].iloc[0]
        gp = sub[sub["Method"] == "GNN+"].iloc[0]
        winner = "Bottleneck" if bn["New MLU"] <= gp["New MLU"] else "GNN+"
        rows.append(
            {
                "Topology": topo,
                "Scenario": scenario,
                "Old BN MLU": fmt_num(float(bn["Old MLU"])),
                "Old GNN+ MLU": fmt_num(float(gp["Old MLU"])),
                "New BN MLU": fmt_num(float(bn["New MLU"])),
                "New GNN+ MLU": fmt_num(float(gp["New MLU"])),
                "New BN Rec (ms)": fmt_ms(float(bn["Old Recovery (ms)"] if pd.isna(bn["New Recovery (ms)"]) else bn["New Recovery (ms)"])),
                "New GNN+ Rec (ms)": fmt_ms(float(gp["Old Recovery (ms)"] if pd.isna(gp["New Recovery (ms)"]) else gp["New Recovery (ms)"])),
                "Corrected Winner": winner,
            }
        )
    return pd.DataFrame(rows)


def write_markdown(table_df: pd.DataFrame) -> Path:
    caption = (
        "Caption: After rebuilding post-failure path libraries and enforcing surviving-path "
        "validity, the corrected failure pipeline removes the catastrophic stale-path MLUs; "
        "the table below compares only Bottleneck and GNN+ on the key seminar cases."
    )
    out = OUT_DIR / "failure_bn_vs_gnnplus_slide_table.md"
    lines = [
        "# Bottleneck vs GNN+ Failure Comparison",
        "",
        caption,
        "",
        table_df.to_markdown(index=False),
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_csv(table_df: pd.DataFrame) -> Path:
    out = OUT_DIR / "failure_bn_vs_gnnplus_slide_table.csv"
    table_df.to_csv(out, index=False)
    return out


def write_png(table_df: pd.DataFrame) -> Path:
    out = OUT_DIR / "failure_bn_vs_gnnplus_slide_table.png"
    fig_h = 1.3 + 0.55 * len(table_df)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        else:
            if c == len(table_df.columns) - 1:
                winner = table_df.iloc[r - 1, c]
                cell.set_facecolor("#d9ead3" if winner == "Bottleneck" else "#d0e0e3")
            elif r % 2 == 1:
                cell.set_facecolor("#f7f7f7")
        cell.set_edgecolor("#666666")

    fig.suptitle("Failure Comparison: Bottleneck vs GNN+", fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.01,
        0.02,
        "Corrected pipeline rebuilds surviving paths after link failures; values are real old-vs-new results.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    df = pd.read_csv(SRC)
    table_df = build_table(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = write_markdown(table_df)
    csv = write_csv(table_df)
    png = write_png(table_df)
    print(md)
    print(csv)
    print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
