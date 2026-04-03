#!/usr/bin/env python3
"""Build a Sarah-style MetaGate report with GNN+ as the learned expert."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

OUTPUT_DIR = PROJECT_ROOT / "results" / "dynamic_metagate_gnnplus"
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DOC = OUTPUT_DIR / "MLP_MetaGate_GNNPLUS_Final_Report_ZeroShotObjective_Updated.docx"
AUDIT_MD = OUTPUT_DIR / "report_audit_zeroshot_objective_updated.md"

RESULTS_CSV = OUTPUT_DIR / "metagate_results.csv"
DECISIONS_CSV = OUTPUT_DIR / "metagate_decisions.csv"
SUMMARY_CSV = OUTPUT_DIR / "metagate_summary.csv"
TIMING_CSV = OUTPUT_DIR / "metagate_timing.csv"
CALIB_CSV = OUTPUT_DIR / "calibration_priors.csv"
TRAIN_DIST_CSV = OUTPUT_DIR / "train_oracle_distribution.csv"
TRAINING_SUMMARY_JSON = OUTPUT_DIR / "training_summary.json"
ORACLE_CSV = OUTPUT_DIR / "test_oracle.csv"
FAILURE_RESULTS_CSV = OUTPUT_DIR / "metagate_failure_results.csv"
FAILURE_SUMMARY_CSV = OUTPUT_DIR / "metagate_failure_summary.csv"
FAILURE_CALIB_CSV = OUTPUT_DIR / "metagate_failure_calibration.csv"
ZERO_SHOT_UNSEEN_SUMMARY_CSV = OUTPUT_DIR / "zero_shot_unseen_summary.csv"

TOPOLOGY_ORDER = [
    "abilene",
    "cernet",
    "geant",
    "ebone",
    "sprintlink",
    "tiscali",
    "germany50",
    "vtlwavenet2011",
]

TOPOLOGY_INFO = {
    "abilene": {"dataset": "abilene", "display": "Abilene", "type": "known", "nodes": 12},
    "cernet": {"dataset": "cernet", "display": "CERNET", "type": "known", "nodes": 41},
    "geant": {"dataset": "geant", "display": "GEANT", "type": "known", "nodes": 22},
    "ebone": {"dataset": "ebone", "display": "Ebone", "type": "known", "nodes": 23},
    "sprintlink": {"dataset": "sprintlink", "display": "Sprintlink", "type": "known", "nodes": 44},
    "tiscali": {"dataset": "tiscali", "display": "Tiscali", "type": "known", "nodes": 49},
    "germany50": {"dataset": "germany50", "display": "Germany50", "type": "UNSEEN", "nodes": 50},
    "vtlwavenet2011": {"dataset": "vtlwavenet2011", "display": "VtlWavenet2011", "type": "UNSEEN", "nodes": 92},
}

DATASET_TO_CANON = {
    "abilene": "abilene",
    "cernet": "cernet",
    "geant": "geant",
    "rocketfuel_ebone": "ebone",
    "rocketfuel_sprintlink": "sprintlink",
    "rocketfuel_tiscali": "tiscali",
    "germany50": "germany50",
    "topologyzoo_vtlwavenet2011": "vtlwavenet2011",
}

METHOD_COLORS = {
    "bottleneck": "#2ca02c",
    "topk": "#9467bd",
    "sensitivity": "#8c564b",
    "gnnplus": "#ff7f0e",
}
FAILURE_SCENARIO_ORDER = [
    "single_link_failure",
    "random_link_failure_1",
    "random_link_failure_2",
    "capacity_degradation_50",
    "traffic_spike_2x",
]
FAILURE_SCENARIO_LABELS = {
    "single_link_failure": "Single Link Failure",
    "random_link_failure_1": "Random Link Failure (1)",
    "random_link_failure_2": "Random Link Failure (2)",
    "capacity_degradation_50": "Capacity Degradation (50%)",
    "traffic_spike_2x": "Traffic Spike (2x)",
}


def canon(name: str) -> str:
    return DATASET_TO_CANON.get(str(name).strip().lower(), str(name).strip().lower())


def topo_label(topo: str) -> str:
    info = TOPOLOGY_INFO[topo]
    return f"{info['display']} ({info['nodes']}n)"


def topo_type(topo: str) -> str:
    return TOPOLOGY_INFO[topo]["type"]


def scenario_label(name: str) -> str:
    return FAILURE_SCENARIO_LABELS.get(str(name).strip().lower(), str(name))


def pretty_expert(name: str) -> str:
    mapping = {
        "bottleneck": "Bottleneck",
        "topk": "TopK",
        "sensitivity": "Sensitivity",
        "gnnplus": "GNN+",
    }
    return mapping.get(str(name).strip().lower(), str(name))


def fmt(value, digits: int = 4) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(v):
        return "N/A"
    if abs(v) >= 1e6 or (0 < abs(v) < 1e-4):
        return f"{v:.2e}"
    return f"{v:.{digits}f}"


def fmt_pct(value, digits: int = 2) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(v):
        return "N/A"
    if abs(v) >= 1e6 or (0 < abs(v) < 1e-3):
        return f"{v:+.2e}%"
    return f"{v:+.{digits}f}%"


def style_doc(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = 1.15
    for level, size in [(1, 16), (2, 14), (3, 12)]:
        h = doc.styles[f"Heading {level}"]
        h.font.name = "Arial"
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = RGBColor(0x1A, 0x3C, 0x6E)


def add_table(doc: Document, headers, rows, font_size: int = 10):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, header in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(str(header))
        run.bold = True
        run.font.size = Pt(font_size)
        run.font.name = "Arial"
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(value))
            run.font.size = Pt(font_size)
            run.font.name = "Arial"
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("")


def bold_run(paragraph, text):
    run = paragraph.add_run(text)
    run.bold = True
    return run


def add_image(doc: Document, path: Path, caption: str, width: float = 6.0):
    if not path.exists():
        doc.add_paragraph(f"[Missing image: {path}]")
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def load_inputs():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing evaluation output: {RESULTS_CSV}")
    results = pd.read_csv(RESULTS_CSV)
    decisions = pd.read_csv(DECISIONS_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    timing = pd.read_csv(TIMING_CSV)
    calib = pd.read_csv(CALIB_CSV)
    train_dist = pd.read_csv(TRAIN_DIST_CSV)
    oracle = pd.read_csv(ORACLE_CSV)
    training_summary = json.loads(TRAINING_SUMMARY_JSON.read_text(encoding="utf-8"))
    failure_results = pd.read_csv(FAILURE_RESULTS_CSV) if FAILURE_RESULTS_CSV.exists() else pd.DataFrame()
    failure_summary = pd.read_csv(FAILURE_SUMMARY_CSV) if FAILURE_SUMMARY_CSV.exists() else pd.DataFrame()
    failure_calib = pd.read_csv(FAILURE_CALIB_CSV) if FAILURE_CALIB_CSV.exists() else pd.DataFrame()
    zero_shot_unseen = pd.read_csv(ZERO_SHOT_UNSEEN_SUMMARY_CSV) if ZERO_SHOT_UNSEEN_SUMMARY_CSV.exists() else pd.DataFrame()

    for frame in [results, decisions, summary, timing, calib, train_dist, oracle, failure_results, failure_summary, failure_calib, zero_shot_unseen]:
        if "dataset" in frame.columns:
            frame["topology"] = frame["dataset"].map(canon)

    return results, decisions, summary, timing, calib, train_dist, oracle, training_summary, failure_results, failure_summary, failure_calib, zero_shot_unseen


def plot_selector_distribution(results: pd.DataFrame) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / "selector_distribution_gnnplus.png"
    counts = (
        results.groupby(["topology", "metagate_selector"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=TOPOLOGY_ORDER, fill_value=0)
        .reindex(columns=["bottleneck", "topk", "sensitivity", "gnnplus"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    for method in ["bottleneck", "topk", "sensitivity", "gnnplus"]:
        vals = counts[method].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=METHOD_COLORS[method], label=method.capitalize() if method != "gnnplus" else "GNN+")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([TOPOLOGY_INFO[t]["display"] for t in counts.index], rotation=20, ha="right")
    ax.set_ylabel("Selected Timesteps")
    ax.set_title("MetaGate Expert Selection Distribution (GNN+ version)")
    ax.legend(ncols=4, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_accuracy_gap(summary: pd.DataFrame) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / "accuracy_gap_gnnplus.png"
    df = summary.copy()
    df["gap_pct"] = (df["metagate_mlu"] - df["oracle_mlu"]) / df["oracle_mlu"] * 100.0
    df = df.set_index("topology").reindex(TOPOLOGY_ORDER).reset_index()
    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(10.5, 4.8))
    ax1.bar(x, df["gap_pct"], color="#4c78a8", alpha=0.85, label="Oracle Gap (%)")
    ax1.set_ylabel("Oracle Gap (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([TOPOLOGY_INFO[t]["display"] for t in df["topology"]], rotation=20, ha="right")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax2 = ax1.twinx()
    ax2.plot(x, df["accuracy"] * 100.0, color="#d62728", marker="o", linewidth=2.0, label="Accuracy (%)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)
    ax1.set_title("MetaGate Accuracy and Oracle Gap by Topology (GNN+ version)")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_germany50_selection(results: pd.DataFrame, oracle: pd.DataFrame) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / "germany50_selection_gnnplus.png"
    pred = (
        results[results["topology"] == "germany50"]["metagate_selector"]
        .value_counts()
        .reindex(["bottleneck", "topk", "sensitivity", "gnnplus"], fill_value=0)
    )
    gold = (
        oracle[oracle["topology"] == "germany50"]["oracle_selector"]
        .value_counts()
        .reindex(["bottleneck", "topk", "sensitivity", "gnnplus"], fill_value=0)
    )
    x = np.arange(4)
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.bar(x - width / 2, pred.values, width=width, label="Predicted", color="#1f77b4")
    ax.bar(x + width / 2, gold.values, width=width, label="Oracle", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(["Bottleneck", "TopK", "Sensitivity", "GNN+"])
    ax.set_ylabel("Timesteps")
    ax.set_title("Germany50: Predicted vs Oracle Expert Counts (GNN+ version)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_failure_selector_mix(failure_results: pd.DataFrame) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / "failure_selector_mix_gnnplus.png"
    counts = (
        failure_results.groupby(["scenario", "metagate_selector"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=FAILURE_SCENARIO_ORDER, fill_value=0)
        .reindex(columns=["bottleneck", "topk", "sensitivity", "gnnplus"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    x = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    for method in ["bottleneck", "topk", "sensitivity", "gnnplus"]:
        vals = counts[method].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=METHOD_COLORS[method], label=pretty_expert(method))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([scenario_label(s) for s in counts.index], rotation=20, ha="right")
    ax.set_ylabel("Selected Timesteps")
    ax.set_title("MetaGate Expert Selection Under Failure Scenarios")
    ax.legend(ncols=4, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def build_report():
    results, decisions, summary, timing, calib, train_dist, oracle, training_summary, failure_results, failure_summary, failure_calib, zero_shot_unseen = load_inputs()

    summary = summary.set_index("topology").reindex(TOPOLOGY_ORDER).reset_index()
    timing = timing.set_index("topology").reindex(TOPOLOGY_ORDER).reset_index()
    calib = calib.set_index("topology").reindex(TOPOLOGY_ORDER).reset_index()

    selector_plot = plot_selector_distribution(results)
    acc_gap_plot = plot_accuracy_gap(summary)
    germany_plot = plot_germany50_selection(results, oracle)
    failure_plot = plot_failure_selector_mix(failure_results) if not failure_results.empty else None

    overall_acc = float(training_summary["overall_test_accuracy"]) * 100.0
    known_acc = float(training_summary["known_test_accuracy"]) * 100.0
    unseen_acc = float(training_summary["unseen_test_accuracy"]) * 100.0
    mean_decision = float(results["t_decision_ms"].mean())
    mean_total = float(results["t_total_ms"].mean())

    germany_summary = summary[summary["topology"] == "germany50"].iloc[0]
    germany_pred = results[results["topology"] == "germany50"]["metagate_selector"].value_counts()
    germany_oracle = oracle[oracle["topology"] == "germany50"]["oracle_selector"].value_counts()
    germany_gnnplus_pct = 100.0 * germany_pred.get("gnnplus", 0) / max(len(results[results["topology"] == "germany50"]), 1)

    doc = Document()
    style_doc(doc)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("MLP Meta-Gate with Few-Shot Bayesian Calibration")
    run.bold = True
    run.font.size = Pt(20)
    run.font.color.rgb = RGBColor(0x1A, 0x3C, 0x6E)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Final Evaluation Report (GNN+ Expert Version)")
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    subtitle2 = doc.add_paragraph()
    subtitle2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle2.add_run(
        "Architecture: 4-Expert Selection via MLP Gate with Per-Topology Bayesian Calibration\n"
        "Expert pool: Bottleneck, TopK, Sensitivity, GNN+"
    )
    run.font.size = Pt(11)
    run.italic = True

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(
        "We present an MLP-based meta-gate that dynamically selects among four traffic engineering "
        "expert strategies (Bottleneck, Top-K by Demand, Sensitivity Analysis, and GNN+-based selection) "
        "on a per-traffic-matrix basis. The gate is a 3-layer MLP (128-128-64-4) with BatchNorm and "
        "dropout, trained on pooled oracle labels from 6 known topologies. Germany50 and VtlWavenet2011 "
        "are held out entirely from gate-weight training."
    )
    doc.add_paragraph(
        "A lightweight few-shot Bayesian calibration phase adapts the gate to each target topology "
        "at deployment time by running 10 validation traffic matrices through all 4 experts, counting "
        "which expert achieves the lowest MLU, and fusing this topology-specific prior with the MLP "
        "softmax output. No gradient updates occur during calibration."
    )
    p = doc.add_paragraph()
    bold_run(p, "Key results: ")
    p.add_run(
        f"Calibrated Germany50 (unseen): GNN+ selected {germany_gnnplus_pct:.1f}% of timesteps, "
        f"oracle gap {((germany_summary['metagate_mlu'] - germany_summary['oracle_mlu']) / germany_summary['oracle_mlu']) * 100.0:+.2f}%, "
        f"accuracy {germany_summary['accuracy'] * 100.0:.1f}%. Overall accuracy {overall_acc:.1f}% across 8 topologies. "
        f"Mean selector decision overhead {mean_decision:.1f} ms; mean end-to-end time (including LP) {mean_total:.1f} ms."
    )

    doc.add_heading("1. Contributions", level=1)
    items = [
        "Base objective preserved: zero-shot generalization to unseen topologies, evaluated without any target-topology calibration.",
        "Additional contribution: lightweight few-shot Bayesian calibration before inference, used as a deployment-time enhancement rather than weight retraining.",
        "Honest MLP Meta-Gate architecture with a real 4-expert pool and per-timestep learned routing-strategy selection.",
        "GNN+ replaces the Original GNN as the learned expert inside the MetaGate evaluation itself.",
        "Unified gate training on known topologies only, with Germany50 and VtlWavenet2011 held out from gate-weight training.",
        "Per-component timing breakdown (BN + TopK + Sens + GNN+ + features + MLP + LP) reported per timestep.",
    ]
    for idx, item in enumerate(items, 1):
        p = doc.add_paragraph()
        bold_run(p, f"{idx}. ")
        p.add_run(item)

    doc.add_heading("2. Architecture", level=1)
    doc.add_heading("2.1 Overview", level=2)
    steps = [
        "Run all 4 experts (Bottleneck, TopK, Sensitivity, GNN+) to produce 4 candidate OD-pair selections, each of size K_crit = 40.",
        "Extract a 49-dimensional feature vector from the TM, expert outputs, GNN+ diagnostics, ECMP baseline, and topology metrics.",
        "The MLP gate predicts which expert will achieve the lowest MLU.",
        "If Bayesian calibration is active, fuse MLP softmax with topology-specific prior.",
        "Route the predicted expert's selection through LP optimization to obtain the final MLU.",
    ]
    for step in steps:
        doc.add_paragraph(step, style="List Number")

    doc.add_heading("2.2 MLP Gate Architecture", level=2)
    doc.add_paragraph(
        "3-layer MLP: Linear(49, 128) -> BatchNorm -> ReLU -> Dropout(0.3) -> "
        "Linear(128, 128) -> BatchNorm -> ReLU -> Dropout(0.3) -> "
        "Linear(128, 64) -> ReLU -> Linear(64, 4). "
        "Output: softmax over 4 expert classes {0=Bottleneck, 1=TopK, 2=Sensitivity, 3=GNN+}."
    )

    doc.add_heading("2.3 Four Experts", level=2)
    add_table(
        doc,
        ["Expert", "Selection Strategy"],
        [
            ["Bottleneck", "Selects K_crit OD pairs traversing the most-utilized link under ECMP routing"],
            ["Top-K by Demand", "Selects K_crit OD pairs with the highest traffic demand"],
            ["Sensitivity", "Selects K_crit OD pairs whose rerouting has the largest impact on max link utilization"],
            ["GNN+ (GraphSAGE variant)", "Learned flow selector using enriched node/edge/OD features with fixed K=40"],
        ],
    )

    doc.add_heading("3. Training Details", level=1)
    total_samples = int(sum(train_dist["samples"]))
    train_rows = []
    for topo in TOPOLOGY_ORDER:
        row = train_dist[train_dist["topology"] == topo]
        if row.empty:
            continue
        rec = row.iloc[0]
        train_rows.append([
            TOPOLOGY_INFO[topo]["display"],
            str(TOPOLOGY_INFO[topo]["nodes"]),
            str(int(rec["samples"])),
            str(int(rec["bottleneck"])),
            str(int(rec["topk"])),
            str(int(rec["sensitivity"])),
            str(int(rec["gnnplus"])),
        ])
    add_table(doc, ["Topology", "Nodes", "Samples", "BN Oracle", "TopK Oracle", "Sens Oracle", "GNN+ Oracle"], train_rows)
    doc.add_paragraph(
        "The gate is trained on pooled data from 6 known topologies only. "
        "Germany50 and VtlWavenet2011 are completely excluded from training."
    )
    overall_counts = training_summary["train_class_counts"]
    doc.add_paragraph(
        "Pooled training distribution: "
        f"Bottleneck={overall_counts['bottleneck']}/{total_samples}, "
        f"TopK={overall_counts['topk']}/{total_samples}, "
        f"Sensitivity={overall_counts['sensitivity']}/{total_samples}, "
        f"GNN+={overall_counts['gnnplus']}/{total_samples}."
    )
    add_table(
        doc,
        ["Parameter", "Value"],
        [
            ["Hidden dimension", str(training_summary["metagate_config"]["hidden_dim"])],
            ["Dropout", str(training_summary["metagate_config"]["dropout"])],
            ["Learning rate", str(training_summary["metagate_config"]["learning_rate"])],
            ["Epochs", str(training_summary["metagate_config"]["num_epochs"])],
            ["Batch size", str(training_summary["metagate_config"]["batch_size"])],
            ["Train accuracy", f"{float(training_summary['train_accuracy']) * 100.0:.1f}%"],
            ["Validation accuracy", f"{float(training_summary['val_accuracy']) * 100.0:.1f}%"],
            ["GNN+ checkpoint", training_summary["gnnplus_checkpoint"]],
        ],
    )

    doc.add_heading("4. Few-Shot Bayesian Calibration", level=1)
    doc.add_paragraph(
        "Before evaluating on any topology, a calibration phase runs 10 validation traffic matrices "
        "through all 4 experts with LP optimization, counting which expert achieves the lowest MLU. "
        "This produces a topology-specific prior that is fused with the MLP softmax at inference time."
    )
    calib_rows = []
    for topo in TOPOLOGY_ORDER:
        row = calib[calib["topology"] == topo]
        if row.empty:
            continue
        rec = row.iloc[0]
        calib_rows.append([
            TOPOLOGY_INFO[topo]["display"],
            topo_type(topo),
            fmt(rec["bottleneck_prior"], 2),
            fmt(rec["topk_prior"], 2),
            fmt(rec["sensitivity_prior"], 2),
            fmt(rec["gnnplus_prior"], 2),
        ])
    add_table(doc, ["Topology", "Type", "BN Prior", "TopK Prior", "Sens Prior", "GNN+ Prior"], calib_rows)
    doc.add_heading("4.1 Objective Framing: Zero-Shot Baseline and Few-Shot Extension", level=2)
    doc.add_paragraph(
        "The original project objective is zero-shot generalization to unseen topologies. To keep that objective "
        "scientifically valid, the report distinguishes between a zero-shot baseline and a stronger deployment mode "
        "that adds a lightweight few-shot Bayesian calibration prior before inference."
    )
    doc.add_paragraph(
        "In this framing, the base MetaGate weights are always trained only on the 6 known topologies. The zero-shot "
        "setting performs direct inference on Germany50 and VtlWavenet2011 with no target-topology adaptation. The "
        "few-shot calibrated setting uses 10 validation traffic matrices from the target topology to estimate a prior, "
        "but it still performs no gradient-based fine-tuning or weight updates."
    )
    if not zero_shot_unseen.empty:
        comparison_rows = []
        for topo in ["germany50", "vtlwavenet2011"]:
            zrow = zero_shot_unseen[zero_shot_unseen["topology"] == topo]
            crow = summary[summary["topology"] == topo]
            if zrow.empty or crow.empty:
                continue
            zrec = zrow.iloc[0]
            crec = crow.iloc[0]
            zgap = (float(zrec["metagate_mlu"]) - float(zrec["oracle_mlu"])) / float(zrec["oracle_mlu"]) * 100.0
            cgap = (float(crec["metagate_mlu"]) - float(crec["oracle_mlu"])) / float(crec["oracle_mlu"]) * 100.0
            comparison_rows.append([
                TOPOLOGY_INFO[topo]["display"],
                f"{float(zrec['accuracy']) * 100.0:.1f}%",
                fmt_pct(zgap),
                f"{float(crec['accuracy']) * 100.0:.1f}%",
                fmt_pct(cgap),
            ])
        add_table(
            doc,
            ["Unseen Topology", "Zero-Shot Accuracy", "Zero-Shot Gap", "Calibrated Accuracy", "Calibrated Gap"],
            comparison_rows,
        )
        doc.add_paragraph(
            "This table is included so the report can keep zero-shot generalization as the base objective, while also "
            "showing that the few-shot calibration module is an additional contribution that improves deployment on the unseen topologies."
        )

    doc.add_heading("5. Expert Selection, Accuracy, and Training Notes", level=1)
    expert_rows = []
    mix_rows = []
    for topo in TOPOLOGY_ORDER:
        row = summary[summary["topology"] == topo]
        if row.empty:
            continue
        rec = row.iloc[0]
        sub = results[results["topology"] == topo]
        counts = sub["metagate_selector"].value_counts()
        total = max(len(sub), 1)
        dominant_name = counts.idxmax() if not counts.empty else "n/a"
        dominant_pct = 100.0 * counts.max() / total if not counts.empty else 0.0
        switches = int((sub["metagate_selector"].values[1:] != sub["metagate_selector"].values[:-1]).sum()) if len(sub) > 1 else 0
        gap = ((rec["metagate_mlu"] - rec["oracle_mlu"]) / rec["oracle_mlu"]) * 100.0
        expert_rows.append([
            topo_label(topo),
            f"{switches}/{max(len(sub) - 1, 1)}",
            f"{pretty_expert(dominant_name)} ({dominant_pct:.0f}%)",
            f"{rec['accuracy'] * 100.0:.1f}%",
            f"{gap:+.2f}%",
        ])
        mix_rows.append([
            topo_label(topo),
            f"{100.0 * counts.get('bottleneck', 0) / total:.0f}%",
            f"{100.0 * counts.get('topk', 0) / total:.0f}%",
            f"{100.0 * counts.get('sensitivity', 0) / total:.0f}%",
            f"{100.0 * counts.get('gnnplus', 0) / total:.0f}%",
        ])
    add_table(doc, ["Topology", "Switches", "Dominant Expert", "Accuracy", "Oracle Gap"], expert_rows)
    add_table(doc, ["Topology", "BN%", "TopK%", "Sens%", "GNN+%"], mix_rows)
    add_image(doc, selector_plot, "MetaGate expert selection distribution with GNN+ as the learned expert.")
    add_image(doc, acc_gap_plot, "MetaGate accuracy and oracle gap by topology (GNN+ version).")

    doc.add_heading("6. Results -- All 8 Topologies", level=1)
    result_rows = []
    for topo in TOPOLOGY_ORDER:
        row = summary[summary["topology"] == topo]
        if row.empty:
            continue
        rec = row.iloc[0]
        sub = results[results["topology"] == topo]
        counts = sub["metagate_selector"].value_counts()
        gnnplus_pct = 100.0 * counts.get("gnnplus", 0) / max(len(sub), 1)
        switches = int((sub["metagate_selector"].values[1:] != sub["metagate_selector"].values[:-1]).sum()) if len(sub) > 1 else 0
        gap = ((rec["metagate_mlu"] - rec["oracle_mlu"]) / rec["oracle_mlu"]) * 100.0
        result_rows.append([
            TOPOLOGY_INFO[topo]["display"],
            topo_type(topo),
            f"{rec['accuracy'] * 100.0:.1f}%",
            fmt(rec["metagate_mlu"]),
            fmt(rec["oracle_mlu"]),
            f"{gap:+.2f}%",
            f"{gnnplus_pct:.0f}%",
            str(switches),
        ])
    add_table(doc, ["Topology", "Type", "Accuracy", "MetaGate MLU", "Oracle MLU", "Gap", "GNN+%", "Switches"], result_rows)
    p = doc.add_paragraph()
    bold_run(p, "Overall accuracy: ")
    p.add_run(f"{overall_acc:.1f}%. ")
    bold_run(p, "Known topology accuracy: ")
    p.add_run(f"{known_acc:.1f}%. ")
    bold_run(p, "Unseen topology accuracy: ")
    p.add_run(f"{unseen_acc:.1f}%.")

    if not failure_results.empty and not failure_summary.empty:
        doc.add_heading("7. Failure Scenario Evaluation", level=1)
        doc.add_paragraph(
            "Unlike the standalone packet-SDN failure tables, this section runs the integrated MetaGate+GNN+ "
            "chooser under the failure scenarios themselves. For each failed timestep, the gate predicts an "
            "expert, and the report logs whether that chosen expert changes relative to normal conditions."
        )
        scenario_rows = []
        normal_dominant = {
            topo: results[results["topology"] == topo]["metagate_selector"].value_counts().idxmax()
            for topo in TOPOLOGY_ORDER
            if not results[results["topology"] == topo].empty
        }
        for scenario in FAILURE_SCENARIO_ORDER:
            sub = failure_results[failure_results["scenario"] == scenario]
            if sub.empty:
                continue
            counts = sub["metagate_selector"].value_counts()
            total = max(len(sub), 1)
            ssum = failure_summary[failure_summary["scenario"] == scenario]
            changed = 0
            compared = 0
            for topo in TOPOLOGY_ORDER:
                topo_sub = sub[sub["topology"] == topo]
                if topo_sub.empty or topo not in normal_dominant:
                    continue
                compared += 1
                failure_dom = topo_sub["metagate_selector"].value_counts().idxmax()
                if failure_dom != normal_dominant[topo]:
                    changed += 1
            scenario_rows.append([
                scenario_label(scenario),
                f"{ssum['accuracy'].mean() * 100.0:.1f}%",
                fmt_pct(ssum["metagate_vs_oracle_gap_pct"].mean()),
                f"{100.0 * counts.get('bottleneck', 0) / total:.0f}%",
                f"{100.0 * counts.get('topk', 0) / total:.0f}%",
                f"{100.0 * counts.get('sensitivity', 0) / total:.0f}%",
                f"{100.0 * counts.get('gnnplus', 0) / total:.0f}%",
                f"{changed}/{compared}",
            ])
        add_table(doc, ["Scenario", "Accuracy", "Oracle Gap", "BN%", "TopK%", "Sens%", "GNN+%", "Changed Topologies"], scenario_rows)
        collapsed = []
        for scenario in FAILURE_SCENARIO_ORDER:
            sub = failure_results[failure_results["scenario"] == scenario]
            if sub.empty:
                continue
            counts = sub["metagate_selector"].value_counts()
            if len(counts) == 1 and counts.index[0] == "gnnplus":
                collapsed.append(scenario_label(scenario))
        if collapsed:
            doc.add_paragraph(
                "Important observation: the harshest link-failure scenarios drive the gate to a degenerate "
                f"GNN+-only choice on {', '.join(collapsed)}. This is a real out-of-distribution behavior in "
                "the current failure evaluation, not a formatting artifact."
            )
        if failure_plot is not None:
            add_image(doc, failure_plot, "MetaGate expert mix across all failure timesteps (GNN+ integrated).")

        for idx, scenario in enumerate(FAILURE_SCENARIO_ORDER, start=1):
            sub = failure_results[failure_results["scenario"] == scenario]
            if sub.empty:
                continue
            doc.add_heading(f"7.{idx} {scenario_label(scenario)}", level=2)
            rows = []
            for topo in TOPOLOGY_ORDER:
                topo_sub = sub[sub["topology"] == topo]
                if topo_sub.empty:
                    continue
                counts = topo_sub["metagate_selector"].value_counts()
                total = max(len(topo_sub), 1)
                dominant = counts.idxmax()
                gap = ((topo_sub["metagate_mlu"].mean() - topo_sub["oracle_mlu"].mean()) / topo_sub["oracle_mlu"].mean()) * 100.0
                rows.append([
                    topo_label(topo),
                    pretty_expert(dominant),
                    f"{100.0 * counts.get('bottleneck', 0) / total:.0f}%",
                    f"{100.0 * counts.get('topk', 0) / total:.0f}%",
                    f"{100.0 * counts.get('sensitivity', 0) / total:.0f}%",
                    f"{100.0 * counts.get('gnnplus', 0) / total:.0f}%",
                    f"{topo_sub['correct'].mean() * 100.0:.1f}%",
                    fmt_pct(gap),
                ])
            add_table(doc, ["Topology", "Dominant", "BN%", "TopK%", "Sens%", "GNN+%", "Accuracy", "Gap"], rows)

    doc.add_heading("8. Germany50 (Unseen) -- Deep Dive", level=1)
    doc.add_paragraph(
        "Germany50 is the clearest unseen-topology case in the current GNN+ integrated MetaGate run. "
        "The table and figure below show the predicted selection mix and the oracle distribution "
        "when GNN+ replaces the Original GNN in the expert pool."
    )
    germany_rows = [
        ["Predicted Bottleneck selections", str(int(germany_pred.get("bottleneck", 0)))],
        ["Predicted GNN+ selections", str(int(germany_pred.get("gnnplus", 0)))],
        ["Oracle Bottleneck wins", str(int(germany_oracle.get("bottleneck", 0)))],
        ["Oracle GNN+ wins", str(int(germany_oracle.get("gnnplus", 0)))],
        ["MetaGate MLU", fmt(germany_summary["metagate_mlu"])],
        ["Oracle MLU", fmt(germany_summary["oracle_mlu"])],
        ["Oracle gap", f"{((germany_summary['metagate_mlu'] - germany_summary['oracle_mlu']) / germany_summary['oracle_mlu']) * 100.0:+.2f}%"],
        ["Accuracy", f"{germany_summary['accuracy'] * 100.0:.1f}%"],
    ]
    add_table(doc, ["Metric", "Value"], germany_rows)
    add_image(doc, germany_plot, "Germany50 predicted vs oracle expert counts with GNN+ integrated.")

    doc.add_heading("9. VtlWavenet2011 (Unseen) -- Analysis", level=1)
    vtl = summary[summary["topology"] == "vtlwavenet2011"].iloc[0]
    vtl_pred = results[results["topology"] == "vtlwavenet2011"]["metagate_selector"].value_counts()
    doc.add_paragraph(
        f"VtlWavenet2011 accuracy is {vtl['accuracy'] * 100.0:.1f}% with oracle gap "
        f"{((vtl['metagate_mlu'] - vtl['oracle_mlu']) / vtl['oracle_mlu']) * 100.0:+.2f}%. "
        f"Predicted selections: Bottleneck={int(vtl_pred.get('bottleneck', 0))}, "
        f"GNN+={int(vtl_pred.get('gnnplus', 0))}, TopK={int(vtl_pred.get('topk', 0))}, "
        f"Sensitivity={int(vtl_pred.get('sensitivity', 0))}."
    )

    doc.add_heading("10. Per-Component Timing Breakdown", level=1)
    timing_rows = []
    for topo in TOPOLOGY_ORDER:
        row = timing[timing["topology"] == topo]
        if row.empty:
            continue
        rec = row.iloc[0]
        timing_rows.append([
            TOPOLOGY_INFO[topo]["display"],
            fmt(rec["t_bn_ms"], 1),
            fmt(rec["t_topk_ms"], 1),
            fmt(rec["t_sens_ms"], 1),
            fmt(rec["t_gnnplus_ms"], 1),
            fmt(rec["t_features_ms"], 2),
            fmt(rec["t_mlp_ms"], 3),
            fmt(rec["t_lp_ms"], 1),
            fmt(rec["t_total_ms"], 1),
        ])
    add_table(doc, ["Topology", "BN ms", "TopK ms", "Sens ms", "GNN+ ms", "Feat ms", "MLP ms", "LP ms", "Total ms"], timing_rows)
    doc.add_paragraph(
        f"Mean decision time (selector overhead before LP): {mean_decision:.1f} ms. "
        f"Mean total time (end-to-end): {mean_total:.1f} ms."
    )

    doc.add_heading("11. Limitations (Honest Assessment)", level=1)
    limitations = [
        "This report validates MetaGate with GNN+ as the learned expert, but it does not include a separate Stable MetaGate extension.",
        "Paper baselines such as FlexDATE, FlexEntry, and ERODRL are still unavailable in the current runnable repository.",
        "The calibration phase uses 10 validation traffic matrices from the target topology, so the framing remains zero-shot gate training with few-shot calibration rather than pure zero-shot deployment.",
        "The reported results are limited to the current runnable bundle and should not be mixed numerically with earlier report branches unless the scope is stated explicitly.",
    ]
    for item in limitations:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("12. Exact Method Description for Thesis", level=1)
    doc.add_paragraph(
        "We use a two-stage expert-selection framework for traffic engineering. An MLP MetaGate "
        "(3-layer, 128-128-64-4 with BatchNorm and dropout 0.3) selects among four routing experts: "
        "Bottleneck, Top-K by Demand, Sensitivity Analysis, and GNN+-based selection, all operating "
        "with K_crit = 40 critical flows. Oracle labels are generated by running all four experts "
        "through LP optimization and selecting the expert with minimum MLU for each traffic matrix. "
        "The gate is trained on pooled oracle labels from 6 known topologies only. Before deployment "
        "on each topology, a lightweight few-shot calibration phase evaluates 10 validation traffic matrices "
        "through all four experts to build a topology-specific Bayesian prior, which is fused with the "
        "MLP softmax during test-time inference. Germany50 and VtlWavenet2011 are held out from gate-weight "
        "training and are used as unseen-topology tests."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_DOC)

    audit_lines = [
        "# MetaGate GNN+ Report Audit",
        "",
        f"- Output report: `{OUTPUT_DOC.relative_to(PROJECT_ROOT)}`",
        "- This report is built only from `results/dynamic_metagate_gnnplus/*` outputs.",
        "- The learned expert in this report is GNN+, not the old Original GNN.",
        "- Selector percentages and accuracy values come from the new integrated MetaGate+GNN+ evaluation.",
        f"- Failure section included: {'yes' if not failure_results.empty else 'no'}",
    ]
    AUDIT_MD.write_text("\n".join(audit_lines), encoding="utf-8")
    print(f"Report saved to {OUTPUT_DOC}")


if __name__ == "__main__":
    build_report()
