const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, LevelFormat
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "results", "gnn_plus");
const PLOTS = path.join(ROOT, "plots");
const OUT = path.join(ROOT, "GNN_Plus_Screening_Report.docx");

// --- Data ---
const summary = [
  { topo: "Abilene", steps: 75, mlu_o: 0.0546, mlu_p: 0.0546, pr_o: 0.000, pr_p: 0.000,
    time_o: 2.04, time_p: 2.93, dist_o: 0.093, dist_p: 0.091, k_mean: 39, k_min: 39, k_max: 39,
    wins_p: 27, wins_o: 27, ties: 21, improve: "~0%" },
  { topo: "GEANT", steps: 75, mlu_o: 0.1615, mlu_p: 0.1630, pr_o: 0.80, pr_p: 1.70,
    time_o: 3.95, time_p: 6.85, dist_o: 0.120, dist_p: 0.098, k_mean: 39, k_min: 39, k_max: 39,
    wins_p: 15, wins_o: 53, ties: 7, improve: "-0.94%" },
  { topo: "Germany50", steps: 44, mlu_o: 18.941, mlu_p: 19.270, pr_o: -1.65, pr_p: 0.25,
    time_o: 16.45, time_p: 29.22, dist_o: 0.268, dist_p: 0.174, k_mean: 39, k_min: 39, k_max: 39,
    wins_p: 2, wins_o: 42, ties: 0, improve: "-1.74%" },
  { topo: "AGGREGATE", steps: 194, mlu_o: 4.379, mlu_p: 4.455, pr_o: -0.07, pr_p: 0.71,
    time_o: 6.05, time_p: 10.41, dist_o: 0.143, dist_p: 0.113, k_mean: 39, k_min: 39, k_max: 39,
    wins_p: 44, wins_o: 122, ties: 28, improve: "-1.72%" },
];

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function hdrCell(text, w) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: { fill: "2E4057", type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, font: "Arial", size: 18, color: "FFFFFF" })] })],
  });
}

function cell(text, w, opts = {}) {
  const shade = opts.shade || undefined;
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: shade ? { fill: shade, type: ShadingType.CLEAR } : undefined,
    margins: cellMargins,
    children: [new Paragraph({ alignment: opts.align || AlignmentType.CENTER,
      children: [new TextRun({ text: String(text), font: "Arial", size: 18,
        bold: opts.bold || false, color: opts.color || "000000" })] })],
  });
}

function heading(text, level) {
  return new Paragraph({ heading: level, spacing: { before: 300, after: 150 },
    children: [new TextRun({ text, font: "Arial", bold: true })] });
}

function para(text, opts = {}) {
  return new Paragraph({ spacing: { after: 120 }, alignment: opts.align || AlignmentType.LEFT,
    children: [new TextRun({ text, font: "Arial", size: 22, ...opts })] });
}

function img(name, w, h) {
  const fp = path.join(PLOTS, name);
  if (!fs.existsSync(fp)) return para(`[Image not found: ${name}]`);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 100 },
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(fp),
      transformation: { width: w, height: h },
      altText: { title: name, description: name, name },
    })] });
}

// ---------- Build document ----------
const children = [];

// Title
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
  children: [new TextRun({ text: "GNN+ Screening Experiment Report", font: "Arial", size: 36, bold: true })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: "AI-Driven Traffic Routing with MLP Meta-Gate", font: "Arial", size: 24, color: "555555" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: "Date: 2026-03-30  |  Branch: gnn-plus-extension  |  Type: Screening (not full ablation)", font: "Arial", size: 20, color: "888888" })] }));

// Disclaimer
children.push(new Paragraph({ spacing: { after: 200 },
  shading: { fill: "FFF3CD", type: ShadingType.CLEAR },
  children: [new TextRun({ text: "IMPORTANT: ", font: "Arial", size: 20, bold: true, color: "856404" }),
    new TextRun({ text: "This is a screening experiment testing the combined GNN+ upgrade (richer features + dynamic K). It does NOT isolate the individual effects of features vs. dynamic K. The original GNN baseline is unchanged.", font: "Arial", size: 20, color: "856404" })] }));

// Section 1: Motivation
children.push(heading("1. Motivation", HeadingLevel.HEADING_1));
children.push(para("The original GNN expert in our MetaGate pipeline uses 34 effective input features (with 4 placeholder zeros in node features). It operates with a fixed K=40 critical flow budget regardless of topology or traffic conditions. This screening experiment tests whether enriching the GNN input features and enabling dynamic K prediction can improve routing quality."));

// Section 2: Old GNN Limitations
children.push(heading("2. Original GNN Limitations", HeadingLevel.HEADING_1));
children.push(para("Identified weaknesses in the current GNN expert:"));

const limitations = [
  "4 wasted node feature slots (zero padding at indices 12-15)",
  "No ECMP per-OD demand contribution to node features",
  "No path overlap / shared bottleneck features between OD pairs",
  "No temporal features (traffic change between t and t-1)",
  "No hop count / path length in OD features (only weighted cost)",
  "No source/destination local congestion in OD scoring",
  "K_pred head exists but is dead code (force_default_k=True at inference)",
];

limitations.forEach(t => {
  children.push(new Paragraph({ spacing: { after: 60 },
    indent: { left: 360 },
    children: [new TextRun({ text: "\u2022 ", font: "Arial", size: 20 }),
      new TextRun({ text: t, font: "Arial", size: 20 })] }));
});

// Section 3: New Feature Design
children.push(heading("3. GNN+ Feature Design", HeadingLevel.HEADING_1));
children.push(para("GNN+ enriches all three feature groups while maintaining backward compatibility:"));

// Node features table
children.push(heading("3.1 Node Features [V, 16]", HeadingLevel.HEADING_2));
children.push(para("Indices 0-11 unchanged. Indices 12-15 replaced with real features:"));
const nfW = [800, 3400, 2500, 2660];
children.push(new Table({
  width: { size: 9360, type: WidthType.DXA }, columnWidths: nfW,
  rows: [
    new TableRow({ children: [hdrCell("Idx", nfW[0]), hdrCell("Feature", nfW[1]), hdrCell("Source", nfW[2]), hdrCell("Replaces", nfW[3])] }),
    new TableRow({ children: [cell("12", nfW[0]), cell("ECMP demand through node", nfW[1], {align: AlignmentType.LEFT}), cell("TM + OD pairs", nfW[2]), cell("zero", nfW[3], {shade:"FFECB3"})] }),
    new TableRow({ children: [cell("13", nfW[0]), cell("Congested neighbor fraction", nfW[1], {align: AlignmentType.LEFT}), cell("Adjacency + util", nfW[2]), cell("zero", nfW[3], {shade:"FFECB3"})] }),
    new TableRow({ children: [cell("14", nfW[0]), cell("Max residual capacity", nfW[1], {align: AlignmentType.LEFT}), cell("cap - load", nfW[2]), cell("zero", nfW[3], {shade:"FFECB3"})] }),
    new TableRow({ children: [cell("15", nfW[0]), cell("Clustering coefficient", nfW[1], {align: AlignmentType.LEFT}), cell("Topology triangles", nfW[2]), cell("zero", nfW[3], {shade:"FFECB3"})] }),
  ]
}));

// Edge features table
children.push(heading("3.2 Edge Features [E, 8 -> 12]", HeadingLevel.HEADING_2));
const efW = [800, 3500, 2500, 2560];
children.push(new Table({
  width: { size: 9360, type: WidthType.DXA }, columnWidths: efW,
  rows: [
    new TableRow({ children: [hdrCell("Idx", efW[0]), hdrCell("Feature", efW[1]), hdrCell("Source", efW[2]), hdrCell("Status", efW[3])] }),
    new TableRow({ children: [cell("0-7", efW[0]), cell("Original 8 features", efW[1], {align: AlignmentType.LEFT}), cell("Unchanged", efW[2]), cell("Kept", efW[3])] }),
    new TableRow({ children: [cell("8", efW[0]), cell("OD paths sharing edge", efW[1], {align: AlignmentType.LEFT}), cell("Path library", efW[2]), cell("NEW", efW[3], {shade:"C8E6C9", bold:true})] }),
    new TableRow({ children: [cell("9", efW[0]), cell("Residual capacity (abs)", efW[1], {align: AlignmentType.LEFT}), cell("cap - load", efW[2]), cell("NEW", efW[3], {shade:"C8E6C9", bold:true})] }),
    new TableRow({ children: [cell("10", efW[0]), cell("Load change ratio (t/t-1)", efW[1], {align: AlignmentType.LEFT}), cell("Temporal", efW[2]), cell("NEW", efW[3], {shade:"C8E6C9", bold:true})] }),
    new TableRow({ children: [cell("11", efW[0]), cell("Is bottleneck indicator", efW[1], {align: AlignmentType.LEFT}), cell("Path analysis", efW[2]), cell("NEW", efW[3], {shade:"C8E6C9", bold:true})] }),
  ]
}));

// OD features table
children.push(heading("3.3 OD Features [num_od, 10 -> 18]", HeadingLevel.HEADING_2));
const ofW = [800, 3500, 2500, 2560];
children.push(new Table({
  width: { size: 9360, type: WidthType.DXA }, columnWidths: ofW,
  rows: [
    new TableRow({ children: [hdrCell("Idx", ofW[0]), hdrCell("Feature", ofW[1]), hdrCell("Source", ofW[2]), hdrCell("Status", ofW[3])] }),
    new TableRow({ children: [cell("0-9", ofW[0]), cell("Original 10 features", ofW[1], {align: AlignmentType.LEFT}), cell("Unchanged", ofW[2]), cell("Kept", ofW[3])] }),
    new TableRow({ children: [cell("10", ofW[0]), cell("Hop count (normalized)", ofW[1], {align: AlignmentType.LEFT}), cell("Path library", ofW[2]), cell("NEW", ofW[3], {shade:"C8E6C9", bold:true})] }),
    new TableRow({ children: [cell("11", ofW[0]), cell("Demand change ratio", ofW[1], {align: AlignmentType.LEFT}), cell("Temporal (t/t-1)", ofW[2]), cell("NEW", ofW[3], {shade:"C8E6C9", bold:true})] }),
    new TableRow({ children: [cell("12", ofW[0]), cell("Source congestion", ofW[1], {align: AlignmentType.LEFT}), cell("Node max_util_out", ofW[2]), cell("NEW", ofW[3], {shade:"C8E6C9", bold:true})] }),
    new TableRow({ children: [cell("13", ofW[0]), cell("Destination congestion", ofW[1], {align: AlignmentType.LEFT}), cell("Node max_util_in", ofW[2]), cell("NEW", ofW[3], {shade:"C8E6C9", bold:true})] }),
    new TableRow({ children: [cell("14", ofW[0]), cell("Path overlap score", ofW[1], {align: AlignmentType.LEFT}), cell("Edge sharing count", ofW[2]), cell("NEW", ofW[3], {shade:"C8E6C9", bold:true})] }),
    new TableRow({ children: [cell("15", ofW[0]), cell("ECMP congestion contribution", ofW[1], {align: AlignmentType.LEFT}), cell("demand * bottleneck", ofW[2]), cell("NEW", ofW[3], {shade:"C8E6C9", bold:true})] }),
    new TableRow({ children: [cell("16", ofW[0]), cell("Alternative path headroom", ofW[1], {align: AlignmentType.LEFT}), cell("Non-best paths", ofW[2]), cell("NEW", ofW[3], {shade:"C8E6C9", bold:true})] }),
    new TableRow({ children: [cell("17", ofW[0]), cell("Demand x hop (normalized)", ofW[1], {align: AlignmentType.LEFT}), cell("Resource proxy", ofW[2]), cell("NEW", ofW[3], {shade:"C8E6C9", bold:true})] }),
  ]
}));

children.push(para("Total effective features: 16 + 12 + 18 = 46 (up from 34)."));

// Section 4: Dynamic Bounded K
children.push(heading("4. Dynamic Bounded K Design", HeadingLevel.HEADING_1));
children.push(para("Rule: K = max(K_min, min(K_pred, 40)) if K_pred is not None else 40"));
children.push(para("K_min = 15. Justification: Smallest topology (Abilene) has 132 OD pairs; 15 is ~11% coverage, the minimum meaningful selection for LP optimization. K_max = 40 matches all prior experiments for fair comparison."));
children.push(para("Implementation: GNN+ model uses force_default_k=False, with k_head output clamped to [15, 40]."));

// Section 5: Training
children.push(heading("5. Training Summary", HeadingLevel.HEADING_1));
const trW = [3600, 5760];
children.push(new Table({
  width: { size: 9360, type: WidthType.DXA }, columnWidths: trW,
  rows: [
    new TableRow({ children: [hdrCell("Parameter", trW[0]), hdrCell("Value", trW[1])] }),
    new TableRow({ children: [cell("Topologies", trW[0], {align:AlignmentType.LEFT}), cell("Abilene, GEANT, Germany50", trW[1])] }),
    new TableRow({ children: [cell("Train / Val samples", trW[0], {align:AlignmentType.LEFT}), cell("120 / 45", trW[1])] }),
    new TableRow({ children: [cell("Max epochs", trW[0], {align:AlignmentType.LEFT}), cell("30 (early stop patience=8)", trW[1])] }),
    new TableRow({ children: [cell("Best epoch", trW[0], {align:AlignmentType.LEFT}), cell("28", trW[1])] }),
    new TableRow({ children: [cell("Best val loss", trW[0], {align:AlignmentType.LEFT}), cell("2.815", trW[1])] }),
    new TableRow({ children: [cell("Final alpha (residual weight)", trW[0], {align:AlignmentType.LEFT}), cell("0.308", trW[1])] }),
    new TableRow({ children: [cell("Val selection overlap", trW[0], {align:AlignmentType.LEFT}), cell("0.790 (79.0% Jaccard with oracle)", trW[1])] }),
    new TableRow({ children: [cell("Training time", trW[0], {align:AlignmentType.LEFT}), cell("72.3 seconds", trW[1])] }),
    new TableRow({ children: [cell("Learning rate", trW[0], {align:AlignmentType.LEFT}), cell("5e-4 (AdamW + cosine annealing)", trW[1])] }),
    new TableRow({ children: [cell("K_pred at convergence", trW[0], {align:AlignmentType.LEFT}), cell("39 (nearly fixed at upper bound)", trW[1])] }),
  ]
}));

// Section 6: Results
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(heading("6. Screening Results", HeadingLevel.HEADING_1));

// Main results table
children.push(heading("6.1 Summary Table", HeadingLevel.HEADING_2));
const rW = [1200, 1000, 1000, 800, 800, 800, 800, 960, 1000];
children.push(new Table({
  width: { size: 9360, type: WidthType.DXA }, columnWidths: rW,
  rows: [
    new TableRow({ children: [
      hdrCell("Topology", rW[0]), hdrCell("MLU Orig", rW[1]), hdrCell("MLU GNN+", rW[2]),
      hdrCell("PR Orig", rW[3]), hdrCell("PR GNN+", rW[4]),
      hdrCell("Dist Orig", rW[5]), hdrCell("Dist GNN+", rW[6]),
      hdrCell("GNN+ Wins", rW[7]), hdrCell("MLU Change", rW[8]),
    ]}),
    ...summary.map(s => {
      const isAgg = s.topo === "AGGREGATE";
      const shade = isAgg ? "E3F2FD" : undefined;
      return new TableRow({ children: [
        cell(s.topo, rW[0], { bold: isAgg, shade }),
        cell(s.mlu_o.toFixed(4), rW[1], { shade }),
        cell(s.mlu_p.toFixed(4), rW[2], { shade }),
        cell(s.pr_o.toFixed(2) + "%", rW[3], { shade }),
        cell(s.pr_p.toFixed(2) + "%", rW[4], { shade }),
        cell(s.dist_o.toFixed(3), rW[5], { shade }),
        cell(s.dist_p.toFixed(3), rW[6], { shade, color: s.dist_p < s.dist_o ? "2E7D32" : "000000" }),
        cell(`${s.wins_p}/${s.steps}`, rW[7], { shade }),
        cell(s.improve, rW[8], { shade, color: s.improve.startsWith("-") ? "C62828" : "2E7D32" }),
      ]});
    }),
  ]
}));

children.push(para(""));
children.push(para("Key finding: GNN+ has WORSE MLU on GEANT (-0.94%) and Germany50 (-1.74%), but BETTER disturbance on all topologies (21% aggregate improvement).", { bold: true }));

// K Distribution
children.push(heading("6.2 Dynamic K Distribution", HeadingLevel.HEADING_2));
const kW = [1500, 1300, 1300, 1300, 1300, 2660];
children.push(new Table({
  width: { size: 9360, type: WidthType.DXA }, columnWidths: kW,
  rows: [
    new TableRow({ children: [hdrCell("Topology", kW[0]), hdrCell("K Mean", kW[1]), hdrCell("K Min", kW[2]), hdrCell("K Max", kW[3]), hdrCell("K Std", kW[4]), hdrCell("Note", kW[5])] }),
    new TableRow({ children: [cell("Abilene", kW[0]), cell("39", kW[1]), cell("39", kW[2]), cell("39", kW[3]), cell("0", kW[4]), cell("Constant", kW[5])] }),
    new TableRow({ children: [cell("GEANT", kW[0]), cell("39", kW[1]), cell("39", kW[2]), cell("39", kW[3]), cell("0", kW[4]), cell("Constant", kW[5])] }),
    new TableRow({ children: [cell("Germany50", kW[0]), cell("39", kW[1]), cell("39", kW[2]), cell("39", kW[3]), cell("0", kW[4]), cell("Constant", kW[5])] }),
  ]
}));

children.push(para(""));
children.push(para("Observation: K_pred converged to 39 for all topologies. The k_head learned to output near-maximum, meaning the model found no benefit from reducing K below 40. The dynamic K mechanism is effectively inactive.", { bold: true }));

// Execution Time
children.push(heading("6.3 Execution Time", HeadingLevel.HEADING_2));
const etW = [2340, 2340, 2340, 2340];
children.push(new Table({
  width: { size: 9360, type: WidthType.DXA }, columnWidths: etW,
  rows: [
    new TableRow({ children: [hdrCell("Topology", etW[0]), hdrCell("Orig GNN (ms)", etW[1]), hdrCell("GNN+ (ms)", etW[2]), hdrCell("Overhead", etW[3])] }),
    new TableRow({ children: [cell("Abilene", etW[0]), cell("2.04", etW[1]), cell("2.93", etW[2]), cell("+44%", etW[3], {color: "E65100"})] }),
    new TableRow({ children: [cell("GEANT", etW[0]), cell("3.95", etW[1]), cell("6.85", etW[2]), cell("+73%", etW[3], {color: "E65100"})] }),
    new TableRow({ children: [cell("Germany50", etW[0]), cell("16.45", etW[1]), cell("29.22", etW[2]), cell("+78%", etW[3], {color: "E65100"})] }),
  ]
}));
children.push(para("GNN+ feature building is ~44-78% slower due to path overlap computation and clustering coefficient calculation. Both remain sub-30ms, well within real-time requirements."));

// Section 7: Plots
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(heading("7. Comparison Plots", HeadingLevel.HEADING_1));

children.push(heading("7.1 MLU CDF: Original GNN vs GNN+", HeadingLevel.HEADING_2));
children.push(img("mlu_cdf_comparison.png", 620, 180));
children.push(para("Figure 1: CDF of MLU across test timesteps. Original GNN dominates on GEANT and Germany50."));

children.push(heading("7.2 Dynamic K Distribution", HeadingLevel.HEADING_2));
children.push(img("k_distribution.png", 620, 180));
children.push(para("Figure 2: K_pred is constant at 39 across all topologies. Dynamic K mechanism did not activate."));

children.push(heading("7.3 Mean MLU Bar Chart", HeadingLevel.HEADING_2));
children.push(img("mean_mlu_bar_chart.png", 500, 280));
children.push(para("Figure 3: Per-topology mean MLU comparison. GNN+ is slightly worse on GEANT and Germany50."));

children.push(heading("7.4 Disturbance CDF", HeadingLevel.HEADING_2));
children.push(img("disturbance_cdf_comparison.png", 620, 180));
children.push(para("Figure 4: GNN+ achieves lower disturbance on all topologies, especially Germany50 (0.174 vs 0.268)."));

// Section 8: Comparison with Original GNN
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(heading("8. Comparison with Original GNN", HeadingLevel.HEADING_1));

const compItems = [
  ["MLU Quality", "Original GNN is BETTER. GNN+ is 0.94-1.74% worse on GEANT and Germany50. On Abilene, they are tied.", "NEGATIVE"],
  ["Disturbance", "GNN+ is BETTER. 21% lower aggregate disturbance (0.113 vs 0.143). Especially strong on Germany50: 35% reduction.", "POSITIVE"],
  ["Dynamic K", "NOT ACTIVE. K_pred converges to 39 for all topologies. The model learned that maximum K is optimal, which is consistent with the existing fixed K=40 design.", "NEUTRAL"],
  ["Execution Time", "GNN+ is 44-78% slower due to richer feature computation. Still sub-30ms, not a practical concern.", "NEUTRAL"],
  ["Training", "Similar convergence: 28-29 epochs, 72s total. Val overlap 0.79 (comparable to original).", "NEUTRAL"],
];

compItems.forEach(([aspect, detail, verdict]) => {
  const color = verdict === "POSITIVE" ? "2E7D32" : verdict === "NEGATIVE" ? "C62828" : "555555";
  children.push(new Paragraph({ spacing: { after: 120 },
    children: [
      new TextRun({ text: `${aspect}: `, font: "Arial", size: 22, bold: true }),
      new TextRun({ text: detail, font: "Arial", size: 22, color }),
    ] }));
});

// Section 9: Honest Limitations
children.push(heading("9. Honest Limitations", HeadingLevel.HEADING_1));

const honestItems = [
  "This is a SCREENING experiment, not a full ablation. We cannot isolate whether the MLU regression is caused by the richer features, the dynamic K, or the retraining from scratch.",
  "GNN+ was trained from random initialization with only 120 samples across 3 topologies. The original GNN was trained on 240 samples across 6 topologies with REINFORCE fine-tuning. This is NOT a fair comparison of feature quality.",
  "The original GNN also had REINFORCE fine-tuning (LP-in-the-loop), which GNN+ did not receive. This stage can significantly improve MLU quality beyond oracle-label supervision.",
  "K_pred stuck at 39 suggests the k_head loss weight (0.01) may be too low, or that fixed K=40 is genuinely optimal for these topologies and traffic patterns.",
  "Only 3 topologies tested (2 known + 1 unseen). More generalization topologies would be needed for definitive conclusions.",
  "Disturbance improvement may simply be due to different learned score distributions, not necessarily better features.",
];

honestItems.forEach(t => {
  children.push(new Paragraph({ spacing: { after: 80 },
    indent: { left: 360 },
    children: [new TextRun({ text: "\u2022 ", font: "Arial", size: 20 }),
      new TextRun({ text: t, font: "Arial", size: 20 })] }));
});

// Section 10: Recommendation
children.push(heading("10. Should GNN+ Be Inserted into MetaGate?", HeadingLevel.HEADING_1));

children.push(para("Verdict: NOT YET.", { bold: true, size: 24 }));
children.push(para(""));
children.push(para("The screening shows that the combined GNN+ upgrade (richer features + dynamic K) does not improve MLU quality over the original GNN. The original GNN wins 63% of timesteps overall."));
children.push(para(""));
children.push(para("However, the disturbance improvement (21% lower) is interesting and warrants further investigation. Before inserting GNN+ into MetaGate, the following steps are recommended:"));
children.push(para(""));

const nextSteps = [
  "Run a proper ablation: test features-only (fixed K=40) vs dynamic-K-only (original features) to isolate effects.",
  "Apply REINFORCE fine-tuning to GNN+ to match the original GNN training protocol.",
  "Train on all 6 known topologies (not just 3) for fair comparison.",
  "Investigate why K_pred converges to 39 and whether the k_head needs architectural changes.",
  "If features-only ablation shows improvement, integrate the enriched features into the original GNN without dynamic K.",
];

nextSteps.forEach((t, i) => {
  children.push(new Paragraph({ spacing: { after: 80 },
    indent: { left: 360 },
    children: [new TextRun({ text: `${i+1}. `, font: "Arial", size: 20, bold: true }),
      new TextRun({ text: t, font: "Arial", size: 20 })] }));
});

// Build doc
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 180, after: 140 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "GNN+ Screening Report", font: "Arial", size: 16, color: "999999", italics: true })]
      })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Page ", font: "Arial", size: 16 }),
          new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16 })]
      })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log(`Written: ${OUT} (${buf.length} bytes)`);
});
