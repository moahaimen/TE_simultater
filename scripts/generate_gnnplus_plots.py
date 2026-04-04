#!/usr/bin/env python3
"""Generate plots and DOCX report for GNN+ Packet-Level SDN Evaluation."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from collections import defaultdict

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

RESULTS_DIR = Path("results/gnnplus_packet_sdn_report")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Load results
df_normal = pd.read_csv(RESULTS_DIR / "packet_sdn_summary.csv")
df_failure = pd.read_csv(RESULTS_DIR / "packet_sdn_failure.csv")

# Create MLU comparison plot
fig, ax = plt.subplots(figsize=(14, 6))
topologies = df_normal['topology'].unique()
methods = ['ecmp', 'bottleneck', 'gnn', 'gnnplus']
x = np.arange(len(topologies))
width = 0.2

for i, method in enumerate(methods):
    mlu_values = []
    for topo in topologies:
        subset = df_normal[(df_normal['topology'] == topo) & (df_normal['method'] == method)]
        if len(subset) > 0:
            mlu_values.append(subset['mean_mlu'].values[0])
        else:
            mlu_values.append(np.nan)
    ax.bar(x + i * width, mlu_values, width, label=method.upper())

ax.set_xlabel('Topology')
ax.set_ylabel('Mean MLU')
ax.set_title('MLU Comparison Across Topologies (Normal Scenario)')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(topologies, rotation=45, ha='right')
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'mlu_comparison_normal.png', dpi=150, bbox_inches='tight')
plt.close()

# Create Throughput comparison plot
fig, ax = plt.subplots(figsize=(14, 6))
for i, method in enumerate(methods):
    tp_values = []
    for topo in topologies:
        subset = df_normal[(df_normal['topology'] == topo) & (df_normal['method'] == method)]
        if len(subset) > 0:
            tp_values.append(subset['throughput'].values[0])
        else:
            tp_values.append(np.nan)
    ax.bar(x + i * width, tp_values, width, label=method.upper())

ax.set_xlabel('Topology')
ax.set_ylabel('Throughput')
ax.set_title('Throughput Comparison Across Topologies (Normal Scenario)')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(topologies, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'throughput_comparison_normal.png', dpi=150, bbox_inches='tight')
plt.close()

# Create Disturbance comparison plot
fig, ax = plt.subplots(figsize=(14, 6))
for i, method in enumerate(methods):
    dist_values = []
    for topo in topologies:
        subset = df_normal[(df_normal['topology'] == topo) & (df_normal['method'] == method)]
        if len(subset) > 0:
            dist_values.append(subset['mean_disturbance'].values[0])
        else:
            dist_values.append(np.nan)
    ax.bar(x + i * width, dist_values, width, label=method.upper())

ax.set_xlabel('Topology')
ax.set_ylabel('Mean Disturbance')
ax.set_title('Routing Disturbance Comparison Across Topologies')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(topologies, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'disturbance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Create Decision Time comparison plot
fig, ax = plt.subplots(figsize=(14, 6))
for i, method in enumerate(methods):
    time_values = []
    for topo in topologies:
        subset = df_normal[(df_normal['topology'] == topo) & (df_normal['method'] == method)]
        if len(subset) > 0:
            time_values.append(subset['decision_time_ms'].values[0])
        else:
            time_values.append(np.nan)
    ax.bar(x + i * width, time_values, width, label=method.upper())

ax.set_xlabel('Topology')
ax.set_ylabel('Decision Time (ms)')
ax.set_title('Decision Time Comparison Across Topologies')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(topologies, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'decision_time_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Create Failure Recovery plot
fig, ax = plt.subplots(figsize=(14, 6))
scenarios = df_failure['scenario'].unique()
x = np.arange(len(topologies))
width = 0.15

for i, scenario in enumerate(scenarios[:4]):  # First 4 scenarios
    rec_values = []
    for topo in topologies:
        subset = df_failure[(df_failure['topology'] == topo) & 
                           (df_failure['scenario'] == scenario) &
                           (df_failure['method'] == 'gnnplus')]
        if len(subset) > 0:
            rec_values.append(subset['failure_recovery_ms'].values[0])
        else:
            rec_values.append(np.nan)
    ax.bar(x + i * width, rec_values, width, label=scenario)

ax.set_xlabel('Topology')
ax.set_ylabel('Recovery Time (ms)')
ax.set_title('GNN+ Failure Recovery Time by Scenario')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(topologies, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'failure_recovery_gnnplus.png', dpi=150, bbox_inches='tight')
plt.close()

# Create GNN+ vs Original GNN comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# MLU comparison
gnn_mlu = []
gnnplus_mlu = []
for topo in topologies:
    gnn_subset = df_normal[(df_normal['topology'] == topo) & (df_normal['method'] == 'gnn')]
    gnnplus_subset = df_normal[(df_normal['topology'] == topo) & (df_normal['method'] == 'gnnplus')]
    if len(gnn_subset) > 0:
        gnn_mlu.append(gnn_subset['mean_mlu'].values[0])
    else:
        gnn_mlu.append(np.nan)
    if len(gnnplus_subset) > 0:
        gnnplus_mlu.append(gnnplus_subset['mean_mlu'].values[0])
    else:
        gnnplus_mlu.append(np.nan)

x = np.arange(len(topologies))
width = 0.35
ax1.bar(x - width/2, gnn_mlu, width, label='Original GNN', color='#1f77b4', alpha=0.8)
ax1.bar(x + width/2, gnnplus_mlu, width, label='GNN+', color='#ff7f0e', alpha=0.8)
ax1.set_xlabel('Topology')
ax1.set_ylabel('Mean MLU')
ax1.set_title('GNN+ vs Original GNN: MLU Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(topologies, rotation=45, ha='right')
ax1.legend()
ax1.set_yscale('log')

# Decision time comparison
gnn_time = []
gnnplus_time = []
for topo in topologies:
    gnn_subset = df_normal[(df_normal['topology'] == topo) & (df_normal['method'] == 'gnn')]
    gnnplus_subset = df_normal[(df_normal['topology'] == topo) & (df_normal['method'] == 'gnnplus')]
    if len(gnn_subset) > 0:
        gnn_time.append(gnn_subset['decision_time_ms'].values[0])
    else:
        gnn_time.append(np.nan)
    if len(gnnplus_subset) > 0:
        gnnplus_time.append(gnnplus_subset['decision_time_ms'].values[0])
    else:
        gnnplus_time.append(np.nan)

ax2.bar(x - width/2, gnn_time, width, label='Original GNN', color='#1f77b4', alpha=0.8)
ax2.bar(x + width/2, gnnplus_time, width, label='GNN+', color='#ff7f0e', alpha=0.8)
ax2.set_xlabel('Topology')
ax2.set_ylabel('Decision Time (ms)')
ax2.set_title('GNN+ vs Original GNN: Decision Time Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(topologies, rotation=45, ha='right')
ax2.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'gnnplus_vs_original_gnn.png', dpi=150, bbox_inches='tight')
plt.close()

# Create summary statistics table
summary_stats = []
for method in methods:
    for topo in topologies:
        subset = df_normal[(df_normal['topology'] == topo) & (df_normal['method'] == method)]
        if len(subset) > 0:
            summary_stats.append({
                'Method': method.upper(),
                'Topology': topo,
                'Status': subset['status'].values[0],
                'Mean MLU': f"{subset['mean_mlu'].values[0]:.4f}",
                'Throughput': f"{subset['throughput'].values[0]:.4f}",
                'Disturbance': f"{subset['mean_disturbance'].values[0]:.4f}",
                'Decision Time (ms)': f"{subset['decision_time_ms'].values[0]:.2f}",
                'Recovery Time (ms)': f"{subset['failure_recovery_ms'].values[0]:.2f}" if 'failure_recovery_ms' in subset.columns else "N/A"
            })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(RESULTS_DIR / 'packet_sdn_sdn_metrics.csv', index=False)

print("Plots saved to:", PLOTS_DIR)
print("Summary CSV saved to:", RESULTS_DIR / 'packet_sdn_sdn_metrics.csv')
