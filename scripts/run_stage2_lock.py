#!/usr/bin/env python3
"""Stage 2 LOCK: Fix learning signal and prove K is actually learned.

Corrections from pilot:
  - K target normalized to [0,1] with sigmoid output
  - Strong K-loss weights W ∈ {5, 10}
  - Proof metrics: corr(K_pred, K_target), MAE, per-topology + overall
  - Isolated in results/gnn_plus/stage2_lock/

Goal: Verify dynamic K is actually learned before moving to advanced Stage 2.

Success criteria:
  - corr > 0.3 (at least partial learning)
  - corr > 0.6 (convincing learning)
  - K_pred not collapsed
  - disturbance improves
  - MLU does not collapse
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from phase1_reactive.eval.common import load_bundle, load_named_dataset, collect_specs
from te.simulator import load_dataset, TEDataset

warnings.filterwarnings("ignore", category=UserWarning)

CONFIG_PATH = PROJECT_ROOT / "configs" / "phase1_reactive_full.yaml"
SEED = 42
DEVICE = "cpu"
LT = 15  # LP timeout seconds for pilot

K_CANDIDATES = [15, 20, 25, 30, 35, 40, 45, 50]
K_MIN, K_MAX = 1, 50

W_VALUES = [5.0, 10.0]  # Strong K-loss weights only

SUP_EPOCHS = 25
SUP_PATIENCE = 6
SUP_LR = 1e-3
BATCH_SIZE = 16

RL_LR = 1e-4
RL_EPOCHS = 8
RL_PATIENCE = 4
RL_EMA = 0.9

MAX_TEST_STEPS = 50

PILOT_TOPOS = {"germany50_real", "geant_core", "abilene_backbone"}
MAX_TRAIN_PER_TOPO = 25
MAX_VAL_PER_TOPO = 15

OUTPUT_ROOT = PROJECT_ROOT / "results" / "gnn_plus" / "stage2_lock"


# ---------------------------------------------------------------------------
#  Dynamic K Head (with normalized output)
# ---------------------------------------------------------------------------

class DynamicKHead(nn.Module):
    """K prediction head: graph_embed + traffic_stats -> K_norm ∈ [0,1]."""

    def __init__(self, hidden_dim: int, k_min: int = 1, k_max: int = 50, dropout: float = 0.1):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.k_range = k_max - k_min
        
        # Input: 4 traffic stats + 32-dim graph embed = 36 dim
        self.fc1 = nn.Linear(hidden_dim + 4, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # Output: normalized K (logit)

    def forward(self, graph_embed: torch.Tensor, traffic_stats: torch.Tensor) -> torch.Tensor:
        """Returns normalized K in [0, 1] via sigmoid."""
        x = torch.cat([graph_embed, traffic_stats], dim=-1)  # [B, hidden+4]
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        logit = self.fc3(x)  # [B, 1]
        k_norm = torch.sigmoid(logit)  # [0, 1]
        return k_norm.squeeze(-1)  # [B]

    def denormalize(self, k_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized K back to actual K in [k_min, k_max]."""
        return k_norm * self.k_range + self.k_min


class GNNWithDynamicK(nn.Module):
    """GNN + DynamicK head with frozen GNN features."""

    def __init__(self, gnn_feature_dim: int = 32, hidden_dim: int = 64, k_min: int = 1, k_max: int = 50):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.k_range = k_max - k_min
        
        # Feature extraction (will be frozen from original GNN)
        self.feature_proj = nn.Sequential(
            nn.Linear(gnn_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Dynamic K head
        self.k_head = DynamicKHead(hidden_dim, k_min, k_max)
        
        # Flow selector (simple MLP for now)
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for K info
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_embed: torch.Tensor, traffic_stats: torch.Tensor, num_flows: int):
        """
        Args:
            graph_embed: [B, gnn_feature_dim] frozen GNN features
            traffic_stats: [B, 4] traffic statistics
            num_flows: number of flows to score
        Returns:
            k_norm: [B] normalized K predictions
            flow_scores: [B, num_flows] flow importance scores
        """
        # Project graph features
        h = self.feature_proj(graph_embed)  # [B, hidden_dim]
        
        # Predict normalized K
        k_norm = self.k_head(h, traffic_stats)  # [B]
        
        # Flow scoring (placeholder - actual flow selection uses external selectors)
        k_expanded = k_norm.unsqueeze(-1).unsqueeze(-1).expand(-1, num_flows, -1)  # [B, num_flows, 1]
        h_expanded = h.unsqueeze(1).expand(-1, num_flows, -1)  # [B, num_flows, hidden]
        flow_input = torch.cat([h_expanded, k_expanded], dim=-1)  # [B, num_flows, hidden+1]
        flow_scores = self.flow_head(flow_input).squeeze(-1)  # [B, num_flows]
        
        return k_norm, flow_scores


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class KSample:
    """Single training sample with normalized K target."""
    topology_key: str
    timestep: int
    traffic_matrix: np.ndarray  # [num_nodes, num_nodes]
    graph_features: np.ndarray  # [gnn_feature_dim]
    traffic_stats: np.ndarray   # [4] (mean util, max util, p99 flow, flow count)
    oracle_k: int               # Raw oracle K
    oracle_k_norm: float        # Normalized oracle K (oracle_k / K_MAX)
    k_candidates: List[int]     # Available K values
    
    
class KPredictionDataset(Dataset):
    """Dataset for K prediction training."""
    
    def __init__(self, samples: List[KSample]):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'graph_features': torch.tensor(s.graph_features, dtype=torch.float32),
            'traffic_stats': torch.tensor(s.traffic_stats, dtype=torch.float32),
            'oracle_k_norm': torch.tensor(s.oracle_k_norm, dtype=torch.float32),
            'oracle_k': torch.tensor(s.oracle_k, dtype=torch.float32),
            'topology': s.topology_key,
            'timestep': s.timestep,
        }


# ---------------------------------------------------------------------------
#  Oracle K computation (MLU-only, same as pilot but track normalized)
# ---------------------------------------------------------------------------

def compute_oracle_k_target(M, ds, t_idx, prev_selections=None):
    """Find best K by sweeping candidates (MLU-only oracle)."""
    from te.baselines import select_topk_by_demand, select_bottleneck_critical, select_sensitivity_critical
    from te.lp_solver import solve_selected_path_lp
    
    tm_t = ds.tm[t_idx]
    pl = ds.path_library
    caps = np.asarray(ds.capacities, dtype=float)
    ecmp_base = M["ecmp_splits"](pl)
    
    best_k, best_mlu = None, float('inf')
    best_selection, best_method = None, None
    
    for k in K_CANDIDATES:
        for selector_name, fn in [
            ('topk', lambda k=k: select_topk_by_demand(tm_t, k)),
            ('bottleneck', lambda k=k: select_bottleneck_critical(tm_t, ecmp_base, pl, caps, k)),
            ('sensitivity', lambda k=k: select_sensitivity_critical(tm_t, ecmp_base, pl, caps, k)),
        ]:
            try:
                sel = fn()
                lp = solve_selected_path_lp(
                    tm_vector=tm_t, selected_ods=sel, base_splits=ecmp_base,
                    path_library=pl, capacities=caps,
                    time_limit_sec=LT)
                mlu = float(lp.routing.mlu)
                if np.isfinite(mlu) and mlu < best_mlu:
                    best_mlu = mlu
                    best_k = k
                    best_selection = sel
                    best_method = selector_name
            except Exception:
                continue
    
    return best_k, best_selection, best_mlu, best_method


# ---------------------------------------------------------------------------
#  Sample collection
# ---------------------------------------------------------------------------

def compute_traffic_stats(M, ds, tm_t):
    """Compute 4-dim traffic statistics."""
    # Simple link utilization estimate
    link_loads = np.zeros(len(ds.edges))
    for i, (src, dst) in enumerate(ds.edges):
        # Approximate load
        if src in ds.nodes and dst in ds.nodes:
            src_idx = ds.nodes.index(src)
            dst_idx = ds.nodes.index(dst)
            link_loads[i] = tm_t[src_idx * len(ds.nodes) + dst_idx]
    
    capacities = ds.capacities[:len(ds.edges)]
    utilizations = link_loads / (capacities + 1e-8)
    
    # Flow statistics
    flows = tm_t[tm_t > 0]
    p99_flow = np.percentile(flows, 99) if len(flows) > 0 else 0
    
    stats = np.array([
        utilizations.mean(),
        utilizations.max(),
        p99_flow,
        len(flows),
    ], dtype=np.float32)
    
    return stats


def collect_samples(M, datasets, split='train', max_per_topo=25):
    """Collect training/validation samples with oracle K targets."""
    samples = []
    
    print(f"\nCollecting {split} samples...")
    for key, ds in datasets.items():
        # Get split indices
        if split == 'train':
            indices = list(range(ds.split['train']))[:max_per_topo]
        elif split == 'val':
            start = ds.split['train']
            end = ds.split['train'] + ds.split['val']
            indices = list(range(start, end))[:max_per_topo]
        else:
            start = ds.split['train'] + ds.split['val']
            indices = list(range(start, len(ds.tm)))[:max_per_topo]
        
        print(f"  {key}: {len(indices)} {split} samples")
        
        for t_idx in indices:
            tm_t = ds.tm[t_idx]
            
            # Compute oracle K
            oracle_k, selection, mlu, method = compute_oracle_k_target(M, ds, t_idx)
            
            # Compute traffic stats
            traffic_stats = compute_traffic_stats(M, ds, tm_t)
            
            # Create synthetic graph features (will be replaced with real GNN features)
            graph_features = np.random.randn(32).astype(np.float32) * 0.1
            
            sample = KSample(
                topology_key=key,
                timestep=t_idx,
                traffic_matrix=tm_t,
                graph_features=graph_features,
                traffic_stats=traffic_stats,
                oracle_k=oracle_k,
                oracle_k_norm=oracle_k / K_MAX,  # NORMALIZED
                k_candidates=K_CANDIDATES,
            )
            samples.append(sample)
    
    return samples


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train_one(config_name, w_k, train_samples, val_samples):
    """Train dynamic K model with normalized targets and strong K-loss."""
    print(f"\n{'='*50}")
    print(f"[Training {config_name} with W={w_k}]")
    print(f"{'='*50}")
    
    out_dir = OUTPUT_ROOT / f"training_{config_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    train_ds = KPredictionDataset(train_samples)
    val_ds = KPredictionDataset(val_samples)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = GNNWithDynamicK().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=SUP_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, SUP_EPOCHS + 1):
        # Training
        model.train()
        train_losses = []
        train_k_losses = []
        train_k_preds = []
        
        for batch in train_loader:
            graph_feat = batch['graph_features'].to(DEVICE)
            traffic_stats = batch['traffic_stats'].to(DEVICE)
            oracle_k_norm = batch['oracle_k_norm'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward (dummy num_flows=50)
            k_norm_pred, _ = model(graph_feat, traffic_stats, 50)
            
            # Loss on NORMALIZED K
            k_loss = F.mse_loss(k_norm_pred, oracle_k_norm)
            loss = w_k * k_loss  # Strong K-loss only
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_k_losses.append(k_loss.item())
            train_k_preds.extend((k_norm_pred * K_MAX).detach().cpu().numpy())
        
        # Validation
        model.eval()
        val_losses = []
        val_k_losses = []
        val_k_preds = []
        val_k_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                graph_feat = batch['graph_features'].to(DEVICE)
                traffic_stats = batch['traffic_stats'].to(DEVICE)
                oracle_k_norm = batch['oracle_k_norm'].to(DEVICE)
                oracle_k = batch['oracle_k'].to(DEVICE)
                
                k_norm_pred, _ = model(graph_feat, traffic_stats, 50)
                
                k_loss = F.mse_loss(k_norm_pred, oracle_k_norm)
                loss = w_k * k_loss
                
                val_losses.append(loss.item())
                val_k_losses.append(k_loss.item())
                val_k_preds.extend((k_norm_pred * K_MAX).detach().cpu().numpy())
                val_k_targets.extend(oracle_k.cpu().numpy())
        
        # Stats
        mean_train_loss = np.mean(train_losses)
        mean_val_loss = np.mean(val_losses)
        mean_k_loss = np.mean(val_k_losses)
        
        train_k_mean = np.mean(train_k_preds)
        train_k_std = np.std(train_k_preds)
        val_k_mean = np.mean(val_k_preds)
        val_k_std = np.std(val_k_preds)
        
        print(f"  Ep {epoch:2d}: loss={mean_train_loss:.4f} vl={mean_val_loss:.4f} "
              f"kl={mean_k_loss:.4f} K={train_k_mean:.1f}±{train_k_std:.1f} "
              f"valK={val_k_mean:.1f}±{val_k_std:.1f}")
        
        # Early stopping
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': mean_val_loss,
            }, out_dir / "supervised.pt")
        else:
            patience_counter += 1
            if patience_counter >= SUP_PATIENCE:
                print(f"  Early stop at {epoch}")
                break
        
        scheduler.step(mean_val_loss)
    
    print(f"  Supervised done, best_ep={best_epoch}")
    return model, best_epoch


# ---------------------------------------------------------------------------
#  Proof metrics: Correlation and MAE
# ---------------------------------------------------------------------------

def compute_learning_proof(k_preds, k_targets, topology_keys, timesteps):
    """Compute correlation and MAE per topology and overall."""
    results = {}
    
    # Overall
    if len(k_preds) > 1 and len(set(k_targets)) > 1:
        corr, pval = pearsonr(k_preds, k_targets)
    else:
        corr, pval = 0.0, 1.0
    
    mae = np.mean(np.abs(np.array(k_preds) - np.array(k_targets)))
    
    results['TOTAL'] = {
        'pearson_corr': corr,
        'p_value': pval,
        'mae': mae,
        'k_pred_mean': np.mean(k_preds),
        'k_pred_std': np.std(k_preds),
        'k_pred_min': np.min(k_preds),
        'k_pred_max': np.max(k_preds),
        'k_target_mean': np.mean(k_targets),
        'k_target_std': np.std(k_targets),
        'n_samples': len(k_preds),
    }
    
    # Per topology
    unique_topos = set(topology_keys)
    for topo in unique_topos:
        indices = [i for i, t in enumerate(topology_keys) if t == topo]
        k_p = [k_preds[i] for i in indices]
        k_t = [k_targets[i] for i in indices]
        
        if len(k_p) > 1 and len(set(k_t)) > 1:
            corr, pval = pearsonr(k_p, k_t)
        else:
            corr, pval = 0.0, 1.0
        
        mae = np.mean(np.abs(np.array(k_p) - np.array(k_t)))
        
        results[topo] = {
            'pearson_corr': corr,
            'p_value': pval,
            'mae': mae,
            'k_pred_mean': np.mean(k_p),
            'k_pred_std': np.std(k_p),
            'k_pred_min': np.min(k_p),
            'k_pred_max': np.max(k_p),
            'k_target_mean': np.mean(k_t),
            'k_target_std': np.std(k_t),
            'n_samples': len(k_p),
        }
    
    return results


# ---------------------------------------------------------------------------
#  Evaluation with proof metrics
# ---------------------------------------------------------------------------

def evaluate_with_proof(model, datasets, w_name):
    """Evaluate model and compute proof metrics."""
    print(f"\n[Evaluating {w_name}]")
    
    model.eval()
    all_results = []
    
    # Collect per-topology results
    k_preds_by_topo = defaultdict(list)
    k_targets_by_topo = defaultdict(list)
    timesteps_by_topo = defaultdict(list)
    
    for key, ds in datasets.items():
        print(f"  {key}:", end=" ")
        
        # Test indices
        start = ds.split['train'] + ds.split['val']
        test_idx = list(range(start, min(start + MAX_TEST_STEPS, len(ds.tm))))
        
        for t_idx in test_idx:
            tm_t = ds.tm[t_idx]
            
            # Get prediction
            traffic_stats = compute_traffic_stats(None, ds, tm_t)
            graph_features = np.random.randn(32).astype(np.float32) * 0.1
            
            with torch.no_grad():
                graph_t = torch.tensor(graph_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                stats_t = torch.tensor(traffic_stats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                k_norm_pred, _ = model(graph_t, stats_t, 50)
                k_pred = (k_norm_pred * K_MAX).item()
            
            # Get oracle
            oracle_k, _, _, _ = compute_oracle_k_target(None, ds, t_idx)
            
            k_preds_by_topo[key].append(k_pred)
            k_targets_by_topo[key].append(oracle_k)
            timesteps_by_topo[key].append(t_idx)
            
            if t_idx % 10 == 0:
                print(f"{t_idx}(K={k_pred:.0f})", end=" ")
        
        print()
    
    # Compute proof metrics
    all_preds = []
    all_targets = []
    all_topos = []
    
    for topo in k_preds_by_topo:
        all_preds.extend(k_preds_by_topo[topo])
        all_targets.extend(k_targets_by_topo[topo])
        all_topos.extend([topo] * len(k_preds_by_topo[topo]))
    
    proof_results = compute_learning_proof(all_preds, all_targets, all_topos, [])
    
    return proof_results, k_preds_by_topo, k_targets_by_topo, timesteps_by_topo


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def generate_plots(k_preds_by_topo, k_targets_by_topo, timesteps_by_topo, w_name):
    """Generate K histograms, over-time plots, and scatter plots."""
    plots_dir = OUTPUT_ROOT / "plots" / w_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for topo in k_preds_by_topo:
        k_preds = np.array(k_preds_by_topo[topo])
        k_targets = np.array(k_targets_by_topo[topo])
        timesteps = np.array(timesteps_by_topo[topo])
        
        # 1. Histogram
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        axes[0].hist(k_preds, bins=20, alpha=0.6, label='K_pred', color='blue', edgecolor='black')
        axes[0].hist(k_targets, bins=20, alpha=0.6, label='K_target', color='red', edgecolor='black')
        axes[0].set_xlabel('K')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{topo}: K Distribution ({w_name})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. K over time
        axes[1].plot(timesteps, k_preds, 'o-', label='K_pred', color='blue', markersize=3, alpha=0.7)
        axes[1].plot(timesteps, k_targets, 's-', label='K_target', color='red', markersize=3, alpha=0.7)
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('K')
        axes[1].set_title(f'{topo}: K over Time ({w_name})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{topo}_k_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Scatter plot: K_pred vs K_target
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(k_targets, k_preds, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Diagonal line (perfect prediction)
        min_val = min(k_targets.min(), k_preds.min())
        max_val = max(k_targets.max(), k_preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        
        # Correlation annotation
        if len(k_preds) > 1:
            corr, _ = pearsonr(k_preds, k_targets)
            ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   verticalalignment='top')
        
        ax.set_xlabel('K_target (Oracle)')
        ax.set_ylabel('K_pred (Model)')
        ax.set_title(f'{topo}: K_pred vs K_target ({w_name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{topo}_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    print("="*60)
    print("STAGE 2 LOCK: Verify K is actually learned")
    print(f"W values: {W_VALUES} (strong K-loss)")
    print(f"K target: NORMALIZED [0,1] with sigmoid output")
    print(f"Output: {OUTPUT_ROOT}")
    print("="*60)
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Load topologies
    print("\n[1] Loading topologies...")
    datasets = {}
    for spec in load_topology_specs():
        if spec.key in PILOT_TOPOS:
            ds = load_dataset_from_spec(spec)
            if ds:
                datasets[spec.key] = ds
                print(f"  {spec.key}: {len(ds.nodes)}N, {len(ds.edges)}E, {len(ds.tm)} TMs")
    
    print(f"\nLoaded: {list(datasets.keys())}")
    
    # Collect samples
    print("\n[2] Collecting samples with oracle K sweep...")
    train_samples = collect_samples(None, datasets, 'train', MAX_TRAIN_PER_TOPO)
    val_samples = collect_samples(None, datasets, 'val', MAX_VAL_PER_TOPO)
    
    if len(train_samples) == 0:
        print("ERROR: No training samples collected!")
        return
    
    # Oracle K stats
    oracle_k_vals = [s.oracle_k for s in train_samples]
    print(f"\nOracle K: mean={np.mean(oracle_k_vals):.1f} std={np.std(oracle_k_vals):.1f} "
          f"min={np.min(oracle_k_vals)} max={np.max(oracle_k_vals)}")
    
    # Train both W values
    results = {}
    for w_val in W_VALUES:
        w_name = f"w{int(w_val)}"
        print(f"\n{'='*60}")
        print(f"[3] Training {w_name} (W={w_val})")
        print(f"{'='*60}")
        
        model, train_time = train_one(w_name, w_val, train_samples, val_samples)
        
        # Evaluate with proof metrics
        proof_results, k_preds, k_targets, timesteps = evaluate_with_proof(model, datasets, w_name)
        results[w_name] = proof_results
        
        # Generate plots
        generate_plots(k_preds, k_targets, timesteps, w_name)
        
        # Print proof metrics
        print(f"\n[Proof Metrics for {w_name}]")
        for topo in ['abilene_backbone', 'geant_core', 'germany50_real', 'TOTAL']:
            if topo in proof_results:
                r = proof_results[topo]
                corr_status = "✓" if r['pearson_corr'] > 0.6 else ("~" if r['pearson_corr'] > 0.3 else "✗")
                print(f"  {topo:20s}: corr={r['pearson_corr']:+.3f} {corr_status}  "
                      f"MAE={r['mae']:.1f}  std={r['k_pred_std']:.1f}  "
                      f"K∈[{r['k_pred_min']:.0f}, {r['k_pred_max']:.0f}]")
    
    # Save results
    print("\n[4] Saving results...")
    
    # Summary CSV
    summary_rows = []
    for w_name in results:
        for topo, metrics in results[w_name].items():
            row = {'w_name': w_name, 'topology': topo}
            row.update(metrics)
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_ROOT / "learning_proof_summary.csv", index=False)
    print(f"  Saved: {OUTPUT_ROOT / 'learning_proof_summary.csv'}")
    
    # Verdict
    print("\n" + "="*60)
    print("STAGE 2 LOCK VERDICT")
    print("="*60)
    
    for w_name in results:
        total_corr = results[w_name]['TOTAL']['pearson_corr']
        if total_corr > 0.6:
            verdict = "PASS - Convincing learning"
        elif total_corr > 0.3:
            verdict = "PARTIAL - Weak learning, needs architecture fix"
        else:
            verdict = "FAIL - No learning detected"
        
        print(f"\n{w_name}: {verdict}")
        print(f"  Overall corr: {total_corr:.3f}")
    
    print(f"\nResults saved to: {OUTPUT_ROOT}")


def load_topology_specs():
    """Load topology specs from config."""
    import yaml
    
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    
    specs = []
    for t in config.get('topologies', []):
        spec = type('Spec', (), {})()
        spec.key = t.get('key')
        spec.dataset_key = t.get('dataset_key', spec.key)
        spec.name = t.get('display_name', spec.key)
        spec.topology_file = t.get('topology_file')
        spec.processed_file = t.get('processed_file')
        specs.append(spec)
    
    return specs


def load_dataset_from_spec(spec):
    """Load dataset from topology spec."""
    try:
        # Build minimal config
        config = {
            'dataset': {
                'key': spec.dataset_key,
                'name': spec.name,
                'data_dir': 'data',
                'processed_file': spec.processed_file,
                'topology_file': spec.topology_file,
            },
            'experiment': {
                'seed': SEED,
                'max_steps': 500,
                'k_paths': 3,
                'k_crit': 20,
                'split': {'train': 0.70, 'val': 0.15, 'test': 0.15},
                'lp_time_limit_sec': LT,
            }
        }
        return load_dataset(config)
    except Exception as e:
        print(f"  Warning: Could not load {spec.key}: {e}")
        return None


if __name__ == "__main__":
    main()
