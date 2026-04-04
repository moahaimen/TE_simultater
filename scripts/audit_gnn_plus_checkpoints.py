#!/usr/bin/env python3
"""Audit all GNN+ checkpoints in the repository."""

import torch
import json
from pathlib import Path
from collections import defaultdict

project_root = Path("/Users/moahaimentalib/Documents/scientfic_papers/aI_sara_network/network_project")

# Find all .pt files in gnn_plus directories
gnn_plus_checkpoints = list(project_root.rglob("results/gnn_plus/**/*.pt"))

print(f"Found {len(gnn_plus_checkpoints)} GNN+ checkpoint files:\n")

audit_results = []

for ckpt_path in sorted(gnn_plus_checkpoints):
    try:
        # Load checkpoint
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # Extract metadata
        meta = {}
        if isinstance(state, dict):
            meta = state.get('metadata', {})
            if not meta and 'config' in state:
                meta = state.get('config', {})
        
        # Get key parameters
        model_type = meta.get('model_type', 'unknown')
        dropout = meta.get('dropout', meta.get('dropout_p', 'unknown'))
        learn_k = meta.get('learn_k_crit', meta.get('adaptive_k', 'unknown'))
        k_crit_min = meta.get('k_crit_min', 'N/A')
        k_crit_max = meta.get('k_crit_max', 'N/A')
        k_crit_fixed = meta.get('k_crit', 'N/A')
        best_epoch = meta.get('best_epoch', meta.get('epoch', 'unknown'))
        best_val_loss = meta.get('best_val_loss', meta.get('val_loss', 'unknown'))
        
        # Determine if fixed K or dynamic K
        if learn_k == True or learn_k == 'true' or learn_k == 1:
            k_type = f"dynamic (min={k_crit_min}, max={k_crit_max})"
        elif k_crit_fixed != 'N/A' and k_crit_fixed is not None:
            k_type = f"fixed K={k_crit_fixed}"
        else:
            k_type = "unknown"
        
        # Check if matches target config
        target_match = False
        if (dropout == 0.2 or dropout == '0.2') and \
           (learn_k == False or learn_k == 'false' or learn_k == 0 or learn_k == 'unknown') and \
           (k_crit_fixed == 40 or k_crit_fixed == '40'):
            target_match = True
        
        audit_results.append({
            'path': str(ckpt_path.relative_to(project_root)),
            'model_type': model_type,
            'dropout': dropout,
            'learn_k': learn_k,
            'k_type': k_type,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'target_match': target_match
        })
        
    except Exception as e:
        audit_results.append({
            'path': str(ckpt_path.relative_to(project_root)),
            'model_type': f'ERROR: {e}',
            'dropout': 'N/A',
            'learn_k': 'N/A',
            'k_type': 'N/A',
            'best_epoch': 'N/A',
            'best_val_loss': 'N/A',
            'target_match': False
        })

# Print audit table
print("=" * 120)
print(f"{'Path':<60} {'Model':<12} {'Dropout':<8} {'Learn K':<8} {'K Type':<25} {'Match':<6}")
print("=" * 120)

for result in audit_results:
    path_short = result['path'][:58]
    match_str = "YES" if result['target_match'] else "NO"
    print(f"{path_short:<60} {result['model_type']:<12} {str(result['dropout']):<8} "
          f"{str(result['learn_k']):<8} {result['k_type']:<25} {match_str:<6}")

print("=" * 120)

# Check if any match target
matching = [r for r in audit_results if r['target_match']]
print(f"\nCheckpoints matching target config (dropout=0.2, fixed K=40, no adaptive K): {len(matching)}")

if not matching:
    print("\n⚠️  NO MATCHING CHECKPOINT FOUND!")
    print("Target config: GNN+ with dropout=0.2, fixed K_crit=40, no adaptive K learning")
    print("\nNext step: Retrain the model with correct configuration.")
else:
    print("\n✓ Found matching checkpoint(s):")
    for m in matching:
        print(f"  - {m['path']}")
