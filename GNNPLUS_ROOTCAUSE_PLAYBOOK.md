# GNN+ Root-Cause Investigation Playbook

> **Purpose**: Reusable step-by-step guide for diagnosing and fixing GNN+ performance regressions.
> Run this whenever GNN+ is underperforming a baseline method on a specific topology/scenario.

---

## 0. Safety: Isolate on a New Branch

```bash
cd /Users/moahaimentalib/Documents/scientfic_papers/aI_sara_network/network_project
git status
git branch --show-current
git rev-parse --short HEAD
git checkout -b gnnplus-<your-tag>-review
git branch --show-current
git status
```

**Rules:**
- All investigation and edits happen ONLY on the new branch
- Do NOT modify the original branch directly
- Do NOT merge anything back without review
- Do NOT overwrite existing result bundles — use new result tags

---

## 1. Map the Codebase (Key Files)

### Core Model Files
| File | What It Does |
|------|-------------|
| `phase1_reactive/drl/gnn_plus_selector.py` | GNN+ feature construction (node 16d, edge 12d, OD 18d) + model wrapper |
| `phase1_reactive/drl/gnn_selector.py` | Original GNN features (node 16d w/ 4 zeros, edge 8d, OD 10d) + base model |
| `phase1_reactive/drl/gnn_training.py` | Supervised + REINFORCE training pipeline |
| `phase1_reactive/drl/reward.py` | RL reward computation |
| `phase1_reactive/drl/teacher_data.py` | Teacher label generation (heuristic + LP) |

### Evaluation Pipeline
| File | What It Does |
|------|-------------|
| `phase1_reactive/eval/core.py` | Evaluation rollout logic (normal mode) |
| `phase1_reactive/failures/failure_runner.py` | Failure evaluation pipeline |
| `te/baselines.py` | Bottleneck / Sensitivity / TopK selectors |
| `te/lp_solver.py` | LP solver for selected-path optimization |
| `te/simulator.py` | Apply routing, compute MLU |

### Experiment Drivers
| File | What It Does |
|------|-------------|
| `scripts/run_gnnplus_improved_fixedk40_experiment.py` | Main GNN+ experiment (train + eval) |
| `scripts/run_gnnplus_packet_sdn_full.py` | SDN packet simulation runner |

---

## 2. Identify the Failing Cases

### Quick Comparison Commands

```bash
# Find all result bundles
ls results/ | grep gnnplus

# Read the failure comparison CSV (old vs new)
cat results/<BUNDLE>/comparison/gnnplus_failure_old_vs_new.csv

# Read method rankings on normal mode
cat results/<BUNDLE>/comparison/new_run_method_ranking.csv

# Read the overall bundle comparison
cat results/<BUNDLE>/comparison/overall_bundle_comparison.csv

# Read the detailed failure data
cat results/<BUNDLE>/packet_sdn_failure.csv

# Filter for a specific topology
grep "vtlwavenet2011" results/<BUNDLE>/packet_sdn_failure.csv

# Filter for a specific failure scenario
grep "random_link_failure_1" results/<BUNDLE>/packet_sdn_failure.csv | grep "vtlwavenet"
```

### Key Metrics to Compare
- `mean_mlu`: Lower is better
- `failure_recovery_ms`: Lower is better
- Compare GNN+ vs: `ecmp`, `bottleneck`, `gnn` (original)

---

## 3. Trace the Code Path for the Failing Case

### For Normal Mode
1. Entry: `benchmark_topology_normal_improved()` in experiment script
2. Per-step: `run_sdn_cycle_gnnplus_improved()` → `gnnplus_select_stateful()`
3. Features: `build_graph_tensors_plus()` + `build_od_features_plus()` 
4. Scoring: `GNNPlusFlowSelector.forward()` (inherited from `GNNFlowSelector`)
5. Selection: `GNNPlusFlowSelector.select_critical_flows()`
6. LP: `solve_selected_path_lp()` → `apply_routing()`

### For Failure Mode
1. Entry: `benchmark_topology_failures_improved()` → `run_failure_scenario_gnnplus_improved()`
2. Failure state: `runner._build_failure_execution_state()` creates modified dataset/paths
3. Features: Same as normal but with `failure_mask` passed
4. **Key difference**: `gnnplus_select_stateful()` detects `has_active_failure` and zeros temporal features

### For Original GNN Under Failure
1. Entry: `failure_runner.py` → `_select_gnn_indices()`
2. Features: `build_graph_tensors()` + `build_od_features()` — **NO failure-aware OD features**
3. Scoring: Same `GNNFlowSelector.forward()` with simpler features

---

## 4. Common Root-Cause Patterns

### Pattern A: Feature Distribution Shift on Unseen Topologies
**Symptom**: GNN+ worse than original GNN on unseen topologies under failure
**Check**: Compare feature dimensions — GNN+ has failure-aware features (path shrink ratio, surviving path count, prev_best_invalid_flag) that are OOD for unseen topologies
**Fix**: Failure-blind fallback — don't pass `failure_mask` to feature builders

### Pattern B: Reward Scale Imbalance in REINFORCE
**Symptom**: GNN+ biased toward high-MLU topologies
**Check**: Inspect `PER_TOPO_REWARD_NORM` flag and MLU scale differences across training topologies
**Fix**: Enable per-topology reward normalization, or use scale-free reward terms only

### Pattern C: Max-Normalization Score Compression
**Symptom**: GNN+ less discriminative under failure
**Check**: In `GNNFlowSelector.forward()`, the `bottleneck / bottleneck.abs().max()` normalization compresses range when extreme values exist
**Fix**: Use mean/std normalization instead (requires retraining)

### Pattern D: Topology Cache Staleness
**Symptom**: Wrong features for capacity degradation scenarios
**Check**: `_get_plus_topology_cache()` key includes `id(path_library)` — verify cache is busted when path library changes
**Fix**: Usually not needed — cache is correctly invalidated for edge-removal failures

### Pattern E: Training Data Coverage Gap
**Symptom**: Model never saw failure features during training
**Check**: Training pipeline in `collect_experiment_samples()` — does it include failure augmentation?
**Fix**: Add synthetic failure augmentation to training (expensive, high risk)

---

## 5. Apply a Fix

### Template: Failure-Blind Fallback Patch

In `gnnplus_select_stateful()` of the experiment script:

```python
# When failure is active, don't pass failure_mask to feature builders
effective_failure_mask = None if has_active_failure else failure_mask
```

Then pass `effective_failure_mask` instead of `failure_mask` to both `build_graph_tensors_plus()` and `build_od_features_plus()`.

### Running the Experiment

```bash
# Set a unique experiment tag (NEVER reuse existing tags)
export GNNPLUS_EXPERIMENT_TAG="gnnplus_<your-fix-name>"

# Run the full experiment (uses existing checkpoint, no retraining)
python scripts/run_gnnplus_improved_fixedk40_experiment.py
```

### Comparing Results

```bash
# Compare failure results
diff <(grep "vtlwavenet" results/<OLD_BUNDLE>/packet_sdn_failure.csv) \
     <(grep "vtlwavenet" results/<NEW_BUNDLE>/packet_sdn_failure.csv)

# Quick summary
cat results/<NEW_BUNDLE>/comparison/overall_bundle_comparison.csv
```

---

## 6. Success / Rollback Criteria

### Success
- Primary failing case improves (e.g., VtlWavenet MLU drops below ECMP baseline)
- No regression on other topologies (known or unseen)
- Normal-mode performance unchanged

### Rollback
- Failing case does not improve → hypothesis falsified, try next solution
- Any known topology degrades >2% → revert patch

---

## 7. Architecture Quick Reference

```
GNN+ Scoring Pipeline:
  
  node_feat[V,16] ─→ node_proj ─→ GraphSAGE x3 ─→ h[V,64]
  edge_feat[E,12] ─→ edge_proj ─→ (used in msg passing)
  
  h[src] || h[dst] || od_feat[18] ─→ od_scorer ─→ gnn_correction[N_od]
  
  graph_embed = mean(h) ─→ blend_head ─→ [w_bn, w_sens]
                         ─→ confidence_head ─→ confidence ∈ [0,1]
  
  base_scores = w_bn * norm(bottleneck) + w_sens * norm(sensitivity)
  final_scores = base_scores + confidence * alpha * norm(gnn_correction)
  
  selected = top-K(final_scores, active_mask)
```

### Feature Variants
| Variant | Node feat 13 | Node feat 15 | OD feat 14 | OD feat 17 |
|---------|-------------|-------------|-----------|-----------|
| `legacy` | congested_nbr_frac | clustering_coeff | path_overlap | demand×hop |
| `section3_physical` | util_delta | demand_delta | bottleneck_delta | demand_delta |
| `section7_temporal` | util_delta + prev_selected | demand_delta + prev_dist | bottleneck_delta + prev_selected + invalid_flag | demand_delta + shrink + prev_dist |
| `lightweight_failure_aware` | util_delta + fail_exposure | demand_delta | bottleneck_delta + shrink + invalid_flag | demand_delta + shrink + invalid_flag |

### Known vs Unseen Topologies
- **Known (6)**: abilene, cernet, geant, ebone, sprintlink, tiscali
- **Unseen (2)**: germany50, vtlwavenet2011
- **Fixed K = 40** for all topologies

---

## 8. Key Config Environment Variables

```bash
GNNPLUS_EXPERIMENT_TAG          # Result bundle name (MUST be unique)
GNNPLUS_FEATURE_VARIANT         # Feature profile (default: section7_temporal)
GNNPLUS_CONTINUITY_BONUS        # Selection continuity weight (default: 0.05)
GNNPLUS_REWARD_MLU              # REINFORCE MLU weight (default: 1.15)
GNNPLUS_REWARD_DISTURBANCE      # REINFORCE disturbance weight (default: 0.15)
GNNPLUS_PER_TOPO_REWARD_NORM    # 0 or 1, per-topology reward normalization
GNNPLUS_NUM_LAYERS              # GraphSAGE layers (default: 3)
GNNPLUS_RL_LR                   # REINFORCE learning rate (default: 7e-5)
GNNPLUS_RL_ENTROPY_WEIGHT       # Entropy regularization (default: 0.002)
```

---

*Last updated: 2026-04-16 — based on branch `gnnplus-rootcause-review` at commit `579f439`*
