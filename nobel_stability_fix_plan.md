# Nobel Stability Fix Plan

Branch: `nobel-stability-fix`
Date: 2026-04-18

## Context

GNN+ is the best method on `Mean MLU` on Nobel-Germany but pays a stability cost:

| Method | Mean MLU | Mean Disturbance | Flow Table Updates | Decision Time (ms) |
|---|---:|---:|---:|---:|
| Bottleneck | 0.780837 | 0.067008 | 7.159091 | 24.448263 |
| Original GNN | 0.794098 | 0.063265 | 6.613636 | 29.446182 |
| **GNN+** | **0.778991** | **0.082761** | **9.295455** | **34.348265** |

GNN+ wins on MLU. GNN+ is worst among intelligent methods on disturbance, flow table updates, and decision time.

The same pattern exists in the broader 8-topology benchmark (disturbance only 4/8 topologies ≤ Original GNN).

Root cause is known precisely from code inspection. Three compounding mechanisms are responsible. All are fixable without changing the GNN model weights.

---

## Root Causes (Confirmed by Code Inspection)

### Cause 1 — Continuity bonus hard-zeroed at inference (bug)

File: `scripts/run_gnnplus_improved_fixedk40_experiment.py`

`CONTINUITY_BONUS = 0.05` is defined at line 103.

Every inference-time call to `continuity_select` inside `run_sdn_cycle_gnnplus_improved` and `run_failure_scenario_gnnplus_improved` passes `continuity_bonus=0.0` — hard-coded. This means the selection stability nudge is completely disabled at inference, despite being enabled during training. Every step re-ranks all 40 ODs from scratch with zero carry-over. This directly causes high churn.

### Cause 2 — Nobel-Germany missing from aggressive tiebreak set (oversight)

File: `scripts/run_gnnplus_improved_fixedk40_experiment.py` line 204.

```python
AGGRESSIVE_TIEBREAK_TOPOLOGIES = {"geant", "tiscali", *UNSEEN_TOPOLOGIES}
```

`UNSEEN_TOPOLOGIES` contains `germany50` and `vtlwavenet2011` but NOT `nobel_germany`. So Nobel-Germany (an unseen zero-shot topology) gets the weak p5 tie-break epsilon instead of the aggressive p25 epsilon. This gives less selection stickiness on the hardest topology.

### Cause 3 — Do-no-harm fallback creates ping-pong routing changes

File: `scripts/run_gnnplus_improved_fixedk40_experiment.py`

Nobel-Germany fallback rate is 6.8%. When the gate fires on step T:
- GNN+ LP is solved (already done)
- Bottleneck LP is solved (extra cost)
- Bottleneck routing is applied

On step T+1, GNN+ resumes. Pattern becomes:
`GNN+, GNN+, FALLBACK→BN, GNN+, GNN+, FALLBACK→BN, ...`

Each fallback-then-return is two routing changes, each counted in disturbance and flow table updates. The double-LP on fallback steps also adds decision time.

`DO_NO_HARM_CACHE_STEPS = 10` means the bottleneck guard LP fires every 10 steps — not amortised enough for unseen topologies.

### Cause 4 — Disturbance reward weight 13× too low (training-side, requires RL rerun)

File: `phase1_reactive/drl/gnn_training.py` line 77 and `scripts/run_gnnplus_improved_fixedk40_experiment.py` line 112.

`w_reward_disturbance = 0.15` vs `w_reward_mlu = 1.15`. The model is trained to trade 1 unit of disturbance for 0.13 units of MLU. With churn multiplier (×3 at max churn), effective disturbance penalty reaches at most 0.45 — still 2.5× less than the MLU weight. The model learned to sacrifice stability aggressively for MLU, which is what the reward asked for.

---

## Fix Plan

### Task A — Inference-only fixes (no retraining, run first)

**Do all three changes in a single commit. No model or training changes.**

#### A1. Re-enable continuity bonus at inference

File: `scripts/run_gnnplus_improved_fixedk40_experiment.py`

Find every call to `continuity_select(...)` inside:
- `run_sdn_cycle_gnnplus_improved`
- `run_failure_scenario_gnnplus_improved`
- any calibration path that uses the inference selector

Change every instance of `continuity_bonus=0.0` to `continuity_bonus=CONTINUITY_BONUS`.

`CONTINUITY_BONUS` is already defined at line 103 as `0.05`. Do not change its value.

The formula inside `continuity_select` (line 1137) is:
```
ranking_scores = active_scores + tie_break_eps * prev_active
```
When continuity_bonus is passed through correctly, the normalized-score path in `select_critical_flows` (lines 892–898) applies:
```
ranking_score = (score - min_score) / score_span + 0.05 * prev_selected_indicator
```
This gives previously-selected ODs a ~5% normalized-score advantage, keeping ~60–70% of the previous selection stable when scores are close. No MLU cost — only near-tied pairs are affected.

Expected improvement: disturbance −15 to −20%, flow table updates −10 to −15%.

#### A2. Add Nobel-Germany to aggressive tiebreak set

File: `scripts/run_gnnplus_improved_fixedk40_experiment.py` line 204.

```python
# Before:
AGGRESSIVE_TIEBREAK_TOPOLOGIES = {"geant", "tiscali", *UNSEEN_TOPOLOGIES}

# After:
AGGRESSIVE_TIEBREAK_TOPOLOGIES = {"geant", "tiscali", "nobel_germany", *UNSEEN_TOPOLOGIES}
```

This switches Nobel-Germany from p5 to p25 tie-break epsilon. Since p25 is already computed for all topologies during calibration, no calibration rerun is needed. Just add the topology name to the set.

Expected improvement: additional selection stickiness on Nobel-Germany specifically.

#### A3. Add fallback hysteresis to eliminate ping-pong

File: `scripts/run_gnnplus_improved_fixedk40_experiment.py`

**Step 1:** Add `guard_fallback_cooldown` to the `gnnplus_state` initializer:
```python
gnnplus_state = {
    ...existing keys...,
    "guard_fallback_cooldown": 0,
}
```

**Step 2:** At the start of `apply_do_no_harm_gate` (or at its call site in `run_sdn_cycle_gnnplus_improved`), before running the gate logic:
```python
# Check cooldown — if we fell back recently, stay on bottleneck
if state.get("guard_fallback_cooldown", 0) > 0:
    state["guard_fallback_cooldown"] -= 1
    # Return the bottleneck selection that was cached from the triggering step
    return cached_bottleneck_selection, "cooldown_hold"
```

**Step 3:** When the gate fires a new fallback (gate condition is True):
```python
state["guard_fallback_cooldown"] = 4   # hold for 4 additional steps
```

**Step 4:** Also increase `DO_NO_HARM_CACHE_STEPS` from 10 to 20 for unseen topologies:
```python
# Before:
DO_NO_HARM_CACHE_STEPS = 10

# After (or make it topology-dependent):
DO_NO_HARM_CACHE_STEPS = 20  # halves the double-LP frequency
```

The cooldown converts `FALLBACK, GNN+, FALLBACK, GNN+` (4 routing changes in 4 steps) into `FALLBACK, BN_hold, BN_hold, BN_hold, BN_hold, GNN+` (2 routing changes in 6 steps). This directly halves fallback-related disturbance.

Expected improvement: fallback rate stays the same but the disturbance contribution per fallback event drops ~50%. Decision time also improves because fewer double-LP steps per 20-step cycle.

---

#### Task A evaluation

After implementing A1 + A2 + A3 in one commit:

1. Re-run Nobel-Germany eval:
   ```
   python scripts/run_nobel_germany_real_eval.py
   ```
   Save outputs to: `results/gnnplus_nobel_stability_taskA/`

2. Also re-run the full 8-topology benchmark to confirm no regression:
   ```
   python scripts/run_gnnplus_packet_sdn_full.py
   ```
   Save to: `results/gnnplus_8topo_stability_taskA/`

3. Report the full comparison table (Task A vs baseline):

   | Metric | Baseline | Task A Target | Pass? |
   |---|---:|---:|---|
   | Nobel Mean MLU | 0.778991 | ≤ 0.787 (+1%) | |
   | Nobel Mean Disturbance | 0.082761 | ≤ 0.072 | |
   | Nobel Flow Table Updates | 9.295455 | ≤ 8.50 | |
   | Nobel Decision Time (ms) | 34.348265 | ≤ 31.0 | |
   | Nobel Fallback Rate | 0.068182 | ≤ 0.040 | |
   | 8-topo Disturbance wins | 4/8 | ≥ 5/8 | |
   | 8-topo Normal MLU wins | 6/8 | 6/8 (no regression) | |

**If Task A passes all gates, stop. No retraining needed. Commit + push.**

**If Task A passes on MLU-preservation but disturbance is still > 0.072, proceed to Task B.**

---

### Task B — RL rerun with higher disturbance weight (retraining, only if Task A not enough)

Do NOT start Task B until Task A eval is complete and reported.

#### B1. Raise disturbance reward weight

File: `scripts/run_gnnplus_improved_fixedk40_experiment.py` line 112:
```python
# Before:
REWARD_DISTURBANCE = float(os.environ.get("GNNPLUS_REWARD_DISTURBANCE", "0.15"))

# After:
REWARD_DISTURBANCE = float(os.environ.get("GNNPLUS_REWARD_DISTURBANCE", "0.60"))
```

With the existing churn multiplier (×3 at max churn): effective peak penalty becomes `0.60 × 3 = 1.80`, higher than the MLU weight `1.15`. The model now trades MLU for stability when the MLU gain is small — exactly the Nobel case where GNN+ beats Bottleneck by only 0.0018 MLU.

#### B2. Enable temporal consistency in the RL training loop

File: `scripts/run_gnnplus_improved_fixedk40_experiment.py`

The temporal consistency loss exists for supervised training (coefficient `TEMPORAL_CONSISTENCY_WEIGHT = 0.02`) but is not included in the RL loss. Add it to the RL reward computation (the block after line 2395):
```python
# Add to RL loss:
if float(TEMPORAL_CONSISTENCY_WEIGHT) > 0.0 and prev_scores is not None:
    tc_loss = TEMPORAL_CONSISTENCY_WEIGHT * _temporal_consistency_loss(
        rollout_scores, prev_selected_indicator=prev_indicator, ...
    )
    rl_loss = rl_loss + tc_loss
```

#### B3. RL run configuration

- Starting checkpoint: Task A inference checkpoint (the supervised model from Task 16, not the Task 14 RL model)
- Max epochs: 10
- Patience: 4
- Early-stop criterion: `val_disturbance + 0.5 * val_failure_mlu` (composite, not just MLU)
- Keep all Task 13c / Task 14 config (state-conditional KL, Gumbel tau=0.5 on failure, running-mean baseline, 5-type failure coverage)
- Save results to: `results/gnnplus_rl_disturbance_taskB/`

#### Task B gate

After RL + Nobel eval:

| Metric | Baseline | Task B Target |
|---|---:|---:|
| Nobel Mean MLU | 0.778991 | ≤ 0.787 |
| Nobel Mean Disturbance | 0.082761 | ≤ 0.068 (≤ Bottleneck) |
| Nobel Flow Table Updates | 9.295455 | ≤ 7.50 |
| 8-topo Disturbance wins | 4/8 | ≥ 6/8 |

If Task B does not move disturbance below 0.072 after 10 RL epochs, **stop and accept the Task A result.** Do not run further RL iterations.

---

## Do Not Do

- Do not retrain from scratch.
- Do not change the GNN model architecture.
- Do not change `K = 40` (fixed budget).
- Do not change the LP formulation.
- Do not create Nobel-specific tuning that breaks the zero-shot story.
- Do not run Task B before Task A eval is complete.
- Do not raise `CONTINUITY_BONUS` above 0.05 without measuring MLU impact — values above 0.10 start pinning stale selections when demand shifts.
- Do not lower `DO_NO_HARM_THRESHOLD_UNSEEN` below 1.00 — it is already at the strict floor.

---

## Files to Change

Summary of exact files touched by Task A:

| File | What changes |
|---|---|
| `scripts/run_gnnplus_improved_fixedk40_experiment.py` | A1: `continuity_bonus=0.0` → `CONTINUITY_BONUS`; A2: add `"nobel_germany"` to set; A3: add cooldown field + cooldown logic + increase cache steps |

Task B additionally changes:

| File | What changes |
|---|---|
| `scripts/run_gnnplus_improved_fixedk40_experiment.py` | B1: raise `REWARD_DISTURBANCE` default; B2: add TC loss to RL loop |

---

## Expected Final State After This Branch

If Task A alone closes the gap:

| Metric | Archfix baseline | Task 17 (last main result) | This branch target |
|---|---:|---:|---:|
| Nobel MLU | not measured | — | ≤ 0.787 |
| Nobel Disturbance | — | — | ≤ 0.072 |
| Nobel Flow Updates | — | — | ≤ 8.50 |
| 8-topo Disturbance wins | 4/8 | 4/8 | ≥ 5/8 |
| 8-topo Normal MLU wins | 2/8 | 6/8 | 6/8 (preserved) |
| Decision time vs Bottleneck | ~2.5× | ~1.28× | ~1.15–1.25× |

The claim after this branch: "GNN+ is competitive with Bottleneck on all three metrics — MLU, disturbance, and decision time — and is best-in-class on failure recovery for large and unseen topologies."
