# Phase 2 Final — Predictive GNN+ Sticky Traffic Engineering

## Working title
*Predictive GNN+ Sticky Traffic Engineering for Proactive Load Balancing
and Failure-Aware Routing in ISP Backbone Networks.*

## Status
Honest result.  Phase 2 Final is a clean **extension** of Phase 1 GNN+
Sticky (not a replacement), but the dominant contribution is the
corrected selector with alternative-path-gain features, **not** the GRU
forecaster.

---

## TL;DR

| Comparison | W / T / L (out of 288) | Win rate |
|---|---|---:|
| Phase 2 Predictive vs Phase 1 GNN+ Sticky | 44 / 236 / 8 | 15.3% |
| Phase 2 Current-State APG vs Phase 1 GNN+ Sticky | 43 / 236 / 9 | 14.9% |
| **Predictive vs Current-State APG (key ablation)** | **2 / 282 / 4** | **0.7%** |

The Predictive method matches Current-State APG to 6 decimal places on
98% of cells.  The full-precision oracle (which sees actual t+d util)
also matches.  Therefore: **the gain over Phase 1 comes from the
selector with alt-path-gain, not from AI prediction.**

This is consistent with Phase 2.6's earlier ablation in the bottleneck-
baseline pipeline.  Composing the predictive feature stack with Phase 1
GNN+ Sticky does not unlock the prediction value either; the GNN+
score (alpha = 0.45) plus alt-path-gain captures whatever is learnable
from the cycle-level state.

---

## Architecture (per professor's spec, fully implemented)

```
Historical traffic + topology + previous utilization
    ↓
Prediction Layer
    - GRU link-utilization forecaster (R² > 0.92, from Phase 2.2b)
    - Heuristic failure-risk derived from predicted util
    ↓
Predictive Feature Builder
    - predicted_bottleneck_score(OD)
    - predicted_hotspot_score_on_path(OD)
    - predicted_failure_risk_on_path(OD)
    - alternative_path_gain_predicted(OD)
    - predicted_demand_growth(OD)
    ↓
Phase 1 GNN+ Selector  ────  same checkpoint, MoE gate, candidate pool
    - returns: scores tensor over candidate pool
    ↓
Score-level fusion
    final_score(OD)  =  α · gnn_score
                      + β · predicted_bottleneck_score
                      + γ · predicted_hotspot_score
                      + δ · predicted_failure_risk
                      + η · alternative_path_gain
                      + ρ · predicted_demand_growth

    Default (balanced): α=0.45  β=0.20  γ=0.15  δ=0.10  η=0.10  ρ=0.05
    Phase 1 recovery:   α=1.00  β=γ=δ=η=ρ=0   (bit-identical to Phase 1)
    ↓
Top-K reselection (K ∈ {10, 20, 40})
    ↓
Sticky Post-Filter        ←  Phase 1 _sticky_compose_selection (unchanged)
    ↓
LP Split Optimization     ←  Phase 1 solve_selected_path_lp_safe (unchanged)
    ↓
Do-No-Harm Gate           ←  Phase 1 apply_do_no_harm_gate (the exact same fn)
    ↓
SDN Simulator
```

The GNN+ score is preserved as the dominant signal (α=0.45).  Phase 1
sticky filter, LP, do-no-harm gate, and failure fallback all run
unmodified.  Phase 2 Final is **bit-identical** to Phase 1 when
α=1, β=γ=δ=η=ρ=0.

---

## Methods compared (4 per cell)

1. **`phase1`** — Phase 1 GNN+ Sticky.  Recovered by `FusionWeights.phase1_only()`.
2. **`current_apg`** — same fusion structure, but predictive features
   are computed from **current** observed util.  Isolates the
   contribution of the alt-path-gain selector with no real prediction.
3. **`predictive`** — full Phase 2 Final.  Predictive features computed
   from GRU forecast at horizon = max(1, delay).
4. **`oracle`** — predictive features computed from actual util[t+d].
   Upper bound on what any forecaster could deliver.

The crucial pair for the ablation is `current_apg` vs `predictive`.
If they tie, prediction adds nothing.

---

## Experimental grid

| Axis | Values |
|---|---|
| Topologies | abilene, cernet, ebone, geant, sprintlink, tiscali, germany50, vtlwavenet2011 |
| Scenarios | normal, spike_2x, ramp_up, flash_crowd |
| K (top-K critical flows) | 10, 20, 40 |
| Actuation delay (cycles) | 0, 1, 2 |
| Cells | 288 |
| Methods per cell | 4 |
| Total method-cell evaluations | 1152 |

Test window per cell: 50 timesteps (consistent with Phase 2.6).
LP time limit: 5 s.  Sticky filter `STICKY_EPS = 0.005`.

---

## Phase 2 Predictive vs Phase 1 GNN+ Sticky — top-line

```
Phase 2 Predictive WINS Phase 1 GNN+ Sticky on:  44 / 288 cells (15.3%)
                                       TIES on: 236 / 288 cells (81.9%)
                                     LOSSES on:   8 / 288 cells ( 2.8%)
```

The dominant outcome is **TIE** — Phase 2 leaves the GNN+ Sticky route
substantially unchanged on 4 out of 5 cells.

### By topology (predictive wins per topology)

| Topology | Wins | Ties | Losses | Win % |
|---|---:|---:|---:|---:|
| germany50 | 14 | 22 | 0 | **38.9%** |
| sprintlink | 12 | 24 | 0 | 33.3% |
| abilene | 11 | 20 | 5 | 30.6% |
| geant | 6 | 30 | 0 | 16.7% |
| tiscali | 1 | 35 | 0 | 2.8% |
| cernet | 0 | 34 | 2 | 0.0% |
| ebone | 0 | 35 | 1 | 0.0% |
| vtlwavenet2011 | 0 | 36 | 0 | 0.0% |

**Pattern:** wins are concentrated on topologies that already
benefited most from Phase 2.6's selector (germany50, sprintlink,
abilene).  Topologies with saturated reactive baselines (ebone,
cernet, vtlwavenet2011) show pure ties — Phase 1 was already
near-optimal there.

### By scenario × delay (predictive wins, all 24-cell sub-grids)

| Scenario | d=0 | d=1 | d=2 |
|---|---:|---:|---:|
| normal | 8.3% | 12.5% | 20.8% |
| spike_2x | 16.7% | 20.8% | 20.8% |
| ramp_up | 4.2% | 16.7% | 12.5% |
| flash_crowd | 12.5% | 16.7% | 20.8% |

Win rate increases monotonically with delay for normal, flash_crowd
and spike_2x — same directional finding as Phase 2.6.  Magnitude is
smaller because GNN+ Sticky already absorbs most of the slack
prediction would provide.

---

## **The key ablation: Predictive vs Current-State APG**

This isolates whether the GRU forecaster adds value, controlling for
the corrected selector.

```
Predictive WINS Current-State APG on:  2 / 288 cells (0.7%)
                              TIES on: 282 / 288 cells (97.9%)
                            LOSSES on:   4 / 288 cells (1.4%)
```

### By scenario × delay

| Scenario | d=0 | d=1 | d=2 |
|---|---:|---:|---:|
| normal | 0/24 W | 0/24 W | **1/24 W (4.2%)** |
| spike_2x | 0/24 W | **1/24 W (4.2%)** | 0/24 W |
| ramp_up | 0/24 W | 0/24 W | 0/24 W |
| flash_crowd | 0/24 W | 0/24 W | 0/24 W |

**The forecaster is statistically equivalent to current-state on 282
of 288 cells.**

### Interpretation

- The GRU forecaster has R² > 0.92 on link util (Phase 2.2b), so its
  predictions are accurate.
- However, on the cycle-level horizons we test (1, 2 cycles), the
  predicted util is so close to current util that the alt-path-gain
  selector picks the same ODs.
- The corrected CFS selector with alt_path_gain captures whatever is
  learnable from the OD-level state at K=10..40.
- In the GNN+ Sticky composition, the GNN+ score (α=0.45) and
  alt-path-gain together dominate.  The forecast cannot improve on
  what the selector already extracts.

This is the same ablation finding as Phase 2.6, now confirmed at the
GNN+ Sticky composition layer.

---

## Failure-risk component (transparency)

We use a heuristic failure-risk derivation, not a separate ML
classifier:

```
risk[link] = clip( (max_predicted_util[link] - 0.7) / (0.95 - 0.7), 0, 1)
```

A separate ML classifier (RandomForest / GBT / small GRU) was
considered but the heuristic provides a stable derived label that
matches the predicted-util signal directly.  We do **not** report
precision / recall / F1 / AUC for failure-risk because:

1. We do not have real ISP failure logs (only simulated stress/failure
   scenarios).  Reporting precision/recall on synthetic-derived labels
   would be circular.
2. The contribution of the failure-risk feature in the fusion is
   weight δ = 0.10 — the per-cell ablation shows the entire fusion
   stack adds zero over current-state, so the failure-risk component
   adds proportionally less.

If a defensible failure-risk classifier becomes a thesis requirement,
we recommend:
- Train a small GRU classifier on labels derived from
  `single_link_failure`, `multiple_link_failure`, `three_link_failure`,
  `capacity_degradation_50`, `traffic_spike_2x` scenarios.
- Use precision / recall / F1 on held-out failure scenarios (not on
  synthetic-derived labels).
- Run as a separate Phase 2.7 sub-experiment.

---

## Phase 2 Final ⊕ Phase 1 — composition properties (verified)

The full Phase 1 stack is preserved and runs unchanged:

| Phase 1 component | Phase 2 Final behavior |
|---|---|
| GNN+ MoE gate | Used unmodified.  Outputs scores α-weighted into fusion. |
| Continuity bonus | Skipped at fusion stage; sticky filter handles continuity. |
| Sticky filter (`_sticky_compose_selection`) | Called unchanged after reselection.  Sticky activation rate observed > 0 on most cells. |
| LP split optimization (`solve_selected_path_lp_safe`) | Called unchanged via `apply_do_no_harm_gate`. |
| Do-No-Harm gate (`apply_do_no_harm_gate`) | Called unchanged with the new top-K. |
| Failure-do-no-harm fallback | Still active when env var enabled. |

Bit-identical Phase 1 fallback: setting weights to
`alpha=1, beta=gamma=delta=eta=rho=0` exactly recovers Phase 1
(verified in the smoke test).

---

## Defensible thesis claims

✅ **"Phase 2 extends Phase 1 GNN+ Sticky with a predictive intelligence
layer composed via score-level fusion.  The full Phase 1 stack
(GNN+ selector, sticky filter, LP, do-no-harm gate, failure fallback)
is preserved unchanged."**

✅ **"Across 288 cells covering 8 topologies × 4 scenarios × 3 K × 3
delays, Phase 2 Predictive GNN+ Sticky wins on 44 cells, ties on 236,
and loses on 8 vs Phase 1 GNN+ Sticky.  The largest gains are on
germany50 (38.9% win rate), sprintlink (33.3%), and abilene (30.6%).
Win rate increases with actuation delay for normal, spike, and
flash-crowd scenarios."**

✅ **"An ablation isolating the AI prediction contribution shows that
Phase 2 Predictive matches Phase 2 Current-State APG on 282 of 288
cells (97.9% ties).  We attribute the gain over Phase 1 primarily to
the alternative-path-gain selector and not to the GRU forecaster."**

❌ **NOT defensible (must NOT claim):**

- "AI prediction improves GNN+ Sticky routing." — the ablation shows
  it does not, on these horizons.
- "Phase 2 achieves universal MLU improvement." — 81.9% of cells tie.
- "Predictive routing is faster than reactive." — Phase 2 adds inference
  cost; the prediction layer adds ~5–15 ms per cycle vs Phase 1.

---

## Honest caveats

1. **Prediction is not free**: Phase 2 adds ~5–15 ms per cycle vs Phase 1
   for the GRU forward pass + feature builder.  When prediction does not
   improve MLU, this is pure overhead.
2. **K=40 saturation persists** — at the largest K, all 4 methods
   collapse to identical MLU.  Same finding as Phase 2.6.
3. **vtlwavenet2011 is the cleanest tie** (0/36 wins) — the topology is
   over-provisioned and Phase 1 already finds near-optimal routes.
4. **No real ISP failure logs** — failure scenarios are synthetic
   perturbations of the test traffic matrices.  We do **not** claim
   real failure-prediction performance.
5. **Phase 2.6 (independent CFS) remains valuable** as a separate
   ablation/prototype — it cleanly demonstrated that the corrected
   selector helps, before the cost of GNN+ + Sticky composition.
6. **Composition with Phase 1 dilutes prediction value**, not amplifies
   it.  The GNN+ score already captures most of the available signal;
   adding prediction as a 20% weight in the fusion does not unlock new
   capacity.

---

## What this means for the paper

This is the **honest answer to the professor's question**:

> "Does Predictive GNN+ Sticky beat Current-State GNN+ Sticky under
> delay and dynamic/failure scenarios?"
>
> **No, not measurably.** Predictive matches Current-State APG on 97.9%
> of cells.  The contribution of Phase 2 over Phase 1 is the selector,
> not the AI prediction.

The professor warned us up-front:

> "If yes: AI prediction contribution is defensible.
>  If no: The contribution is mostly improved future-critical selection,
>         and we must phrase it honestly."

The data says **no**, so we phrase it honestly.

The Phase 2 paper section becomes:

1. **Methodological**: a clean composition framework
   (Phase 2 ⊕ Phase 1) that is bit-identical to Phase 1 when
   prediction weights are zero.
2. **Selector-driven gain**: the alt-path-gain feature adds 14–15%
   wins over Phase 1, attributable to the selector.
3. **Prediction-neutral**: the GRU forecaster, despite R² > 0.92, does
   not measurably improve over current-state in this composition.
4. **Negative result on prediction value**: an honest finding that
   short-horizon (1–2 cycle) link-util prediction is too close to
   current state to add new information at the OD-selection layer
   when GNN+ Sticky is already in place.

---

## Reproducibility

```bash
# 0) Precompute predicted util (one-time, ~5 min)
python scripts/predictive/precompute_predicted_util.py

# 1) Phase 2 Final sweep (288 cells × 4 methods, ~70 min)
GNNPLUS_LP_TIME_LIMIT=5 GNNPLUS_STICKY_EPS=0.005 \
GNNPLUS_RUN_STAGE=eval_reuse_final GNNPLUS_REUSE_SUPERVISED=1 \
python scripts/predictive/eval_phase2_final_predictive_gnnplus_sticky.py \
    --topologies abilene,cernet,ebone,geant,sprintlink,tiscali,germany50,vtlwavenet2011 \
    --scenarios normal,spike_2x,ramp_up,flash_crowd \
    --k 10,20,40 --delay 0,1,2 --max_steps 50

# 2) Aggregate
python scripts/predictive/summarize_phase2_final.py
python scripts/predictive/plot_phase2_final.py
```

**Outputs:**
- `results/phase2_final/phase2_final_routing_results.csv` — per-cell raw metrics
- `results/phase2_final/phase2_final_summary_by_{topology,scenario}.csv`
- `results/phase2_final/phase2_final_ablation_prediction_vs_current.csv` — **the key ablation**
- `results/phase2_final/phase2_final_win_loss_matrix.csv` — full per-cell verdicts
- `results/phase2_final/phase2_final_topline.json`
- `results/phase2_final/figures/*.png` — 8 publication-style plots
