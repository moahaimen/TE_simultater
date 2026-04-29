# Disturbance-Dominance Plan (post-Check-#5)

## Goal

Make GNN+ beat **every** baseline (EroDRL, FlexDate, FlexEntry, CFRRL,
OSPF, ECMP) on **all three** headline metrics on **all 8 topologies**:

- **MLU**: already dominant across the board on Task A checkpoint.
- **Decision time**: dominant; additionally a free ~20 % cut from
  Check #5 (LP 20 s → 5 s, bit-identical MLU).
- **Disturbance**: **currently the only gap.** GNN+ Task A beats
  FlexEntry on 3 / 8 (abilene, geant, vtlwavenet2011) and loses on
  5 / 8 (cernet, ebone, sprintlink, tiscali, germany50). FlexEntry is
  disturbance-optimal by construction; closing the gap is what this
  plan is about.

No metric may regress in pursuit of disturbance. MLU and decision-time
dominance are non-negotiable.

---

## Ground truth (Task A vs external baselines, disturbance column)

Source: `results/requirements_compliant_eval/table_external_baselines.csv`
and `results/gnnplus_8topo_stability_taskA/packet_sdn_summary.csv`.

| topology       | status | GNN+ Task A | FlexEntry | FlexDate | EroDRL | best non-GNN+ | gap to beat |
|----------------|--------|------------:|----------:|---------:|-------:|--------------:|------------:|
| abilene        | seen   | 0.042       | 0.091     | 0.105    | 0.086  | 0.086 (EroDRL) | already wins |
| cernet         | seen   | 0.029       | **0.018** | 0.062    | 0.051  | 0.018 (FlexEntry) | −0.012 |
| ebone          | seen   | 0.048       | **0.039** | 0.087    | 0.072  | 0.039 (FlexEntry) | −0.010 |
| geant          | seen   | 0.058       | 0.134     | 0.152    | 0.113  | 0.113 (EroDRL) | already wins |
| sprintlink     | seen   | 0.016       | **0.013** | 0.044    | 0.031  | 0.013 (FlexEntry) | −0.004 |
| tiscali        | seen   | 0.023       | **0.021** | 0.048    | 0.036  | 0.021 (FlexEntry) | −0.003 |
| germany50      | unseen | 0.192       | **0.114** | 0.221    | 0.178  | 0.114 (FlexEntry) | −0.078 |
| vtlwavenet2011 | unseen | 0.006       | 0.007     | 0.018    | 0.011  | 0.007 (FlexEntry) | already wins |

GNN+ already wins MLU on every single topology vs every baseline —
FlexEntry is a disturbance-optimal baseline that **deliberately trades
MLU for flow-change count**. Our claim is that GNN+ can match its
disturbance without giving back any MLU.

---

## Strategy in one paragraph

GNN+ already has a `continuity_bonus` term in the selector
(`phase1_reactive/drl/gnn_plus_selector.py:892-898`) that rewards
keeping previously selected ODs in the top-K. It defaults to `0.0` at
inference. The rewrite stack also has a `prev_selected_indicator`
threaded through `run_gnnplus_improved_fixedk40_experiment.py:3228`,
so the state we need is already plumbed. The idea is to use that
state at **three new choke points** — (a) continuity-weighted
tie-break inside the selector, (b) a sticky post-filter that refuses
to swap flows for a negligible MLU gain, (c) a disturbance-aware
do-no-harm guard that prefers the lower-churn candidate on near-ties.
All three are env-var-gated so we can OFAT them exactly like the
Check #2 / Check #5 rescue experiments.

If the inference-only layer (Phase 1) gets us past 3 / 5 loses but
not the full 5 / 5, Phase 2 retrains with a stronger
`w_reward_disturbance` term. Phase 3 is the hard fix: add
previous-selection as a GNN+ input feature and train against an
explicit hinge penalty on `|Δselection| − tolerance`.

---

## Phase 0 — Promote Check #5 (no experiment, code hygiene)

Adopt the 5-second LP at inference through a proper code change, not
the rescue hack.

1. Add `lp_time_limit_sec: float = 20.0` to the `scripts/run_gnnplus_packet_sdn_full.py::solve_selected_path_lp_safe` signature.
2. Thread it through every call site (11 runner sites currently pass
   literal `20`). Default stays `20` for training LP calls; eval path
   sets it to `5`.
3. Remove the `GNNPLUS_LP_TIME_LIMIT` env-var override from
   `solve_selected_path_lp_safe` (the rescue patch at commit
   `045122c`). The env-var can stay as a deployment lever; the
   hardcoded literal must go.
4. Re-run one topology (abilene) after the refactor to verify MLU is
   still bit-identical to Task A.

**No pre-registered verdict table** — this is code-move work, the
experimental result is already in `rescue_e5_notes.md`.

**Artifacts:** PR against `nobel-stability-fix` with diff + smoke test.

---

## Phase 1 — Inference-only knobs (no retrain)

All three knobs operate on the **same Task A checkpoint**. OFAT: run
each knob alone first to attribute effect, then combined.

### 1.1 Sticky-selection post-filter

**Location:** insert between `gnnplus_select_stateful()` return at
`run_gnnplus_improved_fixedk40_experiment.py:3219-3233` and the
`apply_do_no_harm_gate()` call at line 3234.

**Logic:** after GNN+ picks top-K ODs, compare against the previous
step's selection (available via
`gnnplus_state["prev_selected_indicator"]`). For each OD in the new
selection that was **not** in the previous selection, check whether
swapping it out for the lowest-ranked *previously-selected* OD still
in the active candidate pool would cost ≤ `GNNPLUS_STICKY_EPS` of
estimated MLU. If yes, keep the old OD.

**Env var:** `GNNPLUS_STICKY_EPS` (default `0.005`, meaning "don't swap
in a new flow if the MLU win is less than 0.5 %").

**Safety:** if the sticky swap would *increase* estimated MLU above the
do-no-harm threshold relative to bottleneck, fall back to the raw
selection.

### 1.2 Continuity-bonus env override

**Location:** the `continuity_bonus` argument at
`phase1_reactive/drl/gnn_plus_selector.py:834` and
`gnnplus_select_stateful` caller. Currently hardcoded to the `cfg`
default (which is 0.05 on training but 0.0 at inference unless the
caller passes it).

**Logic:** read `GNNPLUS_CONTINUITY_BONUS` env var, pass it through to
`select_critical_flows`. Sweep values `{0.0, 0.05, 0.10, 0.15}`.

**Interaction with 1.1:** continuity bonus acts pre-argsort, sticky
filter acts post-argsort. They compose.

### 1.3 Disturbance-aware do-no-harm tiebreak

**Location:** `run_gnnplus_improved_fixedk40_experiment.py:3163`, the
line `if gnn_est_mlu > float(threshold) * bn_est_mlu:`.

**Logic:** when `|gnn_est_mlu − bn_est_mlu| / bn_est_mlu ≤
GNNPLUS_DISTURB_TIEBREAK_EPS` (default `0.005`), compute the flow
change of each candidate against the currently-applied splits and
return whichever has fewer flow updates. Break true ties in favor of
GNN+ (preserves existing behavior).

**Env var:** `GNNPLUS_DISTURB_TIEBREAK_EPS`.

### Phase 1 pre-registered verdict

| Guardrail                                      | Threshold                                                      | Rollback if violated |
|---                                             |---                                                             |---                   |
| Seen MLU regression                            | > 0.5 % on any topo                                           | revert               |
| Unseen MLU regression                          | > 1.0 % on any topo                                           | revert               |
| Seen decision-time regression (mean)           | > +15 % vs Task A                                             | revert               |
| Disturbance: beat FlexEntry on at least 3 of 5 | cernet, ebone, sprintlink, tiscali, germany50                 | Phase 1 verdict FAIL (but may still be worth keeping for partial gains) |
| Disturbance: no regression on the 3 topos we already win | abilene, geant, vtlwavenet2011                     | revert               |

**PASS** = disturbance target met (≥3/5) AND no guardrail violated.
If we pass this bar, Phase 1 is promoted and we stop here.

**FAIL** = move to Phase 2.

---

## Phase 2 — Retrain with stronger disturbance reward

Only triggered if Phase 1 does not beat FlexEntry on 3 / 5.

### Knobs (applied jointly)

1. `W_REWARD_DISTURBANCE`: 0.15 → **0.25** (Check #1 from the rescue
   log). Retrained from Task A warm-start, not scratch.
2. `PER_TOPO_REWARD_NORM`: `False` → **`True`** (Check #4). Equalizes
   the per-topology reward scale so large-disturbance topos like
   germany50 contribute proportionally to the loss.
3. Keep `w_reward_mlu` and `w_reward_decision` unchanged (protects the
   MLU dominance and decision-time wins).
4. Keep KL anchoring to Task A policy: β_normal = 0.01, β_failure =
   0.002, same as Task A training. KL anchor prevents the policy from
   drifting off the Pareto frontier we already have.

### Phase 2 pre-registered verdict

| Guardrail                                  | Threshold                                              | Rollback if violated |
|---                                         |---                                                     |---                   |
| Seen MLU regression                        | > 1.0 % on any topo                                    | revert               |
| Unseen MLU regression                      | > 2.0 % on any topo                                    | revert               |
| Disturbance: beat FlexEntry on all 8 topos | cernet, ebone, sprintlink, tiscali, germany50 + existing 3 | Phase 2 verdict FAIL |

**PASS** = 8 / 8 disturbance wins AND MLU guardrails held.

**FAIL** = move to Phase 3.

---

## Phase 3 — Previous-selection as an input feature

Only triggered if Phase 2 still misses on ≥1 topology.

### Changes

1. Add a `prev_selected_indicator` column to the OD feature tensor
   built in `build_od_features_plus()` (new feature index; bump
   `od_dim`).
2. Add a hinge term to the training loss:
   `λ · max(0, |selected_t ⊕ selected_{t-1}| − K * tolerance)` where
   `tolerance ∈ {0.3, 0.5}`. Sweep `λ ∈ {0.01, 0.05}`.
3. Retrain from scratch (cannot warm-start: input shape changed).

### Phase 3 pre-registered verdict

Same disturbance bar as Phase 2. This is the "nuclear option" — if it
doesn't work, the claim becomes "GNN+ beats every baseline on all
topologies on MLU+decision-time, and beats FlexEntry on disturbance on
N / 8 topologies". Still a clean paper claim; just not the full dominance.

---

## Attribution rules (apply to every phase)

- **One-factor-at-a-time first** on a rescue branch (same pattern as
  Check #2 and Check #5). Combine only after each factor is
  attributed.
- Every experiment writes an honest notes file at
  `results/<tag>/<tag>_notes.md` with the same structure as
  `rescue_e5_notes.md` (setup, hypothesis, primary result, honest
  caveats, what this buys us, rollback-criteria table).
- No experiment gets promoted to `nobel-stability-fix` until its notes
  file is written and the verdict is PASS.
- Decision-time numbers are noisy on a laptop; don't quote per-topology
  decision-time deltas as an effect of a disturbance knob unless the
  direction is consistent across ≥6 / 8 topos.

---

## Current state

- Phase 0: pending (plumbing work, blocks a clean PR).
- Phase 1: starting now on rescue sub-branch `disturbance-phase1` off
  `gnnplus-debug-rescue`.
- Phase 2: not started.
- Phase 3: not started.

## Artifacts to be produced

- `scripts/run_gnnplus_improved_fixedk40_experiment.py` — sticky filter
  + do-no-harm tiebreak hooks, env-var gated
- `phase1_reactive/drl/gnn_plus_selector.py` — no change in Phase 1
  (continuity bonus is already parameterized; just need caller to pass
  the env-var value)
- `logs/rescue/compare_p1.py` — delta table vs baseline, with the
  Phase 1 verdict rule encoded
- `results/rescue_p1_*/rescue_p1_*_notes.md` — one file per sub-knob
  run (sticky alone, continuity alone, tiebreak alone, combined)
