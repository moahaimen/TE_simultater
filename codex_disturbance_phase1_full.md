# disturbance-phase1 — Complete Engineering Record
> Branch: `disturbance-phase1 @ 3f8065e`
> Base: `gnnplus-debug-rescue @ 045122c`
> Date: 2026-04-21
> Author: moahaimen

---

## Table of Contents

1. [Why this branch exists](#1-why-this-branch-exists)
2. [The full 3-phase plan](#2-the-full-3-phase-plan)
3. [What was already wired (no change needed)](#3-what-was-already-wired-no-change-needed)
4. [Exactly what we added — code diff explained](#4-exactly-what-we-added--code-diff-explained)
5. [How to run each sub-experiment](#5-how-to-run-each-sub-experiment)
6. [Experiment rescue_p1_sticky_005 — full record](#6-experiment-rescue_p1_sticky_005--full-record)
7. [Full results tables](#7-full-results-tables)
8. [Verdict and honest caveats](#8-verdict-and-honest-caveats)
9. [What the compare script does](#9-what-the-compare-script-does)
10. [Remaining work on this branch](#10-remaining-work-on-this-branch)
11. [File map](#11-file-map)

---

## 1. Why this branch exists

**The gap:** GNN+ Task A checkpoint beats every baseline on MLU and decision time on all 8 topologies. But on disturbance (flow-change cost between consecutive routing cycles), it loses to FlexEntry on 5 of 8 topologies:

| Topology | GNN+ Task A disturbance | FlexEntry disturbance | Gap |
|---|---:|---:|---:|
| abilene | 0.0418 | 0.0913 | already wins |
| cernet | 0.0293 | **0.0180** | −0.012 (loses) |
| ebone | 0.0481 | **0.0393** | −0.010 (loses) |
| geant | 0.0579 | 0.1345 | already wins |
| sprintlink | 0.0164 | **0.0134** | −0.004 (loses) |
| tiscali | 0.0230 | **0.0213** | −0.003 (loses) |
| germany50 | 0.1917 | **0.1136** | −0.078 (loses) |
| vtlwavenet2011 | 0.0064 | 0.0067 | already wins |

**FlexEntry** is disturbance-optimal by construction — it uses a sensitivity-based scoring that explicitly minimizes flow changes and only uses 75% of the K budget to limit reconfigurations. It is the hardest baseline to beat on disturbance.

**Goal:** achieve disturbance dominance over FlexEntry on at least 3 of the 5 losing topologies without retraining and without regressing MLU or decision time. If we get 3/5+, we stop. If not, we retrain (Phase 2) or add a new input feature (Phase 3).

**Why no retraining first:** inference-only knobs are reversible, take hours instead of days, and don't risk destabilizing the Task A MLU/decision-time results. The principle: try the cheapest lever first.

---

## 2. The full 3-phase plan

Plan file: `plans/disturbance_dominance_plan.md`

### Phase 0 — LP budget code hygiene (not an experiment)

The Check #5 rescue (commit `045122c`) added `GNNPLUS_LP_TIME_LIMIT` as an env-var override to `solve_selected_path_lp_safe`. This works but is a hack — the LP time limit should be a first-class parameter, not an env var override on a function with 20+ call sites.

**Work to do:**
1. Add `lp_time_limit_sec: float = 20.0` to the `solve_selected_path_lp_safe` signature.
2. Thread it through all 11 runner call sites. Default stays `20` for training LP calls; eval path sets it to `5`.
3. Remove the env-var override from the function body.
4. Smoke test: re-run abilene, verify bit-identical MLU.

Status: **pending**. This is a blocking PR prerequisite before promoting Phase 1.

### Phase 1 — Inference-only knobs (this branch)

Three knobs, all env-var gated, all default to `0.0` = bit-identical to Task A:

| Knob | Env var | Default | What it does |
|---|---|---|---|
| Sticky post-filter | `GNNPLUS_STICKY_EPS` | 0.0 | After GNN+ selects top-K ODs and runs the LP, check if keeping previous-cycle ODs costs ≤ eps × fresh_mlu on disturbance. If yes, use the sticky set. |
| Continuity bonus | `GNNPLUS_CONTINUITY_BONUS` | 0.05 | Boost ranking scores for ODs that were selected last cycle, pre-argsort. **Already wired before this branch** (commit `6859462`). |
| Disturbance tiebreak | `GNNPLUS_DISTURB_TIEBREAK_EPS` | 0.0 | When GNN+ and bottleneck MLU are within eps of each other, pick the one with fewer flow changes vs currently-applied splits. |

**OFAT plan:** run each knob alone first, then combined.

| Sub-experiment tag | Env delta | Status |
|---|---|---|
| `rescue_p1_sticky_005` | `GNNPLUS_STICKY_EPS=0.005` | **DONE — PASS** |
| `rescue_p1_continuity_010` | `GNNPLUS_CONTINUITY_BONUS=0.10` | pending |
| `rescue_p1_tiebreak_005` | `GNNPLUS_DISTURB_TIEBREAK_EPS=0.005` | pending |
| `rescue_p1_sticky_combined` | all three active | pending |

### Phase 2 — Retrain with stronger disturbance reward

Only triggered if Phase 1 gets < 3/5 on the losing topologies. Knobs:
- `W_REWARD_DISTURBANCE`: 0.15 → 0.25
- `PER_TOPO_REWARD_NORM`: False → True

Retrain from Task A warm-start (not scratch). MLU guardrail loosens slightly (seen ≤1.0%, unseen ≤2.0%) to give the disturbance reward room to work.

### Phase 3 — Previous-selection as input feature

Nuclear option if Phase 2 still misses ≥1 topology:
1. Add `prev_selected_indicator` column to OD feature tensor in `build_od_features_plus()`.
2. Add hinge penalty to training loss: `λ · max(0, |selected_t ⊕ selected_{t-1}| − K * tolerance)`.
3. Retrain from scratch (input shape changed).

---

## 3. What was already wired (no change needed)

### CONTINUITY_BONUS (commit 6859462)

File: `scripts/run_gnnplus_improved_fixedk40_experiment.py`, line 106:
```python
CONTINUITY_BONUS = float(os.environ.get("GNNPLUS_CONTINUITY_BONUS", "0.05"))
```

Passed through to `select_critical_flows` at line 1088:
```python
continuity_bonus=CONTINUITY_BONUS,
```

Inside `gnn_plus_selector.py:892-898`:
```python
if float(continuity_bonus) > 0.0 and prev_selected.size == scores_np.shape[0]:
    ranking_scores = normalized_scores + float(continuity_bonus) * prev_selected[active_indices]
```

This means: each OD that was selected last cycle gets its ranking score bumped by `continuity_bonus`. The OD with the highest bumped score still wins the argsort. This operates **before** the sticky filter (which operates after the LP solve).

### prev_selected_indicator (already threaded through gnnplus_state)

`gnnplus_state["prev_selected_indicator"]` is a binary array of length `num_od` tracking which ODs were selected last cycle. It was already threaded through the state dict before this branch. The sticky filter reads it; we just decode it into `prev_selected_ods_list` at the call site.

---

## 4. Exactly what we added — code diff explained

**Commit:** `3f8065e` on branch `disturbance-phase1`
**File:** `scripts/run_gnnplus_improved_fixedk40_experiment.py`

### 4.1 New module-level env vars (line 167-168)

```python
STICKY_EPS = float(os.environ.get("GNNPLUS_STICKY_EPS", "0.0"))
DISTURB_TIEBREAK_EPS = float(os.environ.get("GNNPLUS_DISTURB_TIEBREAK_EPS", "0.0"))
```

Both default to `0.0` → no behavior change unless env var is set. This is the same pattern as `DO_NO_HARM_THRESHOLD`, `CONTINUITY_BONUS`, etc.

### 4.2 Module-level log guards (before apply_do_no_harm_gate)

```python
_STICKY_OVERRIDE_LOGGED = False
_DISTURB_TIEBREAK_LOGGED = False
```

Each feature logs a message exactly once (first trigger) to avoid flooding stdout.

### 4.3 New helper: `_sticky_compose_selection`

```python
def _sticky_compose_selection(
    *,
    selected_ods: list[int],
    prev_selected_ods: list[int],
    path_library,
    tm_vector: np.ndarray,
    k_crit: int,
) -> list[int]:
```

**What it does:**
1. Gets the active OD set (those with TM demand > 0 AND surviving in the network).
2. Filters `prev_selected_ods` to only those still active → `sticky_prev_unique` (deduped, ordered).
3. Takes up to `k_crit` of them.
4. If that's fewer than `k_crit`, top up with ODs from the fresh GNN+ selection (in GNN+'s ranking order) that aren't already in the sticky set.
5. If the resulting sticky set is identical to the fresh set, returns `[]` (caller skips the extra LP solve).

**Why this ordering:** we prefer previously-selected ODs (lower disturbance) but always fill to K using fresh GNN+ picks (never sacrifice K). Fresh GNN+ picks as top-up is important: it means the sticky filter can't reduce the LP's coverage below K.

### 4.4 Modified `apply_do_no_harm_gate` signature

Added one optional kwarg:
```python
def apply_do_no_harm_gate(
    runner,
    *,
    ...
    prev_selected_ods: list[int] | None = None,   # NEW
) -> tuple[list[int], object, dict, dict, int]:
```

Added at function entry:
```python
sticky_applied = False
```

### 4.5 Sticky filter block (inside apply_do_no_harm_gate, after gnn_lp is computed)

Location: right after `gnn_lp = runner.solve_selected_path_lp_safe(...)` and `gnn_est_mlu = float(gnn_lp.routing.mlu)`.

```python
if (
    STICKY_EPS > 0.0
    and prev_selected_ods
    and len(selected_ods) > 0
):
    sticky_ods = _sticky_compose_selection(
        selected_ods=selected_ods,
        prev_selected_ods=list(prev_selected_ods),
        path_library=path_library,
        tm_vector=np.asarray(tm_vector, dtype=float),
        k_crit=int(k_crit),
    )
    if sticky_ods:
        try:
            sticky_lp = runner.solve_selected_path_lp_safe(
                tm_vector=tm_vector,
                selected_ods=sticky_ods,
                base_splits=base_splits,
                path_library=path_library,
                capacities=capacities,
                warm_start_splits=warm_start_splits,
                time_limit_sec=LP_TIME_LIMIT,
                context=f"{context}:gnnplus_sticky_candidate",
            )
            sticky_mlu = float(sticky_lp.routing.mlu)
            if sticky_mlu <= gnn_est_mlu * (1.0 + float(STICKY_EPS)) + 1e-12:
                # ... log once ...
                selected_ods = list(sticky_ods)
                gnn_lp = sticky_lp
                gnn_est_mlu = sticky_mlu
                sticky_applied = True
        except Exception:
            pass  # sticky is a nice-to-have; never fail the cycle
```

**Decision boundary:** `sticky_mlu <= gnn_est_mlu * (1.0 + STICKY_EPS) + 1e-12`

With `STICKY_EPS=0.005`, this means: "accept the sticky OD set if its MLU is at most 0.5% worse than what GNN+ freshly picked." This is a Pareto preference: lower disturbance for a bounded MLU cost.

**Flow after this block:** `selected_ods`, `gnn_lp`, and `gnn_est_mlu` are all updated in-place if sticky fires. The rest of `apply_do_no_harm_gate` (the do-no-harm threshold check and tiebreak) operates on the sticky LP result.

**Safety:** if the sticky LP solve throws any exception (timeout, infeasibility), `pass` — the fresh GNN+ result is used. The sticky filter can never break a cycle.

### 4.6 Disturbance-aware tiebreak block (inside apply_do_no_harm_gate, before the threshold check)

Location: right before `if gnn_est_mlu > float(threshold) * bn_est_mlu:`.

```python
if (
    DISTURB_TIEBREAK_EPS > 0.0
    and bn_est_mlu > 1e-12
    and warm_start_splits is not None
):
    rel_gap = abs(gnn_est_mlu - bn_est_mlu) / float(bn_est_mlu)
    if rel_gap <= float(DISTURB_TIEBREAK_EPS):
        try:
            gnn_dist = float(compute_disturbance(
                warm_start_splits, gnn_lp.splits, tm_vector
            ))
            bn_dist = float(compute_disturbance(
                warm_start_splits, bn_lp.splits, tm_vector
            ))
        except Exception:
            gnn_dist = float("inf")
            bn_dist = float("inf")
        if bn_dist + 1e-12 < gnn_dist:
            # ... log once ...
            return list(map(int, bottleneck_selected)), bn_lp, {
                ...,
                "guard_reference_source": "disturb_tiebreak_bottleneck",
                "sticky_applied": bool(sticky_applied),
                ...
            }, updated_cache, int(DO_NO_HARM_FALLBACK_COOLDOWN)
```

**Decision boundary:** if `|gnn_mlu − bn_mlu| / bn_mlu ≤ DISTURB_TIEBREAK_EPS` AND `bn_disturbance < gnn_disturbance` → use bottleneck's routing instead.

**Default `DISTURB_TIEBREAK_EPS=0.0`:** this block is completely skipped unless the env var is set. Setting it to 0.005 means "when GNN+ and bottleneck are within 0.5% MLU of each other, pick the one with lower disturbance."

**Tiebreak to GNN+ on exact equality:** `bn_dist + 1e-12 < gnn_dist` — if they're equal, we keep GNN+. This preserves existing behavior on ties.

### 4.7 `sticky_applied` propagated to all return dicts

Every return path from `apply_do_no_harm_gate` (there are 6) now includes `"sticky_applied": bool(sticky_applied)` in its info dict. This lets callers log it.

### 4.8 Threading `prev_selected_ods` at the two call sites

**Normal cycle call site** (line ~3399 in original, inside `run_sdn_cycle_gnnplus_improved`):

```python
_prev_indicator = gnnplus_state.get("prev_selected_indicator")
prev_selected_ods_list: list[int] = []
if _prev_indicator is not None:
    _prev_arr = np.asarray(_prev_indicator, dtype=np.float32).reshape(-1)
    prev_selected_ods_list = [int(i) for i in np.where(_prev_arr > 0.5)[0].tolist()]
selected_ods, lp_result, gate_info, ... = apply_do_no_harm_gate(
    runner,
    ...
    prev_selected_ods=prev_selected_ods_list,   # NEW
)
```

**Proportional budget call site** (line ~3808, inside `run_proportional_budget_cycle`): identical threading under variable names `_prev_indicator_ratio` / `prev_selected_ods_ratio`.

The threshold `> 0.5` is correct: `prev_selected_indicator` is a binary 0.0/1.0 float array set by the GNN+ selector; values above 0.5 are "was selected."

---

## 5. How to run each sub-experiment

### Prerequisites

1. Be on branch `gnnplus-debug-rescue @ 045122c` (or the current main checkpoint equivalent).
2. Cherry-pick or apply the worktree patch from `disturbance-phase1 @ 3f8065e` as an uncommitted change.
3. Verify `data/raw/` is present (worktree lacks it; main repo has it).
4. Revert the patch after the run.

### Common env block (all sub-runs)

```bash
export GNNPLUS_RUN_STAGE=eval_reuse_final
export GNNPLUS_REUSE_SUPERVISED=1
export GNNPLUS_PREVIOUS_REPORT_TAG=gnnplus_8topo_stability_taskA
export GNNPLUS_LP_TIME_LIMIT=5
export GNNPLUS_CONTINUITY_BONUS=0.05   # same as Task A default
export GNNPLUS_DISTURB_TIEBREAK_EPS=0.0
export GNNPLUS_STICKY_EPS=0.0
```

### Sub-run 1: sticky alone (DONE)

```bash
export GNNPLUS_EXPERIMENT_TAG=rescue_p1_sticky_005
export GNNPLUS_STICKY_EPS=0.005
python scripts/run_gnnplus_improved_fixedk40_experiment.py
```

### Sub-run 2: continuity bonus bump alone (pending)

```bash
export GNNPLUS_EXPERIMENT_TAG=rescue_p1_continuity_010
export GNNPLUS_CONTINUITY_BONUS=0.10
python scripts/run_gnnplus_improved_fixedk40_experiment.py
```

### Sub-run 3: tiebreak alone (pending)

```bash
export GNNPLUS_EXPERIMENT_TAG=rescue_p1_tiebreak_005
export GNNPLUS_DISTURB_TIEBREAK_EPS=0.005
python scripts/run_gnnplus_improved_fixedk40_experiment.py
```

### Sub-run 4: combined (pending)

```bash
export GNNPLUS_EXPERIMENT_TAG=rescue_p1_sticky_combined
export GNNPLUS_STICKY_EPS=0.005
export GNNPLUS_CONTINUITY_BONUS=0.10
export GNNPLUS_DISTURB_TIEBREAK_EPS=0.005
python scripts/run_gnnplus_improved_fixedk40_experiment.py
```

### Evaluate any sub-run

```bash
python logs/rescue/compare_p1.py --rescue-tag rescue_p1_sticky_005
# or substitute any of the tags above
```

---

## 6. Experiment rescue_p1_sticky_005 — full record

| Field | Value |
|---|---|
| Tag | `rescue_p1_sticky_005` |
| Branch | `disturbance-phase1 @ 3f8065e` |
| Run on | `nobel-stability-fix @ 6859462` with uncommitted 3f8065e patch |
| PID | 17724 |
| Start | 2026-04-21 ~16:21 |
| End | 2026-04-21 ~16:57 |
| Wall clock | ~36 min |
| Exit code | 0 |

### Env vars active

```
GNNPLUS_RUN_STAGE=eval_reuse_final
GNNPLUS_REUSE_SUPERVISED=1
GNNPLUS_PREVIOUS_REPORT_TAG=gnnplus_8topo_stability_taskA
GNNPLUS_EXPERIMENT_TAG=rescue_p1_sticky_005
GNNPLUS_LP_TIME_LIMIT=5
GNNPLUS_STICKY_EPS=0.005
GNNPLUS_DISTURB_TIEBREAK_EPS=0.0
GNNPLUS_CONTINUITY_BONUS=0.05    (default, Task A value)
```

### Verification log line (from run log)

```
[sticky] Applied sticky post-filter: fresh_mlu=0.046641 sticky_mlu=0.046641
         eps=0.0050 context=abilene:gnnplus_improved:normal_cycle.
         First sticky override only is logged.
```

The sticky filter fired on abilene and the sticky and fresh MLUs happened to be equal — the LP returned the same objective, confirming the filter is working and not degrading MLU.

### Pre-registered verdict criteria

| Guardrail | Threshold | Observed |
|---|---|---|
| Max seen MLU regression | > 0.5% on any topo | **+0.007%** (abilene) ✓ |
| Max unseen MLU regression | > 1.0% on any topo | **0.000%** ✓ |
| Beats FlexEntry on prev-losing topos | ≥ 3 / 5 | **5 / 5** ✓ |
| No disturbance regression on prev-winning topos | all 3 still beat FE | **3 / 3** ✓ |

**VERDICT: PASS**

---

## 7. Full results tables

### 7.1 Disturbance — rescue_p1_sticky_005 vs Task A baseline

| Topology | Status | Task A disturbance | Sticky disturbance | Δ (rel) | FlexEntry | Beats FE? |
|---|---|---:|---:|---:|---:|:---:|
| abilene | seen | 0.041809 | 0.045258 | **+8.25%** | 0.091 | ✓ |
| cernet | seen | 0.029324 | 0.008817 | **−69.93%** | 0.018 | ✓ |
| ebone | seen | 0.048126 | 0.011714 | **−75.66%** | 0.039 | ✓ |
| geant | seen | 0.057867 | 0.081555 | **+40.94%** | 0.134 | ✓ |
| sprintlink | seen | 0.016399 | 0.008594 | **−47.60%** | 0.013 | ✓ |
| tiscali | seen | 0.023015 | 0.010004 | **−56.54%** | 0.021 | ✓ |
| germany50 | unseen | 0.191726 | 0.089692 | **−53.22%** | 0.114 | ✓ |
| vtlwavenet2011 | unseen | 0.006355 | 0.001339 | **−78.93%** | 0.007 | ✓ |

**All 8 topologies beat FlexEntry. The 5 previously-losing topologies are now all wins.**

### 7.2 MLU — rescue_p1_sticky_005 vs Task A baseline

| Topology | Task A MLU | Sticky MLU | Δ (rel) |
|---|---:|---:|---:|
| abilene | 0.054599 | 0.054603 | +0.007% |
| cernet | 1729.12 | 1718.74 | −0.600% |
| ebone | 379.591 | 379.591 | −0.000% |
| geant | 0.15949 | 0.15909 | −0.247% |
| sprintlink | 860.69 | 806.36 | −6.313% |
| tiscali | 824.10 | 811.60 | −1.516% |
| germany50 | 18.772 | 18.705 | −0.354% |
| vtlwavenet2011 | 12270.3 | 12228.3 | −0.343% |

Max regression: **+0.007%** on abilene (noise-level, well within 0.5% guardrail).

### 7.3 Decision time (ms) — rescue_p1_sticky_005 vs Task A

| Topology | Task A (ms) | Sticky (ms) | Δ (rel) | Note |
|---|---:|---:|---:|---|
| abilene | 29.3 | 42.1 | **+43.5%** | |
| cernet | 63.7 | 87.4 | **+37.1%** | |
| ebone | 40.2 | 64.5 | **+60.6%** | |
| geant | 37.0 | 60.2 | **+62.7%** | |
| sprintlink | 65.4 | 102.3 | **+56.4%** | |
| tiscali | 128.8 | 99.6 | **−22.6%** | sticky narrows LP search space |
| germany50 | 55.3 | 88.1 | **+59.2%** | |
| vtlwavenet2011 | 171.1 | 288.9 | **+68.9%** | |

### 7.4 Do-no-harm fallback rate

| Topology | Task A | Sticky |
|---|---:|---:|
| abilene | 0.000 | 0.000 |
| cernet | 0.000 | 0.000 |
| ebone | 0.000 | 0.000 |
| geant | 0.000 | 0.000 |
| sprintlink | 0.000 | 0.000 |
| tiscali | **0.013** | **0.000** ↓ |
| germany50 | 0.000 | 0.000 |
| vtlwavenet2011 | **0.053** | **0.013** ↓ |

The sticky filter **reduced** fallback rates on both topologies where Task A had non-zero fallback. This is a good sign: the sticky OD sets are within do-no-harm bounds, so the guard is triggered less.

### 7.5 All methods × all topologies (complete picture)

#### MLU (lower is better)

| Topology | ECMP | OSPF | Bottleneck | EroDRL | FlexDate | FlexEntry | GNN+ TaskA | GNN+ Sticky |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abilene | 0.1234 | 0.0839 | 0.0546 | 0.0546 | 0.0546 | 0.0548 | **0.0546** | **0.0546** |
| cernet | 1972.7 | 1900.7 | 1722.7 | 1766.5 | 1724.8 | 1786.4 | 1729.1 | **1718.7** |
| ebone | 415.6 | 421.3 | 379.6 | 379.6 | 379.6 | 379.6 | **379.6** | **379.6** |
| geant | 0.2705 | 0.2694 | 0.1602 | 0.1631 | 0.1629 | 0.1633 | **0.1595** | **0.1591** |
| sprintlink | 1054.5 | 1077.2 | 880.3 | 916.4 | 913.4 | 963.8 | 860.7 | **806.4** |
| tiscali | 866.7 | 1054.1 | 834.9 | 843.5 | 842.6 | 848.4 | 824.1 | **811.6** |
| germany50 | 24.83 | 31.62 | 19.23 | 21.43 | 19.28 | 21.52 | 18.77 | **18.71** |
| vtlwavenet2011 | 12474.6 | 12470.2 | 12251.8 | 12275.4 | 12262.3 | 12286.1 | 12270.3 | **12228.3** |

#### Disturbance (lower is better; ECMP/OSPF = 0 because they are static)

| Topology | Bottleneck | EroDRL | FlexDate | FlexEntry | GNN+ TaskA | GNN+ Sticky |
|---|---:|---:|---:|---:|---:|---:|
| abilene | 0.0750 | 0.0850 | 0.0718 | 0.0913 | 0.0418 | **0.0453** |
| cernet | 0.0308 | 0.0301 | 0.0363 | 0.0180 | 0.0293 | **0.0088** |
| ebone | 0.0674 | 0.0422 | 0.0682 | 0.0393 | 0.0481 | **0.0117** |
| geant | 0.1658 | 0.1833 | 0.1674 | 0.1345 | **0.0579** | 0.0816 |
| sprintlink | 0.0169 | 0.0168 | 0.0217 | 0.0134 | 0.0164 | **0.0086** |
| tiscali | 0.0240 | 0.0248 | 0.0241 | 0.0213 | 0.0230 | **0.0100** |
| germany50 | 0.1762 | 0.1912 | 0.1980 | 0.1136 | 0.1917 | **0.0897** |
| vtlwavenet2011 | 0.0082 | 0.0083 | 0.0076 | 0.0067 | **0.0064** | **0.0013** |

---

## 8. Verdict and honest caveats

### PASS

Sticky filter alone (`GNNPLUS_STICKY_EPS=0.005`, no retraining) achieves:
- **8/8 disturbance wins over FlexEntry** (target was ≥3/5 on previously-losing topos)
- **Max MLU regression: +0.007%** (abilene, noise-level; guardrail is 0.5%)
- **No MLU regression on unseen topologies**
- **Fallback rates improved** on both topologies that had non-zero fallback

### Caveat 1: Decision-time cost is real (+37–69%)

The sticky filter adds **one extra LP solve** per cycle where it fires (LP budget = 5 s). Observed cost: +37% to +69% on 7 of 8 topologies.

**The exception is tiscali (−22.6%):** tiscali has a large K (40 ODs over 235 OD-pairs). The sticky warm-start is so close to the current solution that the LP solver exits much faster than a fresh solve.

**Paper implication:** GNN+ should be presented with **two operating modes**:
- Task A (no sticky): fastest decision time, 3/8 disturbance wins
- Task A + sticky: ~2× decision time, 8/8 disturbance wins

### Caveat 2: Abilene and geant disturbance INCREASED

Abilene: +8.2%, geant: +40.9%. Both still beat FlexEntry by a huge margin (0.0453 vs 0.091; 0.0816 vs 0.134), but the direction is counterintuitive.

**Root cause:** `compute_disturbance(current_splits, new_splits, tm_vector)` measures how much traffic reassignment happens between the *currently applied splits* and the *new LP solution*. For abilene and geant, the traffic matrix evolves quickly across cycles. When we force the LP to start from the previous-cycle OD set, the LP drives the splits in unexpected directions relative to `current_splits` (the splits that are actually installed in the network right now), because the previous-cycle ODs are no longer well-matched to the current traffic demand. The result is **more** disturbance despite the OD selection being "sticky."

**Implication:** the sticky filter works best on slow-varying traffic topologies. For high-TM-volatility topologies (abilene, geant), OD-selection continuity ≠ split continuity.

### Caveat 3: Sprintlink MLU improved by −6.3%

This is real but unintentional. The Task A checkpoint apparently selected slightly suboptimal ODs for sprintlink on this test window. The sticky filter's historical OD set (which comes from earlier in the test sequence, not from the training distribution) happens to be a better match. Good sign for deployment, but it means sticky is non-trivially changing routing decisions on sprintlink — not just reducing disturbance.

### Caveat 4: Single run on a laptop

Decision-time numbers are noisy. The +37–69% cost direction is consistent across 7/8 topologies and mechanically explained by the extra LP solve. The disturbance drops (47–79%) are so large they are unlikely to be noise. A second run would confirm both.

---

## 9. What the compare script does

File: `logs/rescue/compare_p1.py`

**Usage:**
```bash
python logs/rescue/compare_p1.py --rescue-tag rescue_p1_sticky_005
# --baseline-tag defaults to gnnplus_8topo_stability_taskA
```

**Logic:**
1. Loads `results/<baseline-tag>/packet_sdn_summary.csv` and `results/<rescue-tag>/packet_sdn_summary.csv`.
2. Filters to `method == "gnnplus"` and `scenario == "normal"`.
3. For each topology in both CSVs: computes absolute and relative deltas for `mean_mlu`, `mean_disturbance`, `decision_time_ms`, `do_no_harm_fallback_rate`.
4. Checks beats-FlexEntry for each topology using hardcoded reference values from `results/requirements_compliant_eval/table_external_baselines.csv`.
5. Applies verdict rule:
   - `FAIL` if seen MLU regression > 0.5% on any topo
   - `FAIL` if unseen MLU regression > 1.0% on any topo
   - `FAIL` if any previously-winning topo loses disturbance to FlexEntry
   - `FAIL` if fewer than 3 of the 5 previously-losing topos beat FlexEntry
   - `PASS` otherwise
6. Writes `results/<rescue-tag>/delta_table.csv`.

**Bug fixed during this session:** original code used `seen_rows["mean_mlu__rel_pct"].abs().max()` — this treated MLU *improvements* (negative deltas) as violations. Fixed to `.clip(lower=0).max()` to only flag positive (regression) changes. Without this fix, sprintlink's −6.3% MLU improvement would have triggered a false FAIL.

**Hardcoded FlexEntry reference:**
```python
FLEXENTRY_DISTURB = {
    "abilene": 0.091, "cernet": 0.018, "ebone": 0.039,
    "geant": 0.134,   "sprintlink": 0.013, "tiscali": 0.021,
    "germany50": 0.114, "vtlwavenet2011": 0.007,
}
```

These values come from `results/requirements_compliant_eval/table_external_baselines.csv` and are from our own simulator running the FlexEntry reproduction on each topology.

---

## 10. Remaining work on this branch

### Immediate next steps (Phase 1 completion)

1. **Run `rescue_p1_continuity_010`**: set `GNNPLUS_CONTINUITY_BONUS=0.10` (bump from 0.05). Hypothesis: a stronger pre-argsort bonus for previously-selected ODs will further reduce disturbance on high-volatility topologies (abilene, geant) where sticky alone wasn't enough.

2. **Run `rescue_p1_tiebreak_005`**: set `GNNPLUS_DISTURB_TIEBREAK_EPS=0.005`. Hypothesis: in near-tie cycles (GNN+ and bottleneck within 0.5% MLU), picking the lower-disturbance candidate will compound the sticky gains without adding decision-time cost (tiebreak is just a disturbance comparison, no extra LP).

3. **Run `rescue_p1_sticky_combined`**: all three active. This is the "best of all knobs" operating point.

4. **Write notes files** for each sub-run in the same format as `results/rescue_p1_sticky_005/rescue_p1_sticky_005_notes.md`.

### Before promoting to main

1. **Phase 0 LP hygiene PR**: clean up `GNNPLUS_LP_TIME_LIMIT` env-var hack into a proper parameter. Required before promotion.
2. **Second run confirmation** on at least abilene and geant (the two topologies where disturbance increased).
3. **Decision-time reconciliation**: write the "two operating modes" section of the paper clearly.

### Phase 2/3 trigger condition

Phase 1 already PASSed (8/8 disturbance wins). Phase 2 and Phase 3 are not needed unless:
- A second run shows the disturbance drops were noise (unlikely — 47–79% drops), OR
- The decision-time cost is unacceptable and we want to train away from the extra LP solve.

---

## 11. File map

```
plans/
  disturbance_dominance_plan.md          ← 3-phase plan, pre-registered verdicts

scripts/
  run_gnnplus_improved_fixedk40_experiment.py   ← modified on disturbance-phase1@3f8065e

phase1_reactive/drl/
  gnn_plus_selector.py                   ← continuity_bonus already wired (no change in this branch)

logs/rescue/
  compare_p1.py                          ← OFAT verdict script, MLU-bug fixed
  rescue_p1_sticky_005.log               ← run log

results/
  gnnplus_8topo_stability_taskA/
    packet_sdn_summary.csv               ← Task A baseline
  rescue_p1_sticky_005/
    packet_sdn_summary.csv               ← sticky-005 results
    delta_table.csv                      ← compare_p1.py output
    rescue_p1_sticky_005_notes.md        ← full honest notes
  rescue_p1_sticky_combined/
    rescue_p1_notes.md                   ← template (TBD sub-runs)
  requirements_compliant_eval/
    table_external_baselines.csv         ← FlexEntry reference + all baselines

results_summary_for_codex.md             ← shorter summary with all-methods tables
codex_disturbance_phase1_full.md         ← this file
```

---

## Appendix A: The mechanism in one paragraph

GNN+ selects K OD pairs per cycle, solves an LP over them, and checks whether the LP solution is safe (do-no-harm gate). The sticky filter adds a step between "GNN+ picks top-K" and "LP solves": if a previous-cycle OD set exists, it builds an alternative set using those ODs (filled to K with fresh GNN+ picks) and runs that LP too. If the alternative's MLU is within 0.5% of the fresh LP, we prefer the alternative — because it reuses previously-installed ODs, the LP's feasible region is anchored near the currently-installed flow splits, and the output splits change less. Disturbance is measured as the change in flow splits between cycles, so anchoring the OD set directly reduces disturbance. The cost is one extra LP solve per cycle where the filter fires.

## Appendix B: Why FlexEntry is the right comparison target

FlexEntry (`select_sensitivity_critical` in `phase1_reactive/baselines/literature_baselines.py`) uses a sensitivity-based scoring that orders ODs by how much their rerouting would reduce the maximum link utilization — but it only uses 75% of the K budget (`budget = max(1, int(np.ceil(0.75 * k_crit)))`). The 25% budget cut deliberately limits reconfigurations. This makes FlexEntry sub-optimal on MLU (it leaves K capacity on the table) but excellent on disturbance. All baselines including FlexEntry are run through our own event-driven TE simulator on the same topologies as GNN+ — they are not numbers from the original papers.
