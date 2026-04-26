# Failure Robustness Handoff for Claude Opus 4.7

## Acknowledge the Student Concern

The student raised a valid concern:

> “Please make sure 3-link failure is included.”

This concern is correct. Before the latest patch, the active branch did **not** have an explicit `3-link failure` scenario in the main packet-SDN evaluation path.

What existed before:

- `single_link_failure`
- `random_link_failure_1`
- `random_link_failure_2` which was only a **2-link** case
- `capacity_degradation_50`
- `traffic_spike_2x`

So the failure section was weaker than the intended thesis framing.

---

## What Has Already Been Fixed

I have already patched the active evaluation path so the official five failure scenarios are now:

1. `single_link_failure`
2. `multiple_link_failure` = 2-link failure
3. `three_link_failure` = 3-link failure
4. `capacity_degradation_50`
5. `traffic_spike_2x`

I also kept limited backward compatibility where necessary so older `random_link_failure_2` artifacts do not immediately become unusable.

### Files already patched

- `/Users/moahaimentalib/Documents/scientfic_papers/aI_sara_network/network_project/scripts/run_gnnplus_packet_sdn_full.py`
- `/Users/moahaimentalib/Documents/scientfic_papers/aI_sara_network/network_project/scripts/run_gnnplus_improved_fixedk40_experiment.py`
- `/Users/moahaimentalib/Documents/scientfic_papers/aI_sara_network/network_project/scripts/run_nobel_germany_real_eval.py`
- `/Users/moahaimentalib/Documents/scientfic_papers/aI_sara_network/network_project/scripts/build_disturbance_phase1_sarah_report.py`
- `/Users/moahaimentalib/Documents/scientfic_papers/aI_sara_network/network_project/scripts/build_gnnplus_packet_sdn_report_fixed.py`
- `/Users/moahaimentalib/Documents/scientfic_papers/aI_sara_network/network_project/scripts/run_professor_gnnplus_baselines_zeroshot.py`

I also patched a post-eval crash in the main experiment script so the rerun should finish more cleanly:

- `/Users/moahaimentalib/Documents/scientfic_papers/aI_sara_network/network_project/scripts/run_gnnplus_improved_fixedk40_experiment.py`

The crash was a missing-key issue in the legacy training-summary report path for `plackett_luce_weight_*`.

---

## Current Problem

The **code is patched**, but the **reported results are still based on the old failure scenario set**.

That means the current failure numbers in the report are now stale with respect to the intended scenario definition.

This matters because the thesis argument is not only:

- zero-shot generalization

but also:

- failure robustness

If the failure section is going to be used as a strong contribution claim, it must be rerun under the corrected five-scenario set.

---

## What You Need To Do

Your task is to take over from here and **fix the failure robustness story properly**.

### Required outcomes

1. Rerun the main `Task A` baseline with the corrected five failure scenarios.
2. Rerun the `Sticky rescue` branch with the corrected five failure scenarios.
3. Rebuild the Sarah-style report from the new CSVs.
4. Recompute the failure win/tie totals against Bottleneck from the new scenario set.
5. If needed, improve the failure behavior further so the branch is more defensible for the thesis.

### Important target from the student

The student wants the failure result to reach:

- `45/50` win-or-tie vs Bottleneck

Treat this as the performance target to pursue.

Do **not** fake or overstate the result. If `45/50` is not reachable with the current inference-only sticky logic, then diagnose why and make the smallest technically defensible change that can realistically improve the failure count.

---

## Hard Rules

### Do not rely on stale prose

The previous Sarah-style report had hardcoded phrases like `28/40`. Those were already being cleaned up to compute from the CSVs instead of freezing old numbers in text.

You must keep the failure claims **fully data-driven**.

### Do not use report-layer relabeling as a fake fix

It is not acceptable to merely rename:

- `random_link_failure_2`

to:

- `3-link failure`

The actual evaluator must really execute the 3-link scenario.

### Do not weaken the main scientific claim

The point is to strengthen:

- zero-shot generalization
- failure robustness

without making the report technically dishonest.

---

## Where the Real Failure Logic Lives

These are the most important files for the current active path.

### Main packet-SDN failure execution state

- `/Users/moahaimentalib/Documents/scientfic_papers/aI_sara_network/network_project/scripts/run_gnnplus_packet_sdn_full.py`

Important function:

- `_build_failure_execution_state(...)`

This is where the active scenario definitions are applied.

### Main improved experiment / evaluation driver

- `/Users/moahaimentalib/Documents/scientfic_papers/aI_sara_network/network_project/scripts/run_gnnplus_improved_fixedk40_experiment.py`

Important areas:

- `RL_FAILURE_SCENARIOS`
- `canonical_failure_type(...)`
- `run_failure_scenario_gnnplus_improved(...)`
- `benchmark_topology_failures_improved(...)`

### Current Sarah-style report builder

- `/Users/moahaimentalib/Documents/scientfic_papers/aI_sara_network/network_project/scripts/build_disturbance_phase1_sarah_report.py`

Important areas:

- `FAILURE_ORDER`
- `FAILURE_LABELS`
- failure win/tie aggregation
- failure prose that must remain computed, not hardcoded

### Nobel extension evaluator

- `/Users/moahaimentalib/Documents/scientfic_papers/aI_sara_network/network_project/scripts/run_nobel_germany_real_eval.py`

Only rerun this too if you want the unseen-topology failure story strengthened under the same scenario definition.

---

## Minimum Rerun Set

At minimum, rerun these:

### Baseline Task A

Use the baseline result tag:

- `gnnplus_8topo_stability_taskA`

### Sticky rescue

Use the rescue result tag:

- `rescue_p1_sticky_005`

Recommended environment settings for the sticky rerun are the same as before unless you intentionally change them:

- `GNNPLUS_STICKY_EPS=0.005`
- `GNNPLUS_CONTINUITY_BONUS=0.05`
- `GNNPLUS_DISTURB_TIEBREAK_EPS=0.0`
- `GNNPLUS_LP_TIME_LIMIT=5`
- `GNNPLUS_RUN_STAGE=eval_reuse_final`
- `GNNPLUS_REUSE_SUPERVISED=1`

If needed to save time during reruns, the branch already has support to skip the proportional-budget study:

- `GNNPLUS_SKIP_PROPORTIONAL_STUDY=1`

---

## Strategic Guidance

If the corrected five-scenario rerun still does not get close to `45/50`, focus on the weakest failure cases rather than changing everything blindly.

Things worth checking:

1. Which topologies lose most often under `multiple_link_failure` and `three_link_failure`.
2. Whether the sticky logic helps or hurts under failure compared with plain Task A.
3. Whether failure-time inference controls should be separated from normal-time controls.
4. Whether the do-no-harm gate is too weak or too permissive under link-removal scenarios.
5. Whether a small failure-specific ranking/tiebreak adjustment can raise win/tie count without wrecking normal disturbance.

If a simple inference-only change can get the failure score up, prefer that first.
If not, say so clearly and propose the smallest next step that is technically justified.

---

## Non-Negotiable Final Deliverables

When you are done, I need:

1. The new true failure totals under the corrected five-scenario set.
2. The exact per-scenario win/tie breakdown.
3. Whether `45/50` was achieved or not.
4. If not achieved, the exact blocker and the most plausible next improvement.
5. Updated report files and result directories.

---

## Bottom Line

The student was right to question the failure section.

The branch now has the scenario-definition patch, but the science is not complete until the reruns and updated totals are done.

Please take over from here and make the failure robustness story strong, honest, and thesis-defensible.
