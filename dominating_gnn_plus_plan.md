# Making GNN+ Dominant: Diagnosis of the Archfix Run and a New Execution Plan

Author note: this plan is written after reading `fixing_gnn_plus_approch.md` (the Opus 4.6 plan that was executed on Codex) and the actual post-training results in `results/gnnplus_archfix_fulltrain/ARCHFIX_RUN_SUMMARY.md`. It explains **why the five changes did not produce universal dominance**, and specifies a concrete follow-up plan that can be executed on Codex.

---

## 1. What the Archfix Run Actually Tells Us

Headline numbers from the completed run:

| Metric (bundle mean) | Previous | Archfix | Verdict |
|---|---:|---:|---|
| Mean MLU | 2009.10 | **2011.47** | worse |
| Mean Disturbance | 0.0689 | **0.0785** | worse |
| Mean Decision Time (ms) | 93.31 | **79.68** | better |

Per-topology normal MLU vs Bottleneck:

- GNN+ wins only on `Sprintlink`, `Tiscali`.
- GNN+ loses on `Abilene, CERNET, GEANT, Germany50, VtlWavenet2011, Ebone` (tie or behind Bottleneck).
- The two unseen topologies (`Germany50`, `VtlWavenet2011`) are still Bottleneck-dominated.

Training side (from `training_summary.json`):

- Supervised best epoch **12**, best val loss **2.1154** (mediocre — the supervised head never converged tightly).
- **RL best epoch = 1**, best val MLU `566.96`. This is the single most important signal in the whole run: *RL essentially collapsed after the very first epoch*. That means the new architecture (gated residual + cross-OD attention + synthetic failures) was never effectively fine-tuned by the RL reward. Every downstream number is being produced by a model that is basically a supervised imitator of oracle labels, not an RL-optimized policy.

This is consistent with the observed metrics:

- Decision time improved → because it is a pure code/architecture optimization (prefiltering + warm-start), independent of whether RL converged.
- MLU did not improve → RL is what turns a reasonable supervised selector into an *LP-objective-aware* selector. Without RL, the model cannot learn the combinatorial "which 40 ODs produce the best LP" signal that distinguishes GNN+ from Bottleneck.
- Disturbance got worse → the cross-OD attention changed the ranking each step, but nothing in the loss tells it to be temporally smooth. With no RL signal weighting continuity, the selected set churns.

## 2. Root Causes the Previous Plan Missed

The Opus 4.6 plan focused on **architecture and speed**. It was correct that prefiltering + warm-start were the right speed moves, and they landed. But it underestimated three *training-side* failure modes, and one *inference-side* safety issue:

1. **RL collapse with the new architecture.** Gated residual + cross-OD attention together roughly quadruple the number of parameters that are active at the final scoring stage (gate MLP, attention QKV, attention output). The reward landscape for K=40 top-k selection is already non-smooth (discrete subset). Adding capacity without adjusting RL hyperparameters (entropy, KL to supervised policy, learning rate, clip range) is the textbook setup for early-epoch divergence followed by a degenerate policy. RL stopping at epoch 1 is the symptom.

2. **Synthetic failure probability p=0.2 is too aggressive.** 20% of training batches contain 1–3 failed edges. Feature distributions during failure are very different from normal — utilization spikes on surviving paths, bottleneck identities change, path-shrink features fire. The supervised + RL losses are forced to average across two regimes. On Bottleneck-optimal normal topologies (Abilene, GEANT, CERNET, small GNN-favorable ones), this shifts the selector toward "failure-style" features even when the scenario is normal → it under-weights the steady-state bottleneck signal. That explains why normal-condition MLU regressed on exactly the topologies where Bottleneck was already near-optimal.

3. **Gated residual with no "do-no-harm" floor.** The gate was trained but never *constrained* to pass through Bottleneck when uncertain. If the gate lands at 0.3 on an unseen topology, the model still injects 30% of a noisy GNN correction. The right failure mode is *strict*: gate must be able to go to exactly 0 and stay there on OOD inputs.

4. **Cross-OD attention with no disturbance prior.** The attention layer can freely reshuffle the top-K each step. Nothing in the loss says "if last step's selection was nearly as good, keep it." That is precisely what increased `Mean Disturbance` from 0.069 to 0.079.

5. **Candidate-aware training targets are projected, but the 2K=80 window can drop the oracle.** If the bottleneck prefilter ranks the oracle OD at 85, the supervised loss never sees it. Over training this biases the supervised head to agree with bottleneck, which is another reason the RL head is the only one that could have added value, and it collapsed.

## 3. What "Dominant" Has to Mean, Precisely

Before planning, pin the target. For this project "GNN+ dominates" must mean, on the four-method bundle:

- Normal MLU: GNN+ ≤ Bottleneck on every topology (ties allowed, small losses ≤ 0.2% allowed).
- Normal MLU: GNN+ < Bottleneck by a measurable margin on ≥ 5 of 8 topologies.
- Failure MLU: GNN+ ≤ best-of-{Bottleneck, Original GNN} on every topology × scenario (ties allowed).
- Disturbance: GNN+ ≤ Original GNN on every topology (ECMP is trivially zero and not a target).
- Decision time: GNN+ within 1.5× Bottleneck on every topology.

Archfix already meets the decision-time target. Everything in this plan is aimed at the MLU and Disturbance targets, without regressing decision time.

## 4. The New Plan (Execute in This Order on Codex)

The plan is organized into three tracks. Track A is a **must-land hotfix** to retraining. Track B is **inference-time safety** to stop regressions on OOD topologies. Track C is **architecture/loss refinements** that raise the ceiling.

### Track A — Retraining Hotfix (do this first, the rest is moot without it)

**A1. Fix RL collapse with a KL-anchored, entropy-bonused curriculum.**

- Load the supervised-best checkpoint (`gnn_plus_supervised_improved.pt`) as the RL starting point **and** as a frozen reference policy `π_ref`.
- During RL, add a KL penalty `β · KL(π_θ || π_ref)` to the selector's soft score distribution, with `β` scheduled from `0.10` (epoch 1) down to `0.01` (epoch 10). This is the single change most likely to prevent the epoch-1 collapse.
- Add an entropy bonus `−η · H(π_θ)` with `η = 0.01` to keep exploration of alternative OD sets alive.
- Clip RL gradient updates to the gate and attention parameters by a small factor (`grad_clip = 0.5`) for the first 3 epochs; full `1.0` afterward. Prevents the new heads from destroying the supervised prior in the first few SGD steps.
- Halve the RL learning rate vs archfix (e.g. `3e-5` → `1.5e-5`).
- Require a minimum of 20 RL epochs with early-stop patience ≥ 5. The fact that archfix "best = epoch 1" means patience was too low or divergence was never recovered; the KL anchor above should fix the underlying issue.

**A2. Drop synthetic failure probability and stratify.**

- Change synthetic failure probability from `0.2` to `0.08`.
- Stratify: half of failure samples come from known topologies, half come from an augmented set where we perturb edge capacities by ±20%. This teaches "distributional robustness" without drowning the normal signal.
- Inside the failure sample, reduce `num_failed_edges` to `1..2` (drop 3). Three simultaneous edge failures is rare in the evaluation suite and overrepresents the worst case.

**A3. Expand the candidate window during training only.**

- At training time use `2K = 120` (not 80). At inference use `80`.
- Rationale: the supervised target must actually be inside the candidate window so the loss can learn the correct ranking. A wider training window lowers the "oracle-outside-window" rate (measure it — log `frac_oracle_in_window` each epoch). At inference, 80 is still safe because the model has been trained to score well on the oracle when it appears.

**A4. Add an explicit temporal-continuity term to the RL reward.**

The archfix reward already has a continuity bonus (0.05). That is additive and easy for the policy to discount. Replace it with a *multiplicative* continuity modulation on the disturbance term:

```
reward = r_mlu − w_dist · disturbance · (1 + 2 · churn_ratio)
```

where `churn_ratio = |selected_t \ selected_{t−1}| / K`. This turns disturbance penalty up when churn is high and is the direct mechanism for fixing the 0.069 → 0.079 regression.

**A5. Candidate-space supervised loss fix.**

When the oracle OD is outside the 120-candidate window, do *not* simply drop the sample. Instead:

- Add an auxiliary "recall" loss: cross-entropy over the bottleneck pre-score where the oracle gets a +1 target. This supervises the *prefilter* indirectly — the model is pushed to produce node embeddings that correlate with oracle criticality, which improves the bottleneck-replacement head over time (see C2).
- Log `frac_oracle_in_window` and require it to exceed `0.92` on the validation set before RL starts. If it does not, widen the training window.

### Track B — Inference-Time Safety (no retraining needed; ship even before A finishes)

**B1. Hard "do-no-harm" gate floor.**

At inference, before applying GNN+, compute a cheap sanity check:

```
est_bottleneck_mlu  = cheap LP lower-bound on the bottleneck top-K
est_gnn_mlu         = cheap LP lower-bound on the GNN+ top-K
if est_gnn_mlu > est_bottleneck_mlu · 1.02:
    fall back to bottleneck top-K
```

The "cheap LP lower-bound" can be the primal value of the previous step's warm-started LP with only the differing ODs relaxed (a few tens of ms). On topologies where GNN+ is already winning this gate almost never triggers. On VtlWavenet/Germany50 where it regresses, this gate will flip to Bottleneck and turn the regression into a tie.

**B2. Disturbance-aware tie-breaking.**

In the final top-K selection, after scoring, apply:

```
final_rank_score[i] = score[i] + ε · I[i ∈ selected_{t−1}]
```

with `ε` set to the 5th percentile of the score gap distribution (compute once offline per topology). This keeps previously selected ODs when the score gap is within noise, directly reducing disturbance with no MLU cost on average.

**B3. Per-topology gate temperature calibration.**

Fit a single scalar temperature `τ_top` per topology on a tiny (≤200 timestep) calibration set, minimizing validation MLU. Apply as `gate = σ(logit / τ_top)`. This is the zero-shot-safe version of "make the gate more conservative on OOD topologies" — it adds no parameters and ships as a JSON sidecar.

### Track C — Architecture and Loss Refinements (after Track A retraining lands)

**C1. Mixture-of-experts selector with a Bottleneck expert.**

Replace the gated residual with a 2-expert MoE:

```
expert_0 = bottleneck_only_score          (deterministic, no learnable params)
expert_1 = gnn_full_score                 (current GNN+ pipeline)
w_0, w_1 = softmax(gate_logits)           # per-timestep
final_score = w_0 · expert_0 + w_1 · expert_1
```

Constrain `w_0 ≥ 0.1` (floor) and regularize `−log(w_0)` with a small coefficient. This is structurally the same as the current gated residual, but the bottleneck expert is *exactly Bottleneck* (not "bottleneck plus sensitivity") so `w_0 = 1` collapses to Bottleneck *exactly*. That is the "do no harm" property the current architecture lacks.

**C2. Auxiliary bottleneck-replacement head.**

Add a small head that predicts the Bottleneck pre-score from the GNN node embeddings, trained with MSE as an auxiliary loss. Two benefits:

- The node embeddings are *pushed* to contain the bottleneck signal, so the GNN expert no longer has to re-learn it from scratch.
- At inference we can use this head (not the O(OD) bottleneck compute) as the prefilter. Saves another few ms on VtlWavenet.

**C3. Set-level training target (Plackett–Luce over the top-K).**

The current supervised loss scores each OD against an oracle per-OD label. That is an *independent* loss on a *combinatorial* target — the exact mismatch criticized in the Opus 4.6 plan, Section 3. Replace it with a Plackett–Luce listwise loss over the oracle top-K set:

```
L_PL = − Σ_{k=1..K} log( exp(score[π_k]) / Σ_{j∉{π_1..π_{k-1}}} exp(score[j]) )
```

This directly teaches the network to rank the oracle top-K *as a set*, which is what the LP actually consumes. This single change is the most likely to unlock the "beat bottleneck on multiple topologies" milestone.

**C4. Shrink cross-OD attention to what it must learn.**

Archfix uses 4 heads on 80 candidates. That is overparameterized for the signal it must capture (substitute vs complement ODs sharing bottleneck edges). Simplify:

- 2 heads, not 4.
- Add an edge-sharing bias: `attn_bias[i,j] = +b · I[path_set(i) ∩ path_set(j) ≠ ∅]` with a single learnable `b`. This hard-codes the inductive prior and lets the heads focus on residual signal.

**C5. Disturbance term at the supervised level.**

Even before RL, include a mini temporal consistency term in supervised pretraining: for two consecutive timesteps `t−1, t` in the same trajectory, penalize the symmetric difference of predicted top-K. Coefficient `0.02`. This seeds the pre-RL policy with lower churn so RL does not have to undo bad habits.

## 5. Expected Outcome If All of the Above Lands

Conservative projection, vs the archfix numbers:

| Metric | Archfix | Target after plan |
|---|---:|---:|
| Normal MLU (bundle) | 2011.47 | ≤ 1998 (Bottleneck: ≈ 2000) |
| Disturbance (bundle) | 0.0785 | ≤ 0.060 |
| Decision time (bundle, ms) | 79.68 | ≤ 82 (≈ flat) |
| # topologies where GNN+ is best or tied on normal MLU | 2–4 | ≥ 6 |
| # failure scenarios where GNN+ is best-or-tied | many | similar or better |
| Worst-case OOD regression (VtlWavenet normal MLU) | +22.99 | ≤ +2.00 (B1 gate) |

The single most important deliverable is **A1 + C3 + B1**:

- A1 makes RL actually train.
- C3 makes the supervised loss consistent with the LP target.
- B1 removes all catastrophic OOD regressions at inference.

## 6. Execution Checklist for Codex

Do these in order. Each line is a separate Codex task; do not batch A with C.

1. Implement A2 (synthetic failure prob 0.08, 1–2 edges, ±20% capacity perturbation stratification).
2. Implement A3 (training candidate window 120; inference window 80; log `frac_oracle_in_window`).
3. Implement A5 auxiliary recall loss.
4. Implement A1 (KL anchor to `π_ref` = supervised checkpoint, entropy bonus, grad-clip schedule, lr halved, 20 RL epochs, patience 5).
5. Implement A4 (multiplicative churn-aware disturbance penalty in the RL reward).
6. Retrain. Record `frac_oracle_in_window`, supervised val loss, RL val MLU per epoch, and the RL best epoch. If RL best epoch is again ≤ 2, stop and investigate KL coefficient before proceeding.
7. Implement B1 (LP-lower-bound fallback gate at inference). Test first without retrained weights, as a safety rail on the archfix checkpoint. Expect VtlWavenet regression to disappear.
8. Implement B2 (disturbance-aware tie-breaking at inference).
9. Implement B3 (per-topology gate temperature, calibrated on a 200-step held-out slice).
10. Re-run the full benchmark, regenerate the report, regenerate `overall_bundle_comparison.csv`.
11. If Section 5 targets are met, stop. Otherwise proceed to Track C (C1→C3→C4→C5) in that order, retraining after each.

## 7. What Not to Do

- Do not raise the continuity bonus further. A4 already subsumes it multiplicatively.
- Do not retrain Track C changes without Track A in place. Track C raises the ceiling; Track A raises the floor. Ceiling work on a broken floor is wasted compute.
- Do not widen K beyond 40. K is fixed by the LP budget of the project; changing it invalidates all comparisons.
- Do not remove prefiltering, warm-start, or the existing failure gate. Those archfix changes are correct and should remain.
- Do not train with synthetic failures above p=0.10. The normal-condition regression observed on Abilene/GEANT/CERNET is almost certainly caused by the 0.2 setting.

## 8. Honest Risk Assessment

- **A1 is the single highest-risk step.** If the KL anchor is too strong the RL head never diverges from supervised, and GNN+ remains a supervised imitator. If it is too weak, collapse recurs. Use the schedule specified (0.10 → 0.01) and log the actual KL each step; expected equilibrium KL is 0.05–0.20.
- **C3 (Plackett–Luce) changes the loss shape.** Supervised val loss numbers will not be comparable to archfix's 2.1154; report the *oracle-top-K recall@40* instead.
- **B1 gate can be too aggressive.** Track the fallback rate per topology; if it exceeds 30% on a known-good topology, loosen the 1.02 threshold to 1.05.
- There is a non-trivial chance that even with all of the above, GNN+ on `VtlWavenet2011` normal MLU remains only *tied* with Bottleneck (not better). That is an acceptable outcome for this project — the definition of dominance in Section 3 allows ties, and VtlWavenet is OOD.

---

End of plan.
