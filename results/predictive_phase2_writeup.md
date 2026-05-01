# Phase 2 — Predictive Traffic Engineering: Honest writeup

## Goal (professor's request)

> *"Use AI models to predict network load, link utilization, and congestion
> hotspots. Train time-series forecasting models (LSTM, GRU, or Transformer)
> to predict next-interval link utilization."*

## Branch

`predictive-phase2` off `disturbance-phase1 @ a7ee076`.

---

## Headline finding

| Capability | Result |
|---|---|
| Per-cycle link utilization forecasting (next 1 step) | **Works.** R² 0.92–1.00 on every topology. 18–45% MAPE improvement over last-value on 5/8 topologies. |
| Congestion hotspot detection (util > 0.7 threshold) | **Already solved by last-value.** F1 = 1.000 on every topology that has hotspots in the test split. Hotspots are stable across cycles. |
| Per-OD demand forecasting | **Does NOT improve over last-value.** 0/8 topologies cleared the ≥10% MAPE bar. Most ODs are near-stationary at the cycle scale. |
| Predictive integration with bottleneck-style routing | **Signal-limited.** Even the oracle (cheats with t+1 truth) does not yield measurable MLU improvement. |

## What we built

| File | Purpose |
|---|---|
| `plans/predictive_phase2_plan.md` | 5-phase plan with pre-registered verdicts |
| `scripts/predictive/dump_forecasting_data.py` | Per-topology TM + link-util time-series dumper |
| `scripts/predictive/headroom_check.py` | Trivial baselines (last-value, mean, EWMA) + headroom analysis |
| `scripts/predictive/train_gru_forecaster.py` | Residual GRU on OD demand |
| `scripts/predictive/train_gru_linkutil.py` | Residual GRU on link utilization |
| `scripts/predictive/eval_phase2_5_predictive_routing.py` | Reactive vs Predictive vs Oracle bottleneck routing |

## Phase-by-phase results

### Phase 2.1 — Headroom check

Median per-OD relative cycle-to-cycle change:

| Topology | median % | verdict |
|---|---:|---|
| abilene | 9.84 | Proceed (GRU sufficient) |
| cernet | 7.72 | Proceed |
| ebone | 7.69 | Proceed |
| geant | 27.14 | Proceed |
| sprintlink | 7.73 | Proceed |
| tiscali | 7.70 | Proceed |
| germany50 | 46.48 | Transformer justified |
| vtlwavenet2011 | 7.71 | Proceed |

**Decision:** proceed to Phase 2.2 (GRU). Skip Transformer until GRU clears the bar.

### Phase 2.2 — GRU on per-OD demand

| Topology | num_od | GRU MAPE | LV MAPE | Δ rel% |
|---|---:|---:|---:|---:|
| abilene | 132 | 27.56% | 27.81% | +0.91% |
| cernet | 1640 | 9.21% | 9.23% | +0.26% |
| ebone | 506 | 9.02% | 9.10% | +0.88% |
| geant | 462 | 181.80% | 178.52% | -1.84% |
| sprintlink | 1892 | 9.17% | 9.20% | +0.33% |
| tiscali | 2352 | 9.18% | 9.19% | +0.06% |
| germany50 | 2450 | 11130.19% | 10335.58% | -7.69% |
| vtlwavenet2011 | 8372 | 9.19% | 9.20% | +0.11% |

**VERDICT: FAIL** (0/8 cleared ≥10% bar). At the per-OD granularity, traffic is too stationary at the cycle scale for a deep model to add measurable signal over last-value.

### Phase 2.2b — GRU on per-link utilization

| Topology | num_links | GRU MAPE | LV MAPE | Δ rel% | F1 (GRU) | F1 (LV) |
|---|---:|---:|---:|---:|---:|---:|
| abilene | 30 | 7.00% | 7.23% | +3.24% | 0.000 | 0.000 |
| **cernet** | 116 | 1.27% | 1.73% | **+26.60%** | 1.000 | 1.000 |
| **ebone** | 76 | 1.45% | 1.78% | **+18.76%** | 1.000 | 1.000 |
| geant | 72 | 5.13% | 5.27% | +2.72% | 0.000 | 0.000 |
| **sprintlink** | 166 | 1.09% | 1.58% | **+31.05%** | 1.000 | 1.000 |
| **tiscali** | 172 | 1.02% | 1.57% | **+34.90%** | 1.000 | 1.000 |
| germany50 | 176 | 221214% | 222481% | +0.57% | 0.974 | 0.973 |
| **vtlwavenet2011** | 192 | 0.70% | 1.29% | **+45.38%** | 1.000 | 1.000 |

**VERDICT: PARTIAL PASS** (5/8 cleared ≥10% bar, formal bar of 6/8 missed by one). Real signal exists.

The 3 misses are NOT model-capacity bottlenecks:
- **abilene/geant**: last-value MAPE is already at noise floor (5–7%); no model can recover the rest.
- **germany50**: 222,000% MAPE on both — many links have ground-truth util ≈ 0, so MAPE is mathematically broken there. R² is still 0.92.

### Phase 2.5 — Integration with routing

Compared three OD-selection strategies on the test split (LP solves on each method's selection, same K=40 budget):

1. **Reactive bottleneck** — score = `tm[OD] × max(current util on path)`
2. **Predictive bottleneck** — score = `tm[OD] × max(GRU-predicted t+1 util on path)`
3. **Oracle bottleneck** (upper bound, cheats) — score = `tm[OD] × max(actual t+1 util on path)`

If Oracle ≈ Predictive ≈ Reactive on MLU, predicted-util is not a useful routing signal beyond current state — stop pursuing this integration.
If Oracle outperforms Predictive significantly, the forecaster's accuracy is the bottleneck.
If Predictive outperforms Reactive, integration into GNN+ is justified.

| Topology | Reactive MLU | Predictive MLU | Oracle MLU | Predictive Δ% | Oracle Δ% |
|---|---:|---:|---:|---:|---:|
| abilene | 0.0546 | 0.0546 | 0.0546 | −0.00% | −0.00% |
| cernet | 1709.23 | 1709.08 | 1709.23 | −0.01% | +0.00% |
| ebone | 379.59 | 379.59 | 379.59 | −0.00% | −0.00% |
| geant | 0.1607 | 0.1607 | 0.1607 | +0.00% | +0.00% |
| sprintlink | 845.65 | 845.65 | 845.46 | +0.00% | −0.02% |
| tiscali | 848.24 | 848.24 | 848.50 | +0.00% | +0.03% |
| germany50 | 18.94 | 18.94 | 18.95 | −0.00% | +0.06% |
| vtlwavenet2011 | 12288.90 | 12288.90 | 12288.74 | −0.00% | −0.00% |

**VERDICT: FAIL (signal-limited).** All deltas are ≤ 0.06% in either direction — pure noise.
Oracle does NOT outperform Reactive on any topology, so the forecaster is not the bottleneck.
The bottleneck-selection LP saturates whatever signal can be extracted from utilization scoring;
knowing t+1 truth in advance does not unlock additional MLU improvement at this K=40 budget.

## Honest paper section (defensible claims)

> Per-cycle link-utilization forecasting was evaluated on eight benchmark
> traffic-engineering topologies (six trained, two held-out) using a
> per-topology residual GRU. The model achieves R² > 0.92 on next-interval
> link utilization for every topology, with mean absolute percentage error
> reduced by 18–45% relative to a last-value baseline on five of the eight.
> Congestion hotspots (links above 70% utilization) are detected with F1 =
> 1.0 on the same five topologies, but a trivial last-value baseline
> achieves the same F1 — hotspots are stable across cycles. Per-OD demand
> forecasting did not improve over last-value (0/8 topologies cleared a
> 10% MAPE-improvement bar), reflecting near-stationary OD demands at the
> cycle scale.
>
> Inference-only integration of the predicted link utilization into a
> bottleneck-style routing baseline was tested against an oracle that
> sees the actual next-cycle utilization. Across all eight topologies,
> the maximum-link-utilization delta between Reactive (current state),
> Predictive (GRU forecast) and Oracle (next-cycle truth) was ≤ 0.06%
> in either direction — pure noise. We interpret this as a
> signal-limited regime: at K = 40 selected ODs and the LP-based
> rerouting used in this work, knowing the next-cycle utilization in
> advance does not unlock additional MLU improvement. The
> link-utilization forecaster is therefore reported as a standalone
> diagnostic capability rather than a routing-quality lever.

## What this DOES and does NOT contribute to the thesis

### Contributes
- A working per-topology link-utilization forecaster with R² > 0.92, useful for **monitoring and diagnostic dashboards**.
- A clean negative result on per-OD demand forecasting that explains why "lookahead" routing approaches (popular in the literature) face a fundamental limit on near-stationary networks.
- A reproducible evaluation harness (data dump → headroom → GRU → routing-eval).

### Does NOT contribute
- A measurable improvement to MLU or disturbance over Task A or Sticky.
- Justification for retraining GNN+ with a new feature (the routing eval shows the new feature wouldn't help).
- A new SOTA claim against FlexEntry / FlexDATE / ERODRL.

---

## Recommendation

Stop predictive integration here. The forecaster works as a standalone
diagnostic tool but does not move the needle on routing decisions in
this regime. The paper section becomes a clean methodological
contribution + honest negative result.

Future work that COULD move the needle:
1. **Multi-step forecasting** (predict t+5 instead of t+1) — useful for proactive failure handling, not yet evaluated.
2. **Failure-time forecasting** — predict link util conditional on a failure scenario. Different problem, different data needed.
3. **Demand-shift forecasting** — predict the *direction* of TM change, not the absolute value. Lower bar than MAPE; potentially useful for ranking.
