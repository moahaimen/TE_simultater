# Phase 2 — Predictive Traffic Engineering Plan

## Goal

Use AI time-series forecasting to predict next-interval per-link
utilization, per-OD demand, and congestion hotspots — and feed those
predictions into GNN+ for one-step lookahead routing.

The professor's request decomposes into:
1. Predict next-interval link utilization (per-link time series)
2. Predict per-OD traffic demand (the input that drives link utilization)
3. Detect congestion hotspots (binary classification derived from above)

Cleanest formulation: **forecast the next-interval OD traffic matrix,
then derive per-link util and hotspot labels from it via the routing.**

---

## Phase plan

### Phase 2.0 — Data preparation

Dump per-topology forecasting datasets:
- `data/forecasting/<topo>/tm_series.npz` — shape `(num_timesteps, num_od_pairs)`
- `data/forecasting/<topo>/link_util_series.npz` — shape `(num_timesteps, num_edges)` under ECMP
- `data/forecasting/<topo>/hotspots.npz` — `util > 0.7 * cap` (binary)

Train/val/test boundary respects time order — same split as the supervised stage.

### Phase 2.1 — Honest baselines (gating step)

Three trivial baselines that any deep model must beat:

| Baseline | Formula |
|---|---|
| Last-value | `pred[t+1] = actual[t]` |
| Mean | `pred[t+1] = mean(train)` |
| EWMA (α=0.3) | `pred[t+1] = α × actual[t] + (1-α) × pred[t]` |

Compute MAPE, MAE, R² per topology. Plot the cycle-to-cycle relative
change distribution — the **headroom for any forecaster.**

**Pre-registered headroom verdict:**
- If median relative change < 5% on most topologies → no signal, abandon Phase 2
- If 10–30% → proceed to Phase 2.2 (GRU)
- If >30% → strong opportunity, proceed to Phase 2.3 (Transformer)

### Phase 2.2 — GRU per topology (multivariate per-topology)

Per-topology GRU, hidden 64–128, 2 layers. Input: last 12 cycles of all
OD demands → predict next 1 cycle. Loss: log-space MSE.

**Pre-registered verdict:** GRU must beat last-value baseline by ≥10%
MAPE on at least 6 of 8 topologies. Otherwise reject.

### Phase 2.3 — Topology-conditioned Transformer

Single Transformer encoder with topology embedding. Trained on 6 known
topologies, evaluated zero-shot on Germany50 + VtlWavenet2011.

**Pre-registered verdict:** Transformer must match GRU's per-topology
MAPE within 5% on seen topos AND not regress >25% on unseen topos.

### Phase 2.4 — Graph-conditioned forecasting

GRU/Transformer trunk + GNN over the topology graph. Captures: "if OD
A→B surges, the path A→C→B surges proportionally."

**Pre-registered verdict:** graph-aware forecaster must reduce hotspot
F1 error by ≥20% vs the GRU baseline.

### Phase 2.5 — Integration with GNN+

Add `predicted_util[t+1]` and `predicted_hotspots[t+1]` as new OD
features in `build_od_features_plus()`. Retrain GNN+ supervised+RL
warm-started from Task A.

**Pre-registered Phase 2 PASS condition:**
- Predictive GNN+ reduces MLU by ≥1% on at least 4/8 topos, OR reduces
  disturbance by ≥10% on at least 4/8 topos, OR both
- Decision time +30% max vs Task A
- No regression on Phase 1 guardrails (zero-shot, fallback rates)

---

## Honest risks

- **Traffic may be too stationary**: if median per-cycle change <5%,
  Last-value already wins. Phase 2.1 headroom plot is the abort gate.
- **Forecast horizon**: 1-step ahead matches GNN+ cycle. Multi-step is
  optional later.
- **Compute**: Phase 2.5 retraining is hours per topology, vs minutes
  for Phase 1 inference-only.

## Status

- Phase 2.0: pending (this branch — `predictive-phase2`)
- Phase 2.1: pending
- Phase 2.2 onward: gated on Phase 2.1 headroom verdict
