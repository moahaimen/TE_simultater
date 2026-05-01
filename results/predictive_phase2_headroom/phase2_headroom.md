# Phase 2.1 — Headroom & baseline summary

> Pre-registered verdict thresholds:
> - median per-OD relative change < 5% on ≥5 of 8 topologies → ABANDON
> - 5–30% → proceed to GRU (Phase 2.2)
> - >30% → proceed to Transformer (Phase 2.3)

## Cycle-to-cycle change (per-OD relative |Δ|)

| Topology | median % | p25 % | p75 % | p95 % | verdict |
|---|---:|---:|---:|---:|:---|
| abilene | 9.84 | 4.13 | 21.85 | 69.62 | PROCEED (GRU sufficient) |
| cernet | 7.72 | 3.67 | 13.15 | 22.69 | PROCEED (GRU sufficient) |
| ebone | 7.69 | 3.65 | 13.12 | 22.63 | PROCEED (GRU sufficient) |
| geant | 27.14 | 9.14 | 67.06 | 321.60 | PROCEED (GRU sufficient) |
| sprintlink | 7.73 | 3.66 | 13.15 | 22.62 | PROCEED (GRU sufficient) |
| tiscali | 7.70 | 3.65 | 13.13 | 22.66 | PROCEED (GRU sufficient) |
| germany50 | 46.48 | 14.88 | 99.99 | 1202.60 | PROCEED (Transformer justified) |
| vtlwavenet2011 | 7.71 | 3.65 | 13.14 | 22.65 | PROCEED (GRU sufficient) |

Abandon: 0 / 8
Proceed-GRU: 7 / 8
Proceed-Transformer: 1 / 8

## Baseline MAPE (%) on the test split

| Topology | last-value | mean | EWMA(0.3) | best | best-method |
|---|---:|---:|---:|---:|:---|
| abilene | 27.86 | 99.85 | 36.03 | **27.86** | last-value |
| cernet | 9.23 | 38.71 | 8.44 | **8.44** | EWMA(0.3) |
| ebone | 9.09 | 37.52 | 8.30 | **8.30** | EWMA(0.3) |
| geant | 177.82 | 729.66 | 266.55 | **177.82** | last-value |
| sprintlink | 9.19 | 39.09 | 8.46 | **8.46** | EWMA(0.3) |
| tiscali | 9.19 | 39.76 | 8.49 | **8.49** | EWMA(0.3) |
| germany50 | 10557.76 | 61545.03 | 15864.30 | **10557.76** | last-value |
| vtlwavenet2011 | 9.20 | 40.12 | 8.53 | **8.53** | EWMA(0.3) |

## Aggregate verdict

**PROCEED to Phase 2.3 (Transformer)** — at least one topology has >30% median per-OD change, justifying a higher-capacity model. Run Phase 2.2 (GRU) first as a fair-comparison baseline.
