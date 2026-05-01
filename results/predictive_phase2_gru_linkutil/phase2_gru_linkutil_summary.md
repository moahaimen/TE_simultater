# Phase 2.2b — GRU forecaster on LINK UTILIZATION

> Pre-registered verdict: GRU must beat last-value by ≥10% MAPE on ≥6/8 topologies.

| Topology | num_links | GRU MAPE % | LV MAPE % | Δ rel % | GRU R² | F1(GRU) | F1(LV) |
|---|---:|---:|---:|---:|---:|---:|---:|
| abilene | 30 | 7.00 | 7.23 | +3.24 | 0.994 | 0.000 | 0.000 |
| cernet | 116 | 1.27 | 1.73 | +26.60 | 1.000 | 1.000 | 1.000 |
| ebone | 76 | 1.45 | 1.78 | +18.76 | 0.998 | 1.000 | 1.000 |
| geant | 72 | 5.13 | 5.27 | +2.72 | 0.996 | 0.000 | 0.000 |
| sprintlink | 166 | 1.09 | 1.58 | +31.05 | 0.999 | 1.000 | 1.000 |
| tiscali | 172 | 1.02 | 1.57 | +34.90 | 0.999 | 1.000 | 1.000 |
| germany50 | 176 | 221214.38 | 222480.96 | +0.57 | 0.920 | 0.974 | 0.973 |
| vtlwavenet2011 | 192 | 0.70 | 1.29 | +45.38 | 0.999 | 1.000 | 1.000 |

GRU beats LV by ≥10% MAPE: 5 / 8 topologies

**VERDICT: FAIL** — only 5/8 met the ≥10% threshold.
