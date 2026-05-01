# Phase 2.2 — GRU forecaster summary

> Pre-registered verdict: GRU must beat last-value by ≥10% MAPE on ≥6/8 topologies.

| Topology | num_od | GRU MAPE % | LV MAPE % | Δ (rel %) | R² (GRU) | sec |
|---|---:|---:|---:|---:|---:|---:|
| abilene | 132 | 27.56 | 27.81 | +0.91 | 0.945 | 1.4 |
| cernet | 1640 | 9.21 | 9.23 | +0.26 | 0.937 | 1.2 |
| ebone | 506 | 9.02 | 9.10 | +0.88 | 0.941 | 1.3 |
| geant | 462 | 181.80 | 178.52 | -1.84 | 0.989 | 0.9 |
| sprintlink | 1892 | 9.17 | 9.20 | +0.33 | 0.937 | 1.3 |
| tiscali | 2352 | 9.18 | 9.19 | +0.06 | 0.940 | 1.5 |
| germany50 | 2450 | 11130.19 | 10335.58 | -7.69 | 0.821 | 0.8 |
| vtlwavenet2011 | 8372 | 9.19 | 9.20 | +0.11 | 0.952 | 3.6 |

GRU wins on MAPE: 6 / 8 topologies
GRU beats LV by ≥10% MAPE: 0 / 8 topologies

**VERDICT: FAIL** — only 0/8 met the ≥10% threshold.
