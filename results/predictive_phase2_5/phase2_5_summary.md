# Phase 2.5 — Predictive routing (inference-only)

> Compares Reactive (current util) vs Predictive (GRU forecast) vs Oracle (cheats with t+1 actual) bottleneck-style OD selection.

| Topology | reactive MLU | predictive MLU | oracle MLU | predictive Δ% | oracle Δ% |
|---|---:|---:|---:|---:|---:|
| abilene | 0.0546 | 0.0546 | 0.0546 | -0.00% | -0.00% |
| cernet | 1709.2269 | 1709.0804 | 1709.2312 | -0.01% | +0.00% |
| ebone | 379.5915 | 379.5915 | 379.5915 | -0.00% | -0.00% |
| geant | 0.1607 | 0.1607 | 0.1607 | +0.00% | +0.00% |
| sprintlink | 845.6458 | 845.6458 | 845.4603 | +0.00% | -0.02% |
| tiscali | 848.2444 | 848.2444 | 848.4998 | +0.00% | +0.03% |
| germany50 | 18.9358 | 18.9358 | 18.9479 | -0.00% | +0.06% |
| vtlwavenet2011 | 12288.9028 | 12288.9028 | 12288.7435 | -0.00% | -0.00% |

Predictive reduces MLU on: 0 / 8 topologies
Oracle is materially better than Predictive on: 0 / 8 topologies

**VERDICT: FAIL (signal-limited)** — even the oracle does not materially improve MLU, meaning predicted-util is not a useful routing signal beyond the current state. Stop pursuing.
