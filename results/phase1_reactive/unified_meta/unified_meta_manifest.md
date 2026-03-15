# Unified Meta-Selector — Phase-1 Audit Manifest

Generated: 2026-03-15 04:42:47
Seed: 42

## Meta-Gate Training
- Best epoch: 52
- Best val accuracy: 0.722
- Experts: bottleneck, cfrrl, erodrl, flexdate, flexentry, gnn, sensitivity, topk
- Training time: 0.2s

## Standard Evaluation Results (Mean MLU)

| Topology | meta | oracle | selector | bottleneck | cfrrl | flexdate | sensitivity | ecmp | ospf |
|---|---|---|---|---|---|---|---|---|---|
| abilene | 0.054599 | 0.054599 | 0.054599 | 0.054600 | 0.054600 | 0.054675 | 0.054755 | 0.123386 | 0.083937 |
| geant | 0.157355 | 0.157355 | 0.157355 | 0.158507 | 0.158507 | 0.160365 | 0.161661 | 0.270457 | 0.269423 |
| rocketfuel_ebone | 379.591501 | 379.591501 | 379.591501 | 379.591501 | 379.591501 | 379.591501 | 379.591501 | 415.626448 | 421.260085 |
| rocketfuel_sprintlink | 675.576004 | 673.411670 | 754.430557 | 675.550391 | 675.550391 | 751.667898 | 675.551857 | 1051.347659 | 1077.163128 |
| rocketfuel_tiscali | 722.570728 | 722.484395 | 815.379113 | 722.570728 | 722.570728 | 774.972307 | 793.273612 | 899.391452 | 1054.089168 |

## Generalization Results (Germany50)

| Method | Mean MLU |
|--------|----------|
| bottleneck | 17.930412 |
| cfrrl | 17.930412 |
| ecmp | 26.208130 |
| erodrl | 18.082477 |
| flexdate | 18.068561 |
| flexentry | 18.179599 |
| ospf | 31.620645 |
| our_unified_meta | 17.930412 |
| our_unified_oracle | 17.930412 |
| sensitivity | 18.079004 |
| topk | 18.131683 |

