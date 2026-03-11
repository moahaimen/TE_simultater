# Phase-1 Improved DRL Manifest

## 1. Improvement Flags

1. Teacher pretraining added: YES
2. Curriculum training added: YES
3. Dual-gate added: YES
4. PPO preserved: YES
5. DQN preserved: YES
6. All results come from corrected full config: YES

## 2. Final Split

- train_topologies:
  - abilene_backbone
  - geant_core
  - ebone
  - sprintlink
  - tiscali
  - vtlwavenet2011
- eval_topologies:
  - abilene_backbone
  - geant_core
  - ebone
  - sprintlink
  - tiscali
- generalization_topologies:
  - germany50_real
- germany50_real is unseen in the final generalization run: YES

## 3. LP-optimal Wording

- Exact wording to use in the report: `sampled, runtime-capped LP-optimal upper bound`
- Rationale: the underlying LP in `te/lp_solver.py::solve_full_mcf_min_mlu()` is a true full multicommodity-flow LP for one timestep, but the evaluation pipeline only solves it on a sampled subset of test steps and enforces the CBC time limit via `full_mcf_time_limit_sec`.

## 4. Strongest DRL Method By Scenario

- Abilene: `our_drl_ppo`
  - mean MLU: `0.0443573668825943`
- GEANT: `our_drl_dqn`
  - mean MLU: `0.1478079724773275`
- Germany50: `our_drl_dqn`
  - mean MLU: `31.74724817092436`
- failures: `our_drl_dual_gate`
  - mean post-failure MLU: `604.8735158674416`

## 5. Final Honest Conclusion

- improved DRL beats ECMP: YES
  - Abilene gain: `54.76042799867228%`
  - GEANT gain: `40.14818626166728%`
  - Germany50 gain: `1.4011633897042874%`
- improved DRL beats OSPF: YES
  - Abilene gain: `31.87469985125758%`
  - GEANT gain: `39.84192971875595%`
  - Germany50 gain: `17.044966094643026%`
- improved DRL beats best heuristic+LP baseline: NO
  - Abilene remaining gap: `0.000000014176441739922009%`
  - GEANT remaining gap: `1.0364095565074611%`
  - Germany50 remaining gap: `44.732996237268296%`

## 6. K-path Generation Clarification

- `K = 3` candidate paths are generated in `te/paths.py::build_k_shortest_paths()`.
- Implementation uses `networkx.shortest_simple_paths(...)` on the directed topology graph.
- This is a Yen-style K-shortest simple path enumeration routine over weighted shortest-path subproblems.
- Edge weights come from `dataset.weights` loaded into the graph in `te/simulator.py::_build_graph()`.
- For SNDlib topologies, weights are parsed from the topology file in `te/parser_sndlib.py`; if a weight is missing there, the parser falls back to `1.0`.
- These weights are static topology weights. They are not time-varying congestion weights and they are not learned by the DRL selector.
- In practical terms, the K=3 library is a deterministic fixed-cost path library, closest to OSPF/IGP-style routing weights when the source topology provides them.

## 7. Dynamic Kcrit Clarification

- Kcrit is dynamic: NO
- Current corrected full config uses a fixed `k_crit = 40` from `configs/phase1_reactive_full.yaml`.
- The actual selected count at timestep `t` is:
  - `selected_count(t) = min(Kcrit, active_od_count(t))`
- Active OD count means OD pairs with positive demand and at least one valid candidate path.
- The selected-flow percentage over time is therefore computed as:
  - `selected_percentage(t) = selected_count(t) / active_od_count(t)`
- Because `Kcrit` is fixed and `active_od_count(t)` changes with the traffic matrix and failure state, the selected percentage still varies over time even though the configured Kcrit value itself is not dynamic.

## 8. Teacher / Curriculum / Dual-Gate Notes

- Teacher-guided pretraining uses labels built from:
  - Top-K heuristic
  - bottleneck heuristic
  - sensitivity heuristic
  - sampled LP-optimal marginal signal on a limited subset of training steps
- Curriculum training schedule:
  - Stage 1: C2
  - Stage 2: C3
  - Stage 3: mixed C1 + C2 + C3
- Dual-gate inference runs PPO and DQN on the same current TM(t), solves the same LP for both selected OD sets, and chooses the lower-MLU result, with disturbance and delay tie-breakers.

## 9. File Integrity

- Missing required files in the requested bundle: NONE
- Old demo-scope outputs mixed into this improved bundle: NO
- This bundle is built from the improved full-config Phase-1 outputs only.
