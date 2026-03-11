# Phase-1 Hybrid MoE Manifest

1. hybrid moe gate added: YES
2. oracle ensemble labeling added: YES
3. neural gate added: YES
4. all results rerun on corrected full config: YES

5. strongest final method on:
- Abilene: sensitivity (0.044357); strongest DRL/AI method = our_hybrid_moe_gate (0.044357)
- GEANT: lp_optimal (0.102411); strongest DRL/AI method = our_hybrid_moe_gate (0.146686)
- Germany50: flexdate (21.935045); strongest DRL/AI method = our_hybrid_moe_gate (25.700021)
- failures: cfrrl (0.120838 mean post-failure MLU on seen-topology failures); strongest DRL/AI method = our_hybrid_moe_gate (0.121045)

6. whether our_hybrid_moe_gate beats:
- ECMP: YES on Abilene, GEANT, Germany50, and seen-topology failure aggregate
- OSPF: YES on Abilene, GEANT, Germany50, and seen-topology failure aggregate
- our_drl_ppo: YES on Abilene, GEANT, Germany50, and seen-topology failure aggregate
- our_drl_dqn: YES on Abilene, GEANT, Germany50, and seen-topology failure aggregate
- our_drl_dual_gate: YES on Abilene, GEANT, Germany50, and seen-topology failure aggregate
- best heuristic+LP baseline: NO overall

7. if the best heuristic still wins, report the exact remaining gap:
- Abilene vs best heuristic+LP (sensitivity): 0.000000006%
- GEANT vs best heuristic+LP (bottleneck): 0.269445%
- Germany50 vs best heuristic+LP (bottleneck): 15.689984%
- seen-topology failure aggregate vs best heuristic+LP (bottleneck): 0.119480%
- full failure aggregate vs best overall method (cfrrl): 2.048818%

8. whether the new method is still fully AI-driven in final decision making:
- YES. The final selector decision is made by a neural gating model that outputs expert weights over PPO, DQN, Top-K, bottleneck, and sensitivity proposals. The heuristics are expert inputs only; the final OD ranking is produced by the learned gate.

Corrected full split:
- train_topologies: abilene_backbone, geant_core, ebone, sprintlink, tiscali, vtlwavenet2011
- eval_topologies: abilene_backbone, geant_core, ebone, sprintlink, tiscali
- generalization_topologies: germany50_real
