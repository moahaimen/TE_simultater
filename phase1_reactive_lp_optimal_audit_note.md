# LP-optimal Audit Note

- `phase1_reactive/lp/lp_optimal.py` is only a thin wrapper. The actual implementation lives in `te/lp_solver.py` in `solve_full_mcf_min_mlu()`.
- `solve_full_mcf_min_mlu()` is a true full multicommodity-flow linear program over the full directed edge set for all active OD pairs in one timestep.
- Objective: minimize the scalar `U`, where `U` upper-bounds utilization on every directed link.
- Capacity constraints: for each directed edge `e`, `sum_od x[od,e] <= U * capacity[e]`.
- Flow-conservation constraints: for each active OD pair and each node, outgoing minus incoming flow equals `+demand` at the source, `-demand` at the destination, and `0` at transit nodes.
- Non-negativity: all commodity edge-flow variables `x[od,e] >= 0`, and `U >= 0`.
- There is no OD sampling inside `solve_full_mcf_min_mlu()` itself, and there is no K-path restriction in that LP.
- Runtime capping is used through the CBC solver time limit (`full_mcf_time_limit_sec`).
- Step sampling is used in the evaluation pipeline through `optimality_eval_steps`; LP-optimal is not solved on every test step by default.
- Reported `lp_optimal` numbers in Phase-1 are therefore a sampled, runtime-capped upper bound, not a full always-on production baseline over every test timestep.
