# Phase-1/Phase-2 TE: Abilene + GEANT (SNDlib)

Flow-level Traffic Engineering simulator on SNDlib dynamic traffic matrices.

- Phase-1: reactive TE optimization (no forecasting)
- Phase-2: proactive TE (predict next TM, optimize on prediction, evaluate on actual TM)

Implemented methods:
- `M0` OSPF shortest path
- `M1` ECMP over equal-cost shortest candidate paths
- `M2` LP-optimal full MCF (all ODs, min MLU)
- `M3` Top-K demand heuristic + LP (selected ODs), ECMP for others
- `M4` Bottleneck-contribution heuristic + LP (selected ODs), ECMP for others
- `M5` RL selector + LP (optional, checkpoint-based)

## Repo Layout

```text
data/
scripts/
  download_sndlib.sh
  prepare_data.py
  run_demo.sh
  run_phase2_demo.sh
te/
  parser_sndlib.py
  paths.py
  simulator.py
  baselines.py
  lp_solver.py
  disturbance.py
phase2/
  predictors.py
rl/
  policy.py
  train_rl.py
eval/
  run_all.py
  run_phase2.py
  plots.py
  make_report.py
configs/
  abilene.yaml
  geant.yaml
results/
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Phase-1 Quick Demo

```bash
bash scripts/run_demo.sh
```

Useful overrides:

```bash
MAX_STEPS=300 bash scripts/run_demo.sh
RUN_LP_OPTIMAL=1 MAX_STEPS=120 bash scripts/run_demo.sh
```

## Phase-2 Quick Demo (Proactive)

```bash
bash scripts/run_phase2_demo.sh
```

Useful overrides:

```bash
MAX_STEPS=300 PREDICTOR=ar_ridge PREDICTOR_WINDOW=6 bash scripts/run_phase2_demo.sh
RUN_LP_OPTIMAL=1 MAX_STEPS=180 bash scripts/run_phase2_demo.sh
```

## Full Phase-1 Run (longer)

```bash
# 1) Download datasets
bash scripts/download_sndlib.sh --data_dir data

# 2) Prepare processed files
python scripts/prepare_data.py --data_dir data --dataset all --max_steps 3000

# 3) Evaluate Phase-1 methods
python -m eval.run_all \
  --config configs/abilene.yaml \
  --config configs/geant.yaml \
  --output_dir results/full_phase1 \
  --methods ospf,ecmp,topk,bottleneck,lp_optimal
```

## Full Phase-2 Run (proactive)

```bash
python -m eval.run_phase2 \
  --config configs/abilene.yaml \
  --config configs/geant.yaml \
  --output_dir results/full_phase2 \
  --methods ospf,ecmp,topk_pred,bottleneck_pred,lp_optimal_pred \
  --predictor ar_ridge \
  --predictor_window 6 \
  --predictor_alpha 0.01
```

Optional proactive RL-selection+LP:

```bash
python -m eval.run_phase2 \
  --config configs/abilene.yaml \
  --config configs/geant.yaml \
  --output_dir results/full_phase2_rl \
  --methods ospf,ecmp,topk_pred,bottleneck_pred,rl_lp_pred \
  --predictor ar_ridge \
  --rl_checkpoint results/rl/abilene/policy.pt
```

## Optional RL Training (selector)

```bash
python rl/train_rl.py --config configs/abilene.yaml --epochs 5 --max_steps 1000
```

Checkpoint output:
- `results/rl/abilene/policy.pt`
- `results/rl/abilene/train_history.json`

## Outputs

Phase-1 run output directory (example `results/demo`):
- `summary_all.csv`
- `timeseries_all.csv`
- `report.md`

Phase-2 run output directory (example `results/phase2_demo`):
- `summary_all.csv`
- `timeseries_all.csv`
- `report.md`
- `<dataset>/summary.csv`
- `<dataset>/timeseries.csv`
- `<dataset>/prediction_timeseries.csv`
- `<dataset>/<dataset>_mlu_over_time.png`
- `<dataset>/<dataset>_disturbance_over_time.png`

## Notes

- Chronological split is always 70/15/15 with no shuffling.
- Reported metrics are computed on the test split.
- Deterministic seed is logged in `run_metadata.json`.
- If capacities are missing/invalid, parser applies a documented normalization rule and logs it.
