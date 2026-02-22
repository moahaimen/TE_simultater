# Phase-1 Reactive TE: Abilene + GEANT (SNDlib)

Flow-level Traffic Engineering simulator for reactive routing optimization on SNDlib dynamic traffic matrices.

Implemented methods:
- `M0` OSPF shortest path
- `M1` ECMP over equal-cost shortest candidate paths
- `M2` LP-optimal full MCF (all ODs, min MLU)
- `M3` Top-K demand heuristic + LP (selected ODs), ECMP for others
- `M4` Bottleneck-contribution heuristic + LP (selected ODs), ECMP for others
- `M5` RL selector + LP (optional, checkpoint-based in evaluation)

## Repo Layout

```
data/
scripts/
  download_sndlib.sh
  prepare_data.py
  run_demo.sh
te/
  parser_sndlib.py
  paths.py
  simulator.py
  baselines.py
  lp_solver.py
  disturbance.py
rl/
  policy.py
  train_rl.py
eval/
  run_all.py
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

## Quick Demo (<=10 min target on laptop, depends on solver and network)

This runs end-to-end download + prep + evaluation for both datasets with default `MAX_STEPS=500`:

```bash
bash scripts/run_demo.sh
```

Useful overrides:

```bash
MAX_STEPS=300 bash scripts/run_demo.sh
RUN_LP_OPTIMAL=1 MAX_STEPS=120 bash scripts/run_demo.sh
```

## Full Run (longer)

```bash
# 1) Download datasets
bash scripts/download_sndlib.sh --data_dir data

# 2) Prepare processed files (larger step budget)
python scripts/prepare_data.py --data_dir data --dataset all --max_steps 3000

# 3) Evaluate with LP-optimal and optional RL selector
python -m eval.run_all \
  --config configs/abilene.yaml \
  --config configs/geant.yaml \
  --output_dir results/full \
  --methods ospf,ecmp,topk,bottleneck,lp_optimal,rl_lp \
  --max_steps 3000 \
  --rl_checkpoint results/rl/abilene/policy.pt
```

## Optional RL Training

Train selector policy (REINFORCE) for one dataset config:

```bash
python rl/train_rl.py --config configs/abilene.yaml --epochs 5 --max_steps 1000
```

Checkpoint output:
- `results/rl/abilene/policy.pt`
- `results/rl/abilene/train_history.json`

## Outputs

Per run output directory (example `results/demo`):
- `summary_all.csv`
- `timeseries_all.csv`
- `report.md`
- `<dataset>/summary.csv`
- `<dataset>/timeseries.csv`
- `<dataset>/<dataset>_mlu_over_time.png`
- `<dataset>/<dataset>_disturbance_over_time.png`

## Notes

- Chronological split is always 70/15/15 with no shuffling.
- Final reported numbers are computed on **test split only**.
- Deterministic seed is logged in `run_metadata.json`.
- If topology capacities are missing/invalid, parser applies a documented normalization rule and logs it.
- Error messages explain missing data and next preparation step.
