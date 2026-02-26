#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/phase2_final}"
MAX_STEPS="${MAX_STEPS:-500}"
SEED="${SEED:-42}"
PREDICTORS="${PREDICTORS:-seasonal,lstm,ensemble}"
BLEND_LAMBDAS="${BLEND_LAMBDAS:-0.2,0.5,0.8}"
SAFE_Z_VALUES="${SAFE_Z_VALUES:-0.0,0.5,1.0}"

echo "[1/4] Downloading required SNDlib files into $DATA_DIR"
bash "$ROOT_DIR/scripts/download_sndlib.sh" --data_dir "$DATA_DIR"

echo "[2/4] Preparing processed datasets (max_steps=$MAX_STEPS)"
python "$ROOT_DIR/scripts/prepare_data.py" --data_dir "$DATA_DIR" --dataset all --max_steps "$MAX_STEPS"

echo "[3/4] Running final Phase-2 grid (C2/C3, reactive vs proactive)"
python -m eval.run_phase2_final \
  --config "$ROOT_DIR/configs/abilene.yaml" \
  --config "$ROOT_DIR/configs/geant.yaml" \
  --output_dir "$OUTPUT_DIR" \
  --max_steps "$MAX_STEPS" \
  --seed "$SEED" \
  --predictors "$PREDICTORS" \
  --blend_lambdas "$BLEND_LAMBDAS" \
  --safe_z_values "$SAFE_Z_VALUES"

echo "[4/4] Final artifacts"
echo "- Comparison CSV: $OUTPUT_DIR/FINAL_PHASE2_COMPARISON.csv"
echo "- Report: $OUTPUT_DIR/FINAL_PHASE2_REPORT.md"
echo "- Plots dir: $OUTPUT_DIR/plots"
