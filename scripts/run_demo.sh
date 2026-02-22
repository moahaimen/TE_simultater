#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/demo}"
MAX_STEPS="${MAX_STEPS:-500}"
RUN_LP_OPTIMAL="${RUN_LP_OPTIMAL:-0}"

METHODS="ospf,ecmp,topk,bottleneck"
if [[ "$RUN_LP_OPTIMAL" == "1" ]]; then
  METHODS="$METHODS,lp_optimal"
fi

echo "[1/3] Downloading required SNDlib files into $DATA_DIR"
bash "$ROOT_DIR/scripts/download_sndlib.sh" --data_dir "$DATA_DIR"

echo "[2/3] Preparing processed datasets (max_steps=$MAX_STEPS)"
python "$ROOT_DIR/scripts/prepare_data.py" --data_dir "$DATA_DIR" --dataset all --max_steps "$MAX_STEPS"

echo "[3/3] Running TE demo methods: $METHODS"
python -m eval.run_all \
  --config "$ROOT_DIR/configs/abilene.yaml" \
  --config "$ROOT_DIR/configs/geant.yaml" \
  --output_dir "$OUTPUT_DIR" \
  --methods "$METHODS" \
  --max_steps "$MAX_STEPS"

echo "Demo finished. See: $OUTPUT_DIR"
