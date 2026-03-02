#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/phase3_final}"
MAX_STEPS="${MAX_STEPS:-500}"
METHODS="${METHODS:-ospf,ecmp,topk,bottleneck}"
NUM_FAILED_EDGES="${NUM_FAILED_EDGES:-2}"


echo "[Phase3-Fail] Running failure scenarios"
python -m eval.run_phase3_failures \
  --config "$ROOT_DIR/configs/phase3_topologies.yaml" \
  --output_dir "$OUTPUT_DIR" \
  --methods "$METHODS" \
  --num_failed_edges "$NUM_FAILED_EDGES" \
  --max_steps "$MAX_STEPS"

echo "[Phase3-Fail] Done. Outputs in $OUTPUT_DIR"
