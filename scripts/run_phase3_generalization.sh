#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/phase3_final}"
MAX_STEPS="${MAX_STEPS:-500}"
METHODS="${METHODS:-ospf,ecmp,topk,bottleneck}"

echo "[Phase3-Gen] Running generalization batch"
python -m eval.run_phase3_generalization \
  --config "$ROOT_DIR/configs/phase3_topologies.yaml" \
  --output_dir "$OUTPUT_DIR" \
  --methods "$METHODS" \
  --max_steps "$MAX_STEPS"

echo "[Phase3-Gen] Done. Outputs in $OUTPUT_DIR"
