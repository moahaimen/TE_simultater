#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/phase3_final}"
MAX_STEPS="${MAX_STEPS:-500}"

echo "[1/3] Phase-3 generalization"
OUTPUT_DIR="$OUTPUT_DIR" MAX_STEPS="$MAX_STEPS" \
  bash "$ROOT_DIR/scripts/run_phase3_generalization.sh"

echo "[2/3] Phase-3 failures"
OUTPUT_DIR="$OUTPUT_DIR" MAX_STEPS="$MAX_STEPS" \
  bash "$ROOT_DIR/scripts/run_phase3_failures.sh"

echo "[3/3] Phase-3 final report"
python -m eval.make_phase3_report --output_dir "$OUTPUT_DIR"

echo "Phase-3 finished. See: $OUTPUT_DIR"
