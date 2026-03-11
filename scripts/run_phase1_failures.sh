#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/phase1_reactive_demo.yaml}"
MAX_STEPS="${MAX_STEPS:-180}"

python -m phase1_reactive.failures.failure_runner --config "$CONFIG_PATH" --max_steps "$MAX_STEPS"
