#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 MODEL_OR_CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_OR_CHECKPOINT="$1" \
EVAL_OUTPUT_DIR="$2" \
  bash scripts/run_eval.sh configs/eval_validation.yaml
