#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT_DIR}/configs/eval_dpo_tail_aware.yaml}"
ARGS=(--config "${CONFIG}")

if [[ -n "${MODEL_OR_CHECKPOINT:-}" ]]; then
  ARGS+=(--model "${MODEL_OR_CHECKPOINT}")
fi
if [[ -n "${EVAL_OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${EVAL_OUTPUT_DIR}")
fi

cd "${ROOT_DIR}"
python -m steel_rlvr.evaluate "${ARGS[@]}"
