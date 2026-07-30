#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${STEEL_DATA_PATH:-}" ]]; then
  echo "Set STEEL_DATA_PATH to the private cleaned steel dataset." >&2
  exit 1
fi

bash scripts/capture_environment.sh formal-before-training
bash scripts/prepare_data.sh
bash scripts/run_sft_main.sh
bash scripts/run_grpo_baseline.sh
bash scripts/run_grpo_tail_aware.sh
bash scripts/run_all_evals.sh
bash scripts/capture_environment.sh formal-after-evaluation
