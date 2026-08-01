#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python -m steel_rlvr.check_environment --require-rocm --run-kernel-smoke
python -m pytest
bash scripts/capture_environment.sh gpu-smoke
bash scripts/run_sft_smoke.sh
bash scripts/build_preferences.sh
bash scripts/run_dpo_smoke.sh
bash scripts/run_eval.sh configs/eval_smoke.yaml
