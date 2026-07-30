#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
python -m steel_rlvr.prepare_data \
  --output-dir data/processed \
  --threshold 50 \
  --validation-fraction 0.2 \
  --seed 42
