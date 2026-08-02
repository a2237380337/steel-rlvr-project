#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python -m steel_rlvr.train_continued_sft --config configs/continued_sft_hard_pilot.yaml
