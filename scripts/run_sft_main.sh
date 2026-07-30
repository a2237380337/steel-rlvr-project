#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
accelerate launch --num_processes 1 --module steel_rlvr.train_sft --config configs/sft_main.yaml
