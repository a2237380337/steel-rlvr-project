#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bash scripts/run_eval.sh configs/eval_base.yaml
bash scripts/run_eval.sh configs/eval_sft.yaml
bash scripts/run_eval.sh configs/eval_grpo_baseline.yaml
bash scripts/run_eval.sh configs/eval_main.yaml

python -m steel_rlvr.compare_evaluations \
  --summary base=artifacts/evals/base/summary.json \
  --summary sft=artifacts/evals/sft/summary.json \
  --summary drgrpo=artifacts/evals/drgrpo/summary.json \
  --summary tail_aware=artifacts/evals/tail-aware/summary.json \
  --require-complete \
  --output results/formal_evaluation_matrix.json

python -m steel_rlvr.build_result_card \
  --matrix results/formal_evaluation_matrix.json \
  --paper-baselines reports/paper_baselines.csv \
  --output reports/result_card.generated.md
