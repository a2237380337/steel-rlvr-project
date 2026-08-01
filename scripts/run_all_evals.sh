#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bash scripts/run_eval.sh configs/eval_base_dpo_study.yaml
bash scripts/run_eval.sh configs/eval_sft_dpo_study.yaml
bash scripts/run_eval.sh configs/eval_dpo_baseline.yaml
bash scripts/run_eval.sh configs/eval_dpo_tail_aware.yaml

python -m steel_rlvr.compare_evaluations \
  --summary base=artifacts/evals-dpo/base/summary.json \
  --summary sft=artifacts/evals-dpo/sft/summary.json \
  --summary dpo=artifacts/evals-dpo/dpo-baseline/summary.json \
  --summary frequency_aware_dpo=artifacts/evals-dpo/dpo-tail-aware/summary.json \
  --require-complete \
  --output results/formal_evaluation_matrix.json

python -m steel_rlvr.build_result_card \
  --matrix results/formal_evaluation_matrix.json \
  --paper-baselines reports/paper_baselines.csv \
  --output reports/result_card.generated.md
