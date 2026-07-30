#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 OUTPUT_JSON LABEL=SUMMARY [LABEL=SUMMARY ...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_JSON="$1"
shift
ARGS=()
for item in "$@"; do
  ARGS+=(--summary "${item}")
done

cd "${ROOT_DIR}"
python -m steel_rlvr.select_checkpoint \
  "${ARGS[@]}" \
  --validation-file data/processed/validation.jsonl \
  --output "${OUTPUT_JSON}"
