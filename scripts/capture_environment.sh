#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${1:-manual}"
OUTPUT_DIR="${ROOT_DIR}/artifacts/environment/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

python -m steel_rlvr.check_environment > "${OUTPUT_DIR}/environment.json"
python -m pip freeze > "${OUTPUT_DIR}/pip-freeze.txt"
sha256sum "${ROOT_DIR}"/configs/*.yaml > "${OUTPUT_DIR}/config-sha256.txt"
find "${ROOT_DIR}/src" "${ROOT_DIR}/tests" "${ROOT_DIR}/configs" "${ROOT_DIR}/scripts" \
  -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.sh' \) \
  -print0 | sort -z | xargs -0 sha256sum > "${OUTPUT_DIR}/source-sha256.txt"
sha256sum "${ROOT_DIR}/pyproject.toml" "${ROOT_DIR}/requirements-rocm.txt" \
  >> "${OUTPUT_DIR}/source-sha256.txt"

if command -v rocminfo >/dev/null 2>&1; then
  rocminfo > "${OUTPUT_DIR}/rocminfo.txt" 2>&1 || true
fi
if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "${ROOT_DIR}" rev-parse HEAD > "${OUTPUT_DIR}/project-commit.txt" 2>/dev/null || \
    echo "NO_COMMIT" > "${OUTPUT_DIR}/project-commit.txt"
  git -C "${ROOT_DIR}" status --short > "${OUTPUT_DIR}/project-status.txt"
else
  echo "NOT_A_GIT_REPOSITORY" > "${OUTPUT_DIR}/project-commit.txt"
fi

echo "Environment snapshot: ${OUTPUT_DIR}"
