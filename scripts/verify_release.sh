#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ruff check src tests
pytest -q
python -m steel_rlvr.release_audit --root "${ROOT_DIR}"
