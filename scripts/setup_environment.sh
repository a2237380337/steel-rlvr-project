#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_VENV_PATH="${ROCM_VENV_PATH:-${HOME}/venvs/rocm72-py310}"
export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -f "${ROCM_VENV_PATH}/bin/activate" ]]; then
    echo "ROCm venv not found: ${ROCM_VENV_PATH}" >&2
    echo "Run: bash scripts/install_rocm72_venv.sh" >&2
    exit 1
  fi
  source "${ROCM_VENV_PATH}/bin/activate"
fi

python - <<'PY'
import torch
if torch.version.hip is None:
    raise SystemExit("ROCm PyTorch was not detected in the active venv.")
print(f"torch={torch.__version__}, HIP={torch.version.hip}")
PY

python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements-rocm.txt"
python -m pip install -e "${ROOT_DIR}[dev]"
python -m steel_rlvr.check_environment --require-rocm
