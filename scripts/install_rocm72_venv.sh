#!/usr/bin/env bash
set -euo pipefail

# Frozen environment used for the reported RX 7900 XT experiments.
# The wheels are AMD's official ROCm 7.2 manylinux builds for CPython 3.10.
VENV_PATH="${ROCM_VENV_PATH:-${HOME}/venvs/rocm72-py310}"
WHEEL_CACHE="${ROCM_WHEEL_CACHE:-${HOME}/.cache/rocm72-wheels}"
INSTALL_LOG_DIR="${ROCM_INSTALL_LOG_DIR:-${HOME}/rocm72-install-logs}"
export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"

TORCH_WHEEL="torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp310-cp310-linux_x86_64.whl"
TORCHVISION_WHEEL="torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp310-cp310-linux_x86_64.whl"
TORCHAUDIO_WHEEL="torchaudio-2.9.0+rocm7.2.0.gite3c6ee2b-cp310-cp310-linux_x86_64.whl"
TRITON_WHEEL="triton-3.5.1+rocm7.2.0.gita272dfa8-cp310-cp310-linux_x86_64.whl"
BASE_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2"

mkdir -p "${WHEEL_CACHE}" "${INSTALL_LOG_DIR}" "$(dirname "${VENV_PATH}")"
if [[ -e "${VENV_PATH}" && ! -f "${VENV_PATH}/pyvenv.cfg" ]]; then
  echo "Refusing to overwrite a non-venv path: ${VENV_PATH}" >&2
  exit 1
fi
if [[ ! -f "${VENV_PATH}/pyvenv.cfg" ]]; then
  python3.10 -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install "numpy==1.26.4"

download_wheel() {
  local filename="$1"
  local encoded_name="${filename//+/%2B}"
  if [[ ! -s "${WHEEL_CACHE}/${filename}" ]]; then
    curl --fail --location --retry 3 --continue-at - \
      --output "${WHEEL_CACHE}/${filename}" \
      "${BASE_URL}/${encoded_name}"
  fi
}

download_wheel "${TORCH_WHEEL}"
download_wheel "${TORCHVISION_WHEEL}"
download_wheel "${TORCHAUDIO_WHEEL}"
download_wheel "${TRITON_WHEEL}"

sha256sum \
  "${WHEEL_CACHE}/${TORCH_WHEEL}" \
  "${WHEEL_CACHE}/${TORCHVISION_WHEEL}" \
  "${WHEEL_CACHE}/${TORCHAUDIO_WHEEL}" \
  "${WHEEL_CACHE}/${TRITON_WHEEL}" \
  > "${INSTALL_LOG_DIR}/wheel-sha256.txt"

python -m pip install \
  "${WHEEL_CACHE}/${TORCH_WHEEL}" \
  "${WHEEL_CACHE}/${TORCHVISION_WHEEL}" \
  "${WHEEL_CACHE}/${TORCHAUDIO_WHEEL}" \
  "${WHEEL_CACHE}/${TRITON_WHEEL}"

TORCH_LOCATION="$(python -m pip show torch | awk -F ': ' '$1 == "Location" {print $2}')"
TORCH_LIB="${TORCH_LOCATION}/torch/lib"
case "${TORCH_LIB}" in
  "${VENV_PATH}"/*) ;;
  *)
    echo "Refusing to modify torch outside the dedicated venv: ${TORCH_LIB}" >&2
    exit 1
    ;;
esac

# AMD's WSL package expects the WSL-compatible HSA runtime under /opt/rocm.
# Remove only the wheel-bundled copy, after verifying it is inside this venv.
find "${TORCH_LIB}" -maxdepth 1 -name 'libhsa-runtime64.so*' -print \
  > "${INSTALL_LOG_DIR}/removed-wheel-hsa-libraries.txt"
while IFS= read -r library; do
  [[ -n "${library}" ]] && rm -f -- "${library}"
done < "${INSTALL_LOG_DIR}/removed-wheel-hsa-libraries.txt"

python -m pip freeze > "${INSTALL_LOG_DIR}/pip-freeze-before-project.txt"
python - <<'PY'
import json
import torch

result = {
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "accelerator_available": torch.cuda.is_available(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result["accelerator_available"] or result["hip"] is None:
    raise SystemExit("ROCm GPU was not detected.")
PY

echo "Created ROCm environment: ${VENV_PATH}"
echo "Next: source '${VENV_PATH}/bin/activate' && bash scripts/setup_environment.sh"
