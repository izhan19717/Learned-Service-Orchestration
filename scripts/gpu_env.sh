#!/usr/bin/env bash
# Source this file before Decima/GPU runs in WSL2.
#
# Usage:
#   source scripts/gpu_env.sh
#   ./.venv/bin/cisose-deeprm gpu-check

set -euo pipefail

WSL_GPU_LIB="/usr/lib/wsl/lib"

if [[ -d "${WSL_GPU_LIB}" ]]; then
  export PATH="${WSL_GPU_LIB}:${PATH}"
  CLEAN_LD="${LD_LIBRARY_PATH:-}"
  CLEAN_LD="${CLEAN_LD//:export/}"
  case ":${CLEAN_LD}:" in
    *":${WSL_GPU_LIB}:"*) export LD_LIBRARY_PATH="${CLEAN_LD}" ;;
    *) export LD_LIBRARY_PATH="${WSL_GPU_LIB}:${CLEAN_LD}" ;;
  esac
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GIT_PYTHON_REFRESH="${GIT_PYTHON_REFRESH:-quiet}"

echo "GPU environment prepared."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

