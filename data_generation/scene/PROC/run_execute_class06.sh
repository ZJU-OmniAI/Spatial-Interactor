#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_EXE:-/path/to/workspace/miniconda3/bin/conda}"
export DISPLAY="${DISPLAY:-:99}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export HOME="${HOME:-/path/to/workspace/ai2thor_runtime_home}"
mkdir -p "$HOME"
exec "$CONDA_BIN" run --no-capture-output -n ai2thor_env \
  python /path/to/workspace/SCENE/PROC/execute_class06_bank.py "$@"
