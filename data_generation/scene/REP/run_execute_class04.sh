#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_EXE:-/path/to/workspace/miniconda3/bin/conda}"
exec "$CONDA_BIN" run --no-capture-output -n habitat_hm3d_qa \
  python /path/to/workspace/SCENE/REP/execute_class04_bank.py "$@"
