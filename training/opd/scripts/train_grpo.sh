#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPD_ENABLED=false
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-spatial_interactor_grpo_ablation}"
exec bash "${ROOT_DIR}/train_opd.sh" "$@"

