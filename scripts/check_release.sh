#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONDONTWRITEBYTECODE=1

MAX_BYTES=$((50 * 1024 * 1024))
TREE_BYTES="$(du -sb . | cut -f1)"
if (( TREE_BYTES > MAX_BYTES )); then
  echo "Source tree exceeds the 50 MiB submission limit: ${TREE_BYTES} bytes." >&2
  exit 1
fi

FORBIDDEN_ASSETS="$(find . -type f \( \
  -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.gif' \
  -o -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' \
  -o -iname '*.pt' -o -iname '*.pth' -o -iname '*.safetensors' \
  -o -iname '*.parquet' -o -iname '*.jsonl' -o -iname '*.log' -o -iname '*.pyc' \
\) -print)"
if [[ -n "${FORBIDDEN_ASSETS}" ]]; then
  echo "Non-code experiment assets found:" >&2
  printf '%s\n' "${FORBIDDEN_ASSETS}" >&2
  exit 1
fi

python3 - <<'PY'
from pathlib import Path

for path in sorted(Path(".").rglob("*.py")):
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
PY

while IFS= read -r -d '' script; do
  bash -n "${script}"
done < <(find data_generation data_preparation training/opd/scripts training/sft \
  -type f -name '*.sh' \
  -not -path 'training/sft/LLaMA-Factory/*' -print0)

python3 -m unittest discover -s tests -p 'test_*.py'

if [[ -n "${OPD_PYTHON:-}" ]]; then
  PYTHONPATH="${ROOT}/training/opd/EasyR1" \
    "${OPD_PYTHON}" -m unittest training/opd/tests/test_opd_core.py
fi

if rg -n --hidden -S \
  '(^|[^[:alnum:]_])(sk|ms)-[[:alnum:]]{16,}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY' \
  . \
  -g '!scripts/check_release.sh'; then
  echo "Credential-like string or private key found." >&2
  exit 1
fi

if rg -n --hidden -S \
  '/home/[^/[:space:]]+/|/home2/|/Dataset2/|/data02/home/|/data/I[0-9]{4,}/' \
  . \
  -g '!scripts/check_release.sh' \
  -g '!data_preparation/prepare_release_dataset.py'; then
  echo "Machine-specific absolute path found." >&2
  exit 1
fi

echo "Release checks passed."
