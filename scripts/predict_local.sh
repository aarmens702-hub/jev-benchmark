#!/bin/bash
# Run the free-arms CLI from anywhere without setting PYTHONPATH:
#   scripts/predict_local.sh --stage val --floor
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
PYTHONPATH="$root/src" exec uv run --frozen python -m predict_local "$@"
