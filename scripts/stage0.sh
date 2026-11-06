#!/bin/bash
# Stage 0, the first paid step. Needs Aarmen's yes first, and --paid stage0:
#   scripts/stage0.sh --paid stage0
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
PYTHONPATH="$root/src" exec uv run --frozen python -m stage0 "$@"
