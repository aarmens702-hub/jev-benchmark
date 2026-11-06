#!/bin/bash
# Freeze manifest. Build it, or verify it:
#   scripts/freeze.sh
#   scripts/freeze.sh --verify
# It never tags, commits, pushes or uses the network.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
# -P keeps the repo root off sys.path, so only src/ is importable.
p="$(mktemp -d)"
export PYTHONPYCACHEPREFIX="$p"
PYTHONPATH="$root/src" exec uv run --frozen python -P -m freeze "$@"
