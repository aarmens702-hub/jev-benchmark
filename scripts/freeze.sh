#!/bin/bash
# Freeze manifest. Build it, or verify it:
#   scripts/freeze.sh
#   scripts/freeze.sh --verify
# It never tags, commits, pushes or uses the network.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
PYTHONPATH="$root/src" exec uv run --frozen python -m freeze "$@"
