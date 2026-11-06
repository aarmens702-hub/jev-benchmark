#!/bin/bash
# Freeze manifest. Build, verify, or fill the registration once the tag freeze-v1 exists:
#   scripts/freeze.sh
#   scripts/freeze.sh --verify
#   scripts/freeze.sh --registration
# It never tags, commits, pushes or uses the network.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
PYTHONPATH="$root/src" exec uv run --frozen python -m freeze "$@"
