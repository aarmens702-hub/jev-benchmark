#!/bin/bash
# Stage 1 runner. Validation first, as a full dry run; then once on test, after the freeze.
# Paid arms need Aarmen's yes and --paid stage1. The Mac cannot sleep while it runs (caffeinate),
# and caffeinate ends with the run. A rerun resumes where the last one stopped.
#   scripts/stage1.sh --stage val --paid stage1 --region ca-central
#   scripts/stage1.sh --stage test --paid stage1 --region ca-central
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
# Bytecode goes to a fresh folder for this run only, so no .pyc left in the repo (stale or forged) is ever loaded.
# -P keeps the repo root off sys.path, so only src/ is importable and nothing ignored at the root can shadow it.
p="$(mktemp -d)"
export PYTHONPYCACHEPREFIX="$p"
PYTHONPATH="$root/src" exec caffeinate -ims uv run --frozen python -P -m stage1 "$@"
