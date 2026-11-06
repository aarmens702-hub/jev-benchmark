# Jev benchmark

Pre-registered benchmark. See DESIGN.md for the study, VALIDATION.md for the rules every session follows, and PLAN.md for the build order.

Built in private and made public at the freeze tag `freeze-v1`. The registration is [registration.md](registration.md); `manifest.json` at that tag holds the SHA-256 of every frozen input, and `scripts/freeze.sh --verify` checks them.

## Setup

    uv sync
    uv run pytest          # offline, under 2 minutes
    uv run pytest -m data  # opt-in: may reach Hugging Face hosts only

Test data is not on disk until the freeze. `src/firewall.py` is the only module that may load it.
