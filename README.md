# Jev benchmark

Pre-registered benchmark. See DESIGN.md for the study, VALIDATION.md for the rules every session follows, and PLAN.md for the build order.

Private until Aarmen says otherwise.

## Setup

    uv sync
    uv run pytest          # offline, under 2 minutes
    uv run pytest -m data  # opt-in: may reach Hugging Face hosts only

Test data is not on disk until the freeze. `src/firewall.py` is the only module that may load it.
