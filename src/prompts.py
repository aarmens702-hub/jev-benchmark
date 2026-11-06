"""The frozen prompt files under prompts/. Nothing else in the code holds prompt text.

Jev instructions: the task question from the pinned file of each dataset (see third_party/NOTICE.md).
Small LLM: the same task question plus one answer line; the numbered options with descriptions are
built by llm_client from the same label texts Jev sees. The Jev + examples state shape is in
jev_examples_state.v1.json and is built by jev_client.
"""
import json
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
JEV_FILE = "jev_instructions.v1.json"
LLM_FILE = "small_llm.v1.json"
STATE_FILE = "jev_examples_state.v1.json"


def _load(name: str) -> dict:
    return json.loads((PROMPTS_DIR / name).read_text(encoding="utf-8"))


def jev_instructions(dataset: str) -> str:
    blob = _load(JEV_FILE)
    if dataset not in blob or dataset == "version":
        raise ValueError(f"no Jev instructions for {dataset!r}")
    return blob[dataset]


def small_llm_instructions(dataset: str) -> str:
    """Task line (identical to Jev's) followed by the answer line."""
    return f"{jev_instructions(dataset)} {_load(LLM_FILE)['answer_line']}"


def examples_state_shape() -> dict:
    blob = _load(STATE_FILE)
    return {"state_keys": blob["state_keys"], "example_keys": blob["example_keys"]}


def prompt_files() -> list[Path]:
    return [PROMPTS_DIR / name for name in (JEV_FILE, LLM_FILE, STATE_FILE)]
