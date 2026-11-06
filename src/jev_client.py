"""Jev through OpenRouter's Decisions API.

Shape: tests/fixtures/decisions_choice_*.json. The score is the returned probability of Jev's
choice (DESIGN.md), not its separate `confidence` value.
"""
import hashlib
import json
import os
import threading
from dataclasses import replace

from ledger import heal_torn_tail, split_torn_tail
from openrouter import (
    DECISIONS_URL,
    ORDER_SEED,
    REPO_ROOT,
    Answer,
    Client,
    Decision,
    MalformedResponse,
    ModelChanged,
    _is_number,
    check_provider,
    int_field,
    normalize_options,
    option_order,
    pinned_provider,
    require_private_path,
    safe_name,
)

JEV_MODEL = "typesafe/jev-1.14"
JEV_PROVIDER = "TypeSafe"
JEV_MAKER_PREFIX = "typesafe/"
# Jev rounds to 2 decimals, and TypeSafe calls the choice the top option, so it may trail by rounding only.
ROUNDING_TOLERANCE = 0.011
# DESIGN.md: the 1.14 build seen in Stage 0. Replies carry it with a maker prefix, which is dropped to compare.
EXPECTED_JEV_BUILD = "jev-1.14-20261014"
QUESTION = "label"
# Placeholder until the frozen prompt text is passed in; the prompt is part of the freeze.
DEFAULT_INSTRUCTIONS = "Which option best fits the text?"
DEFAULT_CACHE_PATH = REPO_ROOT / "runs" / "private" / "jev_cache.jsonl"


class JevCache:
    """Answers already paid for. Starts empty, has no way to seed it, and lives only under runs/private."""

    def __init__(self, path=DEFAULT_CACHE_PATH):
        self.path = require_private_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        if self.path.exists():
            lines, _ = split_torn_tail(self.path.read_bytes())
            for line in lines:
                entry = json.loads(line)
                self._entries[entry["key"]] = entry["answer"]
            with open(self.path, "r+b") as handle:
                heal_torn_tail(handle)

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> dict | None:
        return self._entries.get(key)

    def put(self, key: str, answer: dict) -> None:
        with self._lock, open(self.path, "ab") as handle:
            handle.write(json.dumps({"key": key, "answer": answer}, separators=(",", ":")).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
            self._entries[key] = answer


def build_id(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def _state(text, examples):
    if not examples:
        return text
    return {"text": text, "labelled_examples": [{"text": e.text, "label": e.label} for e in examples]}


def build_request(text, options, row_id, order_seed, examples, instructions) -> dict:
    options = normalize_options(options)
    criteria = {label: options[label] for label in option_order(row_id, options, order_seed)}
    return {
        "model": JEV_MODEL,
        "state": _state(text, examples),
        "questions": {QUESTION: {"type": "choice", "instructions": instructions, "criteria": criteria}},
        "provider": pinned_provider(JEV_PROVIDER),
    }


def cache_key(build: str, body: dict) -> str:
    """Model build, text and option order. Also examples, descriptions and instructions, which change the answer."""
    question = body["questions"][QUESTION]
    material = [build, body["state"], question["instructions"], list(question["criteria"].items())]
    return hashlib.sha256(json.dumps(material, ensure_ascii=False).encode("utf-8")).hexdigest()


def parse_answer(data: dict, options: dict, expected_build: str, client: Client) -> Answer:
    """Strict on every field the score depends on. Each MalformedResponse names what was wrong."""
    model = safe_name(data.get("model"), "model")
    if not model.startswith(JEV_MAKER_PREFIX) or build_id(model) != expected_build:
        raise ModelChanged(f"expected Jev build {JEV_MAKER_PREFIX}{expected_build}, reply came from {model}; paused")
    client.pin_model(model)
    check_provider(data, JEV_PROVIDER)

    answers = data.get("answers")
    answer = answers.get(QUESTION) if isinstance(answers, dict) else None
    if not isinstance(answer, dict):
        raise MalformedResponse(f"no answers.{QUESTION}")
    if answer.get("type") == "refusal":
        raise MalformedResponse("refusal")
    if answer.get("type") != "choice":
        raise MalformedResponse("answer type is not choice")
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in options:
        raise MalformedResponse("label_not_in_options")

    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        raise MalformedResponse("no probabilities")
    if set(probabilities) != set(options):
        raise MalformedResponse("probabilities do not cover exactly the options")
    if not all(_is_number(p) and 0 <= p <= 1 for p in probabilities.values()):
        raise MalformedResponse("a probability is not a number from 0 to 1")
    if probabilities[choice] < max(probabilities.values()) - ROUNDING_TOLERANCE:
        raise MalformedResponse("choice is not the top option")

    usage = data["usage"]
    return Answer(
        pred=choice,
        top_prob=probabilities[choice],
        probs={label: probabilities[label] for label in options},
        model=model,
        input_tokens=int_field(usage, "input_tokens"),
        output_tokens=int_field(usage, "output_tokens"),
    )


def jev_decide(
    row_id,
    text,
    options,
    order_seed=ORDER_SEED,
    examples=None,
    *,
    client: Client,
    cache: JevCache | None = None,
    instructions=DEFAULT_INSTRUCTIONS,
    expected_build=EXPECTED_JEV_BUILD,
) -> Decision:
    started = client.clock()
    options = normalize_options(options)
    body = build_request(text, options, row_id, order_seed, examples, instructions)
    n_examples = len(examples) if examples else 0

    key = cache_key(expected_build, body) if cache is not None else None
    if key is not None and (hit := cache.get(key)) is not None:
        client.pin_model(hit["model"])
        # A cache hit is timed too, but it is marked cached and left out of latency summaries.
        return Decision(cost=0.0, failed=False, attempts=0, cached=True, n_examples=n_examples, latency_s=client.clock() - started, **hit)

    decision = client.call(row_id, DECISIONS_URL, body, lambda data: parse_answer(data, options, expected_build, client))
    decision = replace(decision, n_examples=n_examples)
    if key is not None and not decision.failed:
        cache.put(
            key,
            {
                "pred": decision.pred,
                "top_prob": decision.top_prob,
                "probs": decision.probs,
                "model": decision.model,
                "input_tokens": decision.input_tokens,
                "output_tokens": decision.output_tokens,
            },
        )
    return decision
