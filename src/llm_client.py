"""The small LLM through OpenRouter chat completions: integer option ids, enum schema, logprobs.

Shape: tests/fixtures/chat_logprobs_*.json. Top probability = exp(sum of token logprobs along the
chosen id). A reply with no usable logprobs means the arm is "not run" (DESIGN.md), so that raises
ArmNotRun rather than being retried.
"""
import json
import math
import re

from openrouter import (
    CHAT_URL,
    ORDER_SEED,
    Answer,
    ArmNotRun,
    Client,
    Decision,
    MalformedResponse,
    _is_number,
    check_provider,
    int_field,
    normalize_options,
    option_order,
    pinned_provider,
)

MAX_TOKENS = 16
# Placeholder until the frozen prompt text is passed in; the prompt is part of the freeze.
DEFAULT_INSTRUCTIONS = "Which option best fits the text? Answer with the id of one option."
_ID_SPAN = re.compile(r'\s*\{\s*"id"\s*:\s*(\d+)\s*\}\s*')
# Rounding can put a certain token a hair above 0; anything more is not a log probability.
_LOGPROB_SLACK = 1e-6


def build_request(text, ordered, descriptions, model, provider, instructions) -> dict:
    lines = []
    for option_id, label in enumerate(ordered, start=1):
        description = descriptions[label]
        lines.append(f"{option_id}. {label}" if description is None else f"{option_id}. {label}: {description}")
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Text: {text}\n\nOptions:\n" + "\n".join(lines)},
        ],
        "max_tokens": MAX_TOKENS,
        "reasoning": {"effort": "none"},
        "logprobs": True,
        "temperature": 0,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "option_choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"id": {"type": "integer", "enum": list(range(1, len(ordered) + 1))}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
        },
        "provider": pinned_provider(provider),
    }


def top_prob_from_logprobs(content: str, logprobs, row_id) -> float:
    """exp(sum of logprobs of the tokens overlapping the id's digits). Raises ArmNotRun if any token cannot be used."""
    tokens = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(tokens, list) or not tokens:
        raise ArmNotRun(f"no token logprobs on row {row_id}")
    spelled = ""
    total = 0.0
    match = _ID_SPAN.fullmatch(content)
    if match is None:
        raise ArmNotRun(f"cannot locate the id in the reply on row {row_id}")
    start, end = match.span(1)
    for token in tokens:
        text = token.get("token") if isinstance(token, dict) else None
        logprob = token.get("logprob") if isinstance(token, dict) else None
        if not isinstance(text, str) or not _is_number(logprob) or logprob > _LOGPROB_SLACK:
            raise ArmNotRun(f"a token has no usable logprob on row {row_id}")
        token_start, token_end = len(spelled), len(spelled) + len(text)
        if token_start < end and token_end > start:
            total += logprob
        spelled += text
    if spelled != content:
        raise ArmNotRun(f"logprob tokens do not spell the reply on row {row_id}")
    return min(1.0, math.exp(total))


def parse_answer(data: dict, id_to_label: dict, provider: str, client: Client, row_id) -> Answer:
    usage = data["usage"]
    tokens = int_field(usage, "completion_tokens")
    details = usage.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    # Before anything can reject the reply: billed tokens count even when the answer is unusable.
    client.track_output(tokens, reasoning if isinstance(reasoning, int) else 0)

    model = data.get("model")
    if not isinstance(model, str):
        raise MalformedResponse("no model")
    client.pin_model(model)
    check_provider(data, provider)

    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise MalformedResponse("no message content")
    try:
        parsed = json.loads(content)
    except ValueError:
        raise MalformedResponse("content is not JSON") from None
    if not isinstance(parsed, dict) or set(parsed) != {"id"}:
        raise MalformedResponse("content is not one id")
    option_id = parsed["id"]
    if not isinstance(option_id, int) or isinstance(option_id, bool) or option_id not in id_to_label:
        raise MalformedResponse("label_not_in_options")

    top_prob = top_prob_from_logprobs(content, choice.get("logprobs"), row_id)
    pred = id_to_label[option_id]
    return Answer(
        pred=pred,
        top_prob=top_prob,
        probs={pred: top_prob},
        model=model,
        input_tokens=int_field(usage, "prompt_tokens"),
        output_tokens=tokens,
    )


def llm_pick(
    row_id,
    text,
    options,
    model,
    provider,
    *,
    client: Client,
    order_seed=ORDER_SEED,
    instructions=DEFAULT_INSTRUCTIONS,
) -> Decision:
    if ":batch" in model:
        raise ValueError("the :batch variant is not allowed (DESIGN.md)")
    options = normalize_options(options)
    ordered = option_order(row_id, options, order_seed)
    id_to_label = dict(enumerate(ordered, start=1))
    body = build_request(text, ordered, options, model, provider, instructions)
    return client.call(row_id, CHAT_URL, body, lambda data: parse_answer(data, id_to_label, provider, client, row_id))
