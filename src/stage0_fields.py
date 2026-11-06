"""What Stage 0 observes on the wire, and the 22 open fields from the Task 5 report.

A Recorder wraps the transport and keeps only structure: status, whether a field is present, its
type, and the model and provider names after the same allow-list the clients use. No reply text, no
row text and no key ever lands here. A reply whose shape differs from the expected fixture is a
problem recorded at once, so the runner can stop before it spends more.
"""
import json
import threading
from dataclasses import dataclass, field

from jev_client import EXPECTED_JEV_BUILD, JEV_MAKER_PREFIX, QUESTION, ROUNDING_TOLERANCE, build_id
from llm_client import MODEL_SUFFIX, reply_matches_request
from openrouter import CHAT_URL, DECISIONS_URL, MalformedResponse, _is_number, safe_name

CONFIRMED, MISMATCH, UNCONFIRMED, NOT_TESTED = "confirmed", "mismatch", "unconfirmed", "not tested"

FIELD_NAMES = {
    1: "Jev model string is the expected 1.14 build",
    2: "Jev probabilities cover exactly the options",
    3: "Jev choice is the top option",
    4: "usage.cost is present on every reply",
    5: "token counts are present on every reply",
    6: "provider.only spelling is accepted on both surfaces",
    7: "provider is echoed on replies",
    8: "Jev reads criteria in the order sent",
    9: "Jev accepts the provider pin with require_parameters",
    10: "logprobs come back under json_schema",
    11: "reasoning effort none is accepted",
    12: "reasoning tokens stay zero",
    13: "key usage reflects spend",
    14: "usage.cost matches what the key is billed",
    15: "Decisions spend appears in key usage",
    16: "seconds for key usage to post",
    17: "state accepts the examples object",
    18: "null criteria accepted for bare names",
    19: "77 and 151 choices accepted",
    20: "refusals, 400s and 404s are billed",
    21: "small LLM reply model matches the suffix pattern",
    22: "Jev reply model starts with typesafe/",
}


@dataclass
class Obs:
    surface: str  # jev or chat
    phase: str
    status: int
    n_options: int = 0
    null_criteria: bool = False
    state_is_object: bool = False
    request_model: str = ""
    provider_only: tuple = ()
    require_parameters: bool = False
    allow_fallbacks: bool = True
    reasoning_none: bool = False
    # Reply side, only for HTTP 200 with a JSON object body.
    ok_shape: bool = False
    problems: list = field(default_factory=list)
    model: str | None = None
    provider: str | None = None
    cost: float | None = None
    reasoning_tokens: int | None = None
    has_logprobs: bool = False
    refusal: bool = False
    extra_keys: list = field(default_factory=list)


def _printable(name) -> str:
    try:
        return safe_name(name, "name")
    except MalformedResponse:
        return "<unprintable>"


def _kind(value) -> str:
    return "missing" if value is None else type(value).__name__


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_jev_shape(data, criteria: list) -> tuple[list, dict]:
    """(problems, facts). Problems say which path differs from the fixture and what type was found."""
    problems = []
    facts = {"refusal": False}
    if not isinstance(data, dict):
        return [f"reply: expected object, got {_kind(data)}"], facts
    if not isinstance(data.get("model"), str):
        problems.append(f"model: expected string, got {_kind(data.get('model'))}")
    answers = data.get("answers")
    answer = answers.get(QUESTION) if isinstance(answers, dict) else None
    if not isinstance(answer, dict):
        problems.append(f"answers.{QUESTION}: expected object, got {_kind(answer)}")
    elif answer.get("type") == "refusal":
        facts["refusal"] = True
    else:
        if answer.get("type") != "choice":
            problems.append("answers.label.type: expected choice")
        choice, probs = answer.get("choice"), answer.get("probabilities")
        if not isinstance(choice, str) or choice not in criteria:
            problems.append("answers.label.choice: expected one of the options sent")
        if not isinstance(probs, dict):
            problems.append(f"answers.label.probabilities: expected object, got {_kind(probs)}")
        elif set(probs) != set(criteria):
            problems.append("answers.label.probabilities: keys differ from the options sent")
        elif not all(_is_number(p) and 0 <= p <= 1 for p in probs.values()):
            problems.append("answers.label.probabilities: expected numbers from 0 to 1")
        elif isinstance(choice, str) and choice in probs and probs[choice] < max(probs.values()) - ROUNDING_TOLERANCE:
            problems.append("answers.label.choice: is not the top option")
    usage = data.get("usage")
    if not isinstance(usage, dict):
        problems.append(f"usage: expected object, got {_kind(usage)}")
    else:
        if not _is_number(usage.get("cost")):
            problems.append(f"usage.cost: expected number, got {_kind(usage.get('cost'))}")
        for name in ("input_tokens", "output_tokens"):
            if not _is_int(usage.get(name)):
                problems.append(f"usage.{name}: expected integer, got {_kind(usage.get(name))}")
    return problems, facts


def check_chat_shape(data) -> tuple[list, dict]:
    """A missing logprobs object is a candidate failing, not a shape problem, so it only sets a fact."""
    problems = []
    facts = {"has_logprobs": False, "reasoning_tokens": None}
    if not isinstance(data, dict):
        return [f"reply: expected object, got {_kind(data)}"], facts
    if not isinstance(data.get("model"), str):
        problems.append(f"model: expected string, got {_kind(data.get('model'))}")
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        problems.append("choices[0].message.content: expected string")
    logprobs = choice.get("logprobs") if isinstance(choice, dict) else None
    tokens = logprobs.get("content") if isinstance(logprobs, dict) else None
    facts["has_logprobs"] = isinstance(tokens, list) and bool(tokens) and all(
        isinstance(t, dict) and isinstance(t.get("token"), str) and _is_number(t.get("logprob")) for t in tokens
    )
    usage = data.get("usage")
    if not isinstance(usage, dict):
        problems.append(f"usage: expected object, got {_kind(usage)}")
    else:
        if not _is_number(usage.get("cost")):
            problems.append(f"usage.cost: expected number, got {_kind(usage.get('cost'))}")
        for name in ("prompt_tokens", "completion_tokens"):
            if not _is_int(usage.get(name)):
                problems.append(f"usage.{name}: expected integer, got {_kind(usage.get(name))}")
        details = usage.get("completion_tokens_details")
        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if reasoning is not None and not _is_int(reasoning):
            problems.append("usage.completion_tokens_details.reasoning_tokens: expected integer or null")
        else:
            facts["reasoning_tokens"] = reasoning
    return problems, facts


_JEV_TOP = {"id", "model", "provider", "answers", "usage"}
_CHAT_TOP = {"id", "object", "created", "model", "provider", "system_fingerprint", "choices", "usage"}


class Observations:
    """Thread-safe store of what each POST looked like, tagged with the step that sent it."""

    def __init__(self):
        self.phase = "start"
        self._lock = threading.Lock()
        self.items: list[Obs] = []

    def add(self, obs: Obs) -> None:
        with self._lock:
            self.items.append(obs)

    def select(self, surface=None, phase=None, status=None) -> list[Obs]:
        with self._lock:
            return [
                o
                for o in self.items
                if (surface is None or o.surface == surface)
                and (phase is None or o.phase == phase)
                and (status is None or o.status == status)
            ]

    def problems(self) -> list[str]:
        with self._lock:
            return sorted({f"{o.surface}: {p}" for o in self.items for p in o.problems})


def observe(url: str, content, reply, phase: str) -> Obs:
    """Never raises: an unreadable exchange is recorded as a problem, not a crash."""
    surface = "jev" if url == DECISIONS_URL else "chat" if url == CHAT_URL else "other"
    obs = Obs(surface=surface, phase=phase, status=reply.status)
    try:
        body = json.loads(content) if content else {}
        provider = body.get("provider") or {}
        obs.provider_only = tuple(provider.get("only") or ())
        obs.require_parameters = provider.get("require_parameters") is True
        obs.allow_fallbacks = provider.get("allow_fallbacks") is not False
        obs.request_model = _printable(body.get("model"))
        criteria: list = []
        if surface == "jev":
            question = body["questions"][QUESTION]
            raw = question["criteria"]
            criteria = list(raw)
            obs.n_options = len(raw)
            obs.null_criteria = any(v is None for v in raw.values())
            obs.state_is_object = isinstance(body.get("state"), dict)
        else:
            obs.reasoning_none = (body.get("reasoning") or {}).get("effort") == "none"
    except Exception:
        criteria = []
        obs.problems.append("request: could not be read back")
    if reply.status != 200 or surface == "other":
        return obs
    try:
        data = json.loads(reply.body)
    except ValueError:
        obs.problems.append("reply: expected JSON on a 200")
        return obs
    if surface == "jev":
        obs.problems, facts = check_jev_shape(data, criteria)
        obs.refusal = facts["refusal"]
        top = _JEV_TOP
    else:
        obs.problems, facts = check_chat_shape(data)
        obs.has_logprobs = facts["has_logprobs"]
        obs.reasoning_tokens = facts["reasoning_tokens"]
        top = _CHAT_TOP
    if isinstance(data, dict):
        obs.ok_shape = not obs.problems
        obs.model = _printable(data["model"]) if isinstance(data.get("model"), str) else None
        obs.provider = _printable(data["provider"]) if isinstance(data.get("provider"), str) else None
        usage = data.get("usage")
        if isinstance(usage, dict) and _is_number(usage.get("cost")):
            obs.cost = float(usage["cost"])
        obs.extra_keys = sorted(_printable(k) for k in data if k not in top)
    return obs


class Recorder:
    """Wraps a transport. Passes everything through; records the structure of each POST."""

    def __init__(self, inner, observations: Observations):
        self.inner = inner
        self.observations = observations

    def send(self, method, url, content=None):
        reply = self.inner.send(method, url, content)
        if method == "POST":
            self.observations.add(observe(url, content, reply, self.observations.phase))
        return reply


@dataclass
class UsageCheck:
    """From polling the key's usage after the Jev smoke: how long it took to show the spend."""

    seconds: float | None  # None: never showed within the wait
    ledger_spend: float
    reported_spend: float


@dataclass
class Finding:
    number: int
    status: str
    note: str

    @property
    def name(self) -> str:
        return FIELD_NAMES[self.number]


def _all(items, test) -> bool:
    return bool(items) and all(test(o) for o in items)


def evaluate(obs: Observations, usage: UsageCheck | None, candidates: dict) -> list[Finding]:
    """One finding per field. `candidates` maps a small LLM name to {"requested", "provider", "passed"}."""
    jev = obs.select("jev", status=200)
    chat = obs.select("chat", status=200)
    out: dict[int, Finding] = {}

    def put(n, status, note):
        out[n] = Finding(n, status, note)

    def yes(n, ok, note_ok, note_bad, missing=None):
        if missing:
            put(n, NOT_TESTED, missing)
        else:
            put(n, CONFIRMED if ok else MISMATCH, note_ok if ok else note_bad)

    no_jev = None if jev else "no Jev reply was seen"
    shaped = [o for o in jev if not o.refusal]
    builds = {build_id(o.model) for o in jev if o.model}
    yes(1, builds == {EXPECTED_JEV_BUILD}, f"build {EXPECTED_JEV_BUILD}", f"builds seen: {sorted(builds)}", no_jev)
    yes(2, _all(shaped, lambda o: not any("probabilities" in p for p in o.problems)), "keys equal the options on every reply",
        "probabilities differ from the options sent", no_jev)
    yes(3, _all(shaped, lambda o: not any("top option" in p for p in o.problems)), "choice is within rounding of the top", "choice is not the top option", no_jev)
    costed = jev + chat
    yes(4, _all(costed, lambda o: o.cost is not None), "present", "missing on a reply", None if costed else "no reply was seen")
    yes(5, _all(costed, lambda o: not any("tokens" in p for p in o.problems)), "present as integers", "missing or not an integer",
        None if costed else "no reply was seen")
    pinned = lambda o: bool(o.provider_only) and not o.allow_fallbacks  # noqa: E731
    both = bool(jev) and bool(chat)
    yes(6, both and _all(jev + chat, pinned), "an unbilled rejection did not occur; both surfaces answered", "a surface did not answer under the pin",
        None if both else "one surface was not reached")
    providers = {s: sorted({o.provider or "absent" for o in objs}) for s, objs in (("jev", jev), ("chat", chat))}
    put(7, CONFIRMED if both else NOT_TESTED, f"provider on replies: {providers}")
    put(8, NOT_TESTED, "the docs are silent and a test needs extra paid calls; the order sent is shuffled per row")
    yes(9, _all(jev, lambda o: o.require_parameters), "Jev answered with require_parameters true", "Jev rejected or was sent no pin", no_jev)
    with_logprobs = [o for o in chat if o.has_logprobs]
    if with_logprobs:
        put(10, CONFIRMED, f"logprobs seen from: {sorted({o.request_model for o in with_logprobs})}")
    elif chat:
        put(10, UNCONFIRMED, "replies came back with no usable logprobs")
    else:
        put(10, NOT_TESTED, "no chat reply was seen")
    none_effort = [o for o in chat if o.reasoning_none]
    put(11, CONFIRMED if none_effort else UNCONFIRMED, "accepted" if none_effort else "no chat reply with effort none was seen")
    hidden = [o for o in chat if (o.reasoning_tokens or 0) > 0]
    if not chat:
        put(12, NOT_TESTED, "no chat reply was seen")
    else:
        put(12, MISMATCH if hidden else CONFIRMED, "reasoning tokens above zero were billed" if hidden else "null or zero on every reply")
    if usage is None:
        for n in (13, 14, 15, 16):
            put(n, NOT_TESTED, "key usage was not polled")
    else:
        shows = usage.seconds is not None
        put(13, CONFIRMED if shows else MISMATCH, "usage rose with spend" if shows else "usage did not rise within the wait")
        if shows and usage.ledger_spend > 0:
            ratio = usage.reported_spend / usage.ledger_spend
            put(14, CONFIRMED if 0.9 <= ratio <= 1.1 else UNCONFIRMED, f"reported usage was {ratio:.2f} of summed usage.cost")
        else:
            put(14, NOT_TESTED, "no spend to compare")
        put(15, CONFIRMED if shows else MISMATCH, "Decisions spend appeared in key usage" if shows else "Decisions spend did not appear in key usage")
        put(16, CONFIRMED if shows else MISMATCH, f"{usage.seconds:.0f} s" if shows else "not seen within the wait")
    examples = [o for o in jev if o.phase == "examples"]
    yes(17, _all(examples, lambda o: o.state_is_object), "accepted", "not accepted", None if examples else "the examples smoke did not run")
    bare = [o for o in jev if o.phase == "bare"]
    yes(18, _all(bare, lambda o: o.null_criteria), "accepted", "not accepted", None if bare else "the bare-name check did not run")
    wide = [o for o in jev if o.n_options >= 77]
    if wide:
        put(19, UNCONFIRMED, "77 accepted; 151 not tested, no 151-way task is in scope")
    else:
        put(19, NOT_TESTED, "no 77-way call answered")
    rejected = [o for o in obs.select() if o.status in (400, 404)]
    put(20, NOT_TESTED, "no 400 or 404 was seen" if not rejected else f"{len(rejected)} such replies seen; billing not separable here")
    if candidates:
        notes = []
        bad = False
        for name, c in candidates.items():
            seen = sorted({o.model for o in chat if o.request_model == c["requested"] and o.model})
            fits = all(reply_matches_request(m, c["requested"]) for m in seen)
            bad = bad or (bool(seen) and not fits)
            notes.append(f"{name}: replied {seen or 'nothing'}, pattern {MODEL_SUFFIX}")
        put(21, MISMATCH if bad else CONFIRMED if any("replied []" not in n for n in notes) else NOT_TESTED, "; ".join(notes))
    else:
        put(21, NOT_TESTED, "no candidate was tried")
    yes(22, _all(jev, lambda o: (o.model or "").startswith(JEV_MAKER_PREFIX)), "starts with typesafe/", "prefix differs", no_jev)
    return [out[n] for n in sorted(FIELD_NAMES)]
