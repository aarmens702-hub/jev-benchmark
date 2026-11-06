"""Budgeted OpenRouter client: every spend guard lives here, shared by the Jev and small LLM clients.

Without `--paid <stage>` only a fake transport can be used. API shapes are recorded with their
source URLs in tests/fixtures/sources.json. Nothing the server sends is copied into an exception
or a log, so a response echoing the key cannot leak it.

Clients and arms in a stage. A stage has one ledger, shared by every client in it, and may use any
number of clients and arms. A client runs one arm at a time, and may run several arms one after
another (for example the Jev arm, then an LLM arm). Only one client sends at a time, because the
limit of 4 concurrent requests is held per client. Each arm starts with its own model pin, failure
count and output-token mean. Stops that are about one arm end only that arm: a changed model or
provider, missing logprobs, 10 failures in a row, hidden reasoning. Stops that are about money or
the key end the client for good: the budget, a 401/402/403, a reconciliation gap or unreadable
usage, a ledger mismatch, a bug or Ctrl-C. A new client after such a stop re-checks the ledger
against the key's usage before it sends anything. An arm is named by its results file: calling
run_arm again on the same client with the same file continues that arm (first contact, then the
rest, or 100-row chunks), keeping its stop, its pin, its counts and its missing-logprob allowance.
"""
import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol
from urllib.parse import urlsplit

import httpx

import secrets_mac
from ledger import STAGES, BudgetExceeded, Ledger, RunStop, heal_torn_tail, split_torn_tail

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_URL = "https://openrouter.ai/api/v1/key"
API_HOST = "openrouter.ai"

REPO_ROOT = Path(__file__).resolve().parents[1]
# One file for every checkout, so a new clone or a deleted runs/ folder cannot reset the cap.
PAID_LEDGER_PATH = Path.home() / ".jev-benchmark" / "ledger.jsonl"
TEST_LEDGER_ENV = "JEV_TEST_LEDGER_PATH"

MAX_BODY_BYTES = 30_000
MAX_RETRIES = 3
MAX_CONCURRENCY = 4
CONSECUTIVE_FAILURE_LIMIT = 10
LADDER = (1, 20)
RECONCILE_EVERY = 100
RECONCILE_GAP = 0.10
# Below one US cent a percentage gap is noise: usage posting a moment late would read as a huge gap.
# At about US$0.0001 a call this delays detection of a real gap by at most a couple of hundred calls.
RECONCILE_FLOOR_USD = 0.01
MEAN_OUTPUT_LIMIT = 16
# Held against the cap for a call whose cost is not known yet, until a real cost has been seen.
DEFAULT_RESERVE_USD = 0.001
MAX_BACKOFF_SECONDS = 30
MAX_RETRY_AFTER_SECONDS = 60
# httpx's timeouts are per phase and per chunk, so a slow drip never fires them. The deadline is the
# overall limit; a request can overrun it by at most one read timeout.
REQUEST_TIMEOUTS = httpx.Timeout(connect=10, read=30, write=30, pool=30)
REQUEST_DEADLINE_SECONDS = 90
# Waited once before believing OpenRouter reports less than the ledger, since usage can post late.
# Measure the real lag in Stage 0 and set this from it.
DEFAULT_RECONCILE_GRACE_SECONDS = 30
USAGE_UNAVAILABLE = "usage_unavailable"
NO_USABLE_LOGPROBS = "no_usable_logprobs"
# A row that failed on an outage gets its first pass plus 1 more, counted across reruns (ruling R2).
OUTAGE_PASS_LIMIT = 2
_SAFE_NAME = re.compile(r"[\w./:-]{1,80}")
# OpenRouter keys are sk-or-v1- and 64 hex characters. Anything carrying sk- or a long hex run, which
# could be the key's secret part on its own, is refused wherever it sits in the name.
_KEY_LIKE = re.compile(r"sk-|[0-9a-f]{32,}", re.IGNORECASE)
ORDER_SEED = 1105
STOP_STATUSES = {401, 402, 403}


class PaidFlagRequired(RuntimeError):
    """Real transport was asked for without `--paid <stage>`."""


class RequestTooLarge(ValueError):
    """Over MAX_BODY_BYTES, which risks the documented less accurate long-context mode. Never sent."""


class MalformedResponse(Exception):
    """A reply that cannot be used as an answer. The row retries, then is marked failed."""


class TransportFailure(Exception):
    """The request did not get a usable HTTP reply. The message is ours, never the library's."""


class PaidOutsideArm(RuntimeError):
    """A paid call was made outside run_arm, which holds the ladder and the end-of-arm check."""


class _CostNotReported(Exception):
    """A reply with no believable usage.cost. Charged at the estimate, and a failed attempt."""


class RequestDeadlineExceeded(Exception):
    """The whole request passed REQUEST_DEADLINE_SECONDS."""


class KeyOrCreditsStop(RunStop):
    pass


class LedgerMismatch(RunStop):
    """The key's reported usage shows spend the ledger doesn't hold: ledger lost, or spend outside it."""


class AcceptRefused(RunStop, ValueError):
    """accept_key_usage_usd was not a usable amount, or there was nothing to accept. Nothing is written or sent."""


class ReconciliationGap(RunStop):
    pass


class ReconciliationUnavailable(RunStop):
    pass


class ConsecutiveFailures(RunStop):
    pass


class HiddenReasoning(RunStop):
    pass


class ModelChanged(RunStop):
    pass


class ProviderChanged(RunStop):
    pass


class ArmNotRun(RunStop):
    """The provider cannot give usable logprobs under the schema. Raised per row, so selection can count it."""

    cost = 0.0  # what the row had been charged when this was raised


class UnexpectedStop(RunStop):
    pass


# Stops about one arm's model or rows. The next arm on the same client starts clear of them. Every
# other stop (budget, key, reconciliation, ledger mismatch, a bug, Ctrl-C) stays for the client's life.
ARM_SCOPED_STOPS = (ModelChanged, ProviderChanged, ArmNotRun, ConsecutiveFailures, HiddenReasoning)


@dataclass(frozen=True)
class Decision:
    pred: str | None
    top_prob: float | None
    probs: dict
    model: str
    cost: float
    failed: bool
    attempts: int = 1
    cached: bool = False
    reason: str | None = None
    n_examples: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class Answer:
    """What a surface-specific parser returns from a valid reply."""

    pred: str
    top_prob: float
    probs: dict
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class TransportResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class Transport(Protocol):
    def send(self, method: str, url: str, content: bytes | None = None) -> TransportResponse: ...


class _Secret:
    """Holds the key so that printing any object that contains it shows nothing."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "<redacted>"

    __str__ = __repr__


class RealTransport:
    """The only code that sends the key: in one header, over HTTPS, to OpenRouter only."""

    def __init__(self, stage, *, key_reader=None, httpx_transport=None, clock=time.monotonic):
        if stage not in STAGES:
            raise PaidFlagRequired(f"real transport needs --paid with one of {STAGES}")
        self.stage = stage
        self._secret = _Secret((key_reader or secrets_mac.read_key)())
        self._clock = clock
        # trust_env off: no proxy or CA bundle from the environment gets to see the key.
        self._http = httpx.Client(timeout=REQUEST_TIMEOUTS, follow_redirects=False, trust_env=False, transport=httpx_transport)

    def __repr__(self) -> str:
        return f"RealTransport(stage={self.stage!r})"

    def send(self, method: str, url: str, content: bytes | None = None) -> TransportResponse:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.hostname != API_HOST:
            raise TransportFailure("refusing to send the key anywhere but https://openrouter.ai")
        deadline = self._clock() + REQUEST_DEADLINE_SECONDS
        try:
            # The header is built inline so that no local variable in this frame ever holds the key.
            with self._http.stream(
                method,
                url,
                content=content,
                headers={"Authorization": f"Bearer {self._secret.reveal()}", "Content-Type": "application/json"},
            ) as response:
                chunks = []
                for chunk in response.iter_bytes():
                    if self._clock() > deadline:
                        raise RequestDeadlineExceeded
                    chunks.append(chunk)
                return TransportResponse(response.status_code, dict(response.headers), b"".join(chunks))
        except Exception as error:  # any library message could quote the request, so keep the type only
            failure = TransportFailure(f"request failed: {type(error).__name__}")
        # Raised out here so the library's exception is not even attached as hidden context.
        raise failure


def key_usage(transport: Transport) -> float:
    """OpenRouter's reported all-time spend for this key, for reconciliation. Reading it is free."""
    response = transport.send("GET", KEY_URL)
    if response.status in STOP_STATUSES:
        raise KeyOrCreditsStop(_stop_message(response.status, {}))
    try:
        usage = json.loads(response.body)["data"]["usage"]
    except (ValueError, KeyError, TypeError):
        raise TransportFailure("GET /api/v1/key did not return data.usage") from None
    if not _is_number(usage):
        raise TransportFailure("GET /api/v1/key did not return a numeric data.usage")
    return float(usage)


def int_field(usage: dict, name: str) -> int:
    value = usage.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise MalformedResponse(f"no usage.{name}")
    return value


def safe_name(value, field_name: str) -> str:
    """Model and provider names are the only reply text that reaches our output, so they are allow-listed.

    A reply that echoed the key must not get it into a message, a results file or the ledger.
    """
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value) or _KEY_LIKE.search(value):
        raise MalformedResponse(f"{field_name} is not an allowed name")
    return value


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reconciliation_stops(lower: float, upper: float, reported: float) -> bool:
    """True if reported spend is too far outside what the ledger can account for.

    lower is spend the API confirmed. upper adds what the ledger charged to be safe, such as an
    estimate for a 529 that may not have been billed. Reported spend anywhere in between is
    consistent. A gap counts only if it is over 10% of reported spend and over the floor.
    """
    gap = max(0.0, lower - reported, reported - upper)
    if gap <= RECONCILE_FLOOR_USD + 1e-12:
        return False
    return reported <= 0 or gap > RECONCILE_GAP * reported + 1e-12


def encode_body(body: dict) -> bytes:
    """The exact bytes that go on the wire. Key order is preserved, which carries the option order."""
    content = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(content) > MAX_BODY_BYTES:
        raise RequestTooLarge(f"request is {len(content)} bytes; the limit is {MAX_BODY_BYTES}")
    return content


def normalize_options(options) -> dict:
    """Labels in the caller's order, mapped to a description or None. A bare list means bare names."""
    if isinstance(options, Mapping):
        normalized = dict(options)
    else:
        labels = list(options)
        if len(set(labels)) != len(labels):
            raise ValueError("option labels must be unique")
        normalized = {label: None for label in labels}
    if not normalized:
        raise ValueError("options must not be empty")
    return normalized


def option_order(row_id, labels, seed: int = ORDER_SEED) -> list:
    """This row's shuffled order: stable across runs and blind to the order labels arrive in."""
    return sorted(labels, key=lambda label: hashlib.sha256(f"{seed}:{row_id}:{label}".encode()).hexdigest())


def pinned_provider(provider: str) -> dict:
    return {"only": [provider], "allow_fallbacks": False, "require_parameters": True}


def check_provider(data: dict, expected: str) -> None:
    """Docs use both names and slugs, and do not promise this field on chat replies, so absent is allowed."""
    seen = data.get("provider")
    if seen is not None and _squash(safe_name(seen, "provider")) != _squash(expected):
        raise ProviderChanged(f"expected provider {expected}, reply came from {seen}")


def _squash(name) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _stop_message(status: int, data: dict) -> str:
    error = data.get("error") if isinstance(data, dict) else None
    metadata = error.get("metadata") if isinstance(error, dict) else None
    source = metadata.get("limit_source") if isinstance(metadata, dict) else None
    if status == 402 and source == "openrouter_in_flight_budget":
        return (
            "OpenRouter returned HTTP 402 for the in-flight spending budget, which it documents as transient. "
            "Stopped anyway, by rule. Finished rows are kept; rerun to resume."
        )
    meaning = {401: "the key was rejected", 402: "credits or the key limit ran out", 403: "the key is not allowed to do this"}
    return f"OpenRouter returned HTTP {status}: {meaning[status]}. Stopped with no retry. Finished rows are kept; rerun to resume."


_TOKEN_FIELDS = ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens")


def _reported_cost(data) -> float | None:
    """The cost, or None if it can't be believed. Zero for a reply that used tokens counts as not reported."""
    usage = data.get("usage") if isinstance(data, dict) else None
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if not _is_number(cost) or cost < 0:
        return None
    if cost == 0 and any(_is_number(usage.get(name)) and usage[name] > 0 for name in _TOKEN_FIELDS):
        return None
    return float(cost)


def _body_stop_status(data) -> int | None:
    """A 200 can still carry an error object; a stop status there must stop like the real one."""
    error = data.get("error") if isinstance(data, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, int) and code in STOP_STATUSES else None


class Client:
    """Sends rows through every guard. One arm at a time, several arms in turn; the ledger may be shared."""

    def __init__(
        self,
        transport: Transport,
        ledger: Ledger,
        reported_usage: Callable[[], float],
        *,
        sleep=time.sleep,
        max_concurrency: int = MAX_CONCURRENCY,
        reserve_usd: float = DEFAULT_RESERVE_USD,
        reconcile_grace_seconds: float = DEFAULT_RECONCILE_GRACE_SECONDS,
        accept_key_usage_usd: float | None = None,
    ):
        if not 1 <= max_concurrency <= MAX_CONCURRENCY:
            raise ValueError(f"max_concurrency must be 1 to {MAX_CONCURRENCY}")
        self.is_paid = isinstance(transport, RealTransport)
        if self.is_paid and ledger.stage != transport.stage:
            raise ValueError(f"ledger stage {ledger.stage} does not match the --paid stage {transport.stage}")
        if accept_key_usage_usd is not None and not self.is_paid:
            raise ValueError("accept_key_usage_usd only applies to a paid run")
        self.transport = transport
        self.ledger = ledger
        self.reported_usage = reported_usage
        self.max_concurrency = max_concurrency
        self.stop: RunStop | None = None
        self.consecutive_failures = 0
        self._sleep = sleep
        self._reserve_usd = reserve_usd
        self._grace = reconcile_grace_seconds
        self._in_arm = False
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._lock = threading.Lock()
        self._sent = 0
        self._largest_cost = 0.0
        self._output_tokens = 0
        self._output_calls = 0
        self._pinned_model: str | None = None
        self._arm_path: Path | None = None
        usage = reported_usage()
        if self.is_paid:
            self._check_paid_start(usage, accept_key_usage_usd)
        self._base_confirmed = ledger.confirmed()
        self._base_uncertain = ledger.uncertain()
        # Where reported usage reads as zero spend, so that it can be compared with the ledger's total.
        baseline = ledger.baseline()
        self._reported_zero = baseline if baseline is not None else usage - ledger.spent()
        # Spend before this point belongs to earlier runs, on both sides of the comparison. A paid client
        # starts from where usage will be once the ledger's confirmed spend has posted, so an earlier
        # arm's spend posting late is not read as this client being billed more. Earlier spend the
        # ledger only estimated (a 529, a crash mid-call) may or may not post, so it widens the upper
        # bound instead; putting it in the starting point would read an unbilled estimate as a gap.
        self._base_reported = usage
        self._pre_start_slack = 0.0
        if self.is_paid:
            self._base_reported = max(usage, self._reported_zero + self._base_confirmed)
            self._pre_start_slack = max(0.0, self._reported_zero + ledger.spent() - self._base_reported)
        ledger.note_reported(usage - self._reported_zero)

    def _check_paid_start(self, usage: float, accept_usd) -> None:
        """Compare the key's all-time usage with the ledger, so a lost or swapped ledger cannot reset the cap.

        Only spend above the ledger counts here. Below it is safe, and usage can post late. Spend shown
        by the key and not by the ledger can be accepted by amount, and is then written to the ledger as
        spend, counted against the total and the current stage: accepting never gives room back.
        """
        if accept_usd is not None and (not _is_number(accept_usd) or accept_usd < 0):
            raise AcceptRefused("accept_key_usage_usd must be a finite number of US dollars, zero or more. Nothing sent.")
        baseline = self.ledger.baseline()
        if baseline is None:
            if not self.ledger.is_empty():
                raise LedgerMismatch("this ledger has entries but no baseline, so its spend can't be checked. Nothing sent.")
            if accept_usd is None:
                if usage > RECONCILE_FLOOR_USD:
                    raise LedgerMismatch(
                        f"the ledger is empty but the key shows US${usage:.6f} spent: the ledger was lost, or there is spend "
                        "outside it. Nothing sent. If that spend is known and accepted, pass accept_key_usage_usd with the amount."
                    )
                self.ledger.record_baseline(usage, {"accepted_usd": None})
                return
            self._require_cover(accept_usd, usage, "the key's usage")
            # The ledger now holds all of the key's usage, so reported usage reads as zero spend at 0.
            self.ledger.record_baseline(0.0, {"accepted_usd": accept_usd}, observed=usage)
            self._charge_accepted(accept_usd, usage)
            return
        observed = self.ledger.baseline_observed()
        if usage < max(baseline, observed) - RECONCILE_FLOOR_USD:
            raise LedgerMismatch(
                f"the key shows US${usage:.6f}, below the US${max(baseline, observed):.6f} it showed when this ledger "
                "began: a different key? Nothing sent."
            )
        since = usage - baseline
        held = self.ledger.spent() + self.ledger.excess()
        mismatch = reconciliation_stops(0.0, held, since)
        if accept_usd is None:
            if mismatch:
                raise LedgerMismatch(
                    f"the key shows US${since:.6f} since the baseline and the ledger holds US${held:.6f}: "
                    "the ledger was lost, or there is spend outside it. Nothing sent. If that spend is known and "
                    f"accepted, pass accept_key_usage_usd with the US${since - held:.6f} difference."
                )
            return
        if not mismatch:
            raise AcceptRefused(
                "this ledger already has a baseline and agrees with the key's usage, so there is nothing to accept. "
                "Nothing sent; run again without accept_key_usage_usd (--accept-key-usage-usd)."
            )
        self._require_cover(accept_usd, since - held, "the spend outside the ledger")
        self._charge_accepted(accept_usd, since - held)

    @staticmethod
    def _require_cover(accept_usd: float, amount: float, what: str) -> None:
        """Within a cent either way. Accepting too much would put the ledger above what the key will ever
        report, and every later client would read that as billed less than the ledger and stop."""
        if abs(accept_usd - amount) > RECONCILE_FLOOR_USD + 1e-12:
            raise LedgerMismatch(
                f"accept_key_usage_usd of US${accept_usd:.6f} does not match {what}, US${amount:.6f}, "
                f"to within US${RECONCILE_FLOOR_USD}. Nothing sent."
            )

    def _charge_accepted(self, accept_usd: float, amount: float) -> None:
        """The shown amount, counted against the total and the current stage. Raises BudgetExceeded, after writing, past a cap."""
        self.ledger.charge(amount, {"accepted_key_usage_usd": accept_usd, "unaccounted_usd": amount})

    def latch(self, stop: RunStop) -> None:
        """Record why the run must end. Later sends refuse; the current row's paid answer can still be kept."""
        with self._lock:
            if self.stop is None:
                self.stop = stop

    def _halt(self, stop: RunStop):
        self.latch(stop)
        raise self.stop

    def _raise_if_stopped(self) -> None:
        if self.stop is not None:
            raise self.stop

    def _estimate(self) -> float:
        with self._lock:
            return self._largest_cost or self._reserve_usd

    def pin_model(self, model: str) -> None:
        """The first model seen is the arm's. A later change pauses the run."""
        with self._lock:
            if self._pinned_model is None:
                self._pinned_model = model
                return
            pinned = self._pinned_model
        if model != pinned:
            raise ModelChanged(f"model changed from {pinned} to {model[:80]}; paused")

    def track_output(self, tokens: int, reasoning_tokens: int) -> None:
        """Hidden reasoning is billed as output. Average over 16 a call, or any reported reasoning, stops the run."""
        with self._lock:
            self._output_tokens += tokens
            self._output_calls += 1
            mean = self._output_tokens / self._output_calls
        if mean > MEAN_OUTPUT_LIMIT or reasoning_tokens > 0:
            self.latch(
                HiddenReasoning(f"mean output is {mean:.1f} tokens a call with {reasoning_tokens} reasoning tokens on this call")
            )

    def note_row(self, failed: bool) -> None:
        with self._lock:
            self.consecutive_failures = self.consecutive_failures + 1 if failed else 0
            hit_limit = self.consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT
        if hit_limit:
            self.latch(ConsecutiveFailures(f"{CONSECUTIVE_FAILURE_LIMIT} rows in a row failed"))

    def _after_send(self) -> None:
        with self._lock:
            self._sent += 1
            due = self._sent % RECONCILE_EVERY == 0
        if due:
            self._reconcile()

    def _ledger_bounds(self) -> tuple[float, float]:
        lower = max(0.0, self.ledger.confirmed() - self._base_confirmed)
        return lower, lower + max(0.0, self.ledger.uncertain() - self._base_uncertain) + self._pre_start_slack

    def begin_arm(self, results_path) -> None:
        """Called by run_arm. An arm is named by its results file.

        The same file as this client's last arm continues that arm: its stop, pin and counts are kept,
        so a candidate stopped on its first-contact row stays stopped. A different file is a new arm:
        it gets its own pin and counts, and a stop about the last arm is cleared.
        """
        path = Path(results_path).resolve()
        with self._lock:
            if self._in_arm:
                raise RuntimeError("this client is already running an arm; run arms one after another")
            self._in_arm = True
            if path == self._arm_path:
                return
            self._arm_path = path
            if isinstance(self.stop, ARM_SCOPED_STOPS):
                self.stop = None
            self._pinned_model = None
            self.consecutive_failures = 0
            self._output_tokens = 0
            self._output_calls = 0

    def _read_usage(self) -> float | None:
        try:
            return self.reported_usage()
        except RunStop as stop:
            self.latch(stop)
        except Exception as error:
            self.latch(ReconciliationUnavailable(f"could not read reported usage: {type(error).__name__}"))
        return None

    def _reconcile(self) -> None:
        usage = self._read_usage()
        if usage is None:
            return
        lower, upper = self._ledger_bounds()
        reported = usage - self._base_reported
        if reported < lower and self._grace > 0 and reconciliation_stops(lower, upper, reported):
            # Reporting less may just be lag. Being billed more can't be, so that case never waits. `lower`
            # is kept from before the wait, since rows sent during it can't have posted yet either.
            self._sleep(self._grace)
            usage = self._read_usage()
            if usage is None:
                return
            upper = self._ledger_bounds()[1]
            reported = usage - self._base_reported
        self.ledger.note_reported(usage - self._reported_zero)
        if reconciliation_stops(lower, upper, reported):
            self.latch(
                ReconciliationGap(f"ledger holds US${lower:.6f} to US${upper:.6f} this run; OpenRouter reports US${reported:.6f}")
            )

    def reconcile_final(self) -> None:
        """The last check, for an arm or a client too short to reach a periodic one. A first stop is never replaced."""
        if self._sent:
            self._reconcile()

    def _send_once(self, url: str, content: bytes, meta: dict):
        """One attempt. Returns (cost, reply, data), with reply None if the network failed."""
        self._raise_if_stopped()
        estimate = self._estimate()
        try:
            entry = self.ledger.reserve(estimate, meta)
        except BudgetExceeded as stop:
            self._halt(stop)
        with self._slots:
            if self.stop is not None:
                self.ledger.settle(entry, 0.0, "not_billed", {**meta, "status": "not_sent"})
                raise self.stop
            try:
                reply = self.transport.send("POST", url, content)
            except TransportFailure:
                reply = None

        # Outside the slot: a reconciliation can wait out its grace, and must not hold up other rows.
        if reply is None:
            self.ledger.settle(entry, estimate, "estimate", {**meta, "status": "network"})
            self._after_send()
            return estimate, None, None

        data = None
        if reply.status == 200:
            try:
                data = json.loads(reply.body)
            except ValueError:
                pass
        stop_status = reply.status if reply.status in STOP_STATUSES else _body_stop_status(data)
        if stop_status is not None:
            self.ledger.settle(entry, 0.0, "not_billed", {**meta, "status": stop_status})
            if data is None:
                try:
                    data = json.loads(reply.body)
                except ValueError:
                    data = {}
            self._halt(KeyOrCreditsStop(_stop_message(stop_status, data)))

        cost = _reported_cost(data)
        if cost is not None:
            with self._lock:
                self._largest_cost = max(self._largest_cost, cost)
            spent, basis = cost, "reported"
        elif _is_retryable(reply.status) or reply.status == 200:
            spent, basis = estimate, "estimate"  # retries count as spend even when the cost is not reported
        else:
            spent, basis = 0.0, "not_billed"  # a 400-class refusal: documented as failing before anything is sent upstream
        self.ledger.settle(entry, spent, basis, {**meta, "status": reply.status})
        self._after_send()
        return spent, reply, data

    def call(self, row_id, url: str, body: dict, parse: Callable[[dict], Answer]) -> Decision:
        """Up to 1 try plus MAX_RETRIES, all charged. Returns a Decision, failed if no valid answer came."""
        self._raise_if_stopped()
        if self.is_paid and not self._in_arm:
            raise PaidOutsideArm("paid calls go through run_arm, which holds the ladder and the end-of-arm check")
        content = encode_body(body)
        model = body.get("model", "")
        total = 0.0
        reason = None
        attempts = 0
        for attempt in range(MAX_RETRIES + 1):
            attempts = attempt + 1
            spent, reply, data = self._send_once(url, content, {"row_id": row_id, "model": model, "attempt": attempts})
            total += spent
            retry_after = None
            if reply is None:
                reason = "network"
            elif reply.status == 200:
                try:
                    if not isinstance(data, dict):
                        raise MalformedResponse("not a JSON object")
                    if _reported_cost(data) is None:
                        raise _CostNotReported
                    answer = parse(data)
                except _CostNotReported:
                    reason = USAGE_UNAVAILABLE
                except MalformedResponse as error:
                    reason = f"malformed: {error}"
                except ArmNotRun as stop:
                    stop.cost = total
                    raise  # this row's signal, not the client's: selection counts it, run_arm stops on it
                except RunStop as stop:
                    self._halt(stop)
                else:
                    self.note_row(failed=False)
                    return Decision(
                        pred=answer.pred,
                        top_prob=answer.top_prob,
                        probs=answer.probs,
                        model=answer.model,
                        cost=total,
                        failed=False,
                        attempts=attempts,
                        input_tokens=answer.input_tokens,
                        output_tokens=answer.output_tokens,
                    )
            elif _is_retryable(reply.status):
                reason = f"http_{reply.status}"
                retry_after = _retry_after(reply.headers)
            else:
                reason = f"http_{reply.status}"
                break  # retrying a refused request cannot help and would only burn the row's tries
            self._raise_if_stopped()
            if attempt < MAX_RETRIES:
                self._sleep(retry_after if retry_after is not None else min(MAX_BACKOFF_SECONDS, 2**attempt))
        self.note_row(failed=True)
        return Decision(None, None, {}, model, total, True, attempts=attempts, reason=reason)


def _is_retryable(status: int) -> bool:
    return status in (408, 429) or status >= 500


def _retry_after(headers: Mapping[str, str]) -> int | None:
    for name, value in headers.items():
        if name.lower() == "retry-after":
            try:
                seconds = int(value)
            except (TypeError, ValueError):
                return None  # the HTTP-date form is not worth parsing; normal backoff applies
            return min(MAX_RETRY_AFTER_SECONDS, max(0, seconds))
    return None


def add_paid_argument(parser) -> None:
    parser.add_argument("--paid", choices=STAGES, default=None, help="spend real money on this stage; needs Aarmen's yes")


def paid_ledger_path() -> Path:
    """The fixed ledger. Under pytest it must be overridden, so no test can ever touch the real one."""
    if secrets_mac._under_pytest():
        override = os.environ.get(TEST_LEDGER_ENV)
        if not override:
            raise RuntimeError(f"under pytest set {TEST_LEDGER_ENV}; the real ledger is off limits")
        return Path(override)
    return PAID_LEDGER_PATH


def build_client(
    paid,
    *,
    ledger_path=None,
    fake_transport=None,
    reported_usage=None,
    sleep=time.sleep,
    reconcile_grace_seconds=DEFAULT_RECONCILE_GRACE_SECONDS,
    accept_key_usage_usd=None,
) -> Client:
    """The one way to get a client. `paid` is the value of --paid: None means a fake transport or nothing."""
    if paid is None:
        if fake_transport is None:
            raise PaidFlagRequired("no --paid flag: only a fake transport can be used")
        if isinstance(fake_transport, RealTransport):
            raise PaidFlagRequired("no --paid flag: the real transport cannot be used")
        if ledger_path is None or reported_usage is None:
            raise ValueError("a fake run needs its own ledger_path and reported_usage, so it cannot touch the real ledger")
        if Path(ledger_path).expanduser().resolve() == PAID_LEDGER_PATH.resolve():
            raise ValueError("a fake run cannot write to the real ledger")
        return Client(fake_transport, Ledger(ledger_path), reported_usage, sleep=sleep, reconcile_grace_seconds=reconcile_grace_seconds)
    if paid not in STAGES:
        raise PaidFlagRequired(f"--paid must be one of {STAGES}")
    if fake_transport is not None or reported_usage is not None:
        raise ValueError("--paid and a fake transport or usage reader cannot be combined")
    if ledger_path is not None:
        raise ValueError("a paid run's ledger location is fixed and cannot be passed")
    transport = RealTransport(paid)
    ledger = Ledger(paid_ledger_path(), stage=paid)
    return Client(
        transport,
        ledger,
        lambda: key_usage(transport),
        sleep=sleep,
        reconcile_grace_seconds=reconcile_grace_seconds,
        accept_key_usage_usd=accept_key_usage_usd,
    )


def require_private_path(path) -> Path:
    """Full 77-way Jev vectors may only be written under runs/private, which is gitignored.

    The two parts must be adjacent: macOS temp paths start /private/var, so "private" alone proves nothing.
    Resolved first, so that neither ".." nor a link can lead back out.
    """
    path = Path(path).resolve()
    if ("runs", "private") not in zip(path.parts, path.parts[1:]):
        raise ValueError(f"{path} is not under runs/private")
    return path


@dataclass
class RunReport:
    status: str  # complete, stopped, refused or arm_not_run
    stop_type: str | None = None
    reason: str | None = None
    decisions: dict = field(default_factory=dict)
    not_run: list = field(default_factory=list)
    projections: list = field(default_factory=list)
    retriable: list = field(default_factory=list)  # failed on an outage, with a pass left for a rerun


def is_outage_reason(reason) -> bool:
    """The service failed, not the answer. Gold isn't read at this point, so retrying these can't bias results."""
    return reason is not None and (reason == "network" or reason == USAGE_UNAVAILABLE or reason.startswith("http_"))


class _Results:
    """Append-only row file. A row with an answer or a final failure is finished and is never redone.

    A row that failed on an outage is retried on a rerun, up to OUTAGE_PASS_LIMIT passes. The count
    comes from the file, so it persists across runs.
    """

    def __init__(self, path):
        self.path = require_private_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.finished: dict = {}
        self._outage_passes: dict = {}
        if self.path.exists():
            raw = self.path.read_bytes()
            lines, _ = split_torn_tail(raw)
            for line in lines:
                record = json.loads(line)
                if record["status"] != "not_run":
                    fields = {k: v for k, v in record.items() if k not in ("row_id", "status")}
                    self._record(record["row_id"], Decision(**fields))
            with open(self.path, "r+b") as handle:
                heal_torn_tail(handle)

    def _record(self, row_id, decision: Decision) -> None:
        self.finished[row_id] = decision
        if decision.failed and is_outage_reason(decision.reason):
            self._outage_passes[row_id] = self._outage_passes.get(row_id, 0) + 1

    def retriable(self, row_id) -> bool:
        decision = self.finished.get(row_id)
        return (
            decision is not None
            and decision.failed
            and is_outage_reason(decision.reason)
            and self._outage_passes.get(row_id, 0) < OUTAGE_PASS_LIMIT
        )

    def needs_run(self, row_id) -> bool:
        return row_id not in self.finished or self.retriable(row_id)

    def _write(self, record: dict) -> None:
        with self._lock, open(self.path, "ab") as handle:
            handle.write(json.dumps(record, separators=(",", ":")).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def save(self, row_id, decision: Decision) -> None:
        self._write({"row_id": row_id, "status": "failed" if decision.failed else "done", **asdict(decision)})
        with self._lock:
            self._record(row_id, decision)

    def mark_not_run(self, row_id, reason: str) -> None:
        self._write({"row_id": row_id, "status": "not_run", "reason": reason})


def run_arm(
    rows,
    decide,
    *,
    client: Client,
    results_path,
    expected_per_row=None,
    row_id=lambda row: row["row_id"],
    allowed_arm_not_run=0,
) -> RunReport:
    """Run an arm on the ladder: 1 row, then 20, then only what the measured cost says will fit.

    This is the only way to make paid calls. `expected_per_row` is the projected cost of a row and
    a paid run must give one, even a first contact, so that it can be refused before sending.
    `allowed_arm_not_run` lets small LLM selection count rows without usable logprobs instead of
    stopping on the first. A rerun skips finished rows, so it never pays twice; outage rows get one
    more pass. Whatever stops the run, finished rows stay saved and the rest are marked not run.
    """
    if client.is_paid and expected_per_row is None:
        raise ValueError("a paid run must give expected_per_row, so it can be refused before anything is sent")
    rows = list(rows)
    seen = {row_id(row) for row in rows}
    if len(seen) != len(rows):
        raise ValueError("row ids must be unique, or a row would be paid for twice")
    client.begin_arm(results_path)
    try:
        return _run_arm(rows, decide, client, results_path, expected_per_row, row_id, allowed_arm_not_run, seen)
    finally:
        client._in_arm = False


def _run_arm(rows, decide, client, results_path, expected_per_row, row_id, allowed_arm_not_run, seen) -> RunReport:
    results = _Results(results_path)
    report = RunReport(status="complete")
    todo = [row for row in rows if results.needs_run(row_id(row))]
    counter_lock = threading.Lock()
    # The allowance is the arm's, so misses already saved in its file count against it.
    without_logprobs = [sum(1 for d in results.finished.values() if d.reason == NO_USABLE_LOGPROBS)]

    def run_one(row):
        if client.stop is not None:
            return None
        try:
            decision = decide(row)
        except RequestTooLarge:
            client.note_row(failed=True)
            decision = Decision(None, None, {}, "", 0.0, True, attempts=0, reason="request_too_large")
        except ArmNotRun as stop:
            with counter_lock:
                without_logprobs[0] += 1
                within_allowance = without_logprobs[0] <= allowed_arm_not_run
            if not within_allowance:
                client.latch(stop)
                return None
            client.note_row(failed=True)
            decision = Decision(None, None, {}, "", stop.cost, True, reason=NO_USABLE_LOGPROBS)
        except RunStop as stop:
            client.latch(stop)
            return None
        except Exception as error:  # a bug: stop cleanly, and keep its text out of the message
            client.latch(UnexpectedStop(f"unexpected {type(error).__name__} on row {row_id(row)}"))
            return None
        results.save(row_id(row), decision)
        return decision

    def run_batch(batch) -> list:
        with ThreadPoolExecutor(max_workers=client.max_concurrency) as pool:
            try:
                return [decision for decision in pool.map(run_one, batch) if decision is not None]
            except BaseException:
                # Ctrl-C: leaving this block waits for the queue, so queued rows must find the stop and not send.
                client.latch(UnexpectedStop("interrupted"))
                raise

    def project(n_rows, per_row) -> None:
        total = client.ledger.projected(n_rows, per_row)
        report.projections.append({"rows": n_rows, "per_row": per_row, "total": total})

    # Saved rows set the arm's model, so a resume can't mix builds.
    for decision in results.finished.values():
        if not decision.failed and decision.model:
            try:
                client.pin_model(decision.model)
            except ModelChanged as stop:
                client.latch(stop)

    refused = False
    position = 0
    try:
        if client.stop is None and expected_per_row is not None and todo:
            try:
                project(len(todo), expected_per_row)
            except BudgetExceeded:
                refused = True
                raise
        spent = paid_rows = 0
        for rung, next_rung in zip(LADDER, LADDER[1:] + (None,)):
            if client.stop is not None:
                break
            batch = todo[position:position + rung]
            for decision in run_batch(batch):
                if not decision.cached:
                    spent += decision.cost
                    paid_rows += 1
            position += len(batch)
            remaining = len(todo) - position
            if client.stop is not None or not remaining:
                break
            # Rows served from cache say nothing about price; the per-call check still guards them.
            project(min(next_rung or remaining, remaining), spent / max(1, paid_rows))
        if client.stop is None and position < len(todo):
            run_batch(todo[position:])
    except BudgetExceeded as stop:
        client.latch(stop)
    client.reconcile_final()

    report.decisions = {key: value for key, value in results.finished.items() if key in seen}
    report.not_run = [row_id(row) for row in rows if row_id(row) not in results.finished]
    report.retriable = [row_id(row) for row in rows if results.retriable(row_id(row))]
    if client.stop is not None:
        report.stop_type = type(client.stop).__name__
        report.reason = str(client.stop)
        report.status = "refused" if refused else "arm_not_run" if isinstance(client.stop, ArmNotRun) else "stopped"
        for key in report.not_run:
            results.mark_not_run(key, report.stop_type)
    return report
