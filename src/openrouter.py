"""Budgeted OpenRouter client: every spend guard lives here, shared by the Jev and small LLM clients.

Without `--paid <stage>` only a fake transport can be used. API shapes are recorded with their
source URLs in tests/fixtures/sources.json. Nothing the server sends is copied into an exception
or a log, so a response echoing the key cannot leak it.
"""
import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol
from urllib.parse import urlsplit

import httpx

import secrets_mac
from ledger import BudgetExceeded, Ledger, RunStop, heal_torn_tail, split_torn_tail

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_URL = "https://openrouter.ai/api/v1/key"
API_HOST = "openrouter.ai"

STAGES = ("stage0", "stage1")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER_PATH = REPO_ROOT / "runs" / "private" / "ledger.jsonl"

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
REQUEST_TIMEOUT_SECONDS = 60
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


class KeyOrCreditsStop(RunStop):
    pass


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


class UnexpectedStop(RunStop):
    pass


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

    def __init__(self, stage, *, key_reader=None, httpx_transport=None):
        if stage not in STAGES:
            raise PaidFlagRequired(f"real transport needs --paid with one of {STAGES}")
        self.stage = stage
        self._secret = _Secret((key_reader or secrets_mac.read_key)())
        self._http = httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False, transport=httpx_transport)

    def __repr__(self) -> str:
        return f"RealTransport(stage={self.stage!r})"

    def send(self, method: str, url: str, content: bytes | None = None) -> TransportResponse:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.hostname != API_HOST:
            raise TransportFailure("refusing to send the key anywhere but https://openrouter.ai")
        headers = {"Authorization": f"Bearer {self._secret.reveal()}", "Content-Type": "application/json"}
        try:
            response = self._http.request(method, url, content=content, headers=headers)
        except Exception as error:  # any library message could quote the request, so keep the type only
            failure = TransportFailure(f"request failed: {type(error).__name__}")
        else:
            return TransportResponse(response.status_code, dict(response.headers), response.content)
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
    if seen is not None and _squash(seen) != _squash(expected):
        raise ProviderChanged(f"expected provider {expected}, reply came from {str(seen)[:80]}")


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


def _reported_cost(data) -> float | None:
    usage = data.get("usage") if isinstance(data, dict) else None
    cost = usage.get("cost") if isinstance(usage, dict) else None
    return float(cost) if _is_number(cost) and cost >= 0 else None


def _body_stop_status(data) -> int | None:
    """A 200 can still carry an error object; a stop status there must stop like the real one."""
    error = data.get("error") if isinstance(data, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, int) and code in STOP_STATUSES else None


class Client:
    """Sends one row at a time through every guard. One client per arm; the ledger may be shared."""

    def __init__(
        self,
        transport: Transport,
        ledger: Ledger,
        reported_usage: Callable[[], float],
        *,
        sleep=time.sleep,
        max_concurrency: int = MAX_CONCURRENCY,
        reserve_usd: float = DEFAULT_RESERVE_USD,
    ):
        if not 1 <= max_concurrency <= MAX_CONCURRENCY:
            raise ValueError(f"max_concurrency must be 1 to {MAX_CONCURRENCY}")
        self.transport = transport
        self.ledger = ledger
        self.reported_usage = reported_usage
        self.max_concurrency = max_concurrency
        self.stop: RunStop | None = None
        self.consecutive_failures = 0
        self._sleep = sleep
        self._reserve_usd = reserve_usd
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._lock = threading.Lock()
        self._sent = 0
        self._largest_cost = 0.0
        self._output_tokens = 0
        self._output_calls = 0
        self._pinned_model: str | None = None
        # Spend before this point belongs to earlier runs, on both sides of the comparison.
        self._base_reported = reported_usage()
        self._base_confirmed = ledger.confirmed()
        self._base_uncertain = ledger.uncertain()

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
        if not due:
            return
        try:
            reported = self.reported_usage() - self._base_reported
        except RunStop as stop:
            self.latch(stop)
            return
        except Exception as error:
            self.latch(ReconciliationUnavailable(f"could not read reported usage: {type(error).__name__}"))
            return
        lower = max(0.0, self.ledger.confirmed() - self._base_confirmed)
        upper = lower + max(0.0, self.ledger.uncertain() - self._base_uncertain)
        if reconciliation_stops(lower, upper, reported):
            self.latch(
                ReconciliationGap(f"ledger holds US${lower:.6f} to US${upper:.6f} this run; OpenRouter reports US${reported:.6f}")
            )

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
                    if _reported_cost(data) is None:
                        raise MalformedResponse("no usage.cost")
                    answer = parse(data)
                except MalformedResponse as error:
                    reason = f"malformed: {error}"
                except ArmNotRun:
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


def build_client(paid, *, ledger_path=None, fake_transport=None, reported_usage=None, sleep=time.sleep) -> Client:
    """The one way to get a client. `paid` is the value of --paid: None means a fake transport or nothing."""
    if paid is None:
        if fake_transport is None:
            raise PaidFlagRequired("no --paid flag: only a fake transport can be used")
        if isinstance(fake_transport, RealTransport):
            raise PaidFlagRequired("no --paid flag: the real transport cannot be used")
        if ledger_path is None or reported_usage is None:
            raise ValueError("a fake run needs its own ledger_path and reported_usage, so it cannot touch the real ledger")
        return Client(fake_transport, Ledger(ledger_path), reported_usage, sleep=sleep)
    if paid not in STAGES:
        raise PaidFlagRequired(f"--paid must be one of {STAGES}")
    if fake_transport is not None or reported_usage is not None:
        raise ValueError("--paid and a fake transport or usage reader cannot be combined")
    transport = RealTransport(paid)
    ledger = Ledger(ledger_path or DEFAULT_LEDGER_PATH, stage=paid)
    return Client(transport, ledger, lambda: key_usage(transport), sleep=sleep)


def require_private_path(path) -> Path:
    """Full 77-way Jev vectors may only be written under runs/private, which is gitignored.

    The two parts must be adjacent: macOS temp paths start /private/var, so "private" alone proves nothing.
    """
    path = Path(path)
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


class _Results:
    """Append-only row file. A row with an answer or a final failure is finished and is never redone."""

    def __init__(self, path):
        self.path = require_private_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.finished: dict = {}
        if self.path.exists():
            raw = self.path.read_bytes()
            lines, _ = split_torn_tail(raw)
            for line in lines:
                record = json.loads(line)
                if record["status"] != "not_run":
                    fields = {k: v for k, v in record.items() if k not in ("row_id", "status")}
                    self.finished[record["row_id"]] = Decision(**fields)
            with open(self.path, "r+b") as handle:
                heal_torn_tail(handle)

    def _write(self, record: dict) -> None:
        with self._lock, open(self.path, "ab") as handle:
            handle.write(json.dumps(record, separators=(",", ":")).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def save(self, row_id, decision: Decision) -> None:
        self._write({"row_id": row_id, "status": "failed" if decision.failed else "done", **asdict(decision)})
        with self._lock:
            self.finished[row_id] = decision

    def mark_not_run(self, row_id, reason: str) -> None:
        self._write({"row_id": row_id, "status": "not_run", "reason": reason})


def run_arm(rows, decide, *, client: Client, results_path, expected_per_row=None, row_id=lambda row: row["row_id"]) -> RunReport:
    """Run an arm on the ladder: 1 row, then 20, then only what the measured cost says will fit.

    `expected_per_row` is the projected cost from earlier measurement, or None for a first
    contact. A rerun skips finished rows, so it never pays twice. Whatever stops the run, finished
    rows stay saved and the rest are marked not run.
    """
    rows = list(rows)
    seen = {row_id(row) for row in rows}
    if len(seen) != len(rows):
        raise ValueError("row ids must be unique, or a row would be paid for twice")
    results = _Results(results_path)
    report = RunReport(status="complete")
    todo = [row for row in rows if row_id(row) not in results.finished]

    def run_one(row) -> None:
        if client.stop is not None:
            return
        try:
            decision = decide(row)
        except RequestTooLarge:
            client.note_row(failed=True)
            decision = Decision(None, None, {}, "", 0.0, True, attempts=0, reason="request_too_large")
        except RunStop as stop:
            client.latch(stop)
            return
        except Exception as error:  # a bug: stop cleanly, and keep its text out of the message
            client.latch(UnexpectedStop(f"unexpected {type(error).__name__} on row {row_id(row)}"))
            return
        results.save(row_id(row), decision)

    def run_batch(batch) -> None:
        with ThreadPoolExecutor(max_workers=client.max_concurrency) as pool:
            try:
                list(pool.map(run_one, batch))
            except BaseException:
                # Ctrl-C: leaving this block waits for the queue, so queued rows must find the stop and not send.
                client.latch(UnexpectedStop("interrupted"))
                raise

    def project(n_rows, per_row) -> None:
        total = client.ledger.projected(n_rows, per_row)
        report.projections.append({"rows": n_rows, "per_row": per_row, "total": total})

    refused = False
    position = 0
    try:
        if expected_per_row is not None and todo:
            try:
                project(len(todo), expected_per_row)
            except BudgetExceeded:
                refused = True
                raise
        spent = paid_rows = 0
        for rung, next_rung in zip(LADDER, LADDER[1:] + (None,)):
            batch = todo[position:position + rung]
            run_batch(batch)
            position += len(batch)
            for row in batch:
                decision = results.finished.get(row_id(row))
                if decision is not None and not decision.cached:
                    spent += decision.cost
                    paid_rows += 1
            remaining = len(todo) - position
            if client.stop is not None or not remaining:
                break
            # Rows served from cache say nothing about price; the per-call check still guards them.
            project(min(next_rung or remaining, remaining), spent / max(1, paid_rows))
        if client.stop is None and position < len(todo):
            run_batch(todo[position:])
    except BudgetExceeded as stop:
        client.latch(stop)

    report.decisions = {key: value for key, value in results.finished.items() if key in seen}
    report.not_run = [row_id(row) for row in rows if row_id(row) not in results.finished]
    if client.stop is not None:
        report.stop_type = type(client.stop).__name__
        report.reason = str(client.stop)
        report.status = "refused" if refused else "arm_not_run" if isinstance(client.stop, ArmNotRun) else "stopped"
        for key in report.not_run:
            results.mark_not_run(key, report.stop_type)
    return report
