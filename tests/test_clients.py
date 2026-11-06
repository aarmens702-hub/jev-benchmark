"""Spend guards for the paid clients. Every test uses a fake transport; nothing here spends or reads a key."""
import argparse
import json
import logging
import math
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import httpx
import pytest

import jev_client
import ledger as ledger_module
import llm_client
import openrouter
import retrieval
import secrets_mac
from jev_client import JevCache, jev_decide
from ledger import BudgetExceeded, Ledger, LedgerCorrupt
from llm_client import llm_pick
from openrouter import (
    ArmNotRun,
    Client,
    Decision,
    HiddenReasoning,
    KeyOrCreditsStop,
    ModelChanged,
    PaidFlagRequired,
    ProviderChanged,
    RealTransport,
    RequestTooLarge,
    TransportFailure,
    TransportResponse as FakeResponse,
    build_client,
    key_usage,
    option_order,
    reconciliation_stops,
    run_arm,
)
from retrieval import Example, retrieve
from secrets_mac import KeyUnavailable, keychain_command, read_key

FIXTURES = Path(__file__).parent / "fixtures"
OPTIONS = {"card_arrival": "Waiting for a card.", "lost_card": "Lost a card.", "top_up_failed": "A top up failed."}
BUILD = "typesafe/jev-1.14-20261014"
FAKE_KEY = "sk-or-v1-FAKE-TEST-VALUE-0123456789"


@pytest.fixture(autouse=True)
def _no_subprocess(monkeypatch):
    """A second lock on the Keychain: nothing in this file can start a process, whatever read_key does."""

    def blocked(*args, **kwargs):
        raise AssertionError("subprocess.run is blocked in these tests")

    monkeypatch.setattr(subprocess, "run", blocked)


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def ok(data):
    return FakeResponse(200, {}, json.dumps(data).encode())


def status(code, headers=None, body=b"{}"):
    return FakeResponse(code, headers or {}, body)


def jev_ok(body, choice=None, cost=0.0001, model=BUILD, top=0.84):
    """A valid Decisions reply to `body`, built from the documented shape."""
    data = fixture("decisions_choice_response.json")
    labels = list(body["questions"]["label"]["criteria"])
    choice = choice or labels[0]
    rest = [label for label in labels if label != choice]
    probs = {label: round((1 - top) / len(rest), 6) for label in rest}
    probs[choice] = top
    data["answers"]["label"].update(choice=choice, probabilities=probs)
    data["usage"]["cost"] = cost
    data["model"] = model
    return ok(data)


def llm_ok(body, cost=0.00003, tokens=5, reasoning=0, id_value=None, logprob=-0.25, model="openai/gpt-oss-20b"):
    data = fixture("chat_logprobs_response.json")
    enum = body["response_format"]["json_schema"]["schema"]["properties"]["id"]["enum"]
    id_value = enum[0] if id_value is None else id_value
    text = str(id_value)
    data["model"] = model
    data["provider"] = body["provider"]["only"][0]
    data["choices"][0]["message"]["content"] = '{"id":' + text + "}"
    data["choices"][0]["logprobs"]["content"] = [
        {"token": '{"', "logprob": -0.001, "bytes": None, "top_logprobs": []},
        {"token": "id", "logprob": -0.001, "bytes": None, "top_logprobs": []},
        {"token": '":', "logprob": -0.001, "bytes": None, "top_logprobs": []},
        {"token": text, "logprob": logprob, "bytes": None, "top_logprobs": []},
        {"token": "}", "logprob": -0.001, "bytes": None, "top_logprobs": []},
    ]
    data["usage"].update(cost=cost, completion_tokens=tokens)
    data["usage"]["completion_tokens_details"]["reasoning_tokens"] = reasoning
    return ok(data)


class FakeTransport:
    """Records every request and answers from `handler(body, n)`. A list is played in order, last entry repeating."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.raw = []
        self._lock = threading.Lock()

    def send(self, method, url, content=None):
        with self._lock:
            n = len(self.calls)
            body = json.loads(content) if content else None
            self.calls.append(body)
            self.raw.append(content)
        reply = self.handler(body, n)
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_client(tmp_path, handler, *, ledger=None, reported=None, **kwargs):
    ledger = ledger or Ledger(tmp_path / "ledger.jsonl")
    transport = FakeTransport(handler)
    sleeps = []
    client = Client(transport, ledger, reported or ledger.confirmed, sleep=sleeps.append, **kwargs)
    return client, transport, ledger, sleeps


def rows(n):
    return [{"row_id": f"r{i}", "text": f"message number {i}"} for i in range(n)]


def jev_decider(client, **kwargs):
    return lambda row: jev_decide(row["row_id"], row["text"], OPTIONS, client=client, **kwargs)


def llm_decider(client):
    return lambda row: llm_pick(row["row_id"], row["text"], OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)


def results_path(tmp_path):
    return tmp_path / "runs" / "private" / "arm.jsonl"


def read_results(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# ------------------------------------------------------------------ the key


def test_no_paid_flag_means_fake_transport_only(tmp_path):
    parser = argparse.ArgumentParser()
    openrouter.add_paid_argument(parser)
    assert parser.parse_args([]).paid is None
    assert parser.parse_args(["--paid", "stage0"]).paid == "stage0"
    assert parser.parse_args(["--paid", "stage1"]).paid == "stage1"
    with pytest.raises(SystemExit):
        parser.parse_args(["--paid", "stage2"])

    ledger_path = tmp_path / "ledger.jsonl"
    # No flag and no fake: nothing to send with.
    with pytest.raises(PaidFlagRequired):
        build_client(None, ledger_path=ledger_path)
    # No flag: the real transport cannot be built or smuggled in as the "fake".
    with pytest.raises(PaidFlagRequired):
        RealTransport(None, key_reader=lambda: FAKE_KEY)
    with pytest.raises(PaidFlagRequired):
        RealTransport("stage9", key_reader=lambda: FAKE_KEY)
    real = RealTransport("stage0", key_reader=lambda: FAKE_KEY, httpx_transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(PaidFlagRequired):
        build_client(None, ledger_path=ledger_path, fake_transport=real, reported_usage=lambda: 0.0)
    # A subclass is still the real thing.
    class Disguised(RealTransport):
        pass
    disguised = Disguised("stage0", key_reader=lambda: FAKE_KEY, httpx_transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(PaidFlagRequired):
        build_client(None, ledger_path=ledger_path, fake_transport=disguised, reported_usage=lambda: 0.0)

    # No flag with a fake works, and uses that fake.
    fake = FakeTransport(lambda body, n: jev_ok(body))
    client = build_client(None, ledger_path=ledger_path, fake_transport=fake, reported_usage=lambda: 0.0)
    assert client.transport is fake
    assert jev_decide("r0", "text", OPTIONS, client=client).failed is False

    # A flag and a fake together are ambiguous and refused.
    with pytest.raises(ValueError):
        build_client("stage0", ledger_path=ledger_path, fake_transport=fake, reported_usage=lambda: 0.0)
    # A bad stage is refused before the key is touched.
    with pytest.raises(PaidFlagRequired):
        build_client("stage9", ledger_path=ledger_path)
    # The paid route takes no ledger path at all (ruling R1).
    with pytest.raises(ValueError, match="ledger"):
        build_client("stage0", ledger_path=ledger_path)
    # The flag alone reaches for the key, and read_key refuses under pytest.
    with pytest.raises(KeyUnavailable):
        build_client("stage0")


def test_read_key_raises_under_pytest(monkeypatch):
    """Refusal comes before the subprocess, on either signal that pytest is running."""
    ran = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: ran.append(a))
    with pytest.raises(KeyUnavailable, match="pytest"):
        read_key()
    # Still refused when the per-test env var is gone, because pytest is imported.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with pytest.raises(KeyUnavailable, match="pytest"):
        read_key()
    assert ran == []


def test_keychain_command_is_exact_and_holds_no_secret():
    assert secrets_mac.SERVICE == "jev-benchmark-openrouter"
    assert keychain_command("aarmen") == [
        "/usr/bin/security", "find-generic-password", "-a", "aarmen", "-s", "jev-benchmark-openrouter", "-w",
    ]


def test_key_reader_with_fake_runner_parses_and_never_echoes():
    seen = []

    def good(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=FAKE_KEY + "\n", stderr="")

    assert secrets_mac._read_with_runner(good, "aarmen") == FAKE_KEY
    assert seen[0][0] == keychain_command("aarmen")
    assert seen[0][1].get("capture_output") is True and not seen[0][1].get("shell")
    assert seen[0][1].get("timeout") == secrets_mac.SECURITY_TIMEOUT_SECONDS > 0

    def hung(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"], output=FAKE_KEY)

    with pytest.raises(KeyUnavailable) as err:
        secrets_mac._read_with_runner(hung, "aarmen")
    assert FAKE_KEY not in "".join(traceback.format_exception(err.value))

    def missing(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 44, stdout="", stderr=f"leaky stderr {FAKE_KEY}")

    with pytest.raises(KeyUnavailable) as err:
        secrets_mac._read_with_runner(missing, "aarmen")
    assert FAKE_KEY not in str(err.value)

    def empty(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="\n", stderr="")

    with pytest.raises(KeyUnavailable):
        secrets_mac._read_with_runner(empty, "aarmen")

    def no_binary(cmd, **kwargs):
        raise FileNotFoundError("security")

    with pytest.raises(KeyUnavailable):
        secrets_mac._read_with_runner(no_binary, "aarmen")


def test_key_never_appears_in_repr_logs_exceptions_or_ledger(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    seen_headers = []

    def echoing(request):
        seen_headers.append(request.headers["authorization"])
        return httpx.Response(402, json={"error": {"code": 402, "message": f"bad key {FAKE_KEY}"}})

    transport = RealTransport("stage0", key_reader=lambda: FAKE_KEY, httpx_transport=httpx.MockTransport(echoing))
    ledger = Ledger(tmp_path / "ledger.jsonl", stage="stage0")
    client = Client(transport, ledger, lambda: 0.0, sleep=lambda s: None)
    path = results_path(tmp_path)

    report = run_arm(rows(3), jev_decider(client), client=client, results_path=path, expected_per_row=0.0001)

    assert seen_headers == [f"Bearer {FAKE_KEY}"]  # the key is really used, once, in the header only
    assert report.status == "stopped"
    surfaces = [
        repr(transport), str(transport), repr(vars(transport)), repr(client), repr(vars(client)),
        repr(report), report.reason, caplog.text,
        (tmp_path / "ledger.jsonl").read_text(), path.read_text(),
    ]
    with pytest.raises(KeyOrCreditsStop) as err:
        jev_decide("r9", "text", OPTIONS, client=client)
    surfaces.append("".join(traceback.format_exception(err.value)))
    assert not [s for s in surfaces if FAKE_KEY in s]

    # A network error that quotes the key is stripped too, with no chained cause to print.
    def exploding(request):
        raise httpx.ConnectError(f"cannot connect with {FAKE_KEY}")

    broken = RealTransport("stage0", key_reader=lambda: FAKE_KEY, httpx_transport=httpx.MockTransport(exploding))
    with pytest.raises(TransportFailure) as err:
        broken.send("POST", openrouter.DECISIONS_URL, b"{}")
    assert FAKE_KEY not in "".join(traceback.format_exception(err.value))
    assert err.value.__cause__ is None

    # The key never travels in a URL, and never to any other host.
    sent_urls = []

    def recording(request):
        sent_urls.append(str(request.url))
        return httpx.Response(200, json={})

    clean = RealTransport("stage0", key_reader=lambda: FAKE_KEY, httpx_transport=httpx.MockTransport(recording))
    clean.send("POST", openrouter.DECISIONS_URL, b"{}")
    assert sent_urls == [openrouter.DECISIONS_URL] and FAKE_KEY not in sent_urls[0]
    for bad in ("https://example.com/x", "http://openrouter.ai/api/alpha/decisions", "https://openrouter.ai.example.com/x"):
        with pytest.raises(TransportFailure):
            clean.send("POST", bad, b"{}")
    assert len(sent_urls) == 1


# ------------------------------------------------------------------ the ledger


def test_ledger_refuses_call_past_cap_before_sending(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", cap_usd=0.002)
    ledger.charge(0.002, {"note": "already spent"})
    client, transport, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body), ledger=ledger)
    with pytest.raises(BudgetExceeded):
        jev_decide("r0", "text", OPTIONS, client=client)
    assert transport.calls == []

    # The stage limit refuses in the same way, even with room under the total.
    stage_ledger = Ledger(tmp_path / "stage.jsonl", stage_cap_usd=0.002, stage="stage0")
    stage_ledger.charge(0.002, {"note": "already spent"})
    client, transport, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body), ledger=stage_ledger)
    with pytest.raises(BudgetExceeded):
        jev_decide("r0", "text", OPTIONS, client=client)
    assert transport.calls == []


def test_ledger_persists_across_restart_and_stages(tmp_path):
    path = tmp_path / "ledger.jsonl"
    first = Ledger(path, stage="stage0")
    first.charge(0.40, {"row_id": "a"})
    first.charge(0.05, {"row_id": "b"})

    again = Ledger(path, stage="stage0")
    assert again.spent() == pytest.approx(0.45)
    assert again.remaining() == pytest.approx(0.05)  # stage0 limit, not the total
    with pytest.raises(BudgetExceeded):
        again.reserve(0.06, {})

    # Stage 1 has its own room but still counts stage 0 against the total.
    later = Ledger(path, stage="stage1")
    assert later.spent() == pytest.approx(0.45)
    assert later.stage_spent() == 0.0
    assert later.remaining() == pytest.approx(7.15 - 0.45)
    with pytest.raises(BudgetExceeded):
        later.projected(1, 7.15)


def test_ledger_caps_cannot_be_raised(tmp_path):
    path = tmp_path / "ledger.jsonl"
    assert ledger_module.HARD_CAP_USD == 7.15
    assert ledger_module.STAGE_CAPS_USD["stage0"] == 0.50
    assert Ledger(path).cap_usd == 7.15
    assert Ledger(path, stage="stage0").stage_cap_usd == 0.50
    with pytest.raises(ValueError):
        Ledger(path, cap_usd=7.16)
    with pytest.raises(ValueError):
        Ledger(path, stage="stage0", stage_cap_usd=0.51)
    # A NaN cap compares false with everything, so it used to allow any spend.
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            Ledger(path, cap_usd=bad)
        with pytest.raises(ValueError):
            Ledger(path, stage_cap_usd=bad)
        with pytest.raises(ValueError):
            Ledger(path, stage="stage0", stage_cap_usd=bad)
    with pytest.raises(ValueError):
        Ledger(path, stage_cap_usd=7.16)
    # A misspelt stage used to get no stage limit at all.
    for bad in ("stage_0", "Stage0", "stage2", ""):
        with pytest.raises(ValueError):
            Ledger(path, stage=bad)


def test_ledger_charge_records_then_raises_past_cap(tmp_path):
    """Money already spent is written down even when it breaks the cap."""
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path, cap_usd=0.01)
    with pytest.raises(BudgetExceeded):
        ledger.charge(0.02, {"row_id": "a"})
    assert Ledger(path, cap_usd=0.01).spent() == pytest.approx(0.02)
    for bad in (-1.0, float("nan"), float("inf"), True, "0.1"):
        with pytest.raises(ValueError):
            ledger.charge(bad, {})


def test_ledger_is_append_only(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.charge(0.1, {"row_id": "a"})
    for step in [lambda: ledger.charge(0.1, {}), lambda: ledger.settle(ledger.reserve(0.1, {}), 0.05, "reported", {})]:
        before = path.read_bytes()
        step()
        assert path.read_bytes().startswith(before) and len(path.read_bytes()) > len(before)


def test_crash_between_reserve_and_settle_counts_the_reservation(tmp_path):
    path = tmp_path / "ledger.jsonl"
    Ledger(path).reserve(0.25, {"row_id": "a"})  # process dies here: the call may have been billed
    assert Ledger(path).spent() == pytest.approx(0.25)


def test_ledger_tolerates_a_torn_tail_but_refuses_a_corrupt_middle(tmp_path):
    path = tmp_path / "ledger.jsonl"
    first = Ledger(path)
    first.charge(0.1, {})
    with open(path, "ab") as handle:
        handle.write(b'{"kind":"charge","id":"x","cos')  # a crash mid-write

    again = Ledger(path)
    assert again.spent() == pytest.approx(0.1)
    again.charge(0.2, {})
    assert Ledger(path).spent() == pytest.approx(0.3)  # still readable after healing

    bad = tmp_path / "bad.jsonl"
    Ledger(bad).charge(0.1, {})
    raw = bad.read_bytes()
    bad.write_bytes(b"not json\n" + raw)
    with pytest.raises(LedgerCorrupt):
        Ledger(bad)


def test_concurrent_reservations_cannot_overshoot_the_cap(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", cap_usd=0.05)
    granted = []

    def grab():
        try:
            granted.append(ledger.reserve(0.01, {}))
        except BudgetExceeded:
            pass

    threads = [threading.Thread(target=grab) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(granted) == 5


def test_two_ledger_objects_on_one_file_see_each_others_spend(tmp_path):
    path = tmp_path / "ledger.jsonl"
    one, two = Ledger(path, cap_usd=0.05), Ledger(path, cap_usd=0.05)
    one.reserve(0.05, {})
    with pytest.raises(BudgetExceeded):
        two.reserve(0.01, {})


# ------------------------------------------------------------------ ladder and preflight


def test_ladder_runs_1_then_20_then_projects_before_continuing(tmp_path):
    client, transport, ledger, _ = make_client(tmp_path, lambda body, n: jev_ok(body, cost=0.001))
    seen = []
    real_projected = ledger.projected

    def spy(n_rows, per_row):
        seen.append((len(transport.calls), n_rows, per_row))
        return real_projected(n_rows, per_row)

    ledger.projected = spy
    report = run_arm(rows(60), jev_decider(client), client=client, results_path=results_path(tmp_path))

    assert report.status == "complete" and len(report.decisions) == 60
    # After 1 row: project the next 20. After 21 rows: project the remaining 39, before sending row 22.
    assert [(calls, n) for calls, n, _ in seen] == [(1, 20), (21, 39)]
    assert seen[0][2] == pytest.approx(0.001) and seen[1][2] == pytest.approx(0.001)
    assert len(transport.calls) == 60


def test_ladder_stops_at_21_rows_when_measured_cost_does_not_fit(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", cap_usd=1.0)
    # 0.01 a row: 21 rows fit (0.21) and 100 rows do not.
    client, transport, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body, cost=0.01), ledger=ledger)
    report = run_arm(rows(121), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert len(transport.calls) == 21
    assert report.status == "stopped" and report.stop_type == "BudgetExceeded"
    assert len(report.decisions) == 21 and len(report.not_run) == 100


def test_ladder_measures_retries_as_part_of_the_row_cost(tmp_path):
    """Every row needs a retry, so the projection must use 2 x the call cost."""
    def handler(body, n):
        return jev_ok(body, cost=0.001) if n % 2 else jev_ok(body, choice="not_an_option", cost=0.001)

    client, _, ledger, _ = make_client(tmp_path, handler, max_concurrency=1)
    seen = []
    real_projected = ledger.projected
    ledger.projected = lambda n, p: (seen.append(p), real_projected(n, p))[1]
    run_arm(rows(5), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert seen[0] == pytest.approx(0.002)


def test_arm_refused_when_projected_cost_exceeds_remaining_budget(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", stage="stage0")
    ledger.charge(0.40, {"note": "pilot"})
    client, transport, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body), ledger=ledger)
    path = results_path(tmp_path)

    # 756 rows at 0.0002 is 0.1512, which does not fit in the 0.10 left of stage 0.
    report = run_arm(rows(756), jev_decider(client), client=client, results_path=path, expected_per_row=0.0002)

    assert report.status == "refused" and report.stop_type == "BudgetExceeded"
    assert transport.calls == []
    assert len(report.not_run) == 756 and report.decisions == {}
    assert ledger.spent() == pytest.approx(0.40)

    # The same arm at a price that fits is allowed through.
    client, transport, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body, cost=0.0001), ledger=ledger)
    report = run_arm(rows(30), jev_decider(client), client=client, results_path=path, expected_per_row=0.0002)
    assert report.status == "complete" and len(transport.calls) == 30


def test_ledger_projected_returns_total_and_raises_past_either_cap(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", stage="stage0")
    ledger.charge(0.10, {})
    assert ledger.projected(100, 0.001) == pytest.approx(0.20)
    with pytest.raises(BudgetExceeded):
        ledger.projected(1000, 0.001)  # 1.10 is under the total but past stage 0's 0.50
    assert ledger.spent() == pytest.approx(0.10)  # a projection spends nothing


# ------------------------------------------------------------------ retries


def test_retries_are_charged_to_the_ledger(tmp_path):
    replies = [
        lambda body: status(529),  # not reported, so charged at the estimate
        lambda body: jev_ok(body, choice="not_an_option", cost=0.0003),  # billed, then rejected
        lambda body: jev_ok(body, cost=0.0002),
    ]
    client, transport, ledger, _ = make_client(tmp_path, lambda body, n: replies[n](body), reserve_usd=0.001)
    decision = jev_decide("r0", "text", OPTIONS, client=client)

    assert decision.failed is False and decision.attempts == 3
    assert len(transport.calls) == 3
    assert ledger.spent() == pytest.approx(0.001 + 0.0003 + 0.0002)
    assert decision.cost == pytest.approx(0.001 + 0.0003 + 0.0002)
    assert ledger.confirmed() == pytest.approx(0.0005)  # only the reported part


def test_missing_reported_cost_is_never_counted_as_free(tmp_path):
    def handler(body, n):
        data = json.loads(jev_ok(body).body)
        del data["usage"]["cost"]
        return ok(data)

    client, transport, ledger, _ = make_client(tmp_path, handler, reserve_usd=0.001)
    decision = jev_decide("r0", "text", OPTIONS, client=client)
    assert decision.failed is True and len(transport.calls) == 4
    assert ledger.spent() == pytest.approx(0.004)


def test_endless_529s_stop_at_3_retries_per_row_and_never_loop(tmp_path):
    client, transport, _, sleeps = make_client(tmp_path, lambda body, n: status(529))
    decision = jev_decide("r0", "text", OPTIONS, client=client)
    assert decision.failed is True and decision.pred is None
    assert decision.attempts == 4 and len(transport.calls) == 4  # 1 try + 3 retries
    assert len(sleeps) == 3  # no sleep after the last try

    # Across a whole arm the failure stop ends it: 10 rows x 4 attempts, then nothing.
    client, transport, _, _ = make_client(tmp_path, lambda body, n: status(529), max_concurrency=1)
    report = run_arm(rows(500), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert len(transport.calls) == 40
    assert report.stop_type == "ConsecutiveFailures"


@pytest.mark.parametrize("code", [429, 529])
def test_429_and_529_back_off_and_retry_and_do_not_count_as_failures_unless_final(tmp_path, code):
    # Each row hits the error 3 times, then succeeds on its last allowed try: 12 rows, no failed rows, no stop.
    def handler(body, n):
        return status(code) if n % 4 != 3 else jev_ok(body)

    client, transport, _, sleeps = make_client(tmp_path, handler, max_concurrency=1)
    decisions = [jev_decide(f"r{i}", "text", OPTIONS, client=client) for i in range(12)]
    assert not any(d.failed for d in decisions)
    assert client.stop is None and client.consecutive_failures == 0
    assert len(transport.calls) == 48
    assert sleeps[:3] == [1, 2, 4]  # exponential

    # When it is final, it counts as one failed row.
    client, _, _, _ = make_client(tmp_path, lambda body, n: status(code))
    assert jev_decide("r0", "text", OPTIONS, client=client).failed is True
    assert client.consecutive_failures == 1


def test_retry_after_header_is_honoured_and_capped(tmp_path):
    replies = [status(429, {"Retry-After": "7"}), status(429, {"retry-after": "9999"}), status(429, {"Retry-After": "soon"})]
    client, _, _, sleeps = make_client(tmp_path, lambda body, n: replies[n] if n < 3 else jev_ok(body))
    jev_decide("r0", "text", OPTIONS, client=client)
    assert sleeps == [7, 60, 4]


def test_network_errors_retry_and_are_charged_at_the_estimate(tmp_path):
    client, transport, ledger, _ = make_client(
        tmp_path, lambda body, n: TransportFailure("timeout") if n == 0 else jev_ok(body, cost=0.0001), reserve_usd=0.001
    )
    assert jev_decide("r0", "text", OPTIONS, client=client).failed is False
    assert ledger.spent() == pytest.approx(0.0011)


def test_400_fails_the_row_without_retrying_or_charging(tmp_path):
    client, transport, ledger, sleeps = make_client(tmp_path, lambda body, n: status(400))
    decision = jev_decide("r0", "text", OPTIONS, client=client)
    assert decision.failed is True and len(transport.calls) == 1 and sleeps == []
    assert ledger.spent() == 0.0


def test_10_consecutive_failures_stop_the_run(tmp_path):
    def handler(body, n):
        return jev_ok(body, choice="not_an_option")

    client, transport, _, _ = make_client(tmp_path, handler)
    path = results_path(tmp_path)
    report = run_arm(rows(50), jev_decider(client), client=client, results_path=path)
    assert report.status == "stopped" and report.stop_type == "ConsecutiveFailures"
    assert sum(d.failed for d in report.decisions.values()) >= 10
    assert len(report.decisions) + len(report.not_run) == 50
    assert len(transport.calls) <= 4 * (10 + 3)  # at most 3 others in flight when the stop lands

    # A success in between resets the count: 9 fails, 1 success, 9 fails never stops.
    def mixed(body, n):
        return jev_ok(body) if n % 40 == 36 else jev_ok(body, choice="not_an_option")

    client, _, _, _ = make_client(tmp_path, mixed, max_concurrency=1)
    for i in range(30):
        jev_decide(f"r{i}", "text", OPTIONS, client=client)
    assert client.stop is None


def test_concurrency_never_exceeds_4(tmp_path):
    gauge = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def handler(body, n):
        with lock:
            gauge["now"] += 1
            gauge["peak"] = max(gauge["peak"], gauge["now"])
        time.sleep(0.005)
        with lock:
            gauge["now"] -= 1
        return jev_ok(body)

    client, _, _, _ = make_client(tmp_path, handler)
    report = run_arm(rows(45), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.status == "complete"
    assert gauge["peak"] == 4  # reached, so the test can see the limit, and never passed

    # Direct callers on many threads are held to the same limit by the client itself.
    gauge.update(now=0, peak=0)
    client, _, _, _ = make_client(tmp_path, handler)
    threads = [threading.Thread(target=jev_decide, args=(f"r{i}", "t", OPTIONS), kwargs={"client": client}) for i in range(24)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert gauge["peak"] == 4

    with pytest.raises(ValueError):
        make_client(tmp_path, handler, max_concurrency=5)


# ------------------------------------------------------------------ reconciliation


def test_ledger_gap_over_10_percent_with_reported_usage_stops_the_run(tmp_path):
    # OpenRouter reports 1.2 x our ledger: silently billed more than we recorded.
    holder = {}
    client, transport, ledger, _ = make_client(
        tmp_path,
        lambda body, n: jev_ok(body, cost=0.001),
        reported=lambda: holder["ledger"].confirmed() * 1.2 if holder.get("ledger") else 0.0,
        max_concurrency=1,
    )
    holder["ledger"] = ledger
    path = results_path(tmp_path)
    report = run_arm(rows(300), jev_decider(client), client=client, results_path=path)
    assert report.status == "stopped" and report.stop_type == "ReconciliationGap"
    assert len(transport.calls) == 100  # checked at 100, stopped before call 101
    assert len(report.decisions) == 100  # the paid row that tripped the check is kept
    assert len(report.not_run) == 200


def test_reconciliation_runs_every_100_calls_and_passes_when_books_agree(tmp_path):
    reads = []
    holder = {}

    def reported():
        reads.append(len(holder["t"].calls) if "t" in holder else 0)
        return holder["l"].confirmed() if "l" in holder else 0.0

    client, transport, ledger, _ = make_client(tmp_path, lambda body, n: jev_ok(body, cost=0.001), reported=reported, max_concurrency=1)
    holder.update(t=transport, l=ledger)
    report = run_arm(rows(250), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.status == "complete"
    assert reads == [0, 100, 200, 250]  # baseline, each 100 calls, and the end of the arm


def test_reconciliation_counts_only_this_runs_spend(tmp_path):
    """Prior ledger spend and prior key usage are both excluded by the baseline."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.charge(2.0, {"note": "earlier run"})
    state = {"reported": 2.5}  # key's all-time usage disagrees by a fixed earlier amount

    def reported():
        return state["reported"] + (ledger.confirmed() - 2.0)

    client, _, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body, cost=0.001), ledger=ledger, reported=reported, max_concurrency=1)
    report = run_arm(rows(120), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.status == "complete"


@pytest.mark.parametrize(
    "lower, upper, reported, stops, why",
    [
        (0.0, 0.0, 0.0, False, "nothing spent, and no division by zero"),
        (0.002, 0.002, 0.001, False, "a 100% gap on a fifth of a cent is noise"),
        (0.0, 0.0, 0.009, False, "under the floor"),
        (0.002, 0.002, 0.0, False, "usage not yet posted, under the floor"),
        (0.05, 0.05, 0.0, True, "real ledger spend and none reported"),
        (0.0, 0.0, 0.05, True, "real reported spend and none in the ledger"),
        (1.0, 1.0, 1.0, False, "agree"),
        (0.91, 0.91, 1.0, False, "9% under"),
        (0.90, 0.90, 1.0, False, "exactly 10% is not over 10%"),
        (0.89, 0.89, 1.0, True, "11% under"),
        (1.0, 1.0, 1.2, True, "reported is 20% above the ledger"),
        (1.2, 1.2, 1.0, True, "ledger is 20% above reported"),
        (0.8, 1.2, 1.0, False, "reported sits inside confirmed + estimated"),
        (0.5, 0.6, 1.0, True, "reported is above even the estimated upper bound"),
    ],
)
def test_reconciliation_guards_against_near_zero_false_alarms(lower, upper, reported, stops, why):
    assert reconciliation_stops(lower, upper, reported) is stops, why


def test_near_zero_spend_does_not_stop_a_real_run(tmp_path):
    """100 calls of very cheap spend with usage that has not posted yet must not stop the run."""
    client, _, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body, cost=0.00001), reported=lambda: 0.0, max_concurrency=1)
    report = run_arm(rows(150), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.status == "complete"


@pytest.mark.parametrize("errors_were_billed", [False, True])
def test_estimated_charges_from_errors_do_not_trigger_a_false_gap(tmp_path, errors_were_billed):
    """Error attempts are charged at an estimate and may or may not be billed. Neither outcome is a gap."""
    holder = {}

    def handler(body, n):
        return status(529) if n % 4 == 0 else jev_ok(body, cost=0.0002)

    def reported():
        if "l" not in holder:
            return 0.0
        return holder["l"].spent() if errors_were_billed else holder["l"].confirmed()

    client, _, ledger, _ = make_client(tmp_path, handler, reported=reported, max_concurrency=1, reserve_usd=0.01)
    holder["l"] = ledger
    for i in range(120):
        jev_decide(f"r{i}", "text", OPTIONS, client=client)
    assert ledger.spent() > ledger.confirmed() * 1.1  # the ledger really does hold more than was reported
    assert client.stop is None


def test_unreadable_reported_usage_stops_the_run(tmp_path):
    state = {"calls": 0}

    def reported():
        state["calls"] += 1
        if state["calls"] > 1:
            raise TransportFailure("down")
        return 0.0

    client, transport, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body), reported=reported, max_concurrency=1)
    report = run_arm(rows(150), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.status == "stopped" and report.stop_type == "ReconciliationUnavailable"
    assert len(transport.calls) == 100


def test_key_usage_reads_data_usage_strictly():
    transport = FakeTransport(lambda body, n: ok(fixture("key_response.json")))
    assert key_usage(transport) == 0.5
    for bad in ({"data": {}}, {"data": {"usage": "1"}}, {}, {"data": {"usage": True}}):
        with pytest.raises(TransportFailure):
            key_usage(FakeTransport(lambda body, n, b=bad: ok(b)))
    with pytest.raises(KeyOrCreditsStop):
        key_usage(FakeTransport(lambda body, n: status(401)))


# ------------------------------------------------------------------ the small LLM


def test_llm_sends_max_tokens_16_and_reasoning_off(tmp_path):
    client, transport, _, _ = make_client(tmp_path, lambda body, n: llm_ok(body))
    decision = llm_pick("r0", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
    assert decision.failed is False
    body = transport.calls[0]
    assert body["max_tokens"] == 16
    assert body["reasoning"] == {"effort": "none"}
    assert body["logprobs"] is True
    assert body["stream"] is False
    schema = body["response_format"]["json_schema"]["schema"]
    assert body["response_format"]["type"] == "json_schema"
    assert schema["properties"]["id"] == {"type": "integer", "enum": [1, 2, 3]}
    assert schema["required"] == ["id"] and schema["additionalProperties"] is False

    # The documented shape agrees with what we build.
    documented = fixture("chat_logprobs_request.json")
    assert set(documented) == set(body)
    assert documented["provider"].keys() == body["provider"].keys()


def test_llm_refuses_batch_variant(tmp_path):
    client, transport, _, _ = make_client(tmp_path, lambda body, n: llm_ok(body))
    with pytest.raises(ValueError, match="batch"):
        llm_pick("r0", "text", OPTIONS, "openai/gpt-oss-20b:batch", "Darkbloom", client=client)
    assert transport.calls == []


def test_llm_mean_output_above_16_tokens_stops_the_run(tmp_path):
    # Tokens at exactly 16 are fine. Then 40 per call pushes the mean over 16.
    def handler(body, n):
        return llm_ok(body, tokens=16 if n < 3 else 40)

    client, transport, _, _ = make_client(tmp_path, handler, max_concurrency=1)
    path = results_path(tmp_path)
    report = run_arm(rows(30), llm_decider(client), client=client, results_path=path)
    assert report.status == "stopped" and report.stop_type == "HiddenReasoning"
    assert len(transport.calls) == 4
    assert len(report.decisions) == 4  # the paid answer is kept
    assert len(report.not_run) == 26

    # At 16 on average the run goes on.
    client, transport, _, _ = make_client(tmp_path, lambda body, n: llm_ok(body, tokens=16), max_concurrency=1)
    for i in range(30):
        llm_pick(f"r{i}", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
    assert client.stop is None

    # The line is exact: 17 is over.
    client, _, _, _ = make_client(tmp_path, lambda body, n: llm_ok(body, tokens=17), max_concurrency=1)
    llm_pick("r0", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
    assert isinstance(client.stop, HiddenReasoning)


def test_llm_any_reported_reasoning_token_stops_the_run(tmp_path):
    client, transport, _, _ = make_client(tmp_path, lambda body, n: llm_ok(body, tokens=5, reasoning=2), max_concurrency=1)
    llm_pick("r0", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
    assert isinstance(client.stop, HiddenReasoning)
    with pytest.raises(HiddenReasoning):
        llm_pick("r1", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
    assert len(transport.calls) == 1


def test_hidden_tokens_on_a_truncated_answer_still_count(tmp_path):
    """Truncated output fails to parse, but its billed tokens must still reach the guard."""
    def handler(body, n):
        reply = json.loads(llm_ok(body, tokens=200).body)
        reply["choices"][0]["message"]["content"] = '{"id'
        reply["choices"][0]["finish_reason"] = "length"
        return ok(reply)

    client, transport, _, _ = make_client(tmp_path, handler, max_concurrency=1)
    with pytest.raises(HiddenReasoning):
        llm_pick("r0", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
    assert len(transport.calls) == 1


def test_top_prob_from_exp_sum_of_token_logprobs_along_chosen_id(tmp_path):
    # The chosen id spans two tokens, "1" and "2". Structural tokens must not count.
    options = {f"label_{i}": None for i in range(20)}

    def handler(body, n):
        reply = json.loads(llm_ok(body, id_value=12).body)
        reply["choices"][0]["message"]["content"] = '{"id":12}'
        reply["choices"][0]["logprobs"]["content"] = [
            {"token": '{"id', "logprob": -0.5, "bytes": None, "top_logprobs": []},
            {"token": '":', "logprob": -0.5, "bytes": None, "top_logprobs": []},
            {"token": "1", "logprob": -0.1, "bytes": None, "top_logprobs": []},
            {"token": "2", "logprob": -0.2, "bytes": None, "top_logprobs": []},
            {"token": "}", "logprob": -0.5, "bytes": None, "top_logprobs": []},
        ]
        return ok(reply)

    client, transport, _, _ = make_client(tmp_path, handler)
    decision = llm_pick("r0", "text", options, "openai/gpt-oss-20b", "Darkbloom", client=client)
    assert decision.top_prob == pytest.approx(math.exp(-0.1 - 0.2))
    assert decision.probs == {decision.pred: decision.top_prob}

    # A token straddling the value counts whole, because it is on the path to the id.
    content = '{"id":7}'
    tokens = [('{"id":7', -0.3), ("}", -9.0)]
    prob = llm_client.top_prob_from_logprobs(
        content,
        {"content": [{"token": t, "logprob": lp, "bytes": None, "top_logprobs": []} for t, lp in tokens]},
        "r0",
    )
    assert prob == pytest.approx(math.exp(-0.3))

    # The documented fixture: id 2, logprob -0.25.
    data = fixture("chat_logprobs_response.json")
    choice = data["choices"][0]
    assert llm_client.top_prob_from_logprobs(choice["message"]["content"], choice["logprobs"], "r0") == pytest.approx(math.exp(-0.25))


@pytest.mark.parametrize(
    "breakage",
    [
        lambda c: c.pop("logprobs"),
        lambda c: c.update(logprobs=None),
        lambda c: c["logprobs"].update(content=None),
        lambda c: c["logprobs"].update(content=[]),
        lambda c: c["logprobs"]["content"][3].update(logprob=None),
        lambda c: c["logprobs"]["content"][3].pop("logprob"),
        lambda c: c["logprobs"]["content"][0].update(logprob=None),  # a structural token too: "a token has no logprob"
        lambda c: c["logprobs"]["content"][3].update(logprob="nan"),
        lambda c: c["logprobs"]["content"][3].update(logprob=float("nan")),
        lambda c: c["logprobs"]["content"][3].update(logprob=0.5),  # above 0 is not a log probability
        lambda c: c["logprobs"]["content"].pop(3),  # tokens no longer spell the answer
    ],
)
def test_missing_token_logprob_marks_arm_not_run(tmp_path, breakage):
    def handler(body, n):
        if n == 0:
            return llm_ok(body)
        reply = json.loads(llm_ok(body).body)
        breakage(reply["choices"][0])
        return ok(reply)

    client, transport, ledger, _ = make_client(tmp_path, handler, max_concurrency=1)
    report = run_arm(rows(30), llm_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.status == "arm_not_run" and report.stop_type == "ArmNotRun"
    assert len(transport.calls) == 2  # stopped at the first bad row, no retries: this is not a per-row fault
    assert ledger.spent() == pytest.approx(2 * 0.00003)  # the bad call was still billed
    assert len(report.not_run) == 29


def test_selection_can_count_a_missing_logprob_row_by_row(tmp_path):
    """Selection allows 1 miss in 50, so the signal must be catchable without ending the client."""
    def handler(body, n):
        reply = json.loads(llm_ok(body).body)
        if n == 0:
            reply["choices"][0]["logprobs"] = None
        return ok(reply)

    client, _, _, _ = make_client(tmp_path, handler, max_concurrency=1)
    with pytest.raises(ArmNotRun):
        llm_pick("r0", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
    assert llm_pick("r1", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client).failed is False


def test_provider_pinned_with_no_fallbacks_and_require_parameters(tmp_path):
    expected = {"only": ["Darkbloom"], "allow_fallbacks": False, "require_parameters": True}
    client, transport, _, _ = make_client(tmp_path, lambda body, n: llm_ok(body))
    llm_pick("r0", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
    assert transport.calls[0]["provider"] == expected

    client, transport, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body))
    jev_decide("r0", "text", OPTIONS, client=client)
    assert transport.calls[0]["provider"] == {"only": ["TypeSafe"], "allow_fallbacks": False, "require_parameters": True}
    assert transport.calls[0]["model"] == "typesafe/jev-1.14"

    # A reply from another provider stops the run rather than mixing providers.
    def other(body, n):
        reply = json.loads(llm_ok(body).body)
        reply["provider"] = "SomeoneElse"
        return ok(reply)

    client, _, _, _ = make_client(tmp_path, other)
    with pytest.raises(ProviderChanged):
        llm_pick("r0", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)

    def other_jev(body, n):
        reply = json.loads(jev_ok(body).body)
        reply["provider"] = "SomeoneElse"
        return ok(reply)

    client, _, _, _ = make_client(tmp_path, other_jev)
    with pytest.raises(ProviderChanged):
        jev_decide("r0", "text", OPTIONS, client=client)

    # Provider name matching ignores case and punctuation; the docs use both names and slugs.
    def spaced(body, n):
        reply = json.loads(llm_ok(body).body)
        reply["provider"] = "darkbloom"
        return ok(reply)

    client, _, _, _ = make_client(tmp_path, spaced)
    assert llm_pick("r0", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client).failed is False


# ------------------------------------------------------------------ size limit


def test_request_over_30000_bytes_refuses_to_send(tmp_path):
    client, transport, ledger, _ = make_client(tmp_path, lambda body, n: jev_ok(body))
    with pytest.raises(RequestTooLarge):
        jev_decide("r0", "x" * 30_001, OPTIONS, client=client)
    with pytest.raises(RequestTooLarge):
        llm_pick("r0", "x" * 30_001, OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
    # Multi-byte text counts in bytes, not characters.
    with pytest.raises(RequestTooLarge):
        jev_decide("r0", "é" * 15_500, OPTIONS, client=client)
    assert transport.calls == [] and ledger.spent() == 0.0

    # The measure is the exact bytes sent: a body of exactly 30,000 goes out, 30,001 does not.
    base = len(openrouter.encode_body(jev_client.build_request("", OPTIONS, "r0", 1105, None, jev_client.DEFAULT_INSTRUCTIONS)))
    text = "x" * (30_000 - base)
    assert len(openrouter.encode_body(jev_client.build_request(text, OPTIONS, "r0", 1105, None, jev_client.DEFAULT_INSTRUCTIONS))) == 30_000
    jev_decide("r0", text, OPTIONS, client=client)
    assert len(transport.raw[0]) == 30_000
    with pytest.raises(RequestTooLarge):
        jev_decide("r1", text + "x", OPTIONS, client=client)
    assert len(transport.calls) == 1


def test_oversize_row_is_a_failed_row_not_a_crash(tmp_path):
    client, transport, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body))
    data = rows(3)
    data[1]["text"] = "x" * 40_000
    report = run_arm(data, jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.status == "complete"
    assert report.decisions["r1"].failed is True and report.decisions["r1"].reason == "request_too_large"
    assert len(transport.calls) == 2


# ------------------------------------------------------------------ bad answers


@pytest.mark.parametrize(
    "bad_reply",
    [
        lambda body: jev_ok(body, choice="not_an_option"),
        lambda body: FakeResponse(200, {}, b""),
        lambda body: FakeResponse(200, {}, b"<html>not json</html>"),
        lambda body: FakeResponse(200, {}, b"[]"),
        lambda body: ok({"model": BUILD, "usage": {"cost": 0.0001}}),
        lambda body: ok({**fixture("decisions_refusal_response.json")}),
    ],
)
def test_label_not_in_options_retries_then_marks_failed(tmp_path, bad_reply):
    client, transport, ledger, _ = make_client(tmp_path, lambda body, n: bad_reply(body), max_concurrency=1, reserve_usd=0.001)
    report = run_arm(rows(3), jev_decider(client), client=client, results_path=results_path(tmp_path))

    assert report.status == "complete"  # nothing crashes the run
    assert len(transport.calls) == 12  # 3 rows x (1 try + 3 retries)
    for decision in report.decisions.values():
        assert decision.failed is True and decision.pred is None and decision.top_prob is None
        assert decision.attempts == 4
    assert ledger.spent() > 0


def test_llm_id_not_in_options_retries_then_marks_failed(tmp_path):
    for bad in ('{"id":99}', '{"id":"1"}', '{"id":true}', '{"id":1.5}', '{"id":1,"x":2}', "not json", ""):
        def handler(body, n, bad=bad):
            reply = json.loads(llm_ok(body).body)
            reply["choices"][0]["message"]["content"] = bad
            return ok(reply)

        client, transport, _, _ = make_client(tmp_path, handler, max_concurrency=1)
        decision = llm_pick("r0", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
        assert decision.failed is True and decision.attempts == 4, bad
        assert decision.reason.startswith("malformed")


@pytest.mark.parametrize(
    "probabilities",
    [
        None,
        {},
        {"card_arrival": 0.9},  # options left out: Stage 0 must confirm whether Jev ever does this
        {"card_arrival": 0.9, "lost_card": 0.1, "top_up_failed": 0, "extra": 0},
        {"card_arrival": 1.1, "lost_card": 0, "top_up_failed": 0},
        {"card_arrival": 0.9, "lost_card": -0.1, "top_up_failed": 0},
        {"card_arrival": "0.9", "lost_card": 0, "top_up_failed": 0},
        {"card_arrival": True, "lost_card": 0, "top_up_failed": 0},
        {"card_arrival": 0.1, "lost_card": 0.9, "top_up_failed": 0},  # the choice is not the top option
        {"card_arrival": 0.48, "lost_card": 0.5, "top_up_failed": 0.02},  # 0.02 under the top is more than rounding
    ],
)
def test_jev_parser_is_strict_about_probabilities(tmp_path, probabilities):
    def handler(body, n):
        reply = json.loads(jev_ok(body, choice="card_arrival").body)
        if probabilities is None:
            del reply["answers"]["label"]["probabilities"]
        else:
            reply["answers"]["label"]["probabilities"] = probabilities
        return ok(reply)

    client, transport, _, _ = make_client(tmp_path, handler, max_concurrency=1)
    decision = jev_decide("r0", "text", OPTIONS, client=client)
    assert decision.failed is True and decision.reason.startswith("malformed") and len(transport.calls) == 4


def test_choice_may_trail_the_top_option_only_by_rounding(tmp_path):
    def handler(body, n):
        reply = json.loads(jev_ok(body, choice="card_arrival").body)
        reply["answers"]["label"]["probabilities"] = {"card_arrival": 0.5, "lost_card": 0.51, "top_up_failed": 0}
        return ok(reply)

    client, _, _, _ = make_client(tmp_path, handler)
    decision = jev_decide("r0", "text", OPTIONS, client=client)
    assert decision.failed is False and decision.pred == "card_arrival" and decision.top_prob == 0.5


def test_jev_uses_the_probability_of_its_choice_not_confidence(tmp_path):
    def handler(body, n):
        data = fixture("decisions_choice_response.json")
        data["answers"]["label"]["probabilities"] = {"card_arrival": 0.84, "lost_card": 0.16, "top_up_failed": 0}
        return ok(data)

    client, _, _, _ = make_client(tmp_path, handler)
    decision = jev_decide("r0", "text", OPTIONS, client=client)
    assert decision == Decision(
        pred="card_arrival",
        top_prob=0.84,
        probs={"card_arrival": 0.84, "lost_card": 0.16, "top_up_failed": 0},
        model=BUILD,
        cost=0.0001,
        failed=False,
        attempts=1,
        cached=False,
        reason=None,
        n_examples=0,
        input_tokens=2376,
        output_tokens=70,
    )


def test_key_rejected_or_credits_out_stops_cleanly_and_keeps_rows(tmp_path):
    for code, body in ((401, b"{}"), (402, (FIXTURES / "error_402.json").read_bytes()), (403, b"{}")):
        run_dir = tmp_path / str(code)
        run_dir.mkdir()

        def handler(request_body, n, code=code, body=body):
            return jev_ok(request_body) if n < 5 else status(code, body=body)

        client, transport, ledger, sleeps = make_client(run_dir, handler, max_concurrency=1)
        path = results_path(run_dir)
        report = run_arm(rows(30), jev_decider(client), client=client, results_path=path)

        assert report.status == "stopped" and report.stop_type == "KeyOrCreditsStop"
        assert str(code) in report.reason
        assert len(transport.calls) == 6  # the rejected call was sent once, with no retry
        assert sleeps == []
        assert sorted(report.decisions) == sorted(f"r{i}" for i in range(5))
        assert len(report.not_run) == 25
        assert ledger.spent() == pytest.approx(5 * 0.0001)  # a rejected call is not charged

        lines = read_results(path)
        assert sum(line["status"] == "done" for line in lines) == 5
        assert sum(line["status"] == "not_run" for line in lines) == 25

        # After fixing the key or topping up, a rerun takes only what is left.
        client, transport, _, _ = make_client(run_dir, lambda b, n: jev_ok(b), ledger=ledger, max_concurrency=1)
        report = run_arm(rows(30), jev_decider(client), client=client, results_path=path)
        assert report.status == "complete" and len(transport.calls) == 25 and report.not_run == []


def test_transient_in_flight_402_also_stops_and_says_so(tmp_path):
    """The docs call this 402 transient; the build's rule is to stop on any 402. Resuming costs nothing."""
    body = (FIXTURES / "error_402_in_flight.json").read_bytes()
    client, transport, _, sleeps = make_client(tmp_path, lambda b, n: status(402, {"Retry-After": "3"}, body))
    with pytest.raises(KeyOrCreditsStop, match="in-flight"):
        jev_decide("r0", "text", OPTIONS, client=client)
    assert len(transport.calls) == 1 and sleeps == []


def test_200_body_carrying_an_auth_error_also_stops(tmp_path):
    client, transport, _, _ = make_client(tmp_path, lambda b, n: ok(fixture("error_402.json")))
    with pytest.raises(KeyOrCreditsStop):
        jev_decide("r0", "text", OPTIONS, client=client)
    assert len(transport.calls) == 1


def test_stop_is_final_for_the_client(tmp_path):
    client, transport, _, _ = make_client(tmp_path, lambda b, n: status(401))
    with pytest.raises(KeyOrCreditsStop):
        jev_decide("r0", "text", OPTIONS, client=client)
    with pytest.raises(KeyOrCreditsStop):
        jev_decide("r1", "text", OPTIONS, client=client)
    assert len(transport.calls) == 1


def test_a_bug_in_decide_stops_cleanly_and_keeps_rows(tmp_path):
    client, _, _, _ = make_client(tmp_path, lambda b, n: jev_ok(b), max_concurrency=1)
    good = jev_decider(client)

    def decide(row):
        if row["row_id"] == "r2":
            raise KeyError("a bug, with a value that must not be printed")
        return good(row)

    report = run_arm(rows(5), decide, client=client, results_path=results_path(tmp_path))
    assert report.status == "stopped" and report.stop_type == "UnexpectedStop"
    assert "KeyError" in report.reason and "must not be printed" not in report.reason
    assert "r0" in report.decisions and "r2" not in report.decisions


# ------------------------------------------------------------------ resume and cache


def test_resume_does_not_pay_twice(tmp_path):
    path = results_path(tmp_path)
    ledger_path = tmp_path / "ledger.jsonl"

    # First run dies from a credits stop after 8 rows; one of them is a final failure.
    def first_handler(body, n):
        if n < 7:
            return jev_ok(body, cost=0.001)
        if n < 11:
            return jev_ok(body, choice="not_an_option", cost=0.001)  # answered but invalid: final, never re-bought
        return status(402)

    first_ledger = Ledger(ledger_path)
    client, first, _, _ = make_client(tmp_path, first_handler, ledger=first_ledger, max_concurrency=1)
    report = run_arm(rows(30), jev_decider(client), client=client, results_path=path)
    assert report.stop_type == "KeyOrCreditsStop"
    finished = {k: v for k, v in report.decisions.items()}
    assert len(finished) == 8
    spent_after_first = first_ledger.spent()
    saved_lines = path.read_bytes()

    # Restart: a new process, new ledger object, same files.
    second_ledger = Ledger(ledger_path)
    assert second_ledger.spent() == pytest.approx(spent_after_first)
    client, second, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body, cost=0.001), ledger=second_ledger, max_concurrency=1)
    report = run_arm(rows(30), jev_decider(client), client=client, results_path=path)

    assert report.status == "complete" and report.not_run == []
    assert len(second.calls) == 22  # only the rows never finished
    assert second_ledger.spent() == pytest.approx(spent_after_first + 22 * 0.001)
    # Finished rows are byte-for-byte unchanged and returned as they were.
    assert path.read_bytes().startswith(saved_lines)
    for row_id, decision in finished.items():
        assert report.decisions[row_id] == decision

    # A third run with everything done sends nothing.
    client, third, _, _ = make_client(tmp_path, lambda body, n: status(500), ledger=second_ledger)
    report = run_arm(rows(30), jev_decider(client), client=client, results_path=path)
    assert report.status == "complete" and third.calls == [] and len(report.decisions) == 30


def test_resume_survives_a_torn_last_line(tmp_path):
    path = results_path(tmp_path)
    client, _, ledger, _ = make_client(tmp_path, lambda b, n: jev_ok(b), max_concurrency=1)
    run_arm(rows(3), jev_decider(client), client=client, results_path=path)
    with open(path, "ab") as handle:
        handle.write(b'{"row_id":"r3","status":"do')

    client, transport, _, _ = make_client(tmp_path, lambda b, n: jev_ok(b), ledger=ledger, max_concurrency=1)
    report = run_arm(rows(5), jev_decider(client), client=client, results_path=path)
    assert report.status == "complete" and len(transport.calls) == 2 and len(report.decisions) == 5

    client, transport, _, _ = make_client(tmp_path, lambda b, n: jev_ok(b), ledger=ledger, max_concurrency=1)
    assert run_arm(rows(5), jev_decider(client), client=client, results_path=path).status == "complete"
    assert transport.calls == []


def test_outputs_must_live_under_runs_private(tmp_path):
    client, _, _, _ = make_client(tmp_path, lambda b, n: jev_ok(b))
    # On a Mac tmp_path itself starts /private/var, which must not count.
    assert "private" in Path("/private/var/x").parts
    for bad in (tmp_path / "public" / "out.jsonl", Path("/private/var/out.jsonl"), tmp_path / "private" / "out.jsonl"):
        with pytest.raises(ValueError, match="runs/private"):
            run_arm(rows(1), jev_decider(client), client=client, results_path=bad)
        with pytest.raises(ValueError, match="runs/private"):
            JevCache(bad)


def test_duplicate_row_ids_are_refused_before_paying(tmp_path):
    client, transport, _, _ = make_client(tmp_path, lambda b, n: jev_ok(b))
    with pytest.raises(ValueError, match="unique"):
        run_arm(rows(2) + rows(1), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert transport.calls == []


def test_ctrl_c_stops_queued_rows_from_sending(tmp_path):
    client, transport, _, _ = make_client(tmp_path, lambda b, n: jev_ok(b), max_concurrency=1)
    good = jev_decider(client)

    def decide(row):
        if row["row_id"] == "r3":
            raise KeyboardInterrupt
        return good(row)

    path = results_path(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        run_arm(rows(40), decide, client=client, results_path=path)
    assert len(transport.calls) == 3
    assert [line["row_id"] for line in read_results(path)] == ["r0", "r1", "r2"]


@pytest.mark.parametrize(
    "mangle",
    [
        lambda d: d["answers"]["label"].update(choice=["card_arrival"]),  # unhashable
        lambda d: d["answers"]["label"].update(choice={"a": 1}),
        lambda d: d["answers"]["label"].update(choice=None),
        lambda d: d.update(answers=[]),
        lambda d: d.update(model=7),
        lambda d: d["usage"].update(input_tokens="2376"),
        lambda d: d["usage"].pop("output_tokens"),
        lambda d: d["usage"].update(cost=-1),
        lambda d: d["usage"].update(cost=True),
        lambda d: d.update(error={"code": ["not", "a", "status"]}, usage=[]),
    ],
)
def test_odd_payloads_retry_and_fail_without_crashing(tmp_path, mangle):
    def handler(body, n):
        reply = json.loads(jev_ok(body).body)
        mangle(reply)
        return ok(reply)

    client, transport, _, _ = make_client(tmp_path, handler, max_concurrency=1)
    report = run_arm(rows(2), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.status == "complete" and len(transport.calls) == 8
    assert all(d.failed for d in report.decisions.values())


def test_cache_is_empty_at_start_and_keyed_on_model_build(tmp_path):
    cache_path = tmp_path / "runs" / "private" / "jev_cache.jsonl"
    cache = JevCache(cache_path)
    assert len(cache) == 0
    assert jev_client.DEFAULT_CACHE_PATH.parts[-3:] == ("runs", "private", "jev_cache.jsonl")
    assert "runs/" in (Path(__file__).parents[1] / ".gitignore").read_text().splitlines()

    client, transport, ledger, _ = make_client(tmp_path, lambda b, n: jev_ok(b, cost=0.001))
    first = jev_decide("r0", "text", OPTIONS, client=client, cache=cache)
    assert first.cached is False and first.cost == pytest.approx(0.001) and len(cache) == 1

    # Same build, text and order: served from cache, free, unchanged.
    again = jev_decide("r0", "text", OPTIONS, client=client, cache=cache)
    assert again.cached is True and again.cost == 0.0 and again.pred == first.pred and again.probs == first.probs
    assert len(transport.calls) == 1 and ledger.spent() == pytest.approx(0.001)

    # Survives a restart.
    assert len(JevCache(cache_path)) == 1
    assert jev_decide("r0", "text", OPTIONS, client=client, cache=JevCache(cache_path)).cached is True

    # A different text pays.
    assert jev_decide("r0", "other text", OPTIONS, client=client, cache=cache).cached is False
    # A different option order pays: the row id drives the order, and so does the seed.
    base_order = option_order("r0", OPTIONS, 1105)
    other_row = next(f"r{i}" for i in range(1, 99) if option_order(f"r{i}", OPTIONS, 1105) != base_order)
    other_seed = next(i for i in range(1, 99) if option_order("r0", OPTIONS, i) != base_order)
    assert jev_decide(other_row, "text", OPTIONS, client=client, cache=cache).cached is False
    assert jev_decide("r0", "text", OPTIONS, order_seed=other_seed, client=client, cache=cache).cached is False
    # Different descriptions or examples pay.
    changed = {**OPTIONS, "lost_card": "Something else."}
    assert jev_decide("r0", "text", changed, client=client, cache=cache).cached is False
    examples = [Example("I lost it", "lost_card")]
    assert jev_decide("r0", "text", OPTIONS, examples=examples, client=client, cache=cache).cached is False

    # A cached answer from another build is never served, even for the same text and order.
    calls = len(transport.calls)
    other_build = "jev-1.14-20270101"
    client2, transport2, _, _ = make_client(tmp_path, lambda b, n: jev_ok(b, model="typesafe/" + other_build), ledger=ledger)
    other = jev_decide("r0", "text", OPTIONS, client=client2, cache=cache, expected_build=other_build)
    assert other.cached is False and len(transport2.calls) == 1
    assert len(transport.calls) == calls


def test_failed_rows_are_not_cached(tmp_path):
    cache = JevCache(tmp_path / "runs" / "private" / "cache.jsonl")
    client, _, _, _ = make_client(tmp_path, lambda b, n: status(529))
    assert jev_decide("r0", "text", OPTIONS, client=client, cache=cache).failed is True
    assert len(cache) == 0


def test_model_change_mid_run_pauses(tmp_path):
    other = "typesafe/jev-1.14-20270101"

    def handler(body, n):
        return jev_ok(body, model=BUILD if n < 25 else other)

    cache = JevCache(results_path(tmp_path).parent / "cache.jsonl")
    client, transport, _, _ = make_client(tmp_path, handler, max_concurrency=1)
    report = run_arm(rows(40), jev_decider(client, cache=cache), client=client, results_path=results_path(tmp_path))
    assert report.status == "stopped" and report.stop_type == "ModelChanged"
    assert "20270101" in report.reason and "20261014" in report.reason
    assert len(transport.calls) == 26
    assert len(report.decisions) == 25  # the answer from the wrong build is not kept
    assert "r25" in report.not_run
    assert len(cache) == 25

    # The small LLM pins the first model it sees, for the same reason.
    def llm_handler(body, n):
        return llm_ok(body, model="openai/gpt-oss-20b" if n < 3 else "openai/gpt-oss-20b-v2")

    client, _, _, _ = make_client(tmp_path, llm_handler, max_concurrency=1)
    for i in range(3):
        llm_pick(f"r{i}", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
    with pytest.raises(ModelChanged):
        llm_pick("r3", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)


# ------------------------------------------------------------------ option order


def test_option_order_is_randomised_per_row_with_seed_1105_and_mapped_back(tmp_path):
    labels = [f"label_{i:02d}" for i in range(77)]
    options = {label: f"Description of {label}." for label in labels}

    # Deterministic, a permutation, different per row and per seed, blind to input order.
    first = option_order("row-1", labels, 1105)
    assert sorted(first) == labels
    assert first == option_order("row-1", labels, 1105)
    assert first == option_order("row-1", list(reversed(labels)), 1105)
    assert first != labels
    assert first != option_order("row-2", labels, 1105)
    assert first != option_order("row-1", labels, 1106)
    # Pinned so a change in the shuffle is seen; it would change every paid request.
    assert option_order("row-1", ["a", "b", "c", "d"], 1105) == PINNED_ORDER
    # Each label reaches each position about evenly over many rows.
    first_slot = [option_order(f"row-{i}", labels[:4], 1105)[0] for i in range(4000)]
    assert all(800 < first_slot.count(label) < 1200 for label in labels[:4])

    # Jev: the bytes on the wire hold that order; default seed is 1105; answers come back keyed by label.
    client, transport, _, _ = make_client(tmp_path, lambda b, n: jev_ok(b, choice="label_33"))
    decision = jev_decide("row-1", "text", options, client=client)
    assert list(transport.calls[0]["questions"]["label"]["criteria"]) == first
    raw = transport.raw[0].decode()
    positions = [raw.index(f'"{label}":') for label in first]
    assert positions == sorted(positions)
    assert transport.calls[0]["questions"]["label"]["criteria"]["label_33"] == "Description of label_33."
    assert decision.pred == "label_33"
    assert list(decision.probs) == labels  # back in the caller's order

    # Retries of a row send the same order.
    replies = [status(529), None]
    client, transport, _, _ = make_client(tmp_path, lambda b, n: replies[n] or jev_ok(b))
    jev_decide("row-1", "text", options, client=client)
    assert transport.calls[0] == transport.calls[1]

    # Small LLM: an id is a position in that row's shuffled list, and it maps back to the right label.
    target = "label_50"
    expected_id = first.index(target) + 1
    client, transport, _, _ = make_client(tmp_path, lambda b, n: llm_ok(b, id_value=expected_id))
    decision = llm_pick("row-1", "text", options, "openai/gpt-oss-20b", "Darkbloom", client=client)
    assert decision.pred == target
    user = transport.calls[0]["messages"][-1]["content"]
    assert f"{expected_id}. {target}: Description of {target}." in user
    assert transport.calls[0]["response_format"]["json_schema"]["schema"]["properties"]["id"]["enum"] == list(range(1, 78))
    listed = [line.split(". ", 1)[1].split(":")[0] for line in user.split("Options:\n")[1].splitlines()]
    assert listed == first

    # Another row, same id number, is a different label.
    second = option_order("row-2", labels, 1105)
    client, _, _, _ = make_client(tmp_path, lambda b, n: llm_ok(b, id_value=expected_id))
    assert llm_pick("row-2", "text", options, "openai/gpt-oss-20b", "Darkbloom", client=client).pred == second[expected_id - 1]


# Computed by hand from the rule: sort by SHA-256("seed:row_id:label").
PINNED_ORDER = ['c', 'd', 'a', 'b']


def test_bare_names_send_null_descriptions(tmp_path):
    client, transport, _, _ = make_client(tmp_path, lambda b, n: jev_ok(b))
    jev_decide("r0", "text", ["a", "b", "c"], client=client)
    assert transport.calls[0]["questions"]["label"]["criteria"] == {k: None for k in option_order("r0", ["a", "b", "c"], 1105)}
    client, transport, _, _ = make_client(tmp_path, lambda b, n: llm_ok(b))
    llm_pick("r0", "text", ["a", "b"], "openai/gpt-oss-20b", "Darkbloom", client=client)
    user = transport.calls[0]["messages"][-1]["content"]
    assert ":" not in user.split("Options:\n")[1]
    for bad in (["a", "a"], [], {}):
        with pytest.raises(ValueError):
            jev_decide("r0", "text", bad, client=client)


def test_examples_travel_in_state_and_are_counted(tmp_path):
    client, transport, _, _ = make_client(tmp_path, lambda b, n: jev_ok(b))
    examples = [Example("I lost my card", "lost_card"), Example("where is my card", "card_arrival")]
    decision = jev_decide("r0", "text", OPTIONS, examples=examples, client=client)
    assert transport.calls[0]["state"] == {
        "text": "text",
        "labelled_examples": [{"text": "I lost my card", "label": "lost_card"}, {"text": "where is my card", "label": "card_arrival"}],
    }
    assert decision.n_examples == 2
    # No examples: plain text state, as in the documented shape.
    client, transport, _, _ = make_client(tmp_path, lambda b, n: jev_ok(b))
    jev_decide("r0", "text", OPTIONS, examples=[], client=client)
    assert transport.calls[0]["state"] == "text"


def test_fixtures_agree_with_the_builders_and_parsers(tmp_path):
    request = fixture("decisions_choice_request.json")
    built = jev_client.build_request("t", OPTIONS, "r0", 1105, None, jev_client.DEFAULT_INSTRUCTIONS)
    assert set(built) == set(request)
    assert set(built["questions"]["label"]) == set(request["questions"]["label"])
    assert built["provider"] == request["provider"]

    sources = fixture("sources.json")
    on_disk = {p.name for p in FIXTURES.glob("*.json")} - {"sources.json"}
    assert set(sources["fixtures"]) == on_disk
    assert all(entry["url"].startswith("https://") for entry in sources["fixtures"].values())

    # The response fixtures go through the real parsers, unmodified.
    options = request["questions"]["label"]["criteria"]
    client, _, _, _ = make_client(tmp_path, lambda b, n: ok(fixture("decisions_choice_response.json")))
    decision = jev_decide("r0", "my card has not arrived", options, client=client)
    assert decision.failed is False and decision.pred == "card_arrival" and decision.top_prob == 0.84

    client, _, _, _ = make_client(tmp_path, lambda b, n: ok(fixture("chat_logprobs_response.json")))
    decision = llm_pick("r0", "t", {"a": None, "b": None}, "openai/gpt-oss-20b", "Darkbloom", client=client)
    assert decision.failed is False and decision.top_prob == pytest.approx(math.exp(-0.25))


# ------------------------------------------------------------------ retrieval


FILLER = [Example(f"unrelated filler sentence about topic{i} weather", "filler") for i in range(60)]


def test_retrieval_max_4_per_class_up_to_24_ignores_zero_scores_ties_by_hash_logs_count(caplog):
    caplog.set_level(logging.INFO, logger="retrieval")

    # Max 4 per class, and classes already at 4 are skipped rather than ending the search.
    bank = [Example(f"lost card version {i}", "lost") for i in range(6)]
    bank += [Example("lost card somewhere else entirely today", "other")] + FILLER
    got = retrieve("lost card", bank)
    assert sum(e.label == "lost" for e in got) == 4
    assert sum(e.label == "other" for e in got) == 1

    # Up to 24 in all, with 7 classes each able to give 4.
    many = [Example(f"lost card sample {c} {i}", f"class{c}") for c in range(7) for i in range(5)] + FILLER * 2
    got = retrieve("lost card", many)
    assert len(got) == 24
    assert max(sum(e.label == f"class{c}" for e in got) for c in range(7)) == 4

    # Zero scores are ignored: no shared term, nothing returned.
    assert retrieve("zebra quantum", FILLER) == []
    # Nothing here beyond the single match is returned, though the classes have room.
    got = retrieve("topic7", FILLER)
    assert [e.text for e in got] == ["unrelated filler sentence about topic7 weather"]

    # Score order comes first.
    ranked = [Example("card card card", "a"), Example("card and many other long words in this one", "b")] + FILLER
    assert [e.label for e in retrieve("card", ranked)] == ["a", "b"]

    # Ties by hash of the text: order does not depend on the bank's order, and matches SHA-256 order.
    tied = [Example(f"lost card {w}", f"c{i}") for i, w in enumerate(["alpha", "bravo", "charlie", "delta", "echo"])] + FILLER
    forward = retrieve("lost card", tied)
    backward = retrieve("lost card", list(reversed(tied)))
    assert forward == backward
    assert [e.text for e in forward] == sorted((e.text for e in forward), key=retrieval.tie_hash)
    assert len(forward) == 5

    # The count is logged per row, including a zero.
    caplog.clear()
    retrieve("lost card", bank, row_id="row-7")
    retrieve("zebra quantum", FILLER, row_id="row-8")
    assert "row=row-7 count=5" in caplog.text
    assert "row=row-8 count=0" in caplog.text

    # Degenerate input does not crash.
    assert retrieve("", bank) == []
    assert retrieve("?!", bank) == []
    assert retrieve("lost card", []) == []


def test_retrieval_uses_okapi_1_5_0_75_on_words_and_adjacent_pairs(monkeypatch):
    assert retrieval.tokenize("Lost my CARD!") == ["lost", "my", "card", "lost my", "my card"]
    assert retrieval.tokenize("card") == ["card"]
    assert retrieval.tokenize("") == []

    seen = {}
    real = retrieval.BM25Okapi

    def spy(corpus, **kwargs):
        seen.update(kwargs, corpus=corpus)
        return real(corpus, **kwargs)

    monkeypatch.setattr(retrieval, "BM25Okapi", spy)
    bank = [Example("lost my card", "a")] + FILLER
    retrieve("lost card", bank)
    assert seen["k1"] == 1.5 and seen["b"] == 0.75
    assert seen["corpus"][0] == ["lost", "my", "card", "lost my", "my card"]

    # An adjacent pair outranks the same words apart, at equal length.
    pair = [Example("lost card here now", "pair"), Example("lost here card now", "apart")] + FILLER
    assert [e.label for e in retrieve("lost card", pair)] == ["pair", "apart"]


# ------------------------------------------------------------------ fix round 1: the paid route


LONG_KEY = "sk-or-v1-" + "ab" * 32


@pytest.fixture
def paid(tmp_path, monkeypatch):
    """The real paid route over httpx.MockTransport, with read_key stubbed. Nothing leaves the process."""
    ledger_path = tmp_path / "home" / ".jev-benchmark" / "ledger.jsonl"
    monkeypatch.setenv(openrouter.TEST_LEDGER_ENV, str(ledger_path))
    monkeypatch.setattr(secrets_mac, "read_key", lambda: FAKE_KEY)
    # usage is what the key has been billed; lag is how much of it has not posted yet; billed is the
    # factor OpenRouter bills over the cost it reports; reply, if set, replaces the answer.
    state = {"usage": 0.0, "posts": [], "gets": 0, "headers": [], "path": ledger_path, "cost": 0.0001, "lag": 0.0, "billed": 1.0, "reply": None}

    def handler(request):
        state["headers"].append(request.headers.get("authorization"))
        if request.method == "GET":
            state["gets"] += 1
            return httpx.Response(200, json={"data": {"usage": max(0.0, state["usage"] - state["lag"])}})
        body = json.loads(request.content)
        state["posts"].append(body)
        if state["reply"] is not None:
            return state["reply"](body)
        state["usage"] += state["cost"] * state["billed"]
        return httpx.Response(200, content=jev_ok(body, cost=state["cost"]).body)

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(**{**kw, "transport": httpx.MockTransport(handler)}))
    return state


def no_sleep(seconds):
    pass


def test_paid_ledger_path_is_fixed_and_outside_any_checkout(tmp_path, monkeypatch):
    assert openrouter.PAID_LEDGER_PATH == Path.home() / ".jev-benchmark" / "ledger.jsonl"
    repo = Path(__file__).resolve().parents[1]
    assert repo not in openrouter.PAID_LEDGER_PATH.parents

    # Under pytest the path must be overridden, so a test can never touch the real ledger.
    monkeypatch.delenv(openrouter.TEST_LEDGER_ENV, raising=False)
    with pytest.raises(RuntimeError, match=openrouter.TEST_LEDGER_ENV):
        openrouter.paid_ledger_path()
    other = tmp_path / "x.jsonl"
    monkeypatch.setenv(openrouter.TEST_LEDGER_ENV, str(other))
    assert openrouter.paid_ledger_path() == other

    # Outside pytest the override is ignored. subprocess is blocked in this file and no key is read here.
    monkeypatch.setattr(secrets_mac, "_under_pytest", lambda: False)
    assert openrouter.paid_ledger_path() == openrouter.PAID_LEDGER_PATH


def test_fake_run_cannot_write_to_the_real_ledger():
    with pytest.raises(ValueError, match="real ledger"):
        build_client(None, ledger_path=openrouter.PAID_LEDGER_PATH, fake_transport=FakeTransport(None), reported_usage=lambda: 0.0)


def test_paid_wiring_gives_stage_cap_fixed_ledger_and_usage_reader(paid):
    client = build_client("stage0", sleep=no_sleep)
    assert isinstance(client.transport, RealTransport) and client.transport.stage == "stage0"
    assert client.ledger.path == paid["path"]
    assert client.ledger.stage == "stage0"
    assert client.ledger.stage_cap_usd == 0.50
    assert client.ledger.cap_usd == 7.15
    assert paid["gets"] == 1  # the usage reader really is key_usage over the real transport
    paid["usage"] = 0.123
    assert client.reported_usage() == 0.123
    paid["usage"] = 0.0

    report = run_arm(rows(3), jev_decider(client), client=client, results_path=paid["path"].parent / "runs" / "private" / "a.jsonl", expected_per_row=0.0001)
    assert report.status == "complete" and len(paid["posts"]) == 3
    assert set(paid["headers"]) == {f"Bearer {FAKE_KEY}"}
    assert client.ledger.spent() == pytest.approx(0.0003)

    # Stage 1 has no stage limit, the same file, and counts stage 0's spend.
    later = build_client("stage1", sleep=no_sleep)
    assert later.ledger.stage == "stage1" and later.ledger.stage_cap_usd is None and later.ledger.path == paid["path"]
    assert later.ledger.spent() == pytest.approx(0.0003)


def test_paid_stage_cap_really_stops_sending(paid):
    paid["cost"] = 0.2
    client = build_client("stage0", sleep=no_sleep)
    report = run_arm(rows(5), jev_decider(client), client=client, results_path=paid["path"].parent / "runs" / "private" / "a.jsonl", expected_per_row=0.0001)
    assert report.stop_type == "BudgetExceeded"
    assert len(paid["posts"]) == 1  # one measured row at 0.20, and 4 more would not fit in 0.50
    assert client.ledger.spent() == pytest.approx(0.2)


def test_client_refuses_real_transport_on_a_different_stage(tmp_path):
    mock = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    transport = RealTransport("stage1", key_reader=lambda: FAKE_KEY, httpx_transport=mock)
    for ledger in (Ledger(tmp_path / "a.jsonl", stage="stage0"), Ledger(tmp_path / "b.jsonl")):
        with pytest.raises(ValueError, match="stage"):
            Client(transport, ledger, lambda: 0.0)
    Client(transport, Ledger(tmp_path / "c.jsonl", stage="stage1"), lambda: 0.0)


def test_empty_ledger_records_an_explicit_baseline(paid):
    build_client("stage0", sleep=no_sleep)
    entries = [json.loads(line) for line in paid["path"].read_text().splitlines()]
    assert [e["kind"] for e in entries] == ["baseline"]
    assert entries[0]["usage"] == 0.0


def test_fresh_ledger_with_key_usage_above_baseline_refuses_to_send(paid):
    """Critical 1: 0.49 was spent, then the ledger file changed. The key's usage gives it away."""
    paid["usage"] = 0.49
    with pytest.raises(openrouter.LedgerMismatch, match="0.49"):
        build_client("stage0", sleep=no_sleep)
    assert paid["posts"] == []
    assert Ledger(paid["path"], stage="stage0").baseline() is None  # nothing was accepted


def test_lost_ledger_is_caught_after_real_spend(paid):
    client = build_client("stage0", sleep=no_sleep)
    run_arm(rows(3), jev_decider(client), client=client, results_path=paid["path"].parent / "runs" / "private" / "a.jsonl", expected_per_row=0.0001)
    paid["usage"] = 0.49
    paid["path"].unlink()
    with pytest.raises(openrouter.LedgerMismatch):
        build_client("stage0", sleep=no_sleep)
    assert len(paid["posts"]) == 3


def test_starting_key_usage_can_only_be_accepted_explicitly_and_by_amount(paid):
    paid["usage"] = 0.49
    with pytest.raises(openrouter.LedgerMismatch):
        build_client("stage0", sleep=no_sleep, accept_key_usage_usd=0.10)  # too small to cover it
    client = build_client("stage0", sleep=no_sleep, accept_key_usage_usd=0.49)
    # The accepted spend is in the ledger, against the total and the stage: it is not a fresh start.
    assert client.ledger.baseline() == 0.0 and client.ledger.baseline_observed() == 0.49
    assert client.ledger.spent() == pytest.approx(0.49) and client.ledger.stage_spent() == pytest.approx(0.49)
    assert client.ledger.remaining() == pytest.approx(0.01)
    entries = [json.loads(line) for line in paid["path"].read_text().splitlines()]
    assert [e["kind"] for e in entries] == ["baseline", "charge"]
    assert entries[0]["meta"]["accepted_usd"] == 0.49
    # Once the ledger agrees with the key there is nothing left to accept.
    with pytest.raises(ValueError, match="baseline"):
        build_client("stage0", sleep=no_sleep, accept_key_usage_usd=0.49)


def test_usage_since_baseline_beyond_ledger_spend_refuses_on_restart(paid):
    build_client("stage0", sleep=no_sleep)
    paid["usage"] = 0.20  # spend outside the ledger
    with pytest.raises(openrouter.LedgerMismatch):
        build_client("stage0", sleep=no_sleep)
    assert paid["posts"] == []


def test_normal_restart_passes_and_late_usage_is_not_a_mismatch(paid):
    client = build_client("stage0", sleep=no_sleep)
    run_arm(rows(3), jev_decider(client), client=client, results_path=paid["path"].parent / "runs" / "private" / "a.jsonl", expected_per_row=0.0001)
    build_client("stage0", sleep=no_sleep)
    paid["usage"] = 0.0  # usage not posted yet: the ledger holds more than reported, which is safe at the start
    build_client("stage0", sleep=no_sleep)
    paid["usage"] = 0.009  # under the floor
    build_client("stage0", sleep=no_sleep)


def test_key_usage_below_baseline_means_a_different_key(paid):
    paid["usage"] = 0.30
    build_client("stage0", sleep=no_sleep, accept_key_usage_usd=0.30)
    paid["usage"] = 0.10  # all-time usage cannot fall
    with pytest.raises(openrouter.LedgerMismatch, match="different key"):
        build_client("stage0", sleep=no_sleep)

    # Without an accepted amount the baseline is the usage seen at the first start; a dip within the floor is noise.
    paid["path"].unlink()
    paid["usage"] = 0.008
    build_client("stage0", sleep=no_sleep)
    paid["usage"] = 0.0
    build_client("stage0", sleep=no_sleep)  # within the floor


def test_paid_ledger_with_spend_but_no_baseline_refuses(paid):
    Ledger(paid["path"], stage="stage0").charge(0.1, {"note": "hand edited"})
    with pytest.raises(openrouter.LedgerMismatch, match="no baseline"):
        build_client("stage0", sleep=no_sleep)
    assert paid["posts"] == []


def test_ledger_baseline_cannot_be_recorded_twice(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    assert ledger.baseline() is None and ledger.is_empty()
    ledger.record_baseline(0.25, {})
    assert ledger.baseline() == 0.25 and not ledger.is_empty()
    assert ledger.spent() == 0.0  # a baseline is not spend
    with pytest.raises(ValueError, match="baseline"):
        ledger.record_baseline(0.0, {})
    assert Ledger(path).baseline() == 0.25


# ------------------------------------------------------------------ R6l, R6k: run_arm is the only paid path


def test_paid_calls_outside_run_arm_are_refused(paid):
    client = build_client("stage0", sleep=no_sleep)
    with pytest.raises(openrouter.PaidOutsideArm):
        jev_decide("r0", "text", OPTIONS, client=client)
    with pytest.raises(openrouter.PaidOutsideArm):
        llm_pick("r0", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
    assert paid["posts"] == []
    assert client.ledger.spent() == 0.0


def test_paid_arm_without_expected_cost_is_refused(paid):
    client = build_client("stage0", sleep=no_sleep)
    with pytest.raises(ValueError, match="expected_per_row"):
        run_arm(rows(3), jev_decider(client), client=client, results_path=paid["path"].parent / "runs" / "private" / "a.jsonl")
    assert paid["posts"] == []


def llm_with_missing_logprobs(miss_rows):
    def handler(body, n):
        reply = json.loads(llm_ok(body).body)
        if n in miss_rows:
            reply["choices"][0]["logprobs"] = None
        return ok(reply)

    return handler


def test_selection_runs_a_ladder_past_one_missing_logprob_and_charges_it(tmp_path):
    client, transport, ledger, _ = make_client(tmp_path, llm_with_missing_logprobs({7}), max_concurrency=1)
    report = run_arm(rows(50), llm_decider(client), client=client, results_path=results_path(tmp_path), allowed_arm_not_run=1)
    assert report.status == "complete"
    assert sum(not d.failed for d in report.decisions.values()) == 49
    assert report.decisions["r7"].failed is True and report.decisions["r7"].reason == "no_usable_logprobs"
    assert report.decisions["r7"].cost == pytest.approx(0.00003)
    assert len(transport.calls) == 50
    assert ledger.spent() == pytest.approx(50 * 0.00003)
    assert report.projections[0]["rows"] == 20  # the ladder still ran


def test_selection_stops_when_too_many_rows_lack_logprobs(tmp_path):
    client, transport, _, _ = make_client(tmp_path, llm_with_missing_logprobs({3, 8}), max_concurrency=1)
    report = run_arm(rows(50), llm_decider(client), client=client, results_path=results_path(tmp_path), allowed_arm_not_run=1)
    assert report.status == "arm_not_run" and len(transport.calls) == 9


# ------------------------------------------------------------------ R2: outage rows


def run_jev(tmp_path, handler, n=30, ledger=None, **kwargs):
    client, transport, ledger, _ = make_client(tmp_path, handler, ledger=ledger, max_concurrency=1, **kwargs)
    report = run_arm(rows(n), jev_decider(client), client=client, results_path=results_path(tmp_path))
    return report, transport, ledger


@pytest.mark.parametrize(
    "outage, reason",
    [
        (TransportFailure("down"), "network"),
        (status(529), "http_529"),
        (status(429), "http_429"),
        (status(400), "http_400"),
        (status(404), "http_404"),
    ],
)
def test_outage_rows_are_retried_on_resume_and_can_recover(tmp_path, outage, reason):
    first, _, ledger = run_jev(tmp_path, lambda body, n: outage)
    assert first.stop_type == "ConsecutiveFailures"
    failed = sorted(k for k, d in first.decisions.items() if d.failed)
    assert len(failed) == 10 and first.decisions[failed[0]].reason == reason
    assert sorted(first.retriable) == failed

    second, transport, _ = run_jev(tmp_path, lambda body, n: jev_ok(body), ledger=ledger)
    assert second.status == "complete"
    assert len(transport.calls) == 30  # the 10 outage rows and the 20 never tried
    assert not any(d.failed for d in second.decisions.values())
    assert second.retriable == []


def test_outage_retry_is_bounded_across_reruns_and_the_count_persists(tmp_path):
    outage = lambda body, n: status(529)
    first, _, ledger = run_jev(tmp_path, outage, n=5)
    assert first.status == "complete" and sorted(first.retriable) == [f"r{i}" for i in range(5)]

    # Second pass fails too, from a new process: the rows are now final and failed.
    second, transport, _ = run_jev(tmp_path, outage, n=5, ledger=Ledger(tmp_path / "ledger.jsonl"))
    assert len(transport.calls) == 20 and second.retriable == []
    assert all(d.failed for d in second.decisions.values())

    # Third run: nothing is sent, even with the service healthy.
    third, transport, _ = run_jev(tmp_path, lambda body, n: jev_ok(body), n=5, ledger=Ledger(tmp_path / "ledger.jsonl"))
    assert transport.calls == [] and third.status == "complete"
    assert all(d.failed and d.reason == "http_529" for d in third.decisions.values())


def test_usage_unavailable_row_is_an_outage_row(tmp_path):
    def no_cost(body, n):
        reply = json.loads(jev_ok(body).body)
        del reply["usage"]["cost"]
        return ok(reply)

    first, _, ledger = run_jev(tmp_path, no_cost, n=3)
    assert all(d.reason == "usage_unavailable" for d in first.decisions.values())
    assert sorted(first.retriable) == ["r0", "r1", "r2"]
    second, transport, _ = run_jev(tmp_path, lambda body, n: jev_ok(body), n=3, ledger=ledger)
    assert len(transport.calls) == 3 and not any(d.failed for d in second.decisions.values())


@pytest.mark.parametrize(
    "invalid",
    [
        lambda body: jev_ok(body, choice="not_an_option"),
        lambda body: ok(fixture("decisions_refusal_response.json")),
        lambda body: FakeResponse(200, {}, b"not json"),
    ],
)
def test_answered_but_invalid_rows_stay_final(tmp_path, invalid):
    first, _, ledger = run_jev(tmp_path, lambda body, n: invalid(body), n=3)
    assert first.retriable == [] and all(d.reason.startswith("malformed") for d in first.decisions.values())
    second, transport, _ = run_jev(tmp_path, lambda body, n: jev_ok(body), n=3, ledger=ledger)
    assert transport.calls == [] and all(d.failed for d in second.decisions.values())


def test_outage_row_stays_listed_when_a_stop_prevents_its_retry(tmp_path):
    first, _, ledger = run_jev(tmp_path, lambda body, n: status(529), n=3)
    second, transport, _ = run_jev(tmp_path, lambda body, n: status(401), n=3, ledger=ledger)
    assert len(transport.calls) == 1
    assert second.stop_type == "KeyOrCreditsStop"
    # The stop was on r0, so r0's second pass did not happen, and all three can still be retried.
    assert sorted(second.retriable) == ["r0", "r1", "r2"]
    third, transport, _ = run_jev(tmp_path, lambda body, n: jev_ok(body), n=3, ledger=ledger)
    assert len(transport.calls) == 3 and not any(d.failed for d in third.decisions.values())


# ------------------------------------------------------------------ R3: zero cost, end-of-arm check


@pytest.mark.parametrize("cost", [0, 0.0, -0.0])
def test_zero_cost_with_tokens_counts_as_unreported(tmp_path, cost):
    client, transport, ledger, _ = make_client(tmp_path, lambda body, n: jev_ok(body, cost=cost), reserve_usd=0.001, reported=lambda: 0.0, max_concurrency=1)
    decision = jev_decide("r0", "text", OPTIONS, client=client)
    assert decision.failed is True and decision.reason == "usage_unavailable" and len(transport.calls) == 4
    assert ledger.spent() == pytest.approx(0.004)
    assert ledger.confirmed() == 0.0


def test_zero_cost_on_every_paid_reply_stops_rather_than_running_free(tmp_path):
    """Important 3: 99 calls at cost 0 used to finish with a ledger of 0.0 and projections of US$0."""
    client, transport, ledger, _ = make_client(tmp_path, lambda body, n: jev_ok(body, cost=0), reserve_usd=0.001, reported=lambda: 0.0, max_concurrency=1)
    report = run_arm(rows(99), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.stop_type == "ConsecutiveFailures"
    assert ledger.spent() == pytest.approx(len(transport.calls) * 0.001)
    assert report.projections[0]["per_row"] == pytest.approx(0.004)


def test_zero_cost_with_no_tokens_is_a_real_zero(tmp_path):
    def handler(body, n):
        reply = json.loads(jev_ok(body, cost=0).body)
        reply["usage"].update(input_tokens=0, output_tokens=0)
        return ok(reply)

    client, _, ledger, _ = make_client(tmp_path, handler)
    assert jev_decide("r0", "text", OPTIONS, client=client).failed is False
    assert ledger.spent() == 0.0


def test_short_run_reconciles_at_the_end(tmp_path):
    """Under 100 calls there is no periodic read, so only the end-of-arm check can see this."""
    reads = []

    def reported():
        reads.append(1)
        return 0.0

    client, transport, _, sleeps = make_client(tmp_path, lambda body, n: jev_ok(body, cost=0.001), reported=reported, max_concurrency=1, reconcile_grace_seconds=45)
    report = run_arm(rows(99), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.status == "stopped" and report.stop_type == "ReconciliationGap"
    assert len(report.decisions) == 99 and report.not_run == []  # every row is still kept
    assert len(reads) == 3  # baseline, first end read, read after the grace
    assert sleeps == [45]


def test_end_of_arm_grace_allows_for_usage_posting_late(tmp_path):
    holder = {"posted": False}
    ledger = Ledger(tmp_path / "ledger.jsonl")

    def reported():
        return ledger.confirmed() if holder["posted"] else 0.0

    client, _, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body, cost=0.001), ledger=ledger, reported=reported, max_concurrency=1, reconcile_grace_seconds=30)
    client._sleep = lambda seconds: holder.update(posted=True, waited=seconds)
    report = run_arm(rows(50), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.status == "complete" and holder["waited"] == 30


def test_grace_is_configurable_and_not_used_when_billed_above_the_ledger(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    client, _, _, sleeps = make_client(tmp_path, lambda body, n: jev_ok(body, cost=0.001), ledger=ledger, reported=lambda: 5.0 if ledger.spent() else 0.0, max_concurrency=1, reconcile_grace_seconds=30)
    report = run_arm(rows(20), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.stop_type == "ReconciliationGap" and sleeps == []  # lag cannot explain being billed more


def test_no_end_read_when_nothing_was_sent(tmp_path):
    reads = []
    client, _, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body), reported=lambda: (reads.append(1), 0.0)[1])
    client.reconcile_final()
    assert len(reads) == 1  # the baseline only


def test_short_client_can_reconcile_directly(tmp_path):
    client, _, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body, cost=0.02), reported=lambda: 0.0)
    jev_decide("r0", "text", OPTIONS, client=client)
    assert client.stop is None
    client.reconcile_final()
    assert isinstance(client.stop, openrouter.ReconciliationGap)


# ------------------------------------------------------------------ R6e: the cap uses the larger figure


def test_cap_uses_reported_spend_when_it_is_above_the_ledger(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path, cap_usd=1.0)
    ledger.charge(0.90, {})
    ledger.note_reported(0.99)
    assert ledger.spent() == pytest.approx(0.90)  # the ledger's own figure is unchanged
    assert ledger.remaining() == pytest.approx(0.01)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(0.02, {})
    with pytest.raises(BudgetExceeded):
        ledger.projected(2, 0.01)
    # It persists across a restart.
    assert Ledger(path, cap_usd=1.0).remaining() == pytest.approx(0.01)
    # A lower report never gives room back.
    ledger.note_reported(0.5)
    assert ledger.remaining() == pytest.approx(0.01)


def test_excess_reported_spend_counts_against_the_stage_limit_too(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", stage="stage0")
    ledger.charge(0.40, {})
    ledger.note_reported(0.49)
    assert ledger.remaining() == pytest.approx(0.01)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(0.02, {})


def test_gap_under_10_percent_still_tightens_the_cap(tmp_path):
    """A 7% gap passes reconciliation, but the extra spend must still count against the cap."""
    ledger = Ledger(tmp_path / "ledger.jsonl", cap_usd=0.25)
    client, transport, _, _ = make_client(
        tmp_path, lambda body, n: jev_ok(body, cost=0.002), ledger=ledger,
        reported=lambda: ledger.confirmed() * 1.07, max_concurrency=1,
    )
    report = run_arm(rows(120), jev_decider(client), client=client, results_path=results_path(tmp_path))
    # By the ledger alone 0.25 allows 125 calls, so all 120 would finish. At call 100 the report says
    # 0.214 against 0.200: that 0.014 stays counted as the ledger grows, leaving room for 118 calls.
    assert report.stop_type == "BudgetExceeded"
    assert 117 <= len(transport.calls) <= 118


# ------------------------------------------------------------------ R4: model pin


def test_llm_first_reply_must_match_the_requested_model(tmp_path):
    client, _, _, _ = make_client(tmp_path, lambda body, n: llm_ok(body, model="other/totally-different-model"))
    with pytest.raises(ModelChanged):
        llm_pick("r0", "text", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)


@pytest.mark.parametrize(
    "reply, allowed",
    [
        ("openai/gpt-oss-20b", True),
        ("openai/gpt-oss-20b-20261014", True),
        ("openai/gpt-oss-20b-v1", False),
        ("openai/gpt-oss-20b-2026", False),
        ("openai/gpt-oss-20", False),
        ("openai/gpt-oss-20bx", False),
        ("anthropic/gpt-oss-20b", False),
        ("gpt-oss-20b", False),
    ],
)
def test_llm_model_suffix_pattern_is_exact(tmp_path, reply, allowed):
    client, _, _, _ = make_client(tmp_path, lambda body, n: llm_ok(body, model=reply))
    if allowed:
        assert llm_pick("r0", "t", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client).model == reply
    else:
        with pytest.raises(ModelChanged):
            llm_pick("r0", "t", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)


def test_llm_tagged_request_matches_reply_with_or_without_the_tag(tmp_path):
    for reply in ("z-ai/glm-5.3-flash", "z-ai/glm-5.3-flash:nitro", "z-ai/glm-5.3-flash-20261001"):
        client, _, _, _ = make_client(tmp_path, lambda body, n, r=reply: llm_ok(body, model=r))
        assert llm_pick("r0", "t", OPTIONS, "z-ai/glm-5.3-flash:nitro", "Wafer", client=client).failed is False


def test_mixed_builds_in_one_arm_stop(tmp_path):
    def handler(body, n):
        return llm_ok(body, model="openai/gpt-oss-20b-20261014" if n < 3 else "openai/gpt-oss-20b-20270101")

    client, transport, _, _ = make_client(tmp_path, handler, max_concurrency=1)
    report = run_arm(rows(10), llm_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.stop_type == "ModelChanged" and len(report.decisions) == 3 and len(transport.calls) == 4


def test_resume_with_a_changed_build_stops(tmp_path):
    client, _, ledger, _ = make_client(tmp_path, lambda b, n: llm_ok(b, model="openai/gpt-oss-20b-20261014"), max_concurrency=1)
    run_arm(rows(5), llm_decider(client), client=client, results_path=results_path(tmp_path))
    client, transport, _, _ = make_client(tmp_path, lambda b, n: llm_ok(b, model="openai/gpt-oss-20b-20270101"), ledger=ledger, max_concurrency=1)
    report = run_arm(rows(10), llm_decider(client), client=client, results_path=results_path(tmp_path))
    assert report.stop_type == "ModelChanged"
    assert {d.model for d in report.decisions.values()} == {"openai/gpt-oss-20b-20261014"}
    assert len(report.decisions) == 5 and len(transport.calls) == 1


def test_resume_with_a_changed_jev_build_stops(tmp_path):
    client, _, ledger, _ = make_client(tmp_path, lambda b, n: jev_ok(b), max_concurrency=1)
    run_arm(rows(5), jev_decider(client), client=client, results_path=results_path(tmp_path))
    other = "jev-1.14-20270101"
    client, _, _, _ = make_client(tmp_path, lambda b, n: jev_ok(b, model="typesafe/" + other), ledger=ledger, max_concurrency=1)
    report = run_arm(rows(10), jev_decider(client, expected_build=other), client=client, results_path=results_path(tmp_path))
    assert report.stop_type == "ModelChanged"
    assert {d.model for d in report.decisions.values()} == {BUILD}


def test_saved_rows_that_already_mix_builds_stop_before_sending(tmp_path):
    client, _, ledger, _ = make_client(tmp_path, lambda b, n: llm_ok(b, model="openai/gpt-oss-20b-20261014"), max_concurrency=1)
    path = results_path(tmp_path)
    run_arm(rows(2), llm_decider(client), client=client, results_path=path)
    with open(path, "ab") as handle:
        record = {"row_id": "r2", "status": "done", "pred": "a", "top_prob": 0.5, "probs": {"a": 0.5}, "model": "openai/gpt-oss-20b-20270101", "cost": 0.1, "failed": False}
        handle.write(json.dumps(record).encode() + b"\n")
    client, transport, _, _ = make_client(tmp_path, lambda b, n: llm_ok(b), ledger=ledger, max_concurrency=1)
    report = run_arm(rows(5), llm_decider(client), client=client, results_path=path)
    assert report.stop_type == "ModelChanged" and transport.calls == []


def test_jev_model_must_start_with_typesafe(tmp_path):
    for bad in ("anything/jev-1.14-20261014", "jev-1.14-20261014"):
        client, _, _, _ = make_client(tmp_path, lambda b, n, m=bad: jev_ok(b, model=m))
        with pytest.raises(ModelChanged):
            jev_decide("r0", "text", OPTIONS, client=client)


# ------------------------------------------------------------------ R6a: the key echoed in reply fields


def _jev_echo(field, value):
    def handler(body, n):
        reply = json.loads(jev_ok(body).body)
        reply[field] = value
        return ok(reply)

    return handler


def _llm_echo(field, value):
    def handler(body, n):
        reply = json.loads(llm_ok(body).body)
        reply[field] = value
        return ok(reply)

    return handler


@pytest.mark.parametrize(
    "make_handler, field, value",
    [
        (_jev_echo, "model", LONG_KEY),
        (_jev_echo, "model", "typesafe/" + LONG_KEY),
        (_jev_echo, "provider", LONG_KEY),
        (_jev_echo, "provider", "x/" + LONG_KEY),
        (_llm_echo, "model", LONG_KEY),
        (_llm_echo, "model", "openai/" + LONG_KEY),
        (_llm_echo, "provider", LONG_KEY),
        (_llm_echo, "provider", "SK-" + LONG_KEY[3:]),
    ],
)
def test_key_echoed_in_reply_fields_reaches_no_output(tmp_path, caplog, make_handler, field, value):
    caplog.set_level(logging.DEBUG)
    client, transport, ledger, _ = make_client(tmp_path, make_handler(field, value), max_concurrency=1)
    decide = jev_decider(client) if make_handler is _jev_echo else llm_decider(client)
    path = results_path(tmp_path)
    report = run_arm(rows(2), decide, client=client, results_path=path)
    surfaces = [repr(report), report.reason or "", caplog.text, path.read_text(), (tmp_path / "ledger.jsonl").read_text()]
    needle = value.split("/")[-1][3:]
    assert not [s for s in surfaces if needle in s]
    assert all(d.failed and d.reason.startswith("malformed") for d in report.decisions.values())


@pytest.mark.parametrize(
    "bad",
    [
        "", "x" * 81, "has space", "new\nline", "quote\"", "sk-abc", "SK-abc", "a/sk-abc", "a:sk-abc", "a.sk-abc", 7, None, ["a"],
        # Fix round 2 (attack4): sk- anywhere, not only at the start of a segment, and a bare secret.
        "t/a-" + LONG_KEY, "t/a_" + LONG_KEY, "t/1" + LONG_KEY, "a" + LONG_KEY, "ask-or-v1-abc", "task-model", "x/mask-1",
        "ab" * 32, "AB" * 32, "x/" + "0" * 32, "m-" + "f" * 32 + "-z", "sk-or-", "SK-OR-v1",
    ],
)
def test_safe_name_refuses(bad):
    with pytest.raises(openrouter.MalformedResponse):
        openrouter.safe_name(bad, "model")


@pytest.mark.parametrize(
    "good",
    [
        "openai/gpt-oss-20b", "z-ai/glm-5.3-flash", "typesafe/jev-1.14-20261014", "Darkbloom", "Wafer",
        "TypeSafe", "z-ai/glm-5.3-flash:nitro", "openai/gpt-4o-mini-2024-07-18", "deepseek/deepseek-r1:free", "x" * 80,
        "m-" + "f" * 31 + "-z",  # a hex run one short of the limit
    ],
)
def test_safe_name_allows(good):
    assert openrouter.safe_name(good, "model") == good


# ------------------------------------------------------------------ R6c, d, f: the real transport


def test_real_client_has_timeouts_ignores_env_and_refuses_redirects():
    transport = RealTransport("stage0", key_reader=lambda: FAKE_KEY, httpx_transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    timeout = transport._http.timeout
    assert all(value is not None and value > 0 for value in (timeout.connect, timeout.read, timeout.write, timeout.pool))
    assert transport._http.trust_env is False
    assert transport._http.follow_redirects is False
    assert 0 < openrouter.REQUEST_DEADLINE_SECONDS <= 120


def test_overall_wall_clock_limit_stops_a_slow_drip_reply():
    clock = {"now": 0.0}

    class Drip(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(1000):
                clock["now"] += 10  # each chunk arrives inside the read timeout, so httpx alone never fires
                yield b"x"

    transport = RealTransport(
        "stage0", key_reader=lambda: FAKE_KEY, clock=lambda: clock["now"],
        httpx_transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Drip())),
    )
    with pytest.raises(TransportFailure, match="RequestDeadlineExceeded"):
        transport.send("POST", openrouter.DECISIONS_URL, b"{}")
    assert clock["now"] <= openrouter.REQUEST_DEADLINE_SECONDS + 10


def test_reply_within_the_deadline_is_read_whole():
    transport = RealTransport("stage0", key_reader=lambda: FAKE_KEY, httpx_transport=httpx.MockTransport(lambda r: httpx.Response(201, content=b"abc", headers={"Retry-After": "3"})))
    reply = transport.send("POST", openrouter.DECISIONS_URL, b"{}")
    assert (reply.status, reply.body, reply.headers["retry-after"]) == (201, b"abc", "3")


def test_no_frame_local_in_send_holds_the_key():
    seen = {}

    def inspecting(request):
        frame = sys._getframe()
        while frame is not None:
            if frame.f_code.co_name == "send" and frame.f_code.co_filename.endswith("openrouter.py"):
                seen["locals"] = {name: repr(value) for name, value in frame.f_locals.items()}
            frame = frame.f_back
        return httpx.Response(200)

    transport = RealTransport("stage0", key_reader=lambda: FAKE_KEY, httpx_transport=httpx.MockTransport(inspecting))
    transport.send("POST", openrouter.DECISIONS_URL, b"{}")
    assert seen["locals"] and not [v for v in seen["locals"].values() if FAKE_KEY in v]


# ------------------------------------------------------------------ R6g, h


def test_retrieve_raises_on_the_querys_own_text_in_the_bank():
    bank = [Example("my card never arrived", "card_arrival")] + FILLER
    for query in ("my card never arrived", "  MY  card\tnever ARRIVED ", "my card never arrived\u00a0"):
        with pytest.raises(retrieval.QueryInBank):
            retrieve(query, bank)
    assert retrieve("my card has never arrived", bank)


def test_results_path_is_resolved_before_the_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ("runs/private/../../published/out.jsonl", tmp_path / "runs" / "private" / ".." / ".." / "out.jsonl"):
        with pytest.raises(ValueError, match="runs/private"):
            openrouter.require_private_path(bad)
    # A link inside runs/private that points out is caught too.
    (tmp_path / "runs" / "private").mkdir(parents=True)
    (tmp_path / "public").mkdir()
    (tmp_path / "runs" / "private" / "link").symlink_to(tmp_path / "public")
    with pytest.raises(ValueError, match="runs/private"):
        openrouter.require_private_path("runs/private/link/out.jsonl")
    assert openrouter.require_private_path("runs/private/ok.jsonl") == (tmp_path / "runs" / "private" / "ok.jsonl").resolve()


# ------------------------------------------------------------------ R6i


@pytest.mark.parametrize("value", ["3", 1.5, True, [1], {"a": 1}])
def test_non_integer_reasoning_tokens_fail_closed(tmp_path, value):
    def handler(body, n):
        reply = json.loads(llm_ok(body).body)
        reply["usage"]["completion_tokens_details"]["reasoning_tokens"] = value
        return ok(reply)

    client, transport, _, _ = make_client(tmp_path, handler, max_concurrency=1)
    with pytest.raises(HiddenReasoning):
        llm_pick("r0", "t", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
        llm_pick("r1", "t", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("details", [None, {}, {"reasoning_tokens": None}, {"reasoning_tokens": 0}])
def test_documented_empty_reasoning_fields_are_allowed(tmp_path, details):
    def handler(body, n):
        reply = json.loads(llm_ok(body).body)
        reply["usage"]["completion_tokens_details"] = details
        return ok(reply)

    client, _, _, _ = make_client(tmp_path, handler)
    assert llm_pick("r0", "t", OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client).failed is False
    assert client.stop is None


# ------------------------------------------------------------------ fix round 2, N1: excess, accepted usage


def private(paid, name):
    return paid["path"].parent / "runs" / "private" / f"{name}.jsonl"


def test_recorded_excess_does_not_block_a_later_paid_start(paid):
    """attack7: billed 0.003 a call against a reported 0.002. The gap stops the arm and stays counted as excess."""
    paid["cost"], paid["billed"] = 0.002, 1.5
    client = build_client("stage0", sleep=no_sleep)
    report = run_arm(rows(20), jev_decider(client), client=client, results_path=private(paid, "fee"), expected_per_row=0.002)
    assert report.stop_type == "ReconciliationGap"
    assert client.ledger.spent() == pytest.approx(0.04) and client.ledger.excess() == pytest.approx(0.02)

    # The excess is already held against the caps, so a restart agrees with the key and may send.
    again = build_client("stage0", sleep=no_sleep)
    assert again.ledger.remaining() == pytest.approx(0.50 - 0.06)
    # Nothing is left to accept, so an accepted amount is refused rather than charged.
    with pytest.raises(ValueError, match="nothing to accept"):
        build_client("stage0", sleep=no_sleep, accept_key_usage_usd=0.02)

    # Spend outside the ledger beyond the excess is still caught.
    paid["usage"] += 0.05
    with pytest.raises(openrouter.LedgerMismatch, match="0.05"):
        build_client("stage0", sleep=no_sleep)


def test_accepting_spend_outside_an_existing_ledger_charges_it_and_never_resets(paid):
    client = build_client("stage0", sleep=no_sleep)
    paid["cost"] = 0.1
    run_arm(rows(2), jev_decider(client), client=client, results_path=private(paid, "a"), expected_per_row=0.1)
    assert client.ledger.spent() == pytest.approx(0.2)
    paid["usage"] += 0.15  # spend outside the ledger
    with pytest.raises(openrouter.LedgerMismatch):
        build_client("stage0", sleep=no_sleep)
    with pytest.raises(openrouter.LedgerMismatch, match="cover"):
        build_client("stage0", sleep=no_sleep, accept_key_usage_usd=0.135)  # 10% short: not within a cent
    after = build_client("stage0", sleep=no_sleep, accept_key_usage_usd=0.15)
    assert after.ledger.baseline() == 0.0  # the baseline is untouched
    assert after.ledger.spent() == pytest.approx(0.35) and after.ledger.stage_spent() == pytest.approx(0.35)
    assert after.ledger.remaining() == pytest.approx(0.15)
    build_client("stage0", sleep=no_sleep)  # and the books now agree


@pytest.mark.parametrize("accept", [math.nan, math.inf, -math.inf, -5, -0.01, "0.49", True])
def test_accepted_usage_must_be_a_finite_amount_of_zero_or_more(paid, accept):
    paid["usage"] = 0.49
    with pytest.raises(ValueError, match="finite"):
        build_client("stage0", sleep=no_sleep, accept_key_usage_usd=accept)
    assert Ledger(paid["path"], stage="stage0").is_empty()
    assert paid["posts"] == []


def test_accepted_usage_must_cover_the_key_to_within_a_cent(paid):
    paid["usage"] = 0.40
    for short in (0.9 * 0.40, 0.36, 0.389):
        with pytest.raises(openrouter.LedgerMismatch, match="cover"):
            build_client("stage0", sleep=no_sleep, accept_key_usage_usd=short)
    assert Ledger(paid["path"], stage="stage0").is_empty()
    client = build_client("stage0", sleep=no_sleep, accept_key_usage_usd=0.391)
    assert client.ledger.spent() == pytest.approx(0.40)  # charged at the larger of the accepted and the shown amount


def test_lost_ledger_then_accept_does_not_give_the_stage_a_fresh_cap(paid):
    """attack3 C: spend 0.40, delete the ledger, accept. Stage 0 has 0.10 left, not a new 0.50."""
    paid["cost"] = 0.2
    client = build_client("stage0", sleep=no_sleep)
    report = run_arm(rows(2), jev_decider(client), client=client, results_path=private(paid, "a"), expected_per_row=0.2)
    assert report.status == "complete" and paid["usage"] == pytest.approx(0.40)
    paid["path"].unlink()
    with pytest.raises(openrouter.LedgerMismatch):
        build_client("stage0", sleep=no_sleep)
    client = build_client("stage0", sleep=no_sleep, accept_key_usage_usd=0.40)
    assert client.ledger.remaining() == pytest.approx(0.10)
    report = run_arm(rows(3), jev_decider(client), client=client, results_path=private(paid, "b"), expected_per_row=0.2)
    assert report.status == "refused" and len(paid["posts"]) == 2
    # Stage 1 counts it against the total too.
    assert build_client("stage1", sleep=no_sleep).ledger.remaining() == pytest.approx(7.15 - 0.40)


# ------------------------------------------------------------------ fix round 2, N2: late usage, arms on one client


def test_earlier_arms_spend_posting_late_is_not_read_as_overbilling(paid):
    """attack6: arm 1's last 0.016 posts while arm 2 runs on a new client. Arm 2 must not stop."""
    paid["cost"] = 0.002
    sleeps = []
    client = build_client("stage0", sleep=sleeps.append)
    paid["lag"] = 0.016  # within 10% of arm 1's 0.20
    report = run_arm(rows(100), jev_decider(client), client=client, results_path=private(paid, "a"), expected_per_row=0.002)
    assert report.status == "complete"

    client = build_client("stage0", sleep=sleeps.append)  # starts while 0.016 is still unposted
    paid["lag"] = 0.0
    report = run_arm(rows(20), jev_decider(client), client=client, results_path=private(paid, "b"), expected_per_row=0.002)
    assert report.status == "complete", report.reason
    assert client.ledger.excess() == 0.0

    # Being billed more than the ledger is still caught at once, with no wait.
    paid["billed"] = 2.0
    client = build_client("stage0", sleep=sleeps.append)
    report = run_arm(rows(20), jev_decider(client), client=client, results_path=private(paid, "c"), expected_per_row=0.002)
    assert report.stop_type == "ReconciliationGap"
    assert sleeps == []


def test_unposted_spend_at_start_waits_out_the_grace_like_any_lag(paid):
    paid["cost"] = 0.002
    client = build_client("stage0", sleep=no_sleep)
    run_arm(rows(50), jev_decider(client), client=client, results_path=private(paid, "a"), expected_per_row=0.002)
    paid["lag"] = 0.05  # none of arm 1 has posted when arm 2's client starts
    sleeps = []
    client = build_client("stage0", sleep=lambda s: (sleeps.append(s), paid.update(lag=0.0)))
    report = run_arm(rows(10), jev_decider(client), client=client, results_path=private(paid, "b"), expected_per_row=0.002)
    assert report.status == "complete", report.reason
    assert sleeps == [openrouter.DEFAULT_RECONCILE_GRACE_SECONDS]


def test_earlier_estimates_that_were_never_billed_do_not_stop_the_next_client(paid):
    """529s are charged at an estimate. If they were not billed, the next client must not read that as a gap."""
    paid["reply"] = lambda body: httpx.Response(529)
    client = build_client("stage0", sleep=no_sleep)
    report = run_arm(rows(12), jev_decider(client), client=client, results_path=private(paid, "a"), expected_per_row=0.0001)
    assert report.stop_type == "ConsecutiveFailures"
    assert client.ledger.spent() >= 0.04 and client.ledger.confirmed() == 0.0 and paid["usage"] == 0.0  # 4 attempts a row at 0.001

    paid["reply"] = None
    client = build_client("stage0", sleep=no_sleep)
    report = run_arm(rows(5), jev_decider(client), client=client, results_path=private(paid, "b"), expected_per_row=0.0001)
    assert report.status == "complete", report.reason


def test_earlier_estimates_that_were_billed_may_post_late_without_a_stop(paid):
    def billed_529(body):
        paid["usage"] += 0.001
        return httpx.Response(529)

    paid["reply"] = billed_529
    client = build_client("stage0", sleep=no_sleep)
    run_arm(rows(12), jev_decider(client), client=client, results_path=private(paid, "a"), expected_per_row=0.0001)
    estimated = client.ledger.spent()
    assert estimated == pytest.approx(paid["usage"]) and client.ledger.confirmed() == 0.0

    paid["reply"], paid["lag"] = None, estimated  # none of it has posted when the next client starts
    client = build_client("stage0", sleep=no_sleep)
    paid["lag"] = 0.0  # it all posts during the next arm
    report = run_arm(rows(5), jev_decider(client), client=client, results_path=private(paid, "b"), expected_per_row=0.0001)
    assert report.status == "complete", report.reason


def test_one_client_runs_the_jev_arm_then_the_llm_arm(tmp_path):
    """attack3 B: the model pin and the stop are per arm, so one client can run both arms in turn."""

    def handler(body, n):
        return jev_ok(body) if "questions" in body else llm_ok(body)

    client, transport, _, _ = make_client(tmp_path, handler)
    first = run_arm(rows(3), jev_decider(client), client=client, results_path=tmp_path / "runs" / "private" / "jev.jsonl")
    second = run_arm(rows(3), llm_decider(client), client=client, results_path=tmp_path / "runs" / "private" / "llm.jsonl")
    assert first.status == "complete" and second.status == "complete", second.reason
    assert {d.model for d in first.decisions.values()} == {BUILD}
    assert {d.model for d in second.decisions.values()} == {"openai/gpt-oss-20b"}
    assert len(transport.calls) == 6


def test_each_arm_seeds_its_pin_from_its_own_saved_rows(tmp_path):
    other = "typesafe/jev-1.14-20270101"
    model = {"name": BUILD}
    client, transport, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body, model=model["name"]))
    path_a, path_b = tmp_path / "runs" / "private" / "a.jsonl", tmp_path / "runs" / "private" / "b.jsonl"
    assert run_arm(rows(3), jev_decider(client), client=client, results_path=path_a).status == "complete"
    model["name"] = other
    report = run_arm(rows(3), jev_decider(client, expected_build=other.split("/")[1]), client=client, results_path=path_b)
    assert report.status == "complete", report.reason
    # Back on arm a's file, its saved rows pin the old build, so the first reply from the new one stops it.
    sent = len(transport.calls)
    report = run_arm(rows(5), jev_decider(client, expected_build=other.split("/")[1]), client=client, results_path=path_a)
    assert report.stop_type == "ModelChanged" and len(transport.calls) == sent + 1
    assert report.not_run == ["r3", "r4"]  # the answer from the other build is not kept


def test_an_arm_scoped_stop_ends_only_its_arm(tmp_path):
    replies = {"mode": "fail"}

    def handler(body, n):
        failing = replies["mode"] == "fail" or body["state"] in ("message number 0", "message number 1")
        return status(400) if failing else jev_ok(body)

    client, transport, _, _ = make_client(tmp_path, handler, max_concurrency=1)
    report = run_arm(rows(15), jev_decider(client), client=client, results_path=tmp_path / "runs" / "private" / "a.jsonl")
    assert report.stop_type == "ConsecutiveFailures"
    replies["mode"] = "first two fail"
    report = run_arm(rows(5), jev_decider(client), client=client, results_path=tmp_path / "runs" / "private" / "b.jsonl")
    assert report.status == "complete", report.reason  # the failure count started again at 0
    assert [d.failed for d in report.decisions.values()].count(True) == 2


def test_the_output_token_mean_is_per_arm(tmp_path):
    tokens = {"n": 40}
    client, _, _, _ = make_client(tmp_path, lambda body, n: llm_ok(body, tokens=tokens["n"]), max_concurrency=1)
    report = run_arm(rows(1), llm_decider(client), client=client, results_path=tmp_path / "runs" / "private" / "a.jsonl")
    assert report.stop_type == "HiddenReasoning"
    tokens["n"] = 5
    report = run_arm(rows(1), llm_decider(client), client=client, results_path=tmp_path / "runs" / "private" / "b.jsonl")
    assert report.status == "complete", report.reason  # 5 tokens, not the mean of 40 and 5


@pytest.mark.parametrize("make_stop", ["key", "gap"])
def test_money_and_key_stops_stay_for_the_life_of_the_client(tmp_path, make_stop):
    replies = {"mode": "first"}
    ledger = Ledger(tmp_path / "ledger.jsonl")

    def handler(body, n):
        if replies["mode"] == "first" and make_stop == "key":
            return status(401)
        return jev_ok(body, cost=0.02)

    reported = (lambda: 0.0) if make_stop == "gap" else ledger.confirmed
    client, transport, _, _ = make_client(tmp_path, handler, ledger=ledger, reported=reported, max_concurrency=1)
    first = run_arm(rows(3), jev_decider(client), client=client, results_path=tmp_path / "runs" / "private" / "a.jsonl")
    assert first.stop_type == ("KeyOrCreditsStop" if make_stop == "key" else "ReconciliationGap")
    replies["mode"] = "later"
    sent = len(transport.calls)
    second = run_arm(rows(3), jev_decider(client), client=client, results_path=tmp_path / "runs" / "private" / "b.jsonl")
    assert second.status == "stopped" and second.stop_type == first.stop_type
    assert len(transport.calls) == sent and second.not_run == ["r0", "r1", "r2"]


def test_stop_scopes_are_as_documented():
    for scoped in (ModelChanged, ProviderChanged, ArmNotRun, openrouter.ConsecutiveFailures, HiddenReasoning):
        assert issubclass(scoped, openrouter.ARM_SCOPED_STOPS)
    for lasting in (
        BudgetExceeded, KeyOrCreditsStop, openrouter.LedgerMismatch, openrouter.ReconciliationGap,
        openrouter.ReconciliationUnavailable, openrouter.UnexpectedStop,
    ):
        assert not issubclass(lasting, openrouter.ARM_SCOPED_STOPS)
    assert "one arm at a time" in openrouter.__doc__ and "any" in openrouter.__doc__


def test_a_client_runs_one_arm_at_a_time(tmp_path):
    client, transport, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body))
    client._in_arm = True  # as if another thread were inside run_arm
    with pytest.raises(RuntimeError, match="one after another"):
        run_arm(rows(1), jev_decider(client), client=client, results_path=results_path(tmp_path))
    assert transport.calls == []
