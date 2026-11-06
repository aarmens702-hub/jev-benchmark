"""Latency (Task 7, Step 0): wall-clock seconds per decision in every arm's predictions, and the probes.

Fakes only. No model loads, no network, no key. Paid probes are built here and run later under the
paid-stage approval.
"""
import json

import numpy as np
import pytest

import latency_probe
import openrouter
import predict_local
import predictions
from jev_client import jev_decide
from llm_client import llm_pick
from openrouter import Decision, PaidFlagRequired, TransportResponse, run_arm
from retrieval import Example
from test_arms_local import AG, make_runtime, toy_split
from test_clients import OPTIONS, jev_ok, llm_ok, make_client, results_path


class Ticker:
    """A clock that moves `step` seconds every time it is read."""

    def __init__(self, step=0.25):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


SMALL = {"ag_news": toy_split(AG, 8, 20)}

LOCAL_ARMS = [
    (["--arm", "embed_sim"], ["embed_sim", "embed_sim_t0.01", "embed_sim_t0.2", "embed_sim_raw"]),
    (["--arm", "embed_sim", "--bare"], ["embed_sim_bare"]),
    (["--arm", "lr", "--draw", "1", "--s", "2"], ["lr_d1_s2"]),
    (["--arm", "lr_cv", "--draw", "1", "--s", "2"], ["lr_cv_d1_s2"]),
    (["--arm", "tfidf_lr", "--draw", "1", "--s", "2"], ["tfidf_lr_d1_s2"]),
    (["--arm", "nli"], ["nli"]),
]


def test_latency_is_recorded_for_every_arm(tmp_path):
    # Local arms: every row of every prediction file carries a positive wall-clock time.
    rt = make_runtime(tmp_path, timer=Ticker(), splits=SMALL)
    for args, arms in LOCAL_ARMS:
        assert predict_local.main(["--stage", "val", "--dataset", "ag_news", *args], rt) == 0
        for arm in arms:
            rows, _, _ = predictions.read(arm, "ag_news", stage="val", root=tmp_path / "runs")
            assert "latency_s" in rows.columns, arm
            assert np.isfinite(rows["latency_s"]).all() and (rows["latency_s"] > 0).all(), arm

    # API arms: Jev, Jev + examples and the small LLM each time every decision.
    deciders = {
        "jev": lambda client: lambda row: jev_decide(row["row_id"], row["text"], OPTIONS, client=client),
        "jev_examples": lambda client: lambda row: jev_decide(
            row["row_id"], row["text"], OPTIONS, examples=[Example("a card", "card_arrival")], client=client
        ),
        "small_llm": lambda client: lambda row: llm_pick(
            row["row_id"], row["text"], OPTIONS, "openai/gpt-oss-20b", "Darkbloom", client=client
        ),
    }
    for arm, make in deciders.items():
        handler = (lambda body, n: llm_ok(body)) if arm == "small_llm" else (lambda body, n: jev_ok(body))
        client, _, _, _ = make_client(tmp_path / arm, handler, clock=Ticker())
        rows = [{"row_id": f"r{i}", "text": f"message {i}"} for i in range(5)]
        report = run_arm(rows, make(client), client=client, results_path=tmp_path / arm / "runs" / "private" / "a.jsonl")
        assert all(d.latency_s is not None and d.latency_s > 0 for d in report.decisions.values()), arm
        frame, _ = predictions.decisions_frame([r["row_id"] for r in rows], report.decisions, list(OPTIONS))
        assert (frame["latency_s"] > 0).all(), arm


def test_api_latency_covers_retries_and_survives_a_resume(tmp_path):
    clock = Ticker(0.0)
    replies = iter([TransportResponse(529, {}, b"{}"), None])

    def handler(body, n):
        clock.now += 5.0  # each attempt takes 5 seconds
        reply = next(replies)
        return reply if reply is not None else jev_ok(body)

    client, _, _, _ = make_client(tmp_path, handler, clock=clock)
    path = results_path(tmp_path)
    report = run_arm([{"row_id": "r0", "text": "hi"}], lambda row: jev_decide("r0", "hi", OPTIONS, client=client), client=client, results_path=path)
    first = report.decisions["r0"].latency_s
    assert first == 10.0  # both attempts are inside the decision's time
    saved = json.loads(path.read_text().splitlines()[0])
    assert saved["latency_s"] == first
    client2, transport2, _, _ = make_client(tmp_path / "again", lambda body, n: jev_ok(body), clock=clock)
    again = run_arm([{"row_id": "r0", "text": "hi"}], lambda row: jev_decide("r0", "hi", OPTIONS, client=client2), client=client2, results_path=path)
    assert transport2.calls == [] and again.decisions["r0"].latency_s == first


def test_supervised_latency_excludes_fitting(tmp_path):
    """The bank is embedded and the model fit before the clock starts; rows time prediction only."""
    calls = []

    class Timer:
        now = 0.0

        def __call__(self):
            calls.append("t")
            return self.now

    timer = Timer()
    rt = make_runtime(tmp_path, timer=timer, splits=SMALL)
    import arms_local

    original = arms_local.fit_lr

    def slow_fit(*args, **kwargs):
        timer.now += 1000.0  # if fitting were inside the timed span, rows would show it
        return original(*args, **kwargs)

    arms_local.fit_lr = slow_fit
    try:
        predict_local.main(["--stage", "val", "--dataset", "ag_news", "--arm", "lr", "--draw", "0", "--s", "2"], rt)
    finally:
        arms_local.fit_lr = original
    rows, _, _ = predictions.read("lr_d0_s2", "ag_news", stage="val", root=tmp_path / "runs")
    assert (rows["latency_s"] < 1000.0 / len(rows)).all()


def test_decisions_frame_keeps_usage_and_puts_vectors_only_in_the_private_part():
    ok = Decision("lost_card", 0.7, {"card_arrival": 0.1, "lost_card": 0.7, "top_up_failed": 0.2}, "typesafe/jev-1.14-20261014", 0.0002, False, input_tokens=900, output_tokens=3, latency_s=0.4)
    bad = Decision(None, None, {}, "typesafe/jev-1.14-20261014", 0.0003, True, attempts=4, reason="network", latency_s=2.0)
    frame, vectors = predictions.decisions_frame(["a", "b"], {"a": ok, "b": bad}, list(OPTIONS))
    assert frame["row_id"].tolist() == ["a", "b"]
    assert frame["pred"].tolist() == ["lost_card", ""] and frame["failed"].tolist() == [False, True]
    assert frame["top_prob"].tolist() == [0.7, 0.0]
    assert frame["input_tokens"].tolist()[0] == 900 and frame["cost"].tolist() == [0.0002, 0.0003]
    assert vectors.shape == (2, 3) and vectors[0].tolist() == [0.1, 0.7, 0.2] and np.isnan(vectors[1]).all()
    assert not any(isinstance(v, (list, dict, np.ndarray)) for v in frame.to_numpy().ravel())
    with pytest.raises(ValueError, match="no decision"):
        predictions.decisions_frame(["a", "c"], {"a": ok}, list(OPTIONS))
    # The small LLM gives only its pick's probability, so there is no full vector to keep.
    llm = Decision("lost_card", 0.9, {"lost_card": 0.9}, "openai/gpt-oss-20b", 0.0, False, latency_s=0.1)
    _, none = predictions.decisions_frame(["a"], {"a": llm}, list(OPTIONS))
    assert none is None


def test_old_prediction_files_without_latency_read_as_missing(tmp_path):
    import pandas as pd

    path = tmp_path / "val" / "old__banking77.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"row_id": ["a"], "pred": ["x"], "top_prob": [0.5], "failed": [False]}).to_parquet(path)
    rows, _, _ = predictions.read("old", "banking77", stage="val", root=tmp_path)
    assert list(rows.columns)[:5] == predictions.COLUMNS and np.isnan(rows["latency_s"]).all()


# --------------------------------------------------------------------------- probes


def test_local_probe_keeps_load_cold_and_warm_apart_and_runs_serially(tmp_path):
    clock = Ticker(0.5)
    log = []

    def load():
        log.append("load")
        return lambda text: log.append(text)

    swaps = iter([1.0, 1.5])
    result = latency_probe.probe_local(
        "embed_sim", load, ["t0", "t1", "t2", "t3", "t4", "t5"], warmup=2, clock=clock, read_swap_gb=lambda: next(swaps)
    )
    assert log == ["load", "t0", "t1", "t2", "t3", "t4", "t5"]  # one at a time, in order
    assert result["load_s"] > 0 and result["cold_s"] > 0
    assert result["warmup"] == 2 and result["timed"]["n"] == 3  # 1 cold + 2 warm-up are not in the timed set
    assert result["swap_gb_before"] == 1.0 and result["swap_gb_after"] == 1.5
    assert set(result["timed"]) == {"n", "p50", "p95"}


def test_api_probe_is_serial_and_reads_decision_latency(tmp_path):
    client, _, _, _ = make_client(tmp_path, lambda body, n: jev_ok(body), clock=Ticker(), max_concurrency=1)
    rows = [{"row_id": f"r{i}", "text": f"t{i}"} for i in range(6)]
    result = latency_probe.probe_api(
        "jev",
        rows,
        lambda row: jev_decide(row["row_id"], row["text"], OPTIONS, client=client),
        client=client,
        results_path=tmp_path / "runs" / "private" / "latency" / "jev.jsonl",
        warmup=2,
        read_swap_gb=lambda: 0.0,
    )
    assert result["concurrency"] == 1 and result["timed"]["n"] == 3 and result["cold_s"] > 0
    with pytest.raises(ValueError, match="serial"):
        busy, _, _, _ = make_client(tmp_path / "busy", lambda body, n: jev_ok(body))
        latency_probe.probe_api("jev", rows, lambda row: None, client=busy, results_path=tmp_path / "runs" / "private" / "b.jsonl")


def test_network_probe_sends_a_minimal_request_and_times_each(tmp_path):
    sent = []

    class Transport:
        def send(self, method, url, content=None):
            sent.append((method, url, content))
            return TransportResponse(200, {}, b'{"data": {"usage": 0}}')

    result = latency_probe.network_probe(Transport(), n=4, warmup=1, clock=Ticker())
    assert sent == [("GET", openrouter.KEY_URL, None)] * 5
    assert result["timed"]["n"] == 4 and result["what"] == "GET /api/v1/key"


def test_paid_probes_need_the_paid_flag(tmp_path):
    with pytest.raises(PaidFlagRequired):
        latency_probe.main(["--api", "jev"], runtime=latency_probe.ProbeRuntime(runs_root=tmp_path / "runs"))


def test_report_gives_p50_and_p95_by_condition_and_no_speed_ratio():
    cond = latency_probe.condition(region="ca-central", concurrency=1, hardware="test-mac", date="2026-11-05")
    report = latency_probe.build_report(
        [
            {"arm": "jev", "condition": cond, "timed": latency_probe.percentiles([0.2, 0.4, 0.3])},
            {"arm": "embed_sim", "condition": cond, "timed": latency_probe.percentiles([0.01, 0.02])},
        ]
    )
    text = json.dumps(report).lower()
    assert "ratio" not in text and "faster" not in text and "speedup" not in text
    group = report["by_condition"][0]
    assert group["condition"] == cond and {r["arm"] for r in group["arms"]} == {"jev", "embed_sim"}
    assert group["arms"][0]["timed"]["p50"] == pytest.approx(0.3)
    assert set(cond) == {"hardware", "region", "date", "concurrency"}
