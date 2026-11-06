"""Stage 0 runner, tested end to end with a fake API. Nothing here spends, reads a key or touches the network."""
import hashlib
import json
import subprocess
import threading
from pathlib import Path

import pandas as pd
import pytest
import yaml

import data
import jev_client
import labels
import prompts
import stage0
import stage0_fields as fields
from ledger import Ledger
from openrouter import Client, TransportResponse
from stage0 import Runtime, Stage0, mass_and_cut, repeat_stats, sample_rows, summarize
from test_clients import jev_ok, llm_ok

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_WORDS = ("accuracy", "d_s", "brier", "nll", "acc@", "beats", "better than", "worse than")


def synthetic_split(dataset):
    names = list(labels.label_texts(dataset, bare=False))
    per_val, per_pool = (3, 12) if dataset == "banking77" else (40, 40)
    rows = lambda part, n: [  # noqa: E731
        {"row_id": f"{dataset}:{part}:{i}", "text": f"{label} sample {part} {i}", "label": label}
        for i, label in enumerate(l for l in names for _ in range(n))
    ]
    frame = lambda part, n: pd.DataFrame(rows(part, n)).assign(norm=lambda d: d["text"].map(data.normalize))  # noqa: E731
    return data.Split(val=frame("val", per_val), pool=frame("pool", per_pool))


def h(text):
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


class FakeApi:
    """A transport that answers like the documented API, with knobs for each failure the runner must catch."""

    def __init__(self, *, accuracy=0.9, cost=0.0001, llm_cost=0.00003, top=None, llm_misses=None, flip_rows=(), jev_model=jev_client.JEV_MODEL + "-20261014",
                 mutate_jev=None, bad_fraction=0.0, bad_after=0, reject_null=False, ok_models=("openai/gpt-oss-20b", "z-ai/glm-5.3-flash")):
        self.accuracy, self.cost, self.llm_cost, self.top = accuracy, cost, llm_cost, top
        self.llm_misses = llm_misses or {}  # model -> set of texts that come back without logprobs; "all" for every row
        self.flip_rows, self.jev_model, self.mutate_jev = set(flip_rows), jev_model, mutate_jev
        self.bad_fraction, self.bad_after, self.reject_null, self.ok_models = bad_fraction, bad_after, reject_null, ok_models
        self.gold = {}
        self.calls = []
        self.seen = {}
        self.lock = threading.Lock()

    def learn(self, split):
        for frame in (split.val, split.pool):
            self.gold.update(dict(zip(frame["text"], frame["label"])))

    def send(self, method, url, content=None):
        body = json.loads(content)
        with self.lock:
            self.calls.append((url, body))
            count = self.seen[content] = self.seen.get(content, 0) + 1
        if url.endswith("decisions"):
            return self.jev(body, count)
        return self.llm(body)

    def jev(self, body, count):
        state = body["state"]
        text = state["text"] if isinstance(state, dict) else state
        criteria = list(body["questions"]["label"]["criteria"])
        if self.reject_null and any(v is None for v in body["questions"]["label"]["criteria"].values()):
            return TransportResponse(400, {}, b"{}")
        if self.bad_fraction and len(self.calls) > self.bad_after and h("bad" + text) % 100 < self.bad_fraction * 100:
            return TransportResponse(400, {}, b"{}")
        gold = self.gold.get(text.split(" sample")[0] and text, None)
        right = h("acc" + text) % 1000 < self.accuracy * 1000
        wrong = next(label for label in criteria if label != gold)
        choice = gold if right and gold in criteria else wrong
        if text in self.flip_rows and count == 2:
            choice = wrong if choice == gold else gold
        top = self.top if self.top is not None else 0.5 + (h("top" + text) % 50) / 100
        reply = jev_ok(body, choice=choice, cost=self.cost, model=self.jev_model, top=top)
        if self.mutate_jev:
            payload = json.loads(reply.body)
            self.mutate_jev(payload)
            return TransportResponse(200, {}, json.dumps(payload).encode())
        return reply

    def llm(self, body):
        model = body["model"]
        user = body["messages"][1]["content"]
        text = user.split("\n\nOptions:")[0].removeprefix("Text: ")
        reply = llm_ok(body, cost=self.llm_cost, model=model)
        miss = self.llm_misses.get(model, set())
        if miss == "all" or text in miss:
            payload = json.loads(reply.body)
            payload["choices"][0]["logprobs"] = None
            return TransportResponse(200, {}, json.dumps(payload).encode())
        return reply


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture(autouse=True)
def _no_key_or_process(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("no subprocess in these tests")

    monkeypatch.setattr(subprocess, "run", blocked)


def build(tmp_path, api=None, *, reported=None, clock=None):
    api = api or FakeApi()
    splits = {name: synthetic_split(name) for name in ("banking77", "ag_news")}
    for split in splits.values():
        api.learn(split)
    ledger_path = tmp_path / "ledger.jsonl"
    clock = clock or Clock()
    started = []

    def make_client(reconcile_grace_seconds=None):
        ledger = Ledger(ledger_path, stage="stage0")
        usage = reported(ledger, clock) if reported else ledger.confirmed
        client = Client(api, ledger, usage, sleep=lambda s: None, reconcile_grace_seconds=reconcile_grace_seconds or 0)
        client.asked_grace = reconcile_grace_seconds
        started.append(client)
        return client

    rt = Runtime(make_client=make_client, load_split=lambda d: splits[d], runs_root=tmp_path / "runs", report_dir=tmp_path / "out",
                 clock=clock, sleep=clock.sleep)
    (tmp_path / "out").mkdir()
    return rt, api, ledger_path


def run(tmp_path, api=None, **kwargs):
    rt, api, ledger_path = build(tmp_path, api, **kwargs)
    stage = Stage0(rt)
    code = stage.run_all()
    report = (tmp_path / "out" / "stage0_report.md").read_text()
    return code, report, stage, api, ledger_path


@pytest.fixture(scope="module")
def happy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("happy")
    api = FakeApi(accuracy=0.87, flip_rows=())
    return run(tmp, api)


# ------------------------------------------------------------------ prompts


def test_prompt_files_hold_the_pinned_task_questions():
    assert prompts.jev_instructions("banking77") == labels.question("banking77")
    assert prompts.jev_instructions("ag_news") == labels.question("ag_news")
    assert prompts.jev_instructions("banking77") == "Which banking support intent does the customer_message express? Pick the single best match."
    assert prompts.jev_instructions("ag_news") == "What is this news article mainly about?"


def test_small_llm_prompt_is_the_same_task_line_and_the_answer_line():
    for dataset in ("banking77", "ag_news"):
        text = prompts.small_llm_instructions(dataset)
        assert text.startswith(prompts.jev_instructions(dataset))
        assert text.endswith("Answer with the option number only.")


def test_examples_state_shape_file_matches_what_jev_client_sends():
    shape = prompts.examples_state_shape()
    from retrieval import Example

    state = jev_client._state("a text", [Example("an example", "a_label")])
    assert list(state) == shape["state_keys"]
    assert list(state["labelled_examples"][0]) == shape["example_keys"]
    assert jev_client._state("a text", None) == "a text"


def test_unknown_dataset_has_no_prompt():
    with pytest.raises(ValueError):
        prompts.jev_instructions("version")


def test_no_prompt_text_is_hard_coded_in_stage0():
    source = (ROOT / "src" / "stage0.py").read_text() + (ROOT / "src" / "stage0_fields.py").read_text()
    for line in (prompts.jev_instructions("banking77"), prompts.jev_instructions("ag_news"), "Answer with the option number only."):
        assert line not in source


def test_the_runner_sends_the_frozen_prompts(happy):
    _, _, _, api, _ = happy
    jev_instructions = {b["questions"]["label"]["instructions"] for u, b in api.calls if u.endswith("decisions")}
    assert jev_instructions == {prompts.jev_instructions("banking77"), prompts.jev_instructions("ag_news")}
    system = {b["messages"][0]["content"] for u, b in api.calls if "chat" in u}
    assert system == {prompts.small_llm_instructions("banking77")}


# ------------------------------------------------------------------ the report


def test_stage0_report_contains_no_accuracy_or_comparison_values(happy, capsys):
    code, report, stage, api, _ = happy
    assert code == stage0.EXIT_OK
    lowered = report.lower()
    for word in FORBIDDEN_WORDS:
        assert word not in lowered
    # The fake is right on 87% of rows. That figure, however written, must not appear.
    for text in ("0.87", "87%", "87.0", "0.870"):
        assert text not in report
    assert "Bug floor, Jev on Banking77 validation: PASS" in report
    assert "Overall: PASS" in report
    out = capsys.readouterr().out.lower()
    assert "accuracy" not in out and "87" not in out


def test_report_holds_the_allowed_fields(happy):
    _, report, _, _, _ = happy
    for heading in ("## Jev", "## Repeatability", "## Small LLM selection", "## Tokens, cost and latency", "## Projected Stage 1 spend",
                    "## Fields to confirm"):
        assert heading in report
    assert "1.0 mass" in report and "Score at the 80% cut" in report
    assert "Label agreement: 100.0%" in report and "Median top-probability range" in report
    assert "Latency median s" in report and "Failure rate" in report
    assert "Selected: openai/gpt-oss-20b on Darkbloom" in report


def test_report_holds_no_row_text_or_ids(happy):
    _, report, _, _, _ = happy
    assert "banking77:val" not in report and "sample val" not in report


def test_report_and_frozen_are_the_only_public_outputs(happy, tmp_path_factory):
    _, _, stage, _, _ = happy
    out = Path(stage.rt.report_dir)
    assert sorted(p.name for p in out.iterdir()) == ["frozen.yaml", "stage0_report.md"]
    private = Path(stage.rt.runs_root) / "private" / "stage0"
    assert (private / "jev_banking77_val.jsonl").exists() and (private / "jev_cache.jsonl").exists()
    assert "banking77:val" in (private / "jev_banking77_val.jsonl").read_text()


def test_frozen_yaml_records_the_selections(happy):
    _, _, stage, _, _ = happy
    frozen = yaml.safe_load((Path(stage.rt.report_dir) / "frozen.yaml").read_text())
    assert frozen["jev"]["build"] == "jev-1.14-20261014" and frozen["jev"]["request_model"] == "typesafe/jev-1.14"
    assert frozen["small_llm"]["model"] == "openai/gpt-oss-20b" and frozen["small_llm"]["provider"] == "Darkbloom"
    assert frozen["small_llm"]["provider_options"] == {"only": ["Darkbloom"], "allow_fallbacks": False, "require_parameters": True}
    assert frozen["small_llm"]["reply_model_suffix_pattern"] == r"(?:-\d{8})?"
    assert frozen["nli"] == {"model": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c", "revision": "b2730f16019076bb0009481121efbe4705e0e378",
                             "pairs_per_second": 20.7,
                             "source": "Task 4 Step 5, 2026-11-04: batch 16, 5,000 validation pairs"}
    assert frozen["stage0_complete"] is True


def test_all_22_fields_are_listed_and_confirmed_ones_say_so(happy):
    _, report, stage, _, _ = happy
    numbers = [f.number for f in stage.findings]
    assert numbers == list(range(1, 23))
    status = {f.number: f.status for f in stage.findings}
    for n in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 16, 17, 18, 21, 22):
        assert status[n] == fields.CONFIRMED, (n, status[n])
    assert status[8] == fields.NOT_TESTED and status[19] == fields.UNCONFIRMED
    assert all(f"| {n} |" in report for n in range(1, 23))
    assert "mismatch" not in report


def test_steps_run_in_the_brief_order(happy):
    _, _, stage, _, _ = happy
    names = [n for n, _, _ in stage.steps]
    assert names[:5] == ["1 Jev smoke", "1c key usage", "1b bare-name schema check", "2 small LLM selection", "3a Jev Banking77 validation"]
    assert names[5:] == ["3b Jev AG News validation", "4 repeatability", "5 Jev + examples smoke"]


def test_call_counts_match_the_plan(happy):
    _, _, _, api, _ = happy
    jev = [b for u, b in api.calls if u.endswith("decisions")]
    llm = [b for u, b in api.calls if "chat" in u]
    assert len(llm) == 50  # gpt-oss passes, so GLM is never called
    assert all(b["model"] == "openai/gpt-oss-20b" and b["provider"]["only"] == ["Darkbloom"] for b in llm)
    # smoke 20 + bare 3 + 231 + 160 validation (20 smoke rows are cached) + 300 repeats + 20 examples
    assert len(jev) == 20 + 3 + (231 - 20) + 160 + 300 + 20
    examples = [b for b in jev if isinstance(b["state"], dict)]
    assert len(examples) == 20 and all(0 < len(b["state"]["labelled_examples"]) <= 24 for b in examples)
    bare = [b for b in jev if any(v is None for v in b["questions"]["label"]["criteria"].values())]
    assert len(bare) == 3


def test_projection_is_in_the_report_with_cad(happy):
    _, report, _, _, _ = happy
    assert "Total with the 5.5% credit fee" in report and "CA$" in report
    assert "Fits under the cap: PASS" in report
    assert "routing pilot spend is not measured here" in report


# ------------------------------------------------------------------ stops


def test_stage0_stops_at_stage_cap_of_0_50(tmp_path):
    api = FakeApi(cost=0.02)
    code, report, stage, _, ledger_path = run(tmp_path, api)
    assert code == stage0.EXIT_STOPPED
    assert Ledger(ledger_path, stage="stage0").stage_cap_usd == 0.50
    spent = Ledger(ledger_path, stage="stage0").stage_spent()
    assert spent <= 0.50 + 4 * 0.02
    assert "Overall: STOPPED" in report
    assert "4 repeatability" not in [n for n, _, _ in stage.steps]


def test_the_ledger_itself_stops_a_run_whose_cost_jumps(tmp_path):
    api = FakeApi()
    real = api.send
    state = {"n": 0}

    def costly(method, url, content=None):
        state["n"] += 1
        reply = real(method, url, content)
        if state["n"] > 25 and url.endswith("decisions"):
            payload = json.loads(reply.body)
            payload["usage"]["cost"] = 0.05
            return TransportResponse(200, {}, json.dumps(payload).encode())
        return reply

    api.send = costly
    code, report, stage, _, ledger_path = run(tmp_path, api)
    assert code == stage0.EXIT_STOPPED
    assert Ledger(ledger_path, stage="stage0").stage_spent() <= 0.50 + 4 * 0.05
    assert "STOPPED" in report


def test_a_step_that_does_not_fit_is_refused_before_sending(tmp_path):
    rt, api, ledger_path = build(tmp_path)
    Ledger(ledger_path, stage="stage0").charge(0.4999, {"why": "test"})
    code = Stage0(rt).run_all()
    assert code == stage0.EXIT_STOPPED and api.calls == []
    assert "nothing sent" in (tmp_path / "out" / "stage0_report.md").read_text()


def test_each_step_checks_its_projection_against_what_is_left(tmp_path):
    rt, api, ledger_path = build(tmp_path, FakeApi(cost=0.005))
    # 20 smoke rows at 0.005 leave too little for 231 rows at the measured cost: refused, not sent.
    code = Stage0(rt).run_all()
    assert code == stage0.EXIT_STOPPED
    jev_calls = [b for u, b in api.calls if u.endswith("decisions")]
    assert len(jev_calls) < 40
    assert Ledger(ledger_path, stage="stage0").stage_spent() <= 0.50 + 1e-9


def test_an_answer_shape_that_differs_from_the_fixture_stops_the_run_early(tmp_path):
    def mutate(payload):
        payload["answers"]["label"]["probabilities"] = [0.5, 0.5]

    code, report, _, api, _ = run(tmp_path, FakeApi(mutate_jev=mutate))
    assert code == stage0.EXIT_STOPPED
    assert "differs from the expected fixture" in report and "answers.label.probabilities" in report
    assert "needs Aarmen" in report
    assert len(api.calls) <= 4  # one row, up to 4 attempts, then nothing more


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda p: p["usage"].pop("input_tokens"), "usage.input_tokens"),
        (lambda p: p.pop("answers"), "answers.label"),
        (lambda p: p["answers"]["label"].update(choice="not_a_label"), "answers.label.choice"),
    ],
)
def test_other_shape_differences_name_the_path(tmp_path, mutate, needle):
    code, report, _, api, _ = run(tmp_path, FakeApi(mutate_jev=mutate))
    assert code == stage0.EXIT_STOPPED and needle in report and len(api.calls) <= 4


def test_a_shape_difference_on_the_chat_surface_stops_before_more_spend(tmp_path):
    api = FakeApi()
    real = api.llm

    def odd(body):
        payload = json.loads(real(body).body)
        payload["choices"] = []
        return TransportResponse(200, {}, json.dumps(payload).encode())

    api.llm = odd
    code, report, _, _, _ = run(tmp_path, api)
    assert code == stage0.EXIT_STOPPED and "choices[0].message.content" in report
    assert len([1 for u, _ in api.calls if "chat" in u]) <= 4


def test_a_changed_jev_build_stops_and_asks(tmp_path):
    code, report, _, _, _ = run(tmp_path, FakeApi(jev_model="typesafe/jev-1.15-20261201"))
    assert code == stage0.EXIT_STOPPED and "ModelChanged" in report and "needs Aarmen" in report


def test_missing_probabilities_stop_and_ask(tmp_path):
    code, report, _, _, _ = run(tmp_path, FakeApi(mutate_jev=lambda p: p["answers"]["label"].pop("probabilities")))
    assert code == stage0.EXIT_STOPPED and "needs Aarmen" in report


def test_jev_one_point_zero_mass_at_75_percent_stops_and_asks(tmp_path):
    code, report, stage, _, _ = run(tmp_path, FakeApi(top=1.0))
    assert code == stage0.EXIT_STOPPED
    assert "1.0 mass on Banking77 validation is 75% or higher" in report and "needs Aarmen" in report
    assert "3b Jev AG News validation" not in [n for n, _, _ in stage.steps]
    assert "4 repeatability" not in [n for n, _, _ in stage.steps]
    assert "100.0%" in report  # the mass itself is a permitted figure


def test_mass_just_under_the_limit_passes():
    class D:
        def __init__(self, p):
            self.top_prob, self.failed = p, p is None

    rows = {i: D(1.0 if i < 74 else 0.5) for i in range(100)}
    assert mass_and_cut(rows)["mass"] == 0.74
    rows = {i: D(1.0 if i < 75 else 0.5) for i in range(100)}
    assert mass_and_cut(rows)["mass"] == 0.75


def test_failures_above_one_percent_stop_the_run(tmp_path):
    code, report, _, _, _ = run(tmp_path, FakeApi(bad_fraction=0.05))
    assert code == stage0.EXIT_STOPPED
    assert "FAIL" in report or "STOPPED" in report


def test_bug_floor_fail_stops_and_prints_only_fail(tmp_path, capsys):
    code, report, _, _, _ = run(tmp_path, FakeApi(accuracy=0.1))
    assert code == stage0.EXIT_STOPPED
    assert "Bug floor, Jev on Banking77 validation: FAIL" in report
    assert "mapping bug" in report
    for text in ("0.1", "10%", "10.0"):
        assert f"accuracy {text}" not in report.lower()
    assert "accuracy" not in report.lower()


def test_bare_names_rejected_marks_bare_runs_not_run_and_continues(tmp_path):
    code, report, stage, _, _ = run(tmp_path, FakeApi(reject_null=True))
    names = [n for n, _, _ in stage.steps]
    assert ("1b bare-name schema check", "NOT RUN") in [(n, s) for n, s, _ in stage.steps]
    assert "5 Jev + examples smoke" in names and "the Jev bare-name runs are not run" in report
    assert {f.number: f.status for f in stage.findings}[18] == fields.UNCONFIRMED


def test_bare_check_takes_no_client_argument():
    import inspect

    assert list(inspect.signature(Stage0.bare_check).parameters) == ["self"]


# ------------------------------------------------------------------ selection


def selection_calls(api, model):
    return [b for u, b in api.calls if "chat" in u and b["model"] == model]


def sel_texts(tmp_path_unused, k):
    split = synthetic_split("banking77")
    return [r["text"] for r in sample_rows(split.val, 50, "stage0:select:")][:k]


def test_first_candidate_passing_means_glm_is_never_called(happy):
    _, _, _, api, _ = happy
    assert selection_calls(api, "z-ai/glm-5.3-flash") == []


def test_one_miss_in_fifty_still_passes(tmp_path):
    api = FakeApi(llm_misses={"openai/gpt-oss-20b": set(sel_texts(None, 1))})
    code, report, stage, _, _ = run(tmp_path, api)
    assert stage.selected["requested"] == "openai/gpt-oss-20b" and stage.selected["valid"] == 49
    assert "49 of 50 usable" in report


def test_two_misses_fail_the_candidate_and_glm_is_tried_next(tmp_path):
    api = FakeApi(llm_misses={"openai/gpt-oss-20b": set(sel_texts(None, 2))})
    code, report, stage, _, _ = run(tmp_path, api)
    assert stage.selected["requested"] == "z-ai/glm-5.3-flash" and stage.selected["provider"] == "Wafer"
    assert code == stage0.EXIT_OK
    assert stage.candidates["openai/gpt-oss-20b"]["valid"] <= 48
    frozen = yaml.safe_load((Path(stage.rt.report_dir) / "frozen.yaml").read_text())
    assert frozen["small_llm"]["model"] == "z-ai/glm-5.3-flash" and frozen["small_llm"]["provider"] == "Wafer"
    assert all(b["provider"]["only"] == ["Wafer"] for b in selection_calls(api, "z-ai/glm-5.3-flash"))


def test_cheapest_candidate_is_tried_first(tmp_path):
    api = FakeApi(llm_misses={"openai/gpt-oss-20b": "all", "z-ai/glm-5.3-flash": "all"})
    run(tmp_path, api)
    models = [b["model"] for u, b in api.calls if "chat" in u]
    assert models.index("openai/gpt-oss-20b") < models.index("z-ai/glm-5.3-flash")


def test_no_candidate_passing_marks_the_arm_not_run_and_the_rest_continues(tmp_path):
    api = FakeApi(llm_misses={"openai/gpt-oss-20b": "all", "z-ai/glm-5.3-flash": "all"})
    code, report, stage, _, _ = run(tmp_path, api)
    assert stage.selected is None
    assert code == stage0.EXIT_OK
    assert "The small LLM arm is not run" in report
    assert "NOT RUN" in report and "4 repeatability" in [n for n, _, _ in stage.steps]
    frozen = yaml.safe_load((Path(stage.rt.report_dir) / "frozen.yaml").read_text())
    assert frozen["small_llm"] == {"status": "not run"}
    assert frozen["nli"]["model"].startswith("MoritzLaurer/deberta-v3-large")


def test_a_candidate_that_fails_to_return_logprobs_is_cheap(tmp_path):
    api = FakeApi(llm_misses={"openai/gpt-oss-20b": "all"})
    _, _, stage, _, _ = run(tmp_path, api)
    # The allowance is spent, then the arm stops: a handful of calls, nowhere near 50, on a hopeless candidate.
    assert len(selection_calls(api, "openai/gpt-oss-20b")) <= 8


def test_reply_model_outside_the_suffix_pattern_stops_and_asks(tmp_path):
    api = FakeApi()
    real = api.llm

    def renamed(body):
        payload = json.loads(real(body).body)
        payload["model"] = "openai/gpt-oss-20b-bf16-turbo"
        return TransportResponse(200, {}, json.dumps(payload).encode())

    api.llm = renamed
    code, report, _, _, _ = run(tmp_path, api)
    assert code == stage0.EXIT_STOPPED and "ModelChanged" in report and "needs Aarmen" in report


def test_hidden_reasoning_fails_the_candidate(tmp_path):
    api = FakeApi()
    real = api.llm

    def thinking(body):
        if body["model"].startswith("openai"):
            payload = json.loads(real(body).body)
            payload["usage"]["completion_tokens_details"]["reasoning_tokens"] = 12
            return TransportResponse(200, {}, json.dumps(payload).encode())
        return real(body)

    api.llm = thinking
    _, _, stage, _, _ = run(tmp_path, api)
    assert stage.selected["requested"] == "z-ai/glm-5.3-flash"
    assert any(f.number == 12 and f.status == fields.MISMATCH for f in stage.findings)


# ------------------------------------------------------------------ repeatability


def test_repeatability_counts_label_flips_and_flags_disclosure(tmp_path):
    split = synthetic_split("banking77")
    repeat_texts = [r["text"] for r in sample_rows(split.val, 100, "stage0:repeat:")]
    api = FakeApi(flip_rows=repeat_texts[:5])
    code, report, stage, _, _ = run(tmp_path, api)
    assert "Label agreement: 95.0%" in report
    assert "Below 98% agreement: disclose" in report
    assert code == stage0.EXIT_OK


def test_repeat_calls_are_sent_three_times_each_and_never_served_from_cache(happy):
    _, _, _, api, _ = happy
    split = synthetic_split("banking77")
    repeat_texts = {r["text"] for r in sample_rows(split.val, 100, "stage0:repeat:")}
    per_text = {}
    for u, b in api.calls:
        if u.endswith("decisions") and isinstance(b["state"], str) and b["state"] in repeat_texts:
            per_text[b["state"]] = per_text.get(b["state"], 0) + 1
    assert len(per_text) == 100 and min(per_text.values()) >= 3


def test_repeat_stats_unit():
    class D:
        def __init__(self, pred, p, failed=False):
            self.pred, self.top_prob, self.failed = pred, p, failed

    decisions = {}
    for i in range(10):
        for k in range(3):
            decisions[f"r{i}#{k}"] = D("a" if (i != 0 or k < 2) else "b", 0.8 + 0.01 * k)
    decisions["r10#0"] = D("a", 0.5)
    stats = repeat_stats(decisions)
    assert stats["rows"] == 11 and stats["complete"] == 10
    assert stats["agreement"] == 0.9
    assert stats["median_range"] == pytest.approx(0.02)


# ------------------------------------------------------------------ usage lag and fields


def lagging(delay):
    def factory(ledger, clock):
        started = {}

        def reported():
            started.setdefault("t", clock.now)
            return ledger.confirmed() if clock.now - started["t"] >= delay or clock.now >= delay else 0.0

        return reported

    return factory


def test_usage_lag_is_measured_and_reported(tmp_path):
    clock = Clock()
    stage_report = run(tmp_path, FakeApi(), reported=lambda ledger, clock_: (lambda: ledger.confirmed() if clock.now >= 9 else 0.0), clock=clock)
    code, report, stage, _, _ = stage_report
    assert stage.usage.seconds is not None and 9 <= stage.usage.seconds <= 12
    finding = {f.number: f for f in stage.findings}[16]
    assert finding.status == fields.CONFIRMED and " s" in finding.note


def test_usage_that_never_shows_spend_stops_and_asks(tmp_path):
    clock = Clock()
    code, report, stage, api, _ = run(tmp_path, FakeApi(), reported=lambda ledger, clock_: (lambda: 0.0), clock=clock)
    assert code == stage0.EXIT_STOPPED
    assert "key usage did not rise" in report and "needs Aarmen" in report
    assert len([1 for u, _ in api.calls if "decisions" in u]) == 20  # stopped before the bare check and everything after


def test_unreadable_key_usage_fails_closed(tmp_path):
    def boom():
        raise RuntimeError("no")

    clock = Clock()
    code, report, _, _, _ = run(tmp_path, FakeApi(), reported=lambda ledger, c: (lambda: 0.0 if ledger.confirmed() == 0 else boom()), clock=clock)
    assert code == stage0.EXIT_STOPPED


# ------------------------------------------------------------------ units


def test_summarize_counts_failures_and_costs():
    from openrouter import Decision

    ok = Decision("a", 0.9, {}, "m", 0.001, False, input_tokens=100, output_tokens=10)
    cached = Decision("a", 0.9, {}, "m", 0.0, False, cached=True)
    bad = Decision(None, None, {}, "m", 0.002, True, reason="malformed: no probabilities")
    s = summarize({"a": ok, "b": cached, "c": bad})
    assert s["rows"] == 3 and s["failed"] == 1 and s["cached"] == 1
    assert s["reasons"] == {"malformed": 1} and s["cost"] == pytest.approx(0.003)
    assert s["mean_in"] == 100 and s["mean_out"] == 10


def test_cut_score_ranks_failures_above_the_rest():
    class D:
        def __init__(self, p):
            self.top_prob, self.failed = p, p is None

    rows = {i: D(None if i < 10 else 1 - i / 100) for i in range(100)}
    out = mass_and_cut(rows)
    assert out["n"] == 100
    assert out["cut_score"] == pytest.approx(1 - 79 / 100)  # rank 80 overall, 70th among the 90 answered


def test_sample_rows_is_fixed_and_prefix_specific():
    split = synthetic_split("banking77")
    a = sample_rows(split.val, 20, "stage0:smoke:")
    assert a == sample_rows(split.val.sample(frac=1, random_state=3), 20, "stage0:smoke:")
    assert [r["row_id"] for r in a] != [r["row_id"] for r in sample_rows(split.val, 20, "stage0:select:")]


def test_the_recorder_keeps_structure_and_never_unsafe_names():
    reply = TransportResponse(200, {}, json.dumps({"model": "sk-or-v1-abcdef", "provider": "TypeSafe", "answers": {}, "usage": {}}).encode())
    body = json.dumps({"model": "m", "questions": {"label": {"criteria": {"a": None}}}, "provider": {"only": ["TypeSafe"]}}).encode()
    obs = fields.observe(jev_client.DECISIONS_URL, body, reply, "t")
    assert obs.model == "<unprintable>" and obs.null_criteria is True and not obs.ok_shape


# ------------------------------------------------------------------ the command line


def test_cli_refuses_to_run_without_paid(capsys):
    with pytest.raises(SystemExit) as exit_info:
        stage0.main([])
    assert exit_info.value.code == 2


def test_cli_refuses_stage1(capsys):
    with pytest.raises(SystemExit) as exit_info:
        stage0.main(["--paid", "stage1"])
    assert exit_info.value.code == 2


def test_cli_with_a_fake_runtime_runs_without_the_flag(tmp_path):
    rt, _, _ = build(tmp_path)
    assert stage0.main([], runtime=rt) == stage0.EXIT_OK


def test_the_script_runs_the_module_with_paid_passed_through():
    text = (ROOT / "scripts" / "stage0.sh").read_text()
    assert "-m stage0" in text and '"$@"' in text and "PYTHONPATH" in text


def test_stage0_never_reads_the_key_or_creates_the_real_ledger(tmp_path, monkeypatch):
    import secrets_mac

    def boom(*a, **k):
        raise AssertionError("key read")

    monkeypatch.setattr(secrets_mac, "read_key", boom)
    home = Path.home() / ".jev-benchmark"
    existed = home.exists()
    rt, _, _ = build(tmp_path)
    assert Stage0(rt).run_all() == stage0.EXIT_OK
    assert home.exists() == existed


def test_one_client_per_step_and_none_shared_across_arms(tmp_path):
    rt, api, _ = build(tmp_path)
    made = []
    real = rt.make_client

    def counting(**kwargs):
        client = real(**kwargs)
        made.append(client)
        return client

    rt.make_client = counting
    stage = Stage0(rt)
    assert stage.run_all() == stage0.EXIT_OK
    # smoke, bare check, gpt-oss selection, B77 validation, AG validation, repeatability, examples
    assert len(made) == 7 and len({id(c) for c in made}) == 7


def finding_map(stage):
    return {f.number: f for f in stage.findings}


def test_field_7_is_unconfirmed_when_a_reply_has_no_provider(tmp_path):
    _, _, stage, _, _ = run(tmp_path, FakeApi(mutate_jev=lambda p: p.pop("provider")))
    assert finding_map(stage)[7].status == fields.UNCONFIRMED


def test_field_7_is_confirmed_only_when_every_reply_names_the_pin(happy):
    _, _, stage, _, _ = happy
    assert finding_map(stage)[7].status == fields.CONFIRMED


def test_field_7_is_a_mismatch_for_another_provider():
    obs = fields.Observations()
    for surface, provider, pin in (("jev", "TypeSafe", "TypeSafe"), ("chat", "SomeoneElse", "Darkbloom")):
        obs.add(fields.Obs(surface=surface, phase="x", status=200, provider=provider, provider_only=(pin,), allow_fallbacks=False,
                           ok_shape=True, model="m", cost=0.1))
    assert {f.number: f for f in fields.evaluate(obs, None, {})}[7].status == fields.MISMATCH


def test_request_side_fields_say_so(happy):
    _, _, stage, _, _ = happy
    for n in (6, 9, 17, 18):
        assert "request-side plus HTTP 200 only" in finding_map(stage)[n].note


def test_frozen_small_llm_is_pending_when_stage0_stopped_before_selection(tmp_path):
    code, _, stage, _, _ = run(tmp_path, FakeApi(mutate_jev=lambda p: p["usage"].pop("input_tokens")))
    frozen = yaml.safe_load((Path(stage.rt.report_dir) / "frozen.yaml").read_text())
    assert code == stage0.EXIT_STOPPED and frozen["small_llm"] == {"status": "pending"}


def test_frozen_small_llm_is_pending_when_stopped_mid_selection(tmp_path):
    api = FakeApi()
    real = api.llm
    api.llm = lambda body: TransportResponse(200, {}, json.dumps({**json.loads(real(body).body), "choices": []}).encode())
    _, _, stage, _, _ = run(tmp_path, api)
    frozen = yaml.safe_load((Path(stage.rt.report_dir) / "frozen.yaml").read_text())
    assert frozen["small_llm"] == {"status": "pending"}


def test_a_shape_difference_on_ag_news_stops_after_one_row(tmp_path):
    def mutate(payload):
        if len(payload["answers"]["label"]["probabilities"]) == 4:
            payload["usage"].pop("output_tokens")

    code, report, _, api, _ = run(tmp_path, FakeApi(mutate_jev=mutate))
    four = [b for u, b in api.calls if u.endswith("decisions") and len(b["questions"]["label"]["criteria"]) == 4]
    assert code == stage0.EXIT_STOPPED and "usage.output_tokens" in report
    assert 1 <= len(four) <= 4


def test_failure_rate_is_checked_every_100_rows_and_stops_early(tmp_path):
    api = FakeApi(bad_fraction=0.06, bad_after=40)
    code, report, stage, _, _ = run(tmp_path, api)
    assert code == stage0.EXIT_STOPPED and "so far, above 1%" in report
    banking_texts = {b["state"] for u, b in api.calls if u.endswith("decisions") and isinstance(b["state"], str)}
    assert "3b Jev AG News validation" not in [n for n, _, _ in stage.steps]
    assert len(banking_texts) < 20 + 3 + 231 + 100  # stopped within the first 100-row chunk after smoke, not at the end


def test_an_outage_stop_on_a_candidate_is_reported_as_an_outage(tmp_path):
    api = FakeApi()
    real = api.llm
    api.llm = lambda body: TransportResponse(500, {}, b"{}") if body["model"].startswith("openai") else real(body)
    _, report, stage, _, _ = run(tmp_path, api)
    assert stage.candidates["openai/gpt-oss-20b"]["outage"] is True
    assert "stopped by an outage" in report
    assert stage.selected["requested"] == "z-ai/glm-5.3-flash"


def test_frozen_nli_records_where_the_rate_was_measured(happy):
    _, _, stage, _, _ = happy
    frozen = yaml.safe_load((Path(stage.rt.report_dir) / "frozen.yaml").read_text())
    assert frozen["nli"]["pairs_per_second"] == 20.7 and "Task 4 Step 5" in frozen["nli"]["source"]


# ------------------------------------------------------------------ fix round 3: B2, R7, R8


def test_reasoning_on_the_first_contact_row_only_still_fails_the_candidate(tmp_path):
    """attack9 on the real runner: a stop on first contact must hold for the rest of that candidate's arm."""
    api = FakeApi()
    real = api.llm
    seen = {"openai": 0}

    def first_row_thinks(body):
        reply = real(body)
        if body["model"].startswith("openai"):
            seen["openai"] += 1
            if seen["openai"] == 1:
                payload = json.loads(reply.body)
                payload["usage"]["completion_tokens_details"]["reasoning_tokens"] = 12
                return TransportResponse(200, {}, json.dumps(payload).encode())
        return reply

    api.llm = first_row_thinks
    _, _, stage, api, _ = run(tmp_path, api)
    gpt = stage.candidates["openai/gpt-oss-20b"]
    assert gpt["passed"] is False and gpt["stop"] == "HiddenReasoning" and gpt["valid"] < stage0.SELECT_PASS
    assert len(selection_calls(api, "openai/gpt-oss-20b")) == 1  # the other 49 rows were never sent
    assert stage.selected["requested"] == "z-ai/glm-5.3-flash"


def test_a_jev_first_contact_arm_not_run_is_not_re_sent(tmp_path, monkeypatch):
    from openrouter import ArmNotRun

    real = stage0.jev_decide
    count = {"n": 0}

    def first_fails(*args, **kwargs):
        decision = real(*args, **kwargs)
        count["n"] += 1
        if count["n"] == 1:
            raise ArmNotRun("first contact could not be scored")
        return decision

    monkeypatch.setattr(stage0, "jev_decide", first_fails)
    code, report, stage, api, _ = run(tmp_path, FakeApi())
    assert code == stage0.EXIT_STOPPED
    assert len([1 for u, _ in api.calls if u.endswith("decisions")]) == 1
    assert "ArmNotRun" in report


def test_grace_from_lag_rounds_up_with_floor_and_cap():
    assert stage0.grace_from_lag(0.0) == 30 and stage0.grace_from_lag(29.1) == 30
    assert stage0.grace_from_lag(44.2) == 45 and stage0.grace_from_lag(180.0) == 180
    assert stage0.grace_from_lag(500.0) == stage0.GRACE_CAP_SECONDS == 180


@pytest.mark.parametrize("lag, grace", [(9, 30), (44, 45)])
def test_measured_lag_sets_the_grace_for_every_later_client_and_frozen_yaml(tmp_path, lag, grace):
    clock = Clock()
    rt, api, _ = build(tmp_path, FakeApi(), reported=lambda ledger, clock_: (lambda: ledger.confirmed() if clock.now >= lag else 0.0), clock=clock)
    asked = []
    real = rt.make_client

    def recording(**kwargs):
        asked.append(kwargs.get("reconcile_grace_seconds"))
        return real(**kwargs)

    rt.make_client = recording
    stage = Stage0(rt)
    assert stage.run_all() == stage0.EXIT_OK
    assert stage.grace == grace == stage0.grace_from_lag(stage.usage.seconds)
    assert asked[0] is None and asked[1:] == [grace] * (len(asked) - 1) and len(asked) == 7  # smoke measures; the rest use it
    frozen = yaml.safe_load((tmp_path / "out" / "frozen.yaml").read_text())
    assert frozen["key_usage"]["reconcile_grace_seconds"] == grace
    assert frozen["key_usage"]["lag_seconds"] == pytest.approx(stage.usage.seconds, abs=0.1)


def test_frozen_yaml_has_no_grace_when_the_lag_was_not_measured(tmp_path):
    clock = Clock()
    run(tmp_path, FakeApi(), reported=lambda ledger, clock_: (lambda: 0.0), clock=clock)
    frozen = yaml.safe_load((tmp_path / "out" / "frozen.yaml").read_text())
    assert frozen["key_usage"] == {"lag_seconds": None, "reconcile_grace_seconds": None}


def test_paid_client_factory_passes_the_grace_and_accepts_only_once(monkeypatch):
    calls = []
    monkeypatch.setattr(stage0, "build_client", lambda paid, **kw: calls.append((paid, kw)) or object())
    make = stage0.paid_client_factory("stage0", 0.25)
    make()
    make(reconcile_grace_seconds=45)
    assert calls == [("stage0", {"accept_key_usage_usd": 0.25, "reconcile_grace_seconds": 30}),
                     ("stage0", {"accept_key_usage_usd": None, "reconcile_grace_seconds": 45})]


from test_clients import paid  # noqa: E402,F401  the real paid route over httpx.MockTransport, key reader stubbed


def test_rerun_with_accept_when_the_ledger_agrees_stops_cleanly_and_sends_nothing(tmp_path, paid, capsys):
    """R8: on the real paid route, a second --accept-key-usage-usd is a clean stop message, not a traceback."""
    stage0.paid_client_factory("stage0")()  # an earlier run recorded the baseline; the ledger agrees with the key
    splits = {name: synthetic_split(name) for name in ("banking77", "ag_news")}
    (tmp_path / "out").mkdir()
    rt = Runtime(make_client=stage0.paid_client_factory("stage0", 0.0), load_split=lambda d: splits[d],
                 runs_root=tmp_path / "runs", report_dir=tmp_path / "out", clock=Clock(), sleep=lambda s: None)
    assert Stage0(rt).run_all() == stage0.EXIT_STOPPED
    out = capsys.readouterr().out
    assert "STOPPED at 1 Jev smoke" in out and "nothing to accept" in out and "Traceback" not in out
    assert paid["posts"] == []
    assert "nothing to accept" in (tmp_path / "out" / "stage0_report.md").read_text()
