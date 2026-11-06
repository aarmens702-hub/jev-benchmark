"""Stage 1 runner, end to end with fakes only: a fake API, encoder, NLI scorer and classifiers.

Nothing here spends, reads a key, touches the network or loads test data. The validation run ends in the
real scorer's dry run, so arm names and row order are checked by the code that will score them.
"""
import hashlib
import io
import json
import shutil
import threading
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import arms_local
import data
import firewall
import openrouter
import predictions
import routing
import score
import stage1
from labels import label_texts
from ledger import Ledger
from openrouter import Client, TransportResponse
from stage1 import EXIT_OK, EXIT_REFUSED, EXIT_STOPPED, PaidHalt, Runtime, Stage1
from test_arms_local import FakeEncoder, fake_scorer, keyword
from test_clients import jev_ok, llm_ok
from test_firewall import FakeDownloader, make_repo
from test_score import FAST

GOLD_MARK = "GOLD-"
GRACE = 77  # frozen.yaml's grace in these tests, so the runner is seen to use it
LLM_MODEL, LLM_PROVIDER = "mistralai/mistral-nemo", "DekaLLM"
NOW = datetime(2026, 11, 6, tzinfo=timezone.utc)


def h(text):
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


# --------------------------------------------------------------------------- synthetic data


def frame(dataset, part, per_class, start=0, mark=""):
    rows = []
    for name in label_texts(dataset, bare=False):
        for i in range(per_class):
            rows.append({"row_id": f"{dataset}:train:{part}{len(rows) + start}", "text": f"{keyword(name)} {part} message {i}", "label": mark + name})
    out = pd.DataFrame(rows)
    out["norm"] = out["text"].map(data.normalize)
    return out


def make_splits(mark_val=""):
    """Banking77 and AG News with every real class name. `mark_val` poisons validation labels."""
    return {
        "banking77": data.Split(val=frame("banking77", "v", 1, mark=mark_val), pool=frame("banking77", "p", 5)),
        "ag_news": data.Split(val=frame("ag_news", "v", 5, mark=mark_val), pool=frame("ag_news", "p", 20)),
    }


def unpoisoned(split):
    """The validation split with true labels: routing's generator checks may read validation gold."""
    return data.Split(val=split.val.assign(label=split.val["label"].str.removeprefix(GOLD_MARK)), pool=split.pool)


def clean_val_gold(splits):
    return {d: pd.DataFrame({"row_id": sp.val["row_id"], "label": sp.val["label"].str.removeprefix(GOLD_MARK)}) for d, sp in splits.items()}


def write_frozen(path):
    frozen = {
        "stage0_complete": True,
        "jev": {"request_model": "typesafe/jev-1.14", "provider": "TypeSafe", "build": "jev-1.14-20261014"},
        "small_llm": {"status": "selected", "model": LLM_MODEL, "provider": LLM_PROVIDER, "reasoning_field_sent": False},
        "nli": {"model": arms_local.NLI_MODELS["large"][0], "revision": arms_local.NLI_MODELS["large"][1], "pairs_per_second": 20.7},
        "key_usage": {"lag_seconds": 24.2, "reconcile_grace_seconds": GRACE},
    }
    path.write_text(yaml.safe_dump(frozen))
    return path


STAGE0_COSTS = {
    "jev_banking77_val": (0.0001, 2369),
    "jev_ag_news_val": (0.00002, 417),
    "jev_banking77_examples": (0.00013, 3178),
    "jev_banking77_bare": (0.00009, 1500),
    f"select_{LLM_MODEL.replace('/', '_')}": (0.00005, 1743),
}


def write_stage0(folder, costs=STAGE0_COSTS):
    folder.mkdir(parents=True, exist_ok=True)
    for name, (cost, tokens) in costs.items():
        lines = [json.dumps({"row_id": f"r{i}", "status": "done", "cost": cost, "cached": False, "input_tokens": tokens}) for i in range(5)]
        (folder / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
    return folder


# --------------------------------------------------------------------------- fakes


class FakeApi:
    """Answers like the documented API. Records every request; can fail every request for a while (an outage)."""

    def __init__(self, *, routing_hook=None, jev_cost=0.0001, llm_cost=0.00003):
        self.calls, self.ok = [], {}
        self.leaks = []
        self.outage_after = None
        self.routing_hook = routing_hook
        self.jev_cost, self.llm_cost = jev_cost, llm_cost
        self.lock = threading.Lock()

    def send(self, method, url, content=None):
        if method == "GET":
            return TransportResponse(200, {}, json.dumps({"data": {"usage": 0.0}}).encode())
        body = json.loads(content)
        with self.lock:
            if GOLD_MARK in content.decode():
                self.leaks.append(url)
            self.calls.append((url, body))
            if self.outage_after is not None and sum(self.ok.values()) >= self.outage_after:
                return TransportResponse(503, {}, b"{}")
            self.ok[content] = self.ok.get(content, 0) + 1
        if url == openrouter.DECISIONS_URL:
            criteria = list(body["questions"]["label"]["criteria"])
            if set(criteria) == set(routing.QUEUES) and self.routing_hook is not None:
                self.routing_hook()
            state = body["state"]
            text = state["text"] if isinstance(state, dict) else state
            return jev_ok(body, choice=criteria[h(text) % len(criteria)], cost=self.jev_cost, top=0.5 + (h("t" + text) % 50) / 100)
        return llm_ok(body, cost=self.llm_cost, model=body["model"])

    def kind(self, body):
        """Which arm a request belongs to, from its shape alone."""
        if "messages" in body:
            return "llm"
        criteria = body["questions"]["label"]["criteria"]
        if set(criteria) == set(routing.QUEUES):
            return "jev routing"
        if isinstance(body["state"], dict):
            return "jev examples"
        bare = all(v is None for v in criteria.values())
        return f"jev{' bare' if bare else ''} {'ag_news' if len(criteria) == 4 else 'banking77'}"


class Watch:
    """Fake models that record a leak if a validation label (marked GOLD-) ever reaches them."""

    def __init__(self):
        self.leaks = []

    def check_texts(self, texts):
        if any(GOLD_MARK in t for t in texts):
            self.leaks.append("text")

    def check_labels(self, y):
        if any(str(v).startswith(GOLD_MARK) for v in y):
            self.leaks.append("label")

    def encoder(self):
        inner = FakeEncoder()

        def encode(texts):
            self.check_texts(texts)
            return inner(texts)

        return encode

    def scorer(self):
        inner = fake_scorer()

        def score_pairs(pairs):
            self.check_texts([p for p, _ in pairs])
            return inner(pairs)

        return score_pairs

    def _centroids(self, X, y, classes):
        self.check_labels(y)
        X = np.asarray(X, dtype=float)
        cent = np.stack([X[y == c].mean(axis=0) if (y == c).any() else np.zeros(X.shape[1]) for c in classes])
        return lambda T: arms_local.softmax(np.asarray(T, dtype=float) @ cent.T * 5)

    def lr_fitted(self, train, c=100, classes=None, **_):
        X, y = train
        return self._centroids(X, np.asarray(y), list(classes))

    def lr_cv_fitted(self, train, s, classes=None):
        return self.lr_fitted(train, classes=classes)

    def tfidf_lr_fitted(self, train, c=100, classes=None):
        texts, y = train
        encode = FakeEncoder()
        predict = self._centroids(encode(texts), np.asarray(y), list(classes))
        return lambda t: predict(encode(list(t)))


def make_client_factory(api, ledger_path, *, cap=7.15, graces=None):
    def make_client(*, max_concurrency=4, reconcile_grace_seconds):
        if graces is not None:
            graces.append(reconcile_grace_seconds)
        ledger = Ledger(ledger_path, cap_usd=cap, stage="stage1")
        return Client(api, ledger, ledger.confirmed, sleep=lambda s: None, reconcile_grace_seconds=0, max_concurrency=max_concurrency)

    return make_client


def build(tmp, splits, *, api=None, cap=7.15, graces=None, stage0_costs=STAGE0_COSTS, **overrides):
    api = api or FakeApi()
    watch = Watch()
    rt = Runtime(
        make_client=make_client_factory(api, tmp / "ledger.jsonl", cap=cap, graces=graces),
        runs_root=tmp / "runs",
        frozen_path=write_frozen(tmp / "frozen.yaml"),
        stage0_dir=write_stage0(tmp / "stage0", stage0_costs),
        deviations_path=tmp / "deviations.md",
        load_split=lambda d: splits[d],
        load_test_text=lambda d: (_ for _ in ()).throw(AssertionError("the validation run must not load test")),
        test_file_hash=lambda d: "f" * 64,
        make_encoder=watch.encoder,
        make_nli_scorer=lambda key: watch.scorer(),
        lr_fitted=watch.lr_fitted,
        lr_cv_fitted=watch.lr_cv_fitted,
        tfidf_lr_fitted=watch.tfidf_lr_fitted,
        clock=lambda: 0.0,
        read_swap_gb=lambda: 1.0,
        read_free_disk_gb=lambda path=".": 500.0,
        hardware=lambda: "test-mac",
        dry_run=lambda root: score.main(["--dry-run"], runs_root=root, load_val_gold=lambda: clean_val_gold(splits), settings=FAST,
                                        load_val_text=lambda: splits["banking77"].val[["row_id", "text"]], validity_path=tmp / "validity.json"),
        validity_path=tmp / "validity.json",
        routing_validity=lambda split, encode: routing.run_validity(unpoisoned(split), encode),
        now=lambda: NOW,
    )
    for key, value in overrides.items():
        setattr(rt, key, value)
    (tmp / "deviations.md").write_text("# Deviations\n\n| Date | What changed | Reason | Stage | Approved by |\n|---|---|---|---|---|\n")
    return rt, api, watch


def record_run_arm(monkeypatch, order):
    real = openrouter.run_arm

    def recording(rows, decide, **kwargs):
        order.append(Path(kwargs["results_path"]).stem)
        return real(rows, decide, **kwargs)

    monkeypatch.setattr(openrouter, "run_arm", recording)


PAID_ORDER = [
    "jev", "small_llm",  # latency probes first
    "jev__banking77", "jev__ag_news", "jev_examples__banking77",
    "routing_jev_rule_A__banking77", "routing_jev_rule_B__banking77",
    "routing_small_llm_rule_A__banking77", "routing_small_llm_rule_B__banking77",
    "small_llm__banking77", "small_llm__ag_news", "jev_bare__banking77", "jev_bare__ag_news",
]  # fmt: skip


# --------------------------------------------------------------------------- the validation run


@pytest.fixture(scope="module")
def val_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("val")
    splits = make_splits(mark_val=GOLD_MARK)
    fingerprint_at_first_routing = []
    graces, order = [], []
    path = tmp / "runs" / "val_routing" / "fingerprint.json"
    api = FakeApi(routing_hook=lambda: fingerprint_at_first_routing.append(path.exists()))
    rt, api, watch = build(tmp, splits, api=api, graces=graces)
    with pytest.MonkeyPatch.context() as mp:
        record_run_arm(mp, order)
        out = io.StringIO()
        with redirect_stdout(out):
            code = stage1.main(["--stage", "val"], runtime=rt)
    return {"tmp": tmp, "rt": rt, "api": api, "watch": watch, "code": code, "out": out.getvalue(), "splits": splits,
            "graces": graces, "order": order, "fingerprint_at_first_routing": fingerprint_at_first_routing}  # fmt: skip


def test_validation_run_ends_with_the_real_dry_run_passing(val_run):
    assert val_run["code"] == EXIT_OK
    assert val_run["out"].endswith("DRY RUN OK\n")


def test_arm_names_and_row_order_are_what_the_scorer_takes(val_run):
    runs, splits = val_run["tmp"] / "runs", val_run["splits"]
    assert predictions.read_not_run("val", runs) == {}
    for d in stage1.DATASETS:
        want = splits[d].val["row_id"].tolist()
        names = list(score.expected_single_arms(d)) + [n for f in score.FAMILIES for _, _, n in score.family_files(f)]
        for name in names:
            rows, _, _ = predictions.read(name, d, "val", runs)
            assert rows["row_id"].tolist() == want, name
    for name in ("jev", "jev_bare", "jev_examples", "small_llm", "nli"):
        assert predictions.paths(name, "banking77", "val", runs)[0].exists()


def test_paid_arms_run_in_the_frozen_order_after_the_latency_probes(val_run):
    assert list(dict.fromkeys(val_run["order"])) == PAID_ORDER


def test_each_paid_arm_has_its_own_client_with_the_frozen_grace(val_run):
    # 2 API probes, the network probe, then 13 arms: one client each.
    assert val_run["graces"] == [GRACE] * (3 + len(stage1.paid_plan()))


def test_routing_fingerprint_is_recorded_before_the_first_routing_output(val_run):
    path = val_run["tmp"] / "runs" / "val_routing" / "fingerprint.json"
    record = json.loads(path.read_text())
    assert record["fingerprint"] == routing.rule_fingerprint()
    assert record["first_arm"] == "jev_rule_A"
    assert val_run["fingerprint_at_first_routing"] and val_run["fingerprint_at_first_routing"][0] is True
    runs = val_run["tmp"] / "runs"
    n = len(val_run["splits"]["banking77"].val)
    for name in ("jev_rule_A", "jev_rule_B", "small_llm_rule_A", "small_llm_rule_B", "nli_rule_A", "nli_rule_B", "lr_a",
                 "intent_lr_code_rule_A", "intent_lr_code_rule_B"):  # fmt: skip
        rows, _, _ = predictions.read(name, "banking77", "val_routing", runs)
        assert len(rows) == score.DRY_RUN_ROUTING_ROWS, name  # validation runs a 20-row smoke of each routing arm


def test_validation_run_scores_routing_and_freezes_the_validity_checks(val_run):
    results = json.loads((val_run["tmp"] / "runs" / "dryrun" / "results" / "results.json").read_text())
    assert results["routing"]["n_rows"] == score.DRY_RUN_ROUTING_ROWS and results["routing"]["not_run"] == []
    validity = json.loads((val_run["tmp"] / "validity.json").read_text())
    assert set(validity["checks"]) == set(score.VALIDITY_KEYS) and validity["fingerprint"] == routing.rule_fingerprint()
    assert (val_run["tmp"] / "runs" / "val" / stage1.DRY_RUN_OK).exists()


def test_no_validation_label_reaches_a_predictor(val_run):
    assert val_run["api"].leaks == [] and val_run["watch"].leaks == []


def test_validation_banks_exclude_validation_rows(val_run):
    sizes = json.loads((val_run["tmp"] / "runs" / "val" / "bank_sizes.json").read_text())
    for d, sp in val_run["splits"].items():
        assert sizes[d]["d0_sall"]["rows"] == len(sp.pool)
        assert set(sizes[d]) == {f"d{r}_s{s}" for r, s, _ in score.family_files("lr")}


def test_latency_probes_are_written_for_the_validation_run(val_run):
    lines = (val_run["tmp"] / "runs" / "val" / "latency" / "probes.jsonl").read_text().splitlines()
    arms = [json.loads(line)["arm"] for line in lines]
    assert arms == ["jev", "small_llm", "network", "embed_sim", "lr", "tfidf_lr", "nli"]
    assert not (val_run["tmp"] / "runs" / "latency").exists()  # only the test run writes where the scorer reads


def test_a_rerun_skips_everything_and_pays_nothing(val_run, capsys):
    api, rt = val_run["api"], val_run["rt"]
    calls = len(api.calls)
    spent = Ledger(val_run["tmp"] / "ledger.jsonl").spent()
    assert stage1.main(["--stage", "val"], runtime=rt) == EXIT_OK
    assert len(api.calls) == calls
    assert Ledger(val_run["tmp"] / "ledger.jsonl").spent() == spent
    assert capsys.readouterr().out.endswith("DRY RUN OK\n")


# --------------------------------------------------------------------------- paid-only runs


def paid_only(tmp, splits, **kwargs):
    rt, api, _ = build(tmp, splits, **kwargs)
    runner = Stage1(rt, "val")
    runner.prepare()
    runner.root.mkdir(parents=True, exist_ok=True)
    runner.write_records()
    return runner, api


def test_projection_stop_marks_this_arm_and_every_later_one_not_run(tmp_path):
    costs = dict(STAGE0_COSTS, jev_ag_news_val=(1.0, 417))  # 20 rows at US$1.25 projected cannot fit the cap
    runner, api = paid_only(tmp_path, make_splits(), stage0_costs=costs)
    runner.run_paid()
    runs = tmp_path / "runs"
    assert predictions.paths("jev", "banking77", "val", runs)[0].exists()
    logged = {**predictions.read_not_run("val", runs), **predictions.read_not_run("val_routing", runs)}
    later = stage1.paid_plan()[1:]
    assert set(logged) == {(a.name, a.dataset) for a in later}
    assert all("left under the cap" in logged[(a.name, a.dataset)] for a in later)  # the runner's own projection, before any send
    sent = [api.kind(body) for _, body in api.calls]
    assert "jev ag_news" not in sent and "jev examples" not in sent and "jev routing" not in sent
    assert sent.count("llm") == stage1.PROBE_ROWS  # the small LLM probe only, which ran first


def test_a_halt_resumes_without_paying_twice(tmp_path):
    splits = make_splits()
    runner, api = paid_only(tmp_path, splits)
    api.outage_after = 150  # the probes (100 rows) pass, then the service fails part-way through Jev Banking77
    with pytest.raises(PaidHalt, match="ConsecutiveFailures"):
        runner.run_paid()
    runs = tmp_path / "runs"
    assert not predictions.paths("jev", "banking77", "val", runs)[0].exists()  # a partial arm writes no prediction file
    api.outage_after = None
    runner.run_paid()
    assert max(api.ok.values()) == 1  # no request was answered (and billed) twice
    for arm in stage1.paid_plan():
        stage = "val_routing" if arm.routing else "val"
        assert predictions.paths(arm.name, arm.dataset, stage, runs)[0].exists(), arm.label
    rows, _, _ = predictions.read("jev", "banking77", "val", runs)
    assert rows["row_id"].tolist() == splits["banking77"].val["row_id"].tolist()
    paid = sum(1 for _ in api.ok)
    runner.run_paid()  # everything finished: nothing is sent
    assert sum(1 for _ in api.ok) == paid


def test_without_the_paid_flag_paid_arms_wait(tmp_path):
    runner, api = paid_only(tmp_path, make_splits(), make_client=None)
    with pytest.raises(PaidHalt, match="--paid stage1"):
        runner.run_paid()
    assert api.calls == []


# --------------------------------------------------------------------------- the test stage


def val_ready(val_run, tmp):
    """A copy of the finished validation run and its routing checks, which the test stage requires."""
    shutil.copytree(val_run["tmp"] / "runs", tmp / "runs")
    shutil.copy(val_run["tmp"] / "validity.json", tmp / "validity.json")
    return make_splits()  # same rows and text as the validation run, true labels


def test_test_stage_refuses_without_the_freeze_token(val_run, tmp_path, capsys):
    made = []
    rt, api, _ = build(tmp_path, val_ready(val_run, tmp_path), load_test_text=stage1._test_text)
    real_make = rt.make_client
    rt.make_client = lambda **kw: made.append(1) or real_make(**kw)
    assert stage1.main(["--stage", "test"], runtime=rt) == EXIT_REFUSED
    assert "firewall" in capsys.readouterr().out
    assert not (tmp_path / "runs" / "stage1").exists() and made == [] and api.calls == []


def test_test_stage_refuses_a_wrong_token_before_downloading(val_run, tmp_path):
    splits = val_ready(val_run, tmp_path)
    repo = make_repo(tmp_path)
    downloader = FakeDownloader(tmp_path)
    load = lambda d: firewall._load_test(d, "0" * 64, repo_dir=repo, downloader=downloader)[0]  # noqa: E731
    rt, api, _ = build(tmp_path, splits, load_test_text=load)
    assert stage1.main(["--stage", "test"], runtime=rt) == EXIT_REFUSED
    assert downloader.calls == [] and api.calls == [] and not (tmp_path / "runs" / "stage1").exists()


def test_test_stage_refuses_until_the_validation_run_passed(val_run, tmp_path, capsys):
    loads = []
    splits = val_ready(val_run, tmp_path)
    (tmp_path / "runs" / "val" / stage1.DRY_RUN_OK).unlink()
    rt, api, _ = build(tmp_path, splits, load_test_text=lambda d: loads.append(d))
    assert stage1.main(["--stage", "test"], runtime=rt) == EXIT_REFUSED
    assert "dry run" in capsys.readouterr().out and loads == []
    # The marker alone is not enough: every validation arm must be there too.
    (tmp_path / "runs" / "val" / stage1.DRY_RUN_OK).write_text("{}")
    predictions.paths("nli", "ag_news", "val", tmp_path / "runs")[0].unlink()
    assert stage1.main(["--stage", "test"], runtime=rt) == EXIT_REFUSED
    assert "nli__ag_news" in capsys.readouterr().out and loads == []


class Poison:
    """Stands in for the gold frame. Any use at all is a failure."""

    touched = []

    def __getattribute__(self, name):
        Poison.touched.append(name)
        raise AssertionError("the runner touched gold")


def test_the_firewall_loader_keeps_text_and_drops_gold_unread(monkeypatch):
    texts = pd.DataFrame({"row_id": [0, 1], "text": ["a", "b"]})
    seen = []
    monkeypatch.setattr(score, "freeze_token", lambda: "token from the one helper")
    monkeypatch.setattr(firewall, "load_test", lambda d, token: seen.append((d, token)) or (texts, Poison()))
    Poison.touched.clear()
    assert stage1._test_text("banking77") is texts
    assert Poison.touched == [] and seen == [("banking77", "token from the one helper")]


def test_the_freeze_token_has_one_helper():
    import ast

    for name in ("stage1.py", "score.py"):
        tree = ast.parse((Path(stage1.__file__).parent / name).read_text())
        users = {f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
                 and any(isinstance(n, ast.Attribute) and n.attr == "freeze_token" and getattr(n.value, "id", "") == "firewall" for n in ast.walk(f))}
        assert users <= {"freeze_token"}, name


def make_test_frames(splits):
    """Test text: one fresh row per class, plus rows repeating pool and validation text."""
    texts, gold = {}, {}
    dup = {"banking77": (3, 2), "ag_news": (1, 0)}
    for d, sp in splits.items():
        rows = [(f"{keyword(n)} test message {i}", n) for i, n in enumerate(label_texts(d, bare=False))]
        n_pool, n_val = dup[d]
        rows += list(zip(sp.pool["text"][:n_pool], sp.pool["label"][:n_pool]))
        rows += list(zip(sp.val["text"][:n_val], sp.val["label"][:n_val]))
        texts[d] = pd.DataFrame({"row_id": range(len(rows)), "text": [t for t, _ in rows]})
        gold[d] = pd.DataFrame({"row_id": range(len(rows)), "label": [g for _, g in rows]})
    return texts, gold, dup


@pytest.fixture(scope="module")
def test_run(tmp_path_factory, val_run):
    tmp = tmp_path_factory.mktemp("test")
    splits = val_ready(val_run, tmp)
    texts, gold, dup = make_test_frames(splits)
    scored, banks = [], []

    def score_test(root):
        # What score.py checks first: every arm present, in gold's row order. Gold appears only here.
        assert (tmp / "deviations.md").read_text() == initial, "deviations.md changed before scoring"
        logged = predictions.read_not_run("stage1", root)
        for d in stage1.DATASETS:
            score.check_row_order(score.gold_names(gold[d], None), score.load_arms("stage1", d, root, logged))
        scored.append(root)
        return 0

    rt, api, watch = build(tmp, splits, load_test_text=lambda d: texts[d], score_test=score_test)
    initial = (tmp / "deviations.md").read_text()
    real_bank = data.bank
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data, "bank", lambda *a, **k: banks.append(real_bank(*a, **k)) or banks[-1])
        code = stage1.main(["--stage", "test"], runtime=rt)
    return {"tmp": tmp, "code": code, "scored": scored, "splits": splits, "texts": texts, "dup": dup, "api": api, "watch": watch,
            "banks": banks}  # fmt: skip


def test_test_stage_runs_text_only_and_hands_over_to_the_scorer(test_run):
    assert test_run["code"] == EXIT_OK
    assert len(test_run["scored"]) == 1


def test_duplicates_are_dropped_from_pool_and_validation_before_any_bank(test_run):
    runs, splits, dup = test_run["tmp"] / "runs", test_run["splits"], test_run["dup"]
    record = json.loads((runs / "stage1" / "test_record.json").read_text())["datasets"]
    sizes = json.loads((runs / "stage1" / "bank_sizes.json").read_text())
    for d, (n_pool, n_val) in dup.items():
        assert record[d]["duplicates_removed"] == {"pool": n_pool, "validation": n_val}
        assert record[d]["text_sha256"] and record[d]["file_sha256"] == "f" * 64
        # s=all adds the validation rows back, both sides already de-duplicated.
        assert sizes[d]["d0_sall"]["rows"] == len(splits[d].pool) - n_pool + len(splits[d].val) - n_val
    # Jev + examples ran to the end: retrieval refuses any bank row equal to a test row.
    assert predictions.paths("jev_examples", "banking77", "stage1", runs)[0].exists()


def test_test_stage_records_hashes_duplicates_and_writes_probes_where_the_scorer_reads(test_run):
    text = (test_run["tmp"] / "deviations.md").read_text()
    assert "Test files loaded and hashed" in text and "f" * 64 in text
    assert "banking77: 3 pool, 2 validation" in text
    assert (test_run["tmp"] / "runs" / "latency" / "probes.jsonl").exists()


def sent_text(body):
    """The row text a request carries: Jev's state, the chat message, or the routing message's first line."""
    if "messages" in body:
        text = body["messages"][1]["content"].split("\n\nOptions:")[0].removeprefix("Text: ")
    else:
        state = body["state"]
        text = state["text"] if isinstance(state, dict) else state
    return text.split("\n")[0].removeprefix("Message: ") if text.startswith("Message: ") else text


def test_test_stage_predictors_see_only_test_text(test_run):
    texts = set(test_run["texts"]["banking77"]["text"]) | set(test_run["texts"]["ag_news"]["text"])
    probe = set(test_run["splits"]["banking77"].val["text"][: stage1.PROBE_ROWS])
    kinds = {test_run["api"].kind(body) for _, body in test_run["api"].calls}
    assert {"llm", "jev routing", "jev examples", "jev banking77", "jev ag_news"} <= kinds
    assert {sent_text(body) for _, body in test_run["api"].calls} <= texts | probe  # probes use validation text
    routed = [sent_text(b) for _, b in test_run["api"].calls if test_run["api"].kind(b) == "jev routing"]
    assert set(routed) <= texts and len(set(routed)) == min(routing.N_SAMPLE, len(test_run["texts"]["banking77"]))


def test_no_bank_holds_test_text(test_run):
    norms = {data.normalize(t) for d in stage1.DATASETS for t in test_run["texts"][d]["text"]}
    sizes = {len(b) for b in test_run["banks"]}
    assert len(test_run["banks"]) > 100 and sizes  # every family bank, LR-A's s=35 and Intent LR's s=8 among them
    assert all(not (set(b["norm"]) & norms) for b in test_run["banks"])


def test_test_stage_leaves_deviations_alone_when_the_scorer_refuses(test_run, tmp_path):
    rt, _, _ = build(tmp_path, test_run["splits"], load_test_text=lambda d: test_run["texts"][d], score_test=lambda root: 1)
    runner = Stage1(rt, "test")
    runner.unaccounted = lambda: []
    before = (tmp_path / "deviations.md").read_text()
    assert runner.finish() == 1
    assert (tmp_path / "deviations.md").read_text() == before
    assert not (tmp_path / "runs" / "stage1" / "deviations_written.json").exists()


# --------------------------------------------------------------------------- stops in the paid part


class Twisted(FakeApi):
    """Answers one kind of request (by shape) with `reply(body)` instead, once the probes are done."""

    def __init__(self, kind, reply):
        super().__init__()
        self.twist_kind, self.reply = kind, reply

    def send(self, method, url, content=None):
        if method == "POST":
            body = json.loads(content)
            if self.kind(body) == self.twist_kind:
                with self.lock:
                    self.calls.append((url, body))
                answer = self.reply(body)
                if answer is not None:
                    return answer
        return super().send(method, url, content)


def logged_not_run(runs):
    return {**predictions.read_not_run("val", runs), **predictions.read_not_run("val_routing", runs)}


def test_a_rejected_key_halts_and_resumes_without_using_up_the_run(tmp_path):
    api = Twisted("jev ag_news", lambda body: TransportResponse(401, {}, b'{"error":{"message":"No auth credentials found"}}'))
    runner, api = paid_only(tmp_path, make_splits(), api=api)
    with pytest.raises(PaidHalt, match="key was rejected or credits ran out"):
        runner.run_paid()
    runs = tmp_path / "runs"
    assert logged_not_run(runs) == {}  # nothing is given up on a key problem
    assert predictions.paths("jev", "banking77", "val", runs)[0].exists()
    api.reply = lambda body: None  # the key is fixed
    runner.run_paid()
    assert logged_not_run(runs) == {}
    assert all(runner.paid_done(a) for a in stage1.paid_plan()) and max(api.ok.values()) == 1


def test_failures_above_one_percent_pause_after_saving_the_arm(tmp_path):
    bad = {}

    def malformed(body):
        text = body["state"]
        if text not in bad and len(bad) < 1:
            bad[text] = True
        return jev_ok(body, choice="not an option") if text in bad else None

    api = Twisted("jev ag_news", malformed)
    runner, api = paid_only(tmp_path, make_splits(), api=api)
    with pytest.raises(PaidHalt, match="PAUSE: jev__ag_news has 1 failed rows of 20, above 1%"):
        runner.run_paid()
    runs = tmp_path / "runs"
    rows, _, _ = predictions.read("jev", "ag_news", "val", runs)
    assert rows["failed"].sum() == 1
    assert not predictions.paths("jev_examples", "banking77", "val", runs)[0].exists()
    runner.run_paid()  # a rerun continues with the next arm; the paused arm is not redone
    assert all(runner.paid_done(a) for a in stage1.paid_plan()) and logged_not_run(runs) == {}


def test_a_changed_model_pauses_without_marking_anything_not_run(tmp_path):
    api = Twisted("jev ag_news", lambda body: jev_ok(body, model="typesafe/jev-1.14-20991231"))
    runner, api = paid_only(tmp_path, make_splits(), api=api)
    with pytest.raises(PaidHalt, match="PAUSE: jev__ag_news: the model changed mid-run"):
        runner.run_paid()
    runs = tmp_path / "runs"
    assert not predictions.paths("jev", "ag_news", "val", runs)[0].exists() and logged_not_run(runs) == {}


def test_a_rule_change_stops_spending_on_routing_arms(tmp_path, monkeypatch):
    runner, api = paid_only(tmp_path, make_splits())
    path = tmp_path / "runs" / "val_routing" / "fingerprint.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"fingerprint": "0" * 64, "first_arm": "jev_rule_A"}))
    runner.run_paid()
    logged = predictions.read_not_run("val_routing", tmp_path / "runs")
    assert set(logged) == {(a.name, "banking77") for a in stage1.paid_plan() if a.routing}
    assert "jev routing" not in {api.kind(b) for _, b in api.calls}
    assert (path.parent / score.FINGERPRINT_CHANGES).exists()


def test_rows_to_send_counts_outage_rows_with_a_pass_left(tmp_path):
    path = tmp_path / "runs" / "private" / "r.jsonl"
    path.parent.mkdir(parents=True)
    lines = [{"row_id": "a", "status": "done", "failed": False}, {"row_id": "b", "status": "failed", "failed": True, "reason": "http_503"},
             {"row_id": "c", "status": "failed", "failed": True, "reason": "malformed"},
             {"row_id": "d", "status": "failed", "failed": True, "reason": "network"}, {"row_id": "d", "status": "failed", "failed": True, "reason": "network"}]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert stage1.rows_to_send(path, ["a", "b", "c", "d", "e"]) == ["b", "e"]


def test_giving_up_skips_pending_probes_and_logs_every_unfinished_paid_arm(tmp_path):
    runner, api = paid_only(tmp_path, make_splits())
    runner.give_up_reason = "credits gone"
    runner.run_paid()
    assert api.calls == [] and set(logged_not_run(tmp_path / "runs")) == {(a.name, a.dataset) for a in stage1.paid_plan()}
    assert all(json.loads(runner.probe_marker(p).read_text())["result"] is None for p in ("jev", "small_llm", "network"))


# --------------------------------------------------------------------------- free arms: --max-hours stop and resume


def test_free_arms_stop_on_max_hours_and_resume_without_redoing_saved_work(tmp_path):
    splits = make_splits()
    pairs = []
    scorer = fake_scorer(pairs)
    runner, _ = paid_only(tmp_path, splits, make_nli_scorer=lambda key: scorer)
    done_marker = predictions.paths("embed_sim", "banking77", "val", tmp_path / "runs")[0]
    runner.rt.clock = lambda: 1e9 if done_marker.exists() else 0.0  # the limit passes once embed_sim is saved
    runner.max_hours = 1.0
    with redirect_stdout(io.StringIO()) as out:
        assert runner.run_free() is False
    assert "max-hours" in out.getvalue()
    nli_pairs = sum(pairs)
    assert nli_pairs > 0 and not predictions.paths("lr_d0_s1", "banking77", "val", tmp_path / "runs")[0].exists()
    runner.rt.clock = lambda: 0.0
    assert runner.run_free() is True
    assert sum(pairs) - nli_pairs == stage1.PROBE_ROWS * 77  # only the local NLI latency probe: no NLI row is redone
    paid = {(a.name, "routing" if a.routing else a.dataset) for a in stage1.paid_plan()}
    assert set(runner.unaccounted()) == paid  # every free arm is done, routing's included; only the paid ones wait


def test_giving_up_sets_aside_a_partial_routing_file_and_logs_it(tmp_path):
    runner, _ = paid_only(tmp_path, make_splits())
    rows = runner.routing_rows().head(5)
    frame = pd.DataFrame({"row_id": rows["row_id"], "pred": "amber", "top_prob": 0.5, "failed": False, "latency_s": 0.1})
    public = predictions.write("nli_rule_A", "banking77", frame, "val_routing", root=runner.root)
    assert ("nli_rule_A", "routing") in runner.unaccounted()
    runner.give_up_reason = "out of time"
    runner.give_up([("nli_rule_A", "routing")])
    assert not public.exists() and public.with_name(public.name + ".partial").exists()
    assert "out of time" in predictions.read_not_run("val_routing", runner.root)[("nli_rule_A", "banking77")]
    assert ("nli_rule_A", "routing") not in runner.unaccounted()


def test_a_bad_test_record_is_a_clean_refusal(val_run, tmp_path, capsys):
    splits = val_ready(val_run, tmp_path)
    texts, _, _ = make_test_frames(splits)
    record = tmp_path / "runs" / "stage1" / "test_record.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"gold_cache": {"banking77": "0" * 64}}))  # a scorer run by hand, before this runner
    rt, api, _ = build(tmp_path, splits, load_test_text=lambda d: texts[d])
    assert stage1.main(["--stage", "test"], runtime=rt) == EXIT_REFUSED
    assert "test_record.json" in capsys.readouterr().out and api.calls == []


def test_the_test_stage_refuses_stale_routing_checks(val_run, tmp_path, capsys):
    loads = []
    splits = val_ready(val_run, tmp_path)
    blob = json.loads((tmp_path / "validity.json").read_text())
    (tmp_path / "validity.json").write_text(json.dumps({**blob, "fingerprint": "0" * 64}))
    rt, api, _ = build(tmp_path, splits, load_test_text=lambda d: loads.append(d))
    assert stage1.main(["--stage", "test"], runtime=rt) == EXIT_REFUSED
    assert "validity.json" in capsys.readouterr().out and loads == []


def test_a_rejected_key_during_a_probe_leaves_it_to_run_again(tmp_path):
    api = Twisted("jev banking77", lambda body: TransportResponse(402, {}, b'{"error":{"message":"Insufficient credits"}}'))
    runner, api = paid_only(tmp_path, make_splits(), api=api)
    with pytest.raises(PaidHalt, match="jev latency probe stopped"):
        runner.run_paid()
    assert not runner.probe_marker("jev").exists() and logged_not_run(tmp_path / "runs") == {}
    api.reply = lambda body: None
    runner.run_paid()
    assert json.loads(runner.probe_marker("jev").read_text())["result"]["status"] == "complete"
