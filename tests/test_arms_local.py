"""Free arms and compute guards (Task 4). Fake encoders and fake NLI models only: nothing here downloads,
reads a test split, or calls a paid API.
"""
import ast
import dataclasses
import json
import logging
import re
import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import arms_local
import guards
import predict_local
import predictions
from data import Split, normalize
from guards import (
    BootstrapTooSlow,
    Guard,
    JobLockedError,
    JobStopped,
    ResourceLimitError,
    check_resources,
    model_job_lock,
    project_bootstrap,
    run_with_limit,
)
from labels import label_texts

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
AG = list(label_texts("ag_news", bare=True))  # World, Sports, Business, Sci/Tech


# --------------------------------------------------------------------------- fakes and toy data


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeEncoder:
    """Hashed bag of words, unit length. Class keywords dominate, so toy classes are separable."""

    def __init__(self, dim=128):
        self.dim, self.seen = dim, []

    def __call__(self, texts):
        self.seen.extend(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in tokens(t):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class RandomEncoder:
    """Embedding unrelated to the class, so a classifier on top scores near chance."""

    def __call__(self, texts):
        out = np.stack([np.random.default_rng(zlib.crc32(t.encode())).normal(size=64) for t in texts])
        return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


def keyword(name):
    return "kw" + re.sub(r"[^a-z0-9]", "", name.lower())


def toy_frame(classes, per_class, prefix, start=0):
    rows = []
    for c in classes:
        for i in range(per_class):
            rows.append((f"{prefix}{len(rows) + start}", f"{keyword(c)} filler{i % 7} extra{(i * 3) % 11} item{i}", c))
    df = pd.DataFrame(rows, columns=["row_id", "text", "label"])
    df["norm"] = df.text.map(normalize)
    return df


def toy_split(classes, val_per_class=6, pool_per_class=60, dataset="toy"):
    val = toy_frame(classes, val_per_class, f"{dataset}:train:v")
    pool = toy_frame(classes, pool_per_class, f"{dataset}:train:p")
    return Split(val=val, pool=pool)


def make_runtime(tmp_path, *, encoder=None, scorer=None, clock=None, splits=None, **kw):
    splits = splits or {"banking77": toy_split(["alpha", "beta", "gamma"], 15, 60), "ag_news": toy_split(AG, 200, 60)}
    return predict_local.Runtime(
        runs_root=tmp_path / "runs",
        load_split=lambda dataset: splits[dataset],
        make_encoder=lambda: encoder or FakeEncoder(),
        make_nli_scorer=lambda key: scorer or fake_scorer(),
        clock=clock or (lambda: 0.0),
        read_swap_gb=lambda: 1.0,
        read_free_disk_gb=lambda path=".": 500.0,
        **kw,
    )


def fake_scorer(log=None):
    """Entailment logit = shared words between premise and hypothesis (a stand-in for the NLI model)."""

    def score(pairs):
        if log is not None:
            log.append(len(pairs))
        return [float(len(set(tokens(p)) & set(tokens(h)))) for p, h in pairs]

    return score


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance_hours(self, h):
        self.now += h * 3600.0


# --------------------------------------------------------------------------- LR arms


def toy_xy(n_per_class=20, classes=("a", "b", "c")):
    rng = np.random.default_rng(0)
    centers = np.eye(len(classes), 8) * 3
    X = np.concatenate([centers[i] + rng.normal(size=(n_per_class, 8)) for i in range(len(classes))])
    y = np.repeat(np.array(classes), n_per_class)
    return X, y


def test_lr_uses_c_100_and_fixed_seeded_solver():
    model = arms_local.make_lr()
    p = model.get_params()
    assert (p["C"], p["max_iter"], p["solver"], p["random_state"]) == (100, 5000, "lbfgs", arms_local.LR_SEED)
    assert isinstance(arms_local.LR_SEED, int)
    X, y = toy_xy()
    a = arms_local.lr((X, y), X[:5])
    b = arms_local.lr((X, y), X[:5])
    assert np.array_equal(a, b) and a.shape == (5, 3) and np.allclose(a.sum(axis=1), 1)


def test_lr_default_c_is_100_and_goes_to_the_model(monkeypatch):
    seen = []
    real = arms_local.make_lr
    monkeypatch.setattr(arms_local, "make_lr", lambda c=100, max_iter=5000: seen.append((c, max_iter)) or real(c, max_iter))
    X, y = toy_xy()
    arms_local.lr((X, y), X[:2])
    arms_local.lr((X, y), X[:2], c=10)
    assert seen == [(100, 5000), (10, 5000)]


def test_lr_logs_non_convergence_and_does_not_hide_it(caplog):
    X, y = toy_xy()
    with caplog.at_level(logging.WARNING):
        probs = arms_local.lr((X, y), X[:3], max_iter=1)
    assert probs.shape == (3, 3)
    assert any("did not converge" in r.getMessage() for r in caplog.records)


def test_lr_columns_follow_classes_and_missing_class_gets_zero():
    X, y = toy_xy(classes=("a", "b"))
    probs = arms_local.lr((X, y), X[:4], classes=["c", "b", "a"])
    assert probs.shape == (4, 3) and (probs[:, 0] == 0).all() and np.allclose(probs.sum(axis=1), 1)


def test_lr_cv_uses_stratified_kfold_with_k_min_s_5_and_c100_at_s1():
    assert arms_local.CV_GRID == (1, 10, 100, 1000)
    assert arms_local.cv_plan(1) == arms_local.CVPlan(0, (100,))
    assert [arms_local.cv_plan(s).folds for s in (2, 3, 5, 8, 16, 35, "all")] == [2, 3, 5, 5, 5, 5, 5]
    assert all(arms_local.cv_plan(s).grid == arms_local.CV_GRID for s in (2, 3, 5, 8, 16, 35, "all"))
    assert arms_local.cv_plan(np.int64(3)).folds == 3
    for bad in (0, -1, "most", 2.5, True):
        with pytest.raises(ValueError):
            arms_local.cv_plan(bad)


def test_lr_cv_picks_c_from_inside_the_bank_and_fits_final_model_with_it(monkeypatch):
    calls = []
    monkeypatch.setattr(
        arms_local, "cv_accuracies", lambda X, y, folds, grid: calls.append((folds, grid)) or {1: 0.5, 10: 0.9, 100: 0.6, 1000: 0.7}
    )
    fitted = []
    real = arms_local.lr
    monkeypatch.setattr(arms_local, "lr", lambda train, test, c=100, **kw: fitted.append(c) or real(train, test, c=c, **kw))
    X, y = toy_xy()
    arms_local.lr_cv((X, y), X[:2], s=8)
    assert calls == [(5, (1, 10, 100, 1000))] and fitted == [10]
    arms_local.lr_cv((X, y), X[:2], s=2)
    assert calls[-1][0] == 2
    arms_local.lr_cv((X, y), X[:2], s=1)  # no selection at s=1
    assert len(calls) == 2 and fitted == [10, 10, 100]


def test_cv_accuracies_use_stratified_kfold_with_a_fixed_seed_and_k_fits_per_c(monkeypatch):
    X, y = toy_xy(n_per_class=6)
    acc = arms_local.cv_accuracies(X, y, 5, (1, 100))
    assert set(acc) == {1, 100} and all(0 <= v <= 1 for v in acc.values())
    assert acc == arms_local.cv_accuracies(X, y, 5, (1, 100))
    n_fits = []
    real = arms_local.fit_lr
    monkeypatch.setattr(arms_local, "fit_lr", lambda *a, **k: n_fits.append(1) or real(*a, **k))
    arms_local.cv_accuracies(X, y, 3, (1, 10, 100, 1000))
    assert len(n_fits) == 3 * 4  # k fits per C, never one per row
    with pytest.raises(ValueError):
        arms_local.cv_accuracies(X, y, 1, (100,))


def test_choose_c_ties_prefer_100_then_smaller():
    assert arms_local.choose_c({1: 0.8, 10: 0.8, 100: 0.8, 1000: 0.8}) == 100
    assert arms_local.choose_c({1: 0.9, 10: 0.9, 100: 0.5, 1000: 0.5}) == 1
    assert arms_local.choose_c({1: 0.1, 10: 0.2, 100: 0.3, 1000: 0.4}) == 1000


def test_tfidf_lr_uses_word_1_2_grams_lowercase_no_stop_list_min_df_1_c_100():
    vec = arms_local.make_tfidf()
    p = vec.get_params()
    assert (p["ngram_range"], p["lowercase"], p["stop_words"], p["min_df"]) == ((1, 2), True, None, 1)
    texts = ["Pay my bill now", "pay my bill", "Card arrived late", "my card arrived late", "pay bill today", "late card"]
    labels = ["bill", "bill", "card", "card", "bill", "card"]
    probs = arms_local.tfidf_lr((texts, labels), ["Please pay the bill", "card is late"], classes=["bill", "card"])
    assert probs.argmax(axis=1).tolist() == [0, 1] and np.allclose(probs.sum(axis=1), 1)


# --------------------------------------------------------------------------- embedding similarity


def test_embed_sim_temperature_changes_confidence_not_pick():
    enc = FakeEncoder()
    texts = ["kwworld news today", "kwsports game tonight", "kwbusiness market"]
    labels = ["kwworld", "kwsports", "kwbusiness"]
    cold = arms_local.embed_sim(texts, labels, 0.01, encode=enc)
    mid = arms_local.embed_sim(texts, labels, 0.05, encode=enc)
    warm = arms_local.embed_sim(texts, labels, 0.2, encode=enc)
    for p in (cold, mid, warm):
        assert np.allclose(p.sum(axis=1), 1)
    assert cold.argmax(1).tolist() == mid.argmax(1).tolist() == warm.argmax(1).tolist() == [0, 1, 2]
    assert (cold.max(1) > mid.max(1)).all() and (mid.max(1) > warm.max(1)).all()


def test_embed_sim_sensitivity_rows_come_from_the_same_scores():
    cos = np.array([[0.9, 0.5, 0.1], [0.2, 0.3, 0.25]])
    views = arms_local.sensitivity_views(cos)
    assert list(views) == ["", "_t0.01", "_t0.2", "_raw"]
    assert np.allclose(views[""], arms_local.softmax(cos / 0.05))
    assert np.allclose(views["_t0.01"], arms_local.softmax(cos / 0.01))
    assert np.allclose(views["_t0.2"], arms_local.softmax(cos / 0.2))
    assert np.array_equal(views["_raw"], cos)  # raw cosine: ranked by top cosine, no softmax


def test_embed_sim_default_temperature_constant():
    assert arms_local.EMBED_TEMPERATURE == 0.05 and arms_local.SENSITIVITY_TEMPERATURES == (0.01, 0.2)


# --------------------------------------------------------------------------- NLI


def test_nli_uses_description_as_hypothesis_and_template_for_bare_names():
    desc = arms_local.nli_hypotheses("banking77", bare=False)
    assert desc == list(label_texts("banking77", bare=False).values())  # the description as written
    bare = arms_local.nli_hypotheses("banking77", bare=True)
    assert bare[:2] == [f"This text is about {n}." for n in list(label_texts("banking77", bare=True))[:2]]
    assert "This text is about activate_my_card." in bare
    ag = arms_local.nli_hypotheses("ag_news", bare=False)
    assert ag == list(label_texts("ag_news", bare=False).values())  # unaltered
    assert ag[0] == "world news, international affairs, politics and conflicts"  # the full written text
    assert arms_local.nli_hypotheses("ag_news", bare=True)[3] == "This text is about Sci/Tech."


def test_nli_multi_label_false_softmax_across_entailment():
    logits = {"h0": 2.0, "h1": 0.0, "h2": -1.0}

    def scorer(pairs):
        return [logits[h] for _, h in pairs]

    out = list(arms_local.nli(["a text", "another"], ["h0", "h1", "h2"], batch_size=2, scorer=scorer))
    want = np.exp([2.0, 0.0, -1.0]) / np.exp([2.0, 0.0, -1.0]).sum()
    assert len(out) == 2 and np.allclose(out[0], want) and np.allclose(out[1], want)
    assert np.isclose(out[0].sum(), 1)
    # not an independent per-label sigmoid (multi_label=True), whose values would not sum to 1
    assert not np.allclose(out[0], 1 / (1 + np.exp(-np.array([2.0, 0.0, -1.0]))))


def test_nli_batches_have_a_fixed_size_and_order_is_kept():
    seen = []

    def scorer(pairs):
        seen.append(list(pairs))
        return [float(len(h)) for _, h in pairs]

    hyps = ["x", "yy", "zzz"]
    texts = [f"t{i}" for i in range(7)]
    out = list(arms_local.nli(texts, hyps, batch_size=4, scorer=scorer))
    assert len(out) == 7
    assert [len(b) for b in seen] == [4] * 5 + [1]  # 21 pairs, fixed batches, one short tail
    assert [p for b in seen for p, _ in b] == [t for t in texts for _ in hyps]
    assert all(np.allclose(o, out[0]) for o in out)


def test_nli_entailment_logits_pick_the_entailment_column_and_hf_scorer_truncates_at_256():
    assert arms_local.NLI_MAX_LEN == 256
    calls = {}

    class Tok:
        def __call__(self, premises, hypotheses, **kw):
            calls.update(kw, premises=premises, hypotheses=hypotheses)
            import torch

            return {"input_ids": torch.zeros((len(premises), 3), dtype=torch.long)}

    class Model:
        config = SimpleNamespace(label2id={"entailment": 0, "not_entailment": 1})

        def __call__(self, **inputs):
            import torch

            n = inputs["input_ids"].shape[0]
            return SimpleNamespace(logits=torch.tensor([[float(i), -5.0] for i in range(n)]))

    score = arms_local.make_hf_scorer(Tok(), Model(), device="cpu")
    out = score([("p1", "h1"), ("p2", "h2"), ("p3", "h3")])
    assert out == [0.0, 1.0, 2.0]  # column 0 is entailment in this fake; the other column is ignored
    assert calls["max_length"] == 256 and calls["truncation"] == "only_first" and calls["padding"] is True
    assert calls["premises"] == ["p1", "p2", "p3"] and calls["hypotheses"] == ["h1", "h2", "h3"]


def test_entailment_id_is_found_by_name():
    assert arms_local.entailment_id({"entailment": 1, "not_entailment": 0}) == 1
    assert arms_local.entailment_id({"contradiction": 2, "Entailment": 0, "neutral": 1}) == 0
    with pytest.raises(ValueError):
        arms_local.entailment_id({"yes": 0, "no": 1})


# --------------------------------------------------------------------------- pinned revisions


def test_pinned_revisions_match_design_md():
    design = (REPO / "DESIGN.md").read_text()
    for name, rev in [
        (arms_local.EMBED_MODEL, arms_local.EMBED_REVISION),
        (arms_local.NLI_MODELS["large"][0], arms_local.NLI_MODELS["large"][1]),
        (arms_local.NLI_MODELS["base"][0], arms_local.NLI_MODELS["base"][1]),
    ]:
        assert f"`{name}`" in design and rev in design, name
    assert arms_local.EMBED_REVISION == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def test_real_loaders_pass_revision_explicitly_and_check_disk_first(monkeypatch):
    import sentence_transformers
    import transformers

    seen = {}

    class FakeST:
        def __init__(self, name, **kw):
            seen["st"] = (name, kw)

        def encode(self, texts, **kw):
            seen["encode"] = kw
            return np.ones((len(texts), 2), dtype=np.float32)

    class FakeAuto:
        @staticmethod
        def from_pretrained(name, **kw):
            seen.setdefault("hf", []).append((name, kw))
            return SimpleNamespace(to=lambda d: FakeAuto, eval=lambda: FakeAuto, config=SimpleNamespace(label2id={"entailment": 0}))

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAuto)
    monkeypatch.setattr(transformers, "AutoModelForSequenceClassification", FakeAuto)
    enc = arms_local.load_encoder(read_free_disk_gb=lambda path=".": 100.0)
    enc(["x"])
    assert seen["st"][0] == "BAAI/bge-small-en-v1.5" and seen["st"][1]["revision"] == arms_local.EMBED_REVISION
    assert seen["encode"]["normalize_embeddings"] is True and seen["encode"]["batch_size"] == arms_local.EMBED_BATCH_SIZE
    arms_local.load_nli_scorer("large", read_free_disk_gb=lambda path=".": 100.0)
    assert [(n, kw["revision"]) for n, kw in seen["hf"]] == [("MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c", arms_local.NLI_MODELS["large"][1])] * 2
    seen.clear()
    with pytest.raises(ResourceLimitError):  # under 5 GB free: no download is attempted
        arms_local.load_encoder(read_free_disk_gb=lambda path=".": 1.0)
    with pytest.raises(ResourceLimitError):
        arms_local.load_nli_scorer("base", read_free_disk_gb=lambda path=".": 1.0)
    assert seen == {}


# --------------------------------------------------------------------------- predictions files


def rows_df(ids, preds=None, probs=None):
    return pd.DataFrame(
        {"row_id": ids, "pred": preds or ["a"] * len(ids), "top_prob": probs or [0.5] * len(ids), "failed": [False] * len(ids)}
    )


def test_predictions_write_one_parquet_per_arm_and_full_vector_in_private(tmp_path):
    vec = np.array([[0.7, 0.3], [0.1, 0.9]])
    path = predictions.write("lr_d0_s8", "ag_news", rows_df(["r1", "r2"]), stage="val", vectors=vec, classes=["a", "b"], root=tmp_path)
    assert path == tmp_path / "val" / "lr_d0_s8__ag_news.parquet"
    df = pd.read_parquet(path)
    assert list(df.columns) == ["row_id", "pred", "top_prob", "failed"]
    rows, vecs, classes = predictions.read("lr_d0_s8", "ag_news", stage="val", root=tmp_path)
    assert rows.equals(df) and np.array_equal(vecs, vec) and classes == ["a", "b"]
    private = list((tmp_path / "private").rglob("*"))
    assert any(p.is_file() and p.parent == tmp_path / "private" / "val" for p in private)
    assert [p.suffix for p in (tmp_path / "val").rglob("*")] == [".parquet"]  # nothing private beside the public file


def test_predictions_read_missing_is_none_and_schema_is_enforced(tmp_path):
    assert predictions.read("nope", "ag_news", stage="val", root=tmp_path) is None
    with pytest.raises(ValueError, match="columns"):
        predictions.write("x", "ag_news", rows_df(["r1"]).assign(gold="a"), stage="val", root=tmp_path)
    with pytest.raises(ValueError, match="private"):
        predictions.write("x", "ag_news", rows_df(["r1"]), stage="private", root=tmp_path)
    with pytest.raises(ValueError):
        predictions.write("x", "ag_news", rows_df(["r1", "r1"]), stage="val", root=tmp_path)


# --------------------------------------------------------------------------- run_arm: checkpoints and resume


def counting_predictor(classes, seen):
    def predict(texts):
        seen.extend(texts)
        out = np.zeros((len(texts), len(classes)))
        for i, t in enumerate(texts):
            out[i, len(t) % len(classes)] = 1.0
            out[i] += 0.01
        return out / out.sum(axis=1, keepdims=True)

    return predict


def rows_with_labels(n):
    # a label column is present on purpose: run_arm must never pass it to a predictor
    return pd.DataFrame({"row_id": [f"v{i}" for i in range(n)], "text": [f"text number {i}" for i in range(n)], "label": ["gold"] * n})


def test_predict_resumes_from_saved_rows_without_recomputing(tmp_path):
    classes = ["a", "b", "c"]
    rows = rows_with_labels(23)
    clock = FakeClock()
    seen1 = []
    guard = Guard(max_hours=1.0, clock=clock, read_swap_gb=lambda: 0.0, read_free_disk_gb=lambda path=".": 99.0)

    def slow(texts):
        clock.advance_hours(0.4)
        return counting_predictor(classes, seen1)(texts)

    out = predict_local.run_arm("a1", "ag_news", "val", rows, slow, classes, chunk_size=5, guard=guard, root=tmp_path)
    assert out.stopped == "max_hours" and out.completed == 3  # 0, 0.4, 0.8 h start new chunks; 1.2 h stops
    saved, _, _ = predictions.read("a1", "ag_news", stage="val", root=tmp_path)
    assert saved.row_id.tolist() == [f"v{i}" for i in range(15)]  # the checkpoint was kept

    seen2 = []
    out2 = predict_local.run_arm("a1", "ag_news", "val", rows, counting_predictor(classes, seen2), classes, chunk_size=5, root=tmp_path)
    assert out2.stopped is None and out2.rows_done == 23
    assert sorted(seen2) == sorted(rows.text[15:])  # only the rows not yet saved were computed
    assert not set(seen1) & set(seen2)
    final, vecs, _ = predictions.read("a1", "ag_news", stage="val", root=tmp_path)
    assert sorted(final.row_id) == sorted(rows.row_id) and final.row_id.is_unique and vecs.shape == (23, 3)
    assert final.iloc[:15].equals(saved)  # finished rows unchanged by the resume

    seen3 = []
    predict_local.run_arm("a1", "ag_news", "val", rows, counting_predictor(classes, seen3), classes, chunk_size=5, root=tmp_path)
    assert seen3 == []  # a third run recomputes nothing


def test_run_arm_fills_columns_from_the_vector_and_accepts_rows_with_no_labels(tmp_path):
    classes = ["a", "b"]
    rows = pd.DataFrame({"row_id": ["x", "y"], "text": ["t1", "t2"]})  # test-style rows: no label column at all
    vec = np.array([[0.2, 0.8], [0.6, 0.4]])
    predict_local.run_arm("m", "ag_news", "val", rows, lambda texts: vec, classes, chunk_size=10, root=tmp_path)
    got, _, _ = predictions.read("m", "ag_news", stage="val", root=tmp_path)
    assert got.pred.tolist() == ["b", "a"] and np.allclose(got.top_prob, [0.8, 0.6]) and got.failed.tolist() == [False, False]


def test_run_arm_marks_non_finite_rows_failed_not_crashed(tmp_path):
    rows = pd.DataFrame({"row_id": ["x", "y"], "text": ["t1", "t2"]})
    predict_local.run_arm("m", "ag_news", "val", rows, lambda t: np.array([[np.nan, np.nan], [0.1, 0.9]]), ["a", "b"], root=tmp_path)
    got, _, _ = predictions.read("m", "ag_news", stage="val", root=tmp_path)
    assert got.failed.tolist() == [True, False] and got.pred.tolist() == ["", "b"]


def test_run_arm_refuses_to_resume_with_different_classes(tmp_path):
    rows = pd.DataFrame({"row_id": ["x"], "text": ["t1"]})
    predict_local.run_arm("m", "ag_news", "val", rows, lambda t: np.array([[0.5, 0.5]]), ["a", "b"], root=tmp_path)
    with pytest.raises(ValueError, match="classes"):
        predict_local.run_arm("m", "ag_news", "val", rows.assign(row_id="z"), lambda t: np.array([[0.5, 0.5]]), ["a", "q"], root=tmp_path)


def test_predictors_cannot_read_gold():
    # Predictor sources never touch the firewall or a gold module.
    for name in ("predict_local.py", "predictions.py", "arms_local.py", "guards.py"):
        src = (SRC / name).read_text()
        tree = ast.parse(src)
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[-1] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods |= {(node.module or "").split(".")[-1]} | {a.name for a in node.names}
        assert not mods & {"firewall", "gold", "load_test"}, name
        assert "load_test" not in src and "import gold" not in src, name
    # And the callable boundary: a predictor is handed text only, never labels or ids.
    handed = []
    rows = rows_with_labels(4)

    def spy(texts):
        handed.append(texts)
        return np.full((len(texts), 2), 0.5)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(predictions, "write", lambda *a, **k: Path("unused"))
        mp.setattr(predictions, "read", lambda *a, **k: None)
        predict_local.run_arm("m", "ag_news", "val", rows, spy, ["a", "b"], chunk_size=2, root=Path("unused"))
    assert handed == [["text number 0", "text number 1"], ["text number 2", "text number 3"]]


def test_firewall_predictor_check_covers_the_new_scripts():
    sys.path.insert(0, str(REPO / "tests"))
    from test_firewall import predict_files, reads_gold

    files = predict_files(REPO)
    assert "src/predict_local.py" in files
    assert [f for f in files if reads_gold((REPO / f).read_text())] == []


# --------------------------------------------------------------------------- guards


def test_job_stops_when_swap_grows_over_2gb_or_disk_under_5gb():
    ok = lambda: 1.0  # noqa: E731
    disk = lambda path=".": 50.0  # noqa: E731
    check_resources(ok, disk, baseline=1.0)
    check_resources(lambda: 3.0, disk, baseline=1.0)  # growth of exactly 2 GB is allowed
    with pytest.raises(ResourceLimitError, match="swap"):
        check_resources(lambda: 3.01, disk, baseline=1.0)
    check_resources(ok, lambda path=".": 5.0, baseline=1.0)
    with pytest.raises(ResourceLimitError, match="disk"):
        check_resources(ok, lambda path=".": 4.99, baseline=1.0)
    check_resources(lambda: 0.2, disk, baseline=1.0)  # swap shrinking is fine
    # the swap check is relative to the baseline, not absolute
    check_resources(lambda: 9.0, disk, baseline=7.5)


def test_job_stops_itself_through_run_with_limit_when_swap_jumps():
    swap = {"gb": 0.0}
    guard = Guard(max_hours=None, read_swap_gb=lambda: swap["gb"], read_free_disk_gb=lambda path=".": 99.0)
    done = []

    def process(i):
        done.append(i)
        if i == 2:
            swap["gb"] = 2.5

    flushed = []
    out = run_with_limit(range(10), process, guard, checkpoint=lambda: flushed.append(1))
    assert done == [0, 1, 2] and out.stopped == "resources" and out.completed == 3 and flushed == [1]


def test_parse_swap_used_reads_macos_sysctl_output():
    assert guards.parse_swap_used_gb("total = 2048.00M  used = 1024.00M  free = 1024.00M  (encrypted)") == 1.0
    assert guards.parse_swap_used_gb("total = 4.00G  used = 1.50G  free = 2.50G") == 1.5
    assert guards.parse_swap_used_gb("total = 0.00M  used = 0.00M  free = 0.00M") == 0.0
    with pytest.raises(ValueError):
        guards.parse_swap_used_gb("garbage")


def test_nli_batch_size_is_the_named_frozen_constant():
    assert arms_local.NLI_BATCH_SIZE == 16
    import inspect

    assert inspect.signature(arms_local.nli).parameters["batch_size"].default is arms_local.NLI_BATCH_SIZE


def test_wrapper_script_runs_the_cli_without_pythonpath():
    script = REPO / "scripts" / "predict_local.sh"
    assert script.stat().st_mode & 0o111
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    out = subprocess.run([str(script), "--help"], capture_output=True, text=True, cwd="/", env=env, timeout=50)
    assert out.returncode == 0 and "--floor" in out.stdout and "--time-nli" in out.stdout


def test_default_readers_return_numbers():
    assert guards.free_disk_gb(".") > 0
    assert guards.swap_used_gb() >= 0


def test_long_job_stops_itself_at_max_hours_and_keeps_checkpoint(tmp_path):
    clock = FakeClock()
    guard = Guard(max_hours=1.0, clock=clock, read_swap_gb=lambda: 0.0, read_free_disk_gb=lambda path=".": 99.0)
    ckpt = tmp_path / "ckpt.json"
    state = {"done": []}

    def process(i):
        clock.advance_hours(0.4)
        state["done"].append(i)
        ckpt.write_text(json.dumps(state["done"]))  # the per-item checkpoint a real job writes

    flushes = []
    out = run_with_limit(range(5), process, guard, checkpoint=lambda: flushes.append(len(state["done"])))
    assert out.stopped == "max_hours" and out.completed == 3
    assert json.loads(ckpt.read_text()) == [0, 1, 2]  # kept
    assert flushes == [3]
    # a rerun continues from the checkpoint instead of starting over
    rest = [i for i in range(5) if i not in json.loads(ckpt.read_text())]
    guard2 = Guard(max_hours=1.0, clock=clock, read_swap_gb=lambda: 0.0, read_free_disk_gb=lambda path=".": 99.0)
    out2 = run_with_limit(rest, process, guard2)
    assert out2.stopped is None and json.loads(ckpt.read_text()) == [0, 1, 2, 3, 4]


def test_run_with_limit_checkpoints_on_error_and_reraises():
    guard = Guard(max_hours=None, read_swap_gb=lambda: 0.0, read_free_disk_gb=lambda path=".": 99.0)
    flushed = []

    def boom(i):
        if i == 1:
            raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError, match="crashed"):
        run_with_limit(range(3), boom, guard, checkpoint=lambda: flushed.append(1))
    assert flushed == [1]


def test_guard_without_max_hours_never_expires_and_job_stopped_carries_reason():
    clock = FakeClock()
    g = Guard(max_hours=None, clock=clock, read_swap_gb=lambda: 0.0, read_free_disk_gb=lambda path=".": 99.0)
    clock.advance_hours(1000)
    g()
    with pytest.raises(JobStopped) as e:
        Guard(max_hours=0.5, clock=FakeClock(), read_swap_gb=lambda: 0.0, read_free_disk_gb=lambda path=".": 1.0)()
    assert e.value.reason == "resources"


def test_second_model_job_refuses_to_start_while_one_runs(tmp_path):
    lock = tmp_path / "model_job.lock"
    with model_job_lock(lock):
        assert lock.exists()
        with pytest.raises(JobLockedError, match=str(lock.name)):
            with model_job_lock(lock):
                pytest.fail("second job started")
    assert not lock.exists()  # released
    with model_job_lock(lock):  # and can be taken again
        pass


def test_lock_is_released_after_an_error_and_a_stale_lock_is_reclaimed(tmp_path):
    lock = tmp_path / "sub" / "model_job.lock"
    with pytest.raises(RuntimeError):
        with model_job_lock(lock):
            raise RuntimeError("x")
    assert not lock.exists()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lock.write_text(str(dead.pid))  # left behind by a process that no longer exists
    with model_job_lock(lock):
        assert lock.read_text().strip() == str(__import__("os").getpid())
    lock.write_text("not a pid")
    with pytest.raises(JobLockedError):  # unreadable owner: do not guess, refuse
        with model_job_lock(lock):
            pass


def test_bootstrap_projects_from_100_replicates_and_stops_above_1_hour():
    clock = FakeClock()
    calls = []

    def fast(n_replicates):
        calls.append(n_replicates)
        clock.now += 3.0  # 3 s for 100 -> 300 s for 10,000

    proj = project_bootstrap(fast, clock=clock)
    assert calls == [100] and proj.n_replicates == 100 and proj.full_replicates == 10_000
    assert proj.seconds == 3.0 and proj.projected_hours == pytest.approx(300 / 3600)

    def slow(n_replicates):
        calls.append(n_replicates)
        clock.now += 40.0  # 4,000 s projected = 1.11 h

    with pytest.raises(BootstrapTooSlow, match="1.11"):
        project_bootstrap(slow, clock=clock)
    assert calls == [100, 100]  # it never ran the full bootstrap

    def exactly(n_replicates):
        clock.now += 36.0  # exactly 1.0 h projected: allowed, only above stops

    assert project_bootstrap(exactly, clock=clock).projected_hours == pytest.approx(1.0)
    assert project_bootstrap(slow, limit_hours=2.0, clock=clock).projected_hours > 1.0


# --------------------------------------------------------------------------- CLI: floor


def test_floor_prints_only_pass_and_no_accuracy_anywhere(tmp_path, capsys, caplog):
    rt = make_runtime(tmp_path)
    with caplog.at_level(logging.DEBUG):
        code = predict_local.main(["--stage", "val", "--floor"], rt)
    out = capsys.readouterr()
    assert code == 0 and out.out == "LR s=35 floor: PASS\n" and out.err == ""
    assert_no_accuracy(caplog, tmp_path)


def test_floor_prints_only_fail_and_no_accuracy_anywhere(tmp_path, capsys, caplog):
    rt = make_runtime(tmp_path, encoder=RandomEncoder())
    with caplog.at_level(logging.DEBUG):
        code = predict_local.main(["--stage", "val", "--floor"], rt)
    out = capsys.readouterr()
    assert code == 1 and out.out == "LR s=35 floor: FAIL\n" and out.err == ""
    assert_no_accuracy(caplog, tmp_path)


def assert_no_accuracy(caplog, tmp_path):
    for r in caplog.records:
        msg = r.getMessage()
        assert not re.search(r"\d\.\d|\d%|acc", msg, re.I), msg
    written = [p for p in (tmp_path).rglob("*") if p.is_file()]
    assert written == [], written  # no prediction file, no log file: nothing on disk carries a result


def test_floor_threshold_is_75_percent_and_function_has_no_output_calls():
    assert predict_local.FLOOR_LR_S35 == 0.75
    tree = ast.parse((SRC / "predict_local.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "floor_passes")
    calls = {ast.unparse(c.func) for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert not {c for c in calls if c.split(".")[0] in {"print", "logging", "log", "logger", "open", "sys"} or c.endswith(("write", "write_text", "to_parquet"))}
    assert returns_only_bool(fn)


def returns_only_bool(fn):
    rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    return bool(rets) and all(isinstance(r.value, (ast.Compare, ast.Constant, ast.BoolOp)) for r in rets)


def test_floor_uses_bank_draw_0_s_35_and_only_pool_rows(tmp_path):
    enc = FakeEncoder()
    rt = make_runtime(tmp_path, encoder=enc)
    sp = rt.load_split("banking77")
    predict_local.main(["--stage", "val", "--floor"], rt)
    from data import bank

    bank_texts = set(bank(sp.pool, 0, 35).text)
    assert bank_texts <= set(enc.seen) and set(sp.val.text) <= set(enc.seen)
    assert len(set(enc.seen) - bank_texts - set(sp.val.text)) == 0  # nothing else was embedded


# --------------------------------------------------------------------------- CLI: NLI timing


def test_time_nli_times_5000_pairs_half_per_dataset_and_prints_rate_and_hours(tmp_path, capsys):
    clock = FakeClock()
    batches = []

    def scorer(pairs):
        batches.append(len(pairs))
        clock.now += 0.5  # 0.5 s per batch of NLI_BATCH_SIZE pairs
        return [0.0] * len(pairs)

    rt = make_runtime(tmp_path, scorer=scorer, clock=clock)
    code = predict_local.main(["--stage", "val", "--time-nli"], rt)
    out = capsys.readouterr().out
    assert code == 0 and sum(batches) == 5000
    bs = arms_local.NLI_BATCH_SIZE
    assert all(b <= bs for b in batches) and sum(b < bs for b in batches) <= 2  # fixed size; one short tail per dataset
    rate = 5000 / (0.5 * len(batches))
    hours = predict_local.TOTAL_NLI_PAIRS / rate / 3600
    assert predict_local.TOTAL_NLI_PAIRS == 267_560 == 3080 * 77 + 7600 * 4
    assert re.search(rf"{rate:.1f} pairs/s", out) and re.search(rf"projected {hours:.2f} h", out)
    assert "12-hour rule: kept deberta-v3-large" in out
    log = [json.loads(x) for x in (tmp_path / "runs" / "val" / "nli_timing.jsonl").read_text().splitlines()]
    assert log[-1]["pairs"] == 5000 and log[-1]["use_base"] is False and log[-1]["model"] == "large"


def test_time_nli_splits_pairs_evenly_between_datasets(tmp_path):
    per_dataset = []
    rt = make_runtime(tmp_path, scorer=lambda pairs: per_dataset.append([h for _, h in pairs]) or [0.0] * len(pairs))
    predict_local.main(["--stage", "val", "--time-nli"], rt)
    flat = [h for b in per_dataset for h in b]
    ag_h = set(arms_local.nli_hypotheses("ag_news", bare=False))
    b77_h = set(arms_local.nli_hypotheses("banking77", bare=False))
    assert len(flat) == 5000
    assert sum(h in ag_h for h in flat) == 2500 and sum(h in b77_h for h in flat) == 2500


def test_time_nli_applies_12_hour_rule_and_names_the_pinned_base_model(tmp_path, capsys):
    clock = FakeClock()

    def slow(pairs):
        clock.now += 10.0  # far too slow for 267,560 pairs
        return [0.0] * len(pairs)

    rt = make_runtime(tmp_path, scorer=slow, clock=clock)
    predict_local.main(["--stage", "val", "--time-nli"], rt)
    out = capsys.readouterr().out
    assert "12-hour rule: use MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c at bddf8c5411c34ac3565e16e04384fd68b2618dda" in out
    log = json.loads((tmp_path / "runs" / "val" / "nli_timing.jsonl").read_text().splitlines()[-1])
    assert log["use_base"] is True and log["projected_hours"] > 12


def test_time_nli_prints_no_accuracy_and_never_sees_labels(tmp_path, capsys):
    seen = []
    rt = make_runtime(tmp_path, scorer=lambda pairs: seen.extend(p for p, _ in pairs) or [0.0] * len(pairs))
    predict_local.main(["--stage", "val", "--time-nli"], rt)
    out = capsys.readouterr().out
    assert "acc" not in out.lower() and "%" not in out
    assert not any(t in {"alpha", "beta", "gamma", *AG} for t in seen)  # premises are message texts, not labels


# --------------------------------------------------------------------------- CLI: arms


def test_embed_sim_arm_writes_primary_and_sensitivity_rows_from_one_pass(tmp_path):
    enc = FakeEncoder()
    rt = make_runtime(tmp_path, encoder=enc)
    code = predict_local.main(["--stage", "val", "--arm", "embed_sim", "--dataset", "ag_news"], rt)
    assert code == 0
    for suffix in ["", "_t0.01", "_t0.2", "_raw"]:
        got, vecs, classes = predictions.read(f"embed_sim{suffix}", "ag_news", stage="val", root=tmp_path / "runs")
        assert len(got) == 800 and classes == AG and vecs.shape == (800, 4)
    prim, _, _ = predictions.read("embed_sim", "ag_news", stage="val", root=tmp_path / "runs")
    cold, _, _ = predictions.read("embed_sim_t0.01", "ag_news", stage="val", root=tmp_path / "runs")
    raw, _, _ = predictions.read("embed_sim_raw", "ag_news", stage="val", root=tmp_path / "runs")
    assert prim.pred.tolist() == cold.pred.tolist() == raw.pred.tolist()
    assert (cold.top_prob >= prim.top_prob).all()
    assert (raw.top_prob <= 1.0 + 1e-6).all()
    # labels were embedded once for the whole run, rows only for rows to predict
    assert enc.seen.count(label_texts("ag_news", bare=False)["World"]) >= 1


def test_embed_sim_bare_names_arm_is_named_separately(tmp_path):
    rt = make_runtime(tmp_path)
    predict_local.main(["--stage", "val", "--arm", "embed_sim", "--dataset", "ag_news", "--bare"], rt)
    assert (tmp_path / "runs" / "val" / "embed_sim_bare__ag_news.parquet").exists()
    assert not (tmp_path / "runs" / "val" / "embed_sim__ag_news.parquet").exists()


def test_bare_is_refused_for_arms_other_than_embedding_similarity_and_nli_bare_is_not_run(tmp_path):
    rt = make_runtime(tmp_path)
    for arm in ("nli", "lr", "lr_cv", "tfidf_lr"):
        with pytest.raises(SystemExit):
            predict_local.main(["--stage", "val", "--arm", arm, "--dataset", "ag_news", "--bare"], rt)


@pytest.mark.parametrize("arm", ["lr", "lr_cv", "tfidf_lr"])
def test_supervised_arms_run_from_a_bank_and_write_probability_vectors(tmp_path, arm):
    rt = make_runtime(tmp_path)
    code = predict_local.main(["--stage", "val", "--arm", arm, "--dataset", "ag_news", "--draw", "1", "--s", "2"], rt)
    assert code == 0
    got, vecs, classes = predictions.read(f"{arm}_d1_s2", "ag_news", stage="val", root=tmp_path / "runs")
    assert len(got) == 800 and vecs.shape == (800, 4) and np.allclose(vecs.sum(axis=1), 1) and classes == AG
    # the bank is the only source of labels: a toy keyword task is learnable from 2 rows per class, well above chance
    sp = rt.load_split("ag_news")
    truth = dict(zip(sp.val.row_id, sp.val.label))
    assert np.mean([truth[r] == p for r, p in zip(got.row_id, got.pred)]) > 0.6  # chance is 0.25


def test_lr_s_all_at_validation_stage_uses_the_pool_only(tmp_path):
    enc = FakeEncoder()
    rt = make_runtime(tmp_path, encoder=enc)
    sp = rt.load_split("ag_news")
    # validation rows are the rows being predicted, so they must not also be training rows
    bank_df = predict_local.make_bank(sp, "ag_news", draw=0, s="all", stage="val")
    assert set(bank_df.row_id).isdisjoint(sp.val.row_id) and len(bank_df) == len(sp.pool)


def test_nli_arm_checkpoints_every_500_rows_and_resumes(tmp_path):
    assert predict_local.NLI_CHECKPOINT_ROWS == 500
    pairs_seen = []
    clock = FakeClock()

    def scorer(pairs):
        pairs_seen.extend(p for p, _ in pairs)
        clock.advance_hours(0.01)
        return [float(len(set(tokens(p)) & set(tokens(h)))) for p, h in pairs]

    rt = make_runtime(tmp_path, scorer=scorer, clock=clock)
    # 800 rows x 4 labels = 3200 pairs; the first chunk of 500 rows is 2000 pairs
    code = predict_local.main(["--stage", "val", "--arm", "nli", "--dataset", "ag_news", "--max-hours", "0.0001"], rt)
    assert code == 2  # stopped itself on the limit
    got, _, _ = predictions.read("nli", "ag_news", stage="val", root=tmp_path / "runs")
    assert len(got) == 500  # exactly one checkpoint chunk kept
    first = len(pairs_seen)
    code = predict_local.main(["--stage", "val", "--arm", "nli", "--dataset", "ag_news"], rt)
    assert code == 0
    got, vecs, _ = predictions.read("nli", "ag_news", stage="val", root=tmp_path / "runs")
    assert len(got) == 800 and got.row_id.is_unique and vecs.shape == (800, 4)
    assert len(pairs_seen) - first == 300 * 4  # only the missing rows were scored


def test_cli_refuses_the_test_stage_and_a_second_model_job(tmp_path):
    rt = make_runtime(tmp_path)
    with pytest.raises(SystemExit):
        predict_local.main(["--stage", "test", "--arm", "embed_sim", "--dataset", "ag_news"], rt)
    with model_job_lock(rt.runs_root / "model_job.lock"):
        with pytest.raises(JobLockedError):
            predict_local.main(["--stage", "val", "--floor"], rt)
        with pytest.raises(JobLockedError):
            predict_local.main(["--stage", "val", "--time-nli"], rt)
    assert predict_local.main(["--stage", "val", "--floor"], rt) == 0  # released


def test_cli_stops_on_swap_growth_between_chunks(tmp_path):
    swap = iter([0.0, 0.0])  # baseline, then the check before chunk 1; every later reading is 3 GB
    rt = dataclasses.replace(make_runtime(tmp_path), read_swap_gb=lambda: next(swap, 3.0))
    code = predict_local.main(["--stage", "val", "--arm", "nli", "--dataset", "ag_news"], rt)
    assert code == 2
    got, _, _ = predictions.read("nli", "ag_news", stage="val", root=tmp_path / "runs")
    assert len(got) == 500  # the first chunk was kept; the second never started


def test_embedding_cache_keeps_shards_when_stopped_and_reuses_them(tmp_path):
    texts = [f"text {i}" for i in range(10)]
    enc = FakeEncoder()
    calls = {"n": 0}

    def guard():
        calls["n"] += 1
        if calls["n"] == 3:
            raise guards.MaxHoursExceeded("time")

    cached = arms_local.memoize_encoder(enc, cache_dir=tmp_path / "emb", guard=guard, chunk=4)
    with pytest.raises(JobStopped):
        cached(texts)  # chunks of 4: two are embedded and saved, the third is stopped before it starts
    assert len(list((tmp_path / "emb").glob("shard-*.npz"))) == 2 and len(enc.seen) == 8
    enc2 = FakeEncoder()
    again = arms_local.memoize_encoder(enc2, cache_dir=tmp_path / "emb", chunk=4)
    out = again(texts)
    assert enc2.seen == texts[8:]  # only the missing texts were embedded
    assert np.allclose(out, FakeEncoder()(texts))
    assert len(list((tmp_path / "emb").glob("shard-*.npz"))) == 3
