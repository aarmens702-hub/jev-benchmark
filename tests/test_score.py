"""The one-shot scorer and the blind dry run (Task 7). Synthetic prediction files only (tests/fake_runs.py).

Nothing here reads a dataset, a test split, a model or an API. Values printed or asserted are from
random fake scores, never from a real arm.
"""
import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import guards
import predictions
import report
import score
import verdict as verdict_module
from fake_runs import CLASSES, make_fake_runs
from metrics import GRID, DsResult
from score import RECORD, ScoreRefused, Settings

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
FAST = Settings(b=200, seed=2026, coverage=0.8)  # dry-run tests only; the result of record refuses it


@pytest.fixture(scope="module")
def record_set(tmp_path_factory):
    """One scored stage-of-record run on fake data at B=10,000, shared by the read-only tests."""
    base = tmp_path_factory.mktemp("record")
    gold = make_fake_runs(base / "runs", "stage1", gold_as_ids=True)
    make_fake_runs(base / "runs", "val", seed=1)  # validation scores for realised coverage
    results = score.score("stage1", gold, **paths(base), label_names=CLASSES)
    return base, gold, results


def paths(base):
    return {
        "runs_root": base / "runs",
        "out_dir": base / "results",
        "lock_path": base / "scored.lock",
        "deviations_path": base / "deviations.md",
    }


def copy_runs(src: Path, dst: Path) -> None:
    shutil.copytree(src / "runs", dst / "runs")


def write_deviations(path: Path, rows=()) -> None:
    lines = ["# Deviations", "", "| Date | What changed | Reason | Stage | Approved by |", "|---|---|---|---|---|"]
    lines += [f"| 2026-12-01 | {r} | x | stage1 | Aarmen |" for r in rows]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def fresh(tmp_path, record_set):
    """A stage-of-record set not yet scored, copied from the shared fake runs."""
    base, gold, _ = record_set
    copy_runs(base, tmp_path)
    write_deviations(tmp_path / "deviations.md")
    return tmp_path, gold


# --------------------------------------------------------------------------- refusals before scoring


def test_refuses_when_an_arm_has_neither_predictions_nor_not_run(fresh):
    base, gold = fresh
    public, _ = predictions.paths("nli", "banking77", "stage1", base / "runs")
    public.unlink()
    with pytest.raises(ScoreRefused, match="nli has neither predictions nor a not-run record"):
        score.score("stage1", gold, **paths(base), label_names=CLASSES)
    assert not (base / "scored.lock").exists() and not (base / "results").exists()

    # A family with one draw missing is refused too: a family is all files or a not-run record.
    lr_cv_file, _ = predictions.paths("lr_cv_d3_s5", "ag_news", "stage1", base / "runs")
    lr_cv_file.unlink()
    predictions.mark_not_run("nli", "banking77", "stage1", "test: model failed to load", base / "runs")
    with pytest.raises(ScoreRefused, match="lr_cv is missing 1 of 71 files"):
        score.score("stage1", gold, **paths(base), label_names=CLASSES)


def test_logged_not_run_is_accepted_and_reported(fresh, monkeypatch):
    base, gold = fresh
    predictions.paths("small_llm", "banking77", "stage1", base / "runs")[0].unlink()
    predictions.mark_not_run("small_llm", "banking77", "stage1", "no candidate passed selection", base / "runs")
    monkeypatch.setattr(score, "SECONDARY_COVERAGES", ())  # keep the test quick; covered elsewhere
    results = score.score("stage1", gold, **paths(base), label_names=CLASSES)
    assert {"arm": "small_llm", "dataset": "banking77", "reason": "no candidate passed selection"} in results["not_run"]
    assert results["additions"]["his_claim"]["small LLM"] == {"status": "not run"}


def test_refuses_both_a_file_and_a_not_run_record(fresh):
    base, gold = fresh
    predictions.mark_not_run("jev_bare", "ag_news", "stage1", "test", base / "runs")
    with pytest.raises(ScoreRefused, match="jev_bare has both"):
        score.score("stage1", gold, **paths(base), label_names=CLASSES)


def test_row_id_order_must_match_across_every_arm_and_gold(fresh):
    base, gold = fresh
    rows, vec, classes = predictions.read("embed_sim_t0.2", "ag_news", "stage1", base / "runs")
    flipped = rows.iloc[::-1].reset_index(drop=True)
    predictions.write("embed_sim_t0.2", "ag_news", flipped, "stage1", vec[::-1], classes, base / "runs")
    with pytest.raises(ScoreRefused, match="embed_sim_t0.2 row ids differ from gold's order"):
        score.score("stage1", gold, **paths(base), label_names=CLASSES)
    predictions.write("embed_sim_t0.2", "ag_news", rows.iloc[:-1], "stage1", vec[:-1], classes, base / "runs")
    with pytest.raises(ScoreRefused, match="rows"):
        score.score("stage1", gold, **paths(base), label_names=CLASSES)
    assert not (base / "scored.lock").exists()


def test_result_of_record_requires_b_10000_seed_2026_and_coverage_80(fresh):
    base, gold = fresh
    assert RECORD == Settings(b=10_000, seed=2026, coverage=0.8)
    for bad in (Settings(b=9_999), Settings(seed=1), Settings(coverage=0.7)):
        with pytest.raises(ScoreRefused, match="result of record"):
            score._score("stage1", gold, bad, **paths(base), label_names=CLASSES, val_stage="val",
                         clock=lambda: 0.0, limit_hours=1.0, record=True)  # fmt: skip
    with pytest.raises(ScoreRefused, match="not a test stage"):
        score.score("val", gold, **paths(base), label_names=CLASSES)


def test_refuses_when_the_bootstrap_projects_above_one_hour(fresh):
    base, gold = fresh
    ticks = iter(np.arange(0, 10_000, 40.0))  # every timing call reads 40 s more: 100 replicates -> 4,000 s at B=10,000
    with pytest.raises(guards.BootstrapTooSlow):
        score.score("stage1", gold, **paths(base), label_names=CLASSES, clock=lambda: next(ticks))
    assert not (base / "results").exists() and not (base / "scored.lock").exists()


def test_projection_sums_every_job_against_the_limit():
    calls = []

    def job(n_replicates=None):
        calls.append(n_replicates)

    clock = iter(np.arange(0, 1000, 0.2))  # 0.2 s per 100 replicates -> 20 s each at B=10,000
    jobs = [score.Job(str(i), job) for i in range(200)]  # 200 x 20 s = 4,000 s, over 1 h in total
    with pytest.raises(guards.BootstrapTooSlow, match="scorer's bootstrap work"):
        score.project(jobs, 10_000, lambda: next(clock), 1.0)
    assert set(calls) == {100}


# --------------------------------------------------------------------------- the lock


def test_refuses_rerun_without_deviation_row(fresh, monkeypatch):
    base, gold = fresh
    monkeypatch.setattr(score, "SECONDARY_COVERAGES", ())
    score.score("stage1", gold, **paths(base), label_names=CLASSES)
    lock = json.loads((base / "scored.lock").read_text())
    assert [r["status"] for r in lock["runs"]] == ["complete"] and lock["runs"][0]["settings"] == {"b": 10000, "seed": 2026, "coverage": 0.8}

    with pytest.raises(ScoreRefused, match="needs a new row"):
        score.score("stage1", gold, **paths(base), label_names=CLASSES)
    write_deviations(base / "deviations.md", ["Fixed a typo in the README"])  # a row, but not about scoring
    with pytest.raises(ScoreRefused, match="needs a new row"):
        score.score("stage1", gold, **paths(base), label_names=CLASSES)

    write_deviations(base / "deviations.md", ["Fixed a typo in the README", "Re-scored after a crash in report.py"])
    score.score("stage1", gold, **paths(base), label_names=CLASSES)
    lock = json.loads((base / "scored.lock").read_text())
    assert [r["deviation_rows"] for r in lock["runs"]] == [0, 2]
    with pytest.raises(ScoreRefused, match="needs a new row"):  # that row is used up
        score.score("stage1", gold, **paths(base), label_names=CLASSES)


def test_lock_is_written_before_results_so_a_crash_counts_as_a_run(fresh, monkeypatch):
    base, gold = fresh
    monkeypatch.setattr(score, "SECONDARY_COVERAGES", ())

    def crash(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(score.report, "write_all", crash)
    with pytest.raises(RuntimeError):
        score.score("stage1", gold, **paths(base), label_names=CLASSES)
    assert json.loads((base / "scored.lock").read_text())["runs"][0]["status"] == "started"
    monkeypatch.undo()
    with pytest.raises(ScoreRefused, match="needs a new row"):
        score.score("stage1", gold, **paths(base), label_names=CLASSES)


def test_deviation_rows_skip_the_header_and_rule(tmp_path):
    write_deviations(tmp_path / "d.md", ["one", "two"])
    assert len(score.deviation_rows(tmp_path / "d.md")) == 2
    assert score.deviation_rows(REPO / "deviations.md") == []  # the real table is empty before Stage 1
    assert score.deviation_rows(tmp_path / "missing.md") == []


# --------------------------------------------------------------------------- gold and published files


def test_gold_read_only_inside_score():
    """No module but score.py (and firewall.py, which defines it) loads test gold; report.py sees no gold."""
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text())
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
        }
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        if path.name not in ("score.py", "firewall.py"):
            assert "load_test" not in names | imported, path.name
    report_src = (SRC / "report.py").read_text()
    report_tree = ast.parse(report_src)
    report_imports = {a.name for n in ast.walk(report_tree) if isinstance(n, ast.Import) for a in n.names}
    report_imports |= {n.module for n in ast.walk(report_tree) if isinstance(n, ast.ImportFrom)}
    assert not report_imports & {"firewall", "data", "predictions", "score"}
    assert "gold" not in {a.arg for n in ast.walk(report_tree) if isinstance(n, ast.FunctionDef) for a in n.args.args}
    # score.py reaches the firewall in one place, the command line's _test_gold.
    score_tree = ast.parse((SRC / "score.py").read_text())
    callers = [
        f.name
        for f in ast.walk(score_tree)
        if isinstance(f, ast.FunctionDef) and any(isinstance(n, ast.Attribute) and n.attr == "load_test" for n in ast.walk(f))
    ]
    assert callers == ["_test_gold"]


def test_gold_label_ids_from_the_firewall_map_to_names():
    frame = pd.DataFrame({"row_id": [0, 1, 2], "label": [2, 0, 1]})
    got = score.gold_names(frame, ["a", "b", "c"])
    assert got["label"].tolist() == ["c", "a", "b"]
    with pytest.raises(ScoreRefused, match="label names"):
        score.gold_names(frame, None)


def test_published_files_never_contain_full_vectors(record_set, tmp_path):
    base, _, results = record_set
    out = base / "results"
    files = [p for p in out.rglob("*") if p.is_file()]
    assert files and not [p for p in files if p.suffix in (".npz", ".npy")]
    parquet = [p for p in files if p.suffix == ".parquet"]
    assert len(parquet) == 2 * (10 + 3 * 71) - 1  # every arm on both datasets; Jev + examples on Banking77 only
    for p in parquet:
        frame = pd.read_parquet(p)
        assert list(frame.columns) == score.PUBLISHED_COLUMNS, p.name
        assert all(np.ndim(v) == 0 for v in frame.to_numpy().ravel()), p.name
    jev = pd.read_parquet(out / "predictions" / "banking77" / "jev.parquet")
    assert jev["model"].str.startswith("typesafe/jev-1.14").all() and jev["cost_usd"].gt(0).all()
    assert jev["gold_prob"].between(0, 1).all()
    llm = pd.read_parquet(out / "predictions" / "banking77" / "small_llm.parquet")
    assert llm["gold_prob"].isna().all()  # no vector to read it from, so it isn't made up
    # A JSON or CSV holding a 3-class or 4-class probability list per row would show up as nested lists.
    text = (out / "results.json").read_text()
    assert '"probs"' not in text and '"vectors"' not in text
    # The guard itself refuses a planted vector column or array file.
    planted = tmp_path / "out"
    (planted / "predictions").mkdir(parents=True)
    pd.DataFrame({"row_id": ["a"], "probs": [[0.1, 0.9]]}).to_parquet(planted / "predictions" / "x.parquet")
    with pytest.raises(ScoreRefused, match="outside the published set"):
        score.assert_no_vectors(planted)
    (planted / "predictions" / "x.parquet").unlink()
    np.savez(planted / "v.npz", vectors=np.ones((2, 3)))
    with pytest.raises(ScoreRefused, match="array files"):
        score.assert_no_vectors(planted)


def test_chart_labels_all_with_real_counts_and_has_total_labels_axis(record_set):
    base, _, results = record_set
    assert (base / "results" / "chart.png").stat().st_size > 10_000
    sizes = json.loads((base / "runs" / "stage1" / "bank_sizes.json").read_text())
    for dataset, k in (("banking77", 3), ("ag_news", 4)):
        drawn = results["chart"][dataset]
        total_all = sizes[dataset]["d0_sall"]["rows"]
        assert drawn["x_labels"][:7] == ["1", "2", "3", "5", "8", "16", "35"]
        assert drawn["x_labels"][7] == f"all\n{total_all / k:.1f}/class avg\n{total_all:,} total"
        assert drawn["total_axis_title"] == report.TOTAL_AXIS_TITLE
        mean35 = np.mean([sizes[dataset][f"d{r}_s35"]["rows"] for r in range(10)])
        assert drawn["total_labels"][6] == f"{mean35:,.0f}" and drawn["total_labels"][7] == f"{total_all:,}"
    # The real counts in DESIGN.md come through the same labels.
    real = {s: {"total": 9999.0 if s == "all" else 77.0 * int(s), "per_class": 9999 / 77 if s == "all" else float(s)} for s in report.GRID_LABELS}
    per_class, totals = report.tick_labels(real)
    assert per_class[-1] == "all\n129.9/class avg\n9,999 total" and totals[-1] == "9,999"
    ag = {s: {"total": 119_852.0 if s == "all" else 4.0 * int(s), "per_class": 119_852 / 4 if s == "all" else float(s)} for s in report.GRID_LABELS}
    assert report.tick_labels(ag)[0][-1] == "all\n29963.0/class avg\n119,852 total"


# --------------------------------------------------------------------------- what gets computed


def test_his_claim_uses_embedding_similarity_at_005_against_jev_with_descriptions(fresh, monkeypatch):
    base, gold = fresh
    monkeypatch.setattr(score, "SECONDARY_COVERAGES", ())
    seen = []
    real = score.additions

    def spy(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(score, "additions", spy)
    score.score("stage1", gold, **paths(base), label_names=CLASSES)
    final = seen[-1]
    names = score.gold_names(gold["banking77"], CLASSES["banking77"])["label"].to_numpy()

    def array(arm):
        rows, vectors, classes = predictions.read(arm, "banking77", "stage1", base / "runs")
        return score.build_arm(arm, score.Loaded(rows, vectors, classes), names).array

    assert final["b"] == 10_000 and final["seed"] == 2026 and final["coverage"] == 0.8
    assert np.array_equal(final["jev"], array("jev"))
    assert not np.array_equal(final["jev"], array("jev_bare"))
    claim = final["his_claim_arms"]
    assert set(claim) == set(verdict_module.CLAIM_ARMS)
    assert np.array_equal(claim["embedding similarity"], array("embed_sim"))
    for other in ("embed_sim_t0.01", "embed_sim_t0.2", "embed_sim_raw", "embed_sim_bare"):
        assert not np.array_equal(claim["embedding similarity"], array(other)), other
    assert np.array_equal(claim["NLI"], array("nli")) and np.array_equal(claim["small LLM"], array("small_llm"))
    assert np.array_equal(final["jev_examples"], array("jev_examples"))
    assert isinstance(final["ag_news"], DsResult) and not final["ag_news"].partial


def test_verdict_gets_ds_and_s_all_is_fit_once_and_repeated(fresh, monkeypatch):
    base, gold = fresh
    monkeypatch.setattr(score, "SECONDARY_COVERAGES", ())
    calls, inputs = [], []
    real_verdict, real_bootstrap = score.verdict, score.bootstrap_ds
    monkeypatch.setattr(score, "verdict", lambda s_star, s_j, ds=None: calls.append(ds) or real_verdict(s_star, s_j, ds=ds))

    def spy_bootstrap(lr, jev, classes, *args, **kwargs):
        inputs.append(lr)
        return real_bootstrap(lr, jev, classes, *args, **kwargs)

    monkeypatch.setattr(score, "bootstrap_ds", spy_bootstrap)
    score.score("stage1", gold, **paths(base), label_names=CLASSES)
    assert calls and all(isinstance(ds, DsResult) and ds.n_replicates == 10_000 for ds in calls)
    lr = inputs[-1]
    assert all(np.array_equal(lr["all"][0], lr["all"][r]) for r in range(10))
    assert not all(np.array_equal(lr[8][0], lr[8][r]) for r in range(10))
    # Only draw 0 exists at s=all; other draws are never read there.
    assert [name for _, s, name in score.family_files("lr") if s == "all"] == ["lr_d0_sall"]


def test_all_secondary_rows_are_computed(record_set):
    base, _, results = record_set
    b77 = results["datasets"]["banking77"]
    arms = {row["arm"]: row for row in b77["arms"]}
    for name in ("embed_sim", "embed_sim_t0.01", "embed_sim_t0.2", "embed_sim_raw", "embed_sim_bare", "jev_bare"):
        assert arms[name]["acc_at_80"] is not None, name
    assert set(b77["sensitivity"]) == {"embed_sim", "embed_sim_t0.01", "embed_sim_t0.2", "embed_sim_raw"}
    # Calibration: ECE with its floor, NLL and Brier, except where DESIGN.md says not.
    assert {"ece", "ece_floor", "ece_failed_excluded", "nll", "brier"} <= set(arms["jev"])
    assert "nll" not in arms["small_llm"] and "brier" not in arms["small_llm"] and "ece" in arms["small_llm"]
    assert "ece" not in arms["embed_sim_raw"] and "nll" not in arms["embed_sim_raw"]
    assert "brier" in arms["lr_d3_s8"] and "nll" in arms["nli"]
    # Realised coverage from each arm's validation scores; cost and latency without any ratio.
    assert 0 < arms["jev"]["realised_coverage"] <= 1 and arms["lr_d0_s1"]["realised_threshold"] is not None
    assert arms["jev"]["cost_usd"] > 0 and arms["jev"]["latency_p95_s"] >= arms["jev"]["latency_p50_s"]
    assert arms["nli"]["cost_usd"] is None and arms["nli"]["latency_n"] == 18
    assert "ratio" not in json.dumps(results).lower()
    # s* and s_J at 50, 70, 90 and 100% coverage, with the verdict row compared to 80%.
    assert [c["coverage"] for c in b77["coverages"]] == [0.5, 0.7, 0.9, 1.0]
    assert all("verdict_row" in c and "differs_from_80" in c for c in b77["coverages"])
    # Risk-coverage, the grid with draw range and wins, isolated passes labelled, secondary LR arms.
    assert len(b77["risk_coverage"]["coverage"]) == 20 and "lr_s8_mean" in b77["risk_coverage"]["arms"]
    assert len(b77["grid"]) == 8 and {"gap_min", "gap_max", "draws_beating_jev"} <= set(b77["grid"][0])
    assert all(p["label"] == "not a claim" for p in b77["primary"]["isolated_lr_passes"])
    assert set(b77["secondary_families"]) == {"lr_cv", "tfidf_lr"}
    # AG News by the same rules, compared by verdict row; additions; headline; corrections.
    assert results["datasets"]["ag_news"]["primary"]["verdict_row"] in range(1, 6)
    assert "row_differs" in results["additions"]["ag_news"]
    assert results["headline"]["verdict"] == b77["primary"]["verdict"]
    assert any("fixed labels" in c for c in results["corrections"])
    for name in ("arms.csv", "ds_banking77.csv", "ds_ag_news.csv", "coverage_banking77.csv", "his_claim.csv",
                 "ds_lr_cv_banking77.csv", "ds_tfidf_lr_ag_news.csv", "risk_coverage_banking77.csv", "results.json"):  # fmt: skip
        assert (base / "results" / name).exists(), name


def test_coverage_notes_name_a_verdict_row_change_at_70_or_90_only():
    assert score.coverage_notes({0.5: 1, 0.7: 5, 0.9: 5, 1.0: 2}, 5) == []
    notes = score.coverage_notes({0.5: 5, 0.7: 3, 0.9: 5, 1.0: 5}, 5)
    assert notes == ["At 70% coverage the verdict table row is 3, not 5 as at 80%."]
    assert len(score.coverage_notes({0.7: 2, 0.9: 4}, 5)) == 2


@pytest.mark.parametrize("dataset,below,at", [("banking77", 30, 31), ("ag_news", 75, 76)])
def test_failure_flags_at_31_on_banking77_and_76_on_ag_news(dataset, below, at):
    assert score.failure_flag(dataset, below) is False
    assert score.failure_flag(dataset, at) is True


def test_a_flagged_arm_is_flagged_in_every_table(tmp_path):
    runs = tmp_path / "runs"
    val_gold = make_fake_runs(runs, "val", per_class={"banking77": 11, "ag_news": 20}, failures={("jev", "banking77"): 31, ("nli", "ag_news"): 75}, families=("lr",))
    results = score.dry_run(val_gold, runs_root=runs, settings=FAST, deviations_path=tmp_path / "d.md")
    b77 = results["datasets"]["banking77"]
    assert "jev" in b77["failure_flags"] and "nli" not in results["datasets"]["ag_news"]["failure_flags"]
    out = runs / "dryrun" / "results"
    arms = pd.read_csv(out / "arms.csv")
    assert arms.loc[(arms.arm == "jev") & (arms.dataset == "banking77"), "flagged"].item()
    assert not arms.loc[(arms.arm == "nli") & (arms.dataset == "ag_news"), "flagged"].item()
    assert (pd.read_csv(out / "ds_banking77.csv")["flags"] == "jev").all()
    assert pd.read_csv(out / "coverage_banking77.csv")["flags"].str.contains("jev").all()
    assert any("jev" in n for n in results["headline"]["notes"])
    assert pd.read_csv(out / "his_claim.csv")["jev_flagged"].all()
    assert "jev (flagged)" in pd.read_csv(out / "risk_coverage_banking77.csv").columns
    assert "jev" in results["chart"]["banking77"]["title"]


# --------------------------------------------------------------------------- the dry run


def test_dry_run_permutes_gold_and_writes_to_dryrun_only(tmp_path):
    runs = tmp_path / "runs"
    val_gold = make_fake_runs(runs, "val", families=("lr",))
    (tmp_path / "deviations.md").write_text("unchanged\n")
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    results = score.dry_run(val_gold, runs_root=runs, settings=FAST, deviations_path=tmp_path / "deviations.md")

    after = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    changed = [p for p in after if before.get(p) != after[p]]
    assert changed and all((runs / "dryrun") in p.parents for p in changed)
    assert all(after[p] == before[p] for p in before)  # nothing outside runs/dryrun was touched
    assert (runs / "dryrun" / "scored.lock").exists() and (runs / "dryrun" / "results" / "chart.png").exists()
    assert results["dry_run"] is True and "permuted" in results["gold"]

    # The gold scored is the seed-2026 permutation: same class counts, labels moved.
    for dataset, frame in val_gold.items():
        permuted = score.permute_gold(frame)
        assert permuted["row_id"].tolist() == frame["row_id"].tolist()
        assert sorted(permuted["label"]) == sorted(frame["label"])
        assert (permuted["label"] != frame["label"]).mean() > 0.3
        assert permuted.equals(score.permute_gold(frame, seed=2026))
    pub = pd.read_parquet(runs / "dryrun" / "results" / "predictions" / "banking77" / "lr_d2_s5.parquet")
    rows, vec, classes = predictions.read("lr_d2_s5", "banking77", "val", runs)
    permuted = score.permute_gold(val_gold["banking77"])["label"]
    expected = vec[np.arange(len(vec)), [classes.index(g) for g in permuted]]
    assert np.allclose(pub["gold_prob"], expected)

    # It can be run again: the folder is cleared and the dummy lock starts fresh.
    score.dry_run(val_gold, runs_root=runs, settings=FAST, deviations_path=tmp_path / "deviations.md")
    assert len(json.loads((runs / "dryrun" / "scored.lock").read_text())["runs"]) == 1


def test_dry_run_logs_absent_arms_as_not_run_inside_dryrun_only(tmp_path):
    runs = tmp_path / "runs"
    val_gold = make_fake_runs(runs, "val", families=("lr",), skip={("small_llm", "banking77"), ("jev_examples", "banking77"), ("tfidf_lr", "ag_news")})
    results = score.dry_run(val_gold, runs_root=runs, settings=FAST, deviations_path=tmp_path / "d.md")
    absent = json.loads((runs / "dryrun" / "absent_arms.json").read_text())
    assert {"arm": "small_llm", "dataset": "banking77"} in absent and {"arm": "tfidf_lr", "dataset": "ag_news"} in absent
    assert not (runs / "val" / "not_run.jsonl").read_text().count("dry run")
    assert results["additions"]["jev_examples"] == {"status": "not run"}


def test_dry_run_needs_jev_and_lr_on_banking77(tmp_path):
    runs = tmp_path / "runs"
    val_gold = make_fake_runs(runs, "val", families=("lr",), skip={("jev", "banking77")})
    with pytest.raises(ScoreRefused, match="jev on banking77"):
        score.dry_run(val_gold, runs_root=runs, settings=FAST, deviations_path=tmp_path / "d.md")


def test_dry_run_cli_prints_only_ok_or_a_failure(tmp_path, capsys):
    runs = tmp_path / "runs"
    val_gold = make_fake_runs(runs, "val", families=("lr",))
    code = score.main(["--dry-run"], runs_root=runs, load_val_gold=lambda: val_gold, settings=FAST)
    assert code == 0 and capsys.readouterr().out == "DRY RUN OK\n"
    predictions.paths("jev", "banking77", "val", runs)[0].unlink()
    code = score.main(["--dry-run"], runs_root=runs, load_val_gold=lambda: val_gold, settings=FAST)
    out = capsys.readouterr().out
    assert code == 1 and out.startswith("DRY RUN FAILED:") and out.count("\n") == 1
    assert not any(ch.isdigit() for ch in out.replace("banking77", ""))  # no numbers, so no accuracy


def test_dry_run_at_settings_of_record_on_fake_validation(tmp_path):
    """The dry run's default is the record's B, seed and coverage, end to end."""
    runs = tmp_path / "runs"
    val_gold = make_fake_runs(runs, "val", families=("lr",), per_class={"banking77": 4, "ag_news": 3})
    results = score.dry_run(val_gold, runs_root=runs, deviations_path=tmp_path / "d.md")
    assert results["settings"] == {"b": 10_000, "seed": 2026, "coverage": 0.8}
    assert set(results["datasets"]) == {"banking77", "ag_news"}
