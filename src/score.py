"""The one-shot scorer (DESIGN.md, "Metric", "Outcome rules"; VALIDATION.md, "One-shot scorer").

This is the only code that reads gold labels and produces the published result. Gold reaches it two ways:
the firewall's gold frames at the test stage, or validation labels in the dry run, where they are
permuted first (seed 2026) so the output carries no real comparison.

    PYTHONPATH=src uv run python -m score --dry-run        # prints DRY RUN OK or DRY RUN FAILED: ...
    PYTHONPATH=src uv run python -m score --stage stage1   # the result of record, once

It runs only when every arm has a prediction file or a logged "not run" (predictions.mark_not_run), and
every file holds the gold rows in gold's order. It writes scored.lock before the first result is
computed, so a rerun after a crash counts as a rerun, and a rerun needs a new row in deviations.md whose
Stage is exactly "rescore". Every refusal happens before the lock. The result of record uses B=10,000, seed 2026 and 80% coverage and nothing else.
Published files never hold a full probability vector.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd

import arms_local
import guards
import predictions
import report
from metrics import GRID, N_DRAWS, DsResult, acc_at, bootstrap_ds, brier, ece_with_floor, nll
from verdict import (
    CLAIM_ARMS,
    JEV_SEQUENCE,
    LR_SEQUENCE,
    NOT_A_CLAIM,
    NOT_RUN,
    additions,
    equivalent_within_2,
    isolated_lr_passes,
    jev_ahead,
    lr_ahead,
    verdict,
    verdict_row,
)

REPO = Path(__file__).resolve().parents[1]
RUNS = predictions.RUNS
RESULTS = REPO / "results"
LOCK = REPO / "scored.lock"
DEVIATIONS = REPO / "deviations.md"
DRYRUN = "dryrun"
VAL_STAGE = "val"

DATASETS = ("banking77", "ag_news")
GOLD_SEED = 2026  # the dry run's permutation of validation gold
LIMIT_HOURS = 1.0  # refuse the bootstrap work if it projects above this
PROJECTION_REPLICATES = 100
SECONDARY_COVERAGES = (0.5, 0.7, 0.9, 1.0)
HEADLINE_COVERAGES = (0.7, 0.9)  # a verdict-row change at these goes in the headline
RISK_COVERAGE_POINTS = tuple(round(0.05 * i, 2) for i in range(1, 21))
VALIDATION_KEEP = 0.8  # realised coverage: threshold keeps the top 80% of an arm's validation scores

# DESIGN.md, "Failures": 31 or more on Banking77, 76 or more on AG News, flags the arm in every table.
FAIL_FLAG_AT = {"banking77": 31, "ag_news": 76}

# Arm names, as prediction files are named under runs/<stage>/<arm>__<dataset>.parquet.
JEV, JEV_BARE, JEV_EXAMPLES, SMALL_LLM, NLI = "jev", "jev_bare", "jev_examples", "small_llm", "nli"
EMBED, EMBED_BARE, EMBED_RAW = "embed_sim", "embed_sim_bare", "embed_sim_raw"
EMBED_SENSITIVITY = ("embed_sim_t0.01", "embed_sim_t0.2", EMBED_RAW)
SINGLE_ARMS = (JEV, JEV_BARE, NLI, EMBED, *EMBED_SENSITIVITY, EMBED_BARE, SMALL_LLM)
BANKING77_ONLY = (JEV_EXAMPLES,)
FAMILIES = ("lr", "lr_cv", "tfidf_lr")
PRIMARY_FAMILY = "lr"
API_ARMS = (JEV, JEV_BARE, JEV_EXAMPLES, SMALL_LLM)
# The private file holds raw cosine for these; the score is softmax(cosine / 0.05) (DESIGN.md, Scores).
COSINE_VECTORS = {EMBED: arms_local.EMBED_TEMPERATURE, EMBED_BARE: arms_local.EMBED_TEMPERATURE}
# Ranked by raw top cosine: not a probability, so no ECE, NLL or Brier.
RANK_ONLY = {EMBED_RAW}
# DESIGN.md: the small LLM's NLL and Brier are not reported (it gives only its pick's probability).
NO_PROPER_SCORES = {SMALL_LLM, EMBED_RAW}

PUBLISHED_COLUMNS = ["row_id", "pred", "top_prob", "failed", "gold_prob", "model", "input_tokens", "output_tokens", "cost_usd", "latency_s"]


class ScoreRefused(RuntimeError):
    """A condition for scoring does not hold. Nothing has been written."""


class PublishGuardError(RuntimeError):
    """The scorer's own output broke the publishing rule: a bug, found after the lock, never a refusal."""


@dataclass(frozen=True)
class Settings:
    b: int = 10_000
    seed: int = 2026
    coverage: float = 0.8


RECORD = Settings()


def check_record(settings: Settings) -> None:
    """The result of record uses exactly B=10,000, seed 2026 and 80% coverage (DESIGN.md, Bootstrap)."""
    if settings != RECORD:
        raise ScoreRefused(f"the result of record needs b=10000, seed=2026, coverage=0.8; got {asdict(settings)}")


def failure_flag(dataset: str, n_failed: int) -> bool:
    return int(n_failed) >= FAIL_FLAG_AT[dataset]


def model_id(arm: str) -> str:
    """Model behind a local arm, for the published files. API arms carry their own per row."""
    base = arm.split("_d")[0] if arm.startswith(FAMILIES) else arm
    if base.startswith("embed_sim"):
        return f"{arms_local.EMBED_MODEL}@{arms_local.EMBED_REVISION}"
    if base == "nli_base":
        return "@".join(arms_local.NLI_MODELS["base"])
    if base == NLI:
        return "@".join(arms_local.NLI_MODELS["large"])
    if base in ("lr", "lr_cv"):
        return f"{arms_local.EMBED_MODEL}@{arms_local.EMBED_REVISION} + logistic regression"
    if base == "tfidf_lr":
        return "tf-idf word 1-2 grams + logistic regression"
    return ""


def family_files(family: str) -> list[tuple[int, object, str]]:
    """(draw, s, arm name) for a family. s=all is fit once, as draw 0, and repeated for every draw."""
    out = []
    for s in GRID:
        for draw in (0,) if s == "all" else range(N_DRAWS):
            out.append((draw, s, f"{family}_d{draw}_s{s}"))
    return out


def expected_single_arms(dataset: str) -> tuple:
    return SINGLE_ARMS + (BANKING77_ONLY if dataset == "banking77" else ())


# --------------------------------------------------------------------------- loading


@dataclass
class Loaded:
    rows: pd.DataFrame
    vectors: Optional[np.ndarray]
    classes: Optional[list]


@dataclass
class DatasetInput:
    dataset: str
    single: dict = field(default_factory=dict)  # arm -> Loaded
    families: dict = field(default_factory=dict)  # family -> {arm name: Loaded}
    not_run: dict = field(default_factory=dict)  # arm or family -> reason


def load_arms(stage: str, dataset: str, root: Path, not_run: Mapping) -> DatasetInput:
    """Every expected arm as a prediction file or a not-run record; refuses anything else."""
    out = DatasetInput(dataset)
    problems = []
    for arm in expected_single_arms(dataset):
        loaded = predictions.read(arm, dataset, stage, root)
        logged = not_run.get((arm, dataset))
        if loaded is not None and logged is not None:
            problems.append(f"{arm} has both predictions and a not-run record")
        elif loaded is not None:
            out.single[arm] = Loaded(*loaded)
        elif logged is not None:
            out.not_run[arm] = logged
        else:
            problems.append(f"{arm} has neither predictions nor a not-run record")
    for family in FAMILIES:
        files = {name: predictions.read(name, dataset, stage, root) for _, _, name in family_files(family)}
        present = {name: Loaded(*got) for name, got in files.items() if got is not None}
        logged = not_run.get((family, dataset))
        if logged is not None and present:
            problems.append(f"{family} has both predictions and a not-run record")
        elif logged is not None:
            out.not_run[family] = logged
        elif len(present) == len(files):
            out.families[family] = present
        elif present:
            missing = sorted(set(files) - set(present))
            problems.append(f"{family} is missing {len(missing)} of {len(files)} files, first {missing[0]}")
        else:
            problems.append(f"{family} has neither predictions nor a not-run record")
    if problems:
        raise ScoreRefused(f"{dataset}: " + "; ".join(problems))
    return out


def gold_names(frame: pd.DataFrame, names: Optional[list]) -> pd.DataFrame:
    """Gold as (row_id, label name). The firewall gives label ids; validation gives names."""
    if list(frame.columns) != ["row_id", "label"]:
        raise ScoreRefused(f"gold must have columns row_id, label; got {list(frame.columns)}")
    labels = frame["label"]
    if pd.api.types.is_integer_dtype(labels):
        if names is None:
            raise ScoreRefused("gold holds label ids, so label names are needed")
        labels = labels.map(lambda i: names[int(i)])
    return pd.DataFrame({"row_id": frame["row_id"].to_numpy(), "label": labels.astype(str).to_numpy()})


def check_labels(gold: pd.DataFrame, data: DatasetInput, names: Optional[list]) -> None:
    """Gold and every answered prediction must be a known label name, or a mapping slip scores every row wrong."""
    allowed = set(map(str, names)) if names is not None else set(gold["label"])
    bad_gold = sorted(set(gold["label"]) - allowed)
    if bad_gold:
        raise ScoreRefused(f"{data.dataset}: gold labels outside the label names, e.g. {bad_gold[0]!r}")
    arms = list(data.single.items()) + [item for fam in data.families.values() for item in fam.items()]
    for name, loaded in arms:
        rows = loaded.rows
        answered = rows.loc[~rows["failed"].to_numpy(dtype=bool), "pred"].astype(str)
        outside = sorted(set(answered) - allowed)
        if outside:
            raise ScoreRefused(f"{data.dataset}: {name} predicts labels outside the label names, e.g. {outside[0]!r}")


def check_arm(dataset: str, arm: "Arm") -> None:
    """Every input a metric reads must be usable, so no metric can refuse after the lock."""
    ok = ~arm.failed
    conf = arm.conf[ok]
    if not np.isfinite(conf).all():
        raise ScoreRefused(f"{dataset}: {arm.name} has an answered row with no top probability")
    base = arm.name.split("_d")[0] if arm.name.startswith(FAMILIES) else arm.name
    low = -1.0 if base in RANK_ONLY else 0.0
    if ((conf < low - 1e-9) | (conf > 1 + 1e-9)).any():
        raise ScoreRefused(f"{dataset}: {arm.name} has a top probability outside [{low:g}, 1]")
    if arm.probs is not None and not np.isfinite(arm.probs[ok]).all():
        raise ScoreRefused(f"{dataset}: {arm.name} has an answered row with no probability vector")


def check_row_order(gold: pd.DataFrame, data: DatasetInput) -> None:
    """Every arm must hold exactly gold's rows, in gold's order, before anything is scored."""
    want = gold["row_id"].astype(str).tolist()
    if len(set(want)) != len(want):
        raise ScoreRefused(f"{data.dataset}: gold has repeated row ids")
    arms = list(data.single.items()) + [item for fam in data.families.values() for item in fam.items()]
    for name, loaded in arms:
        got = loaded.rows["row_id"].astype(str).tolist()
        if got != want:
            same_set = sorted(got) == sorted(want)
            raise ScoreRefused(
                f"{data.dataset}: {name} row ids differ from gold's "
                + ("order" if same_set else f"rows ({len(got)} rows, gold has {len(want)})")
            )


# --------------------------------------------------------------------------- per-arm arrays


@dataclass
class Arm:
    name: str
    conf: np.ndarray
    correct: np.ndarray
    failed: np.ndarray
    gold_prob: Optional[np.ndarray]  # None when there's no probability vector to read it from
    probs: Optional[np.ndarray]  # (rows, classes) probabilities, private, for Brier
    gold_index: Optional[np.ndarray]
    rows: pd.DataFrame

    @property
    def array(self) -> np.ndarray:
        """(rows, 3): top probability, correct, failed, as metrics.py takes it."""
        return np.column_stack([np.where(self.failed, 0.0, self.conf), self.correct, self.failed]).astype(float)


def build_arm(name: str, loaded: Loaded, gold_labels: np.ndarray) -> Arm:
    rows = loaded.rows
    failed = rows["failed"].to_numpy(dtype=bool)
    conf = rows["top_prob"].to_numpy(dtype=float)
    correct = (rows["pred"].astype(str).to_numpy() == gold_labels) & ~failed
    probs = gold_prob = gold_index = None
    base = name.split("_d")[0] if name.startswith(FAMILIES) else name
    if loaded.vectors is not None and base not in RANK_ONLY:
        vec = np.asarray(loaded.vectors, dtype=float)
        if base in COSINE_VECTORS:
            safe = np.where(failed[:, None], 0.0, vec)
            vec = np.where(failed[:, None], np.nan, arms_local.softmax(safe / COSINE_VECTORS[base]))
        index = {c: i for i, c in enumerate(loaded.classes)}
        unknown = sorted(set(gold_labels) - set(index))
        if unknown:
            raise ScoreRefused(f"{name}: gold labels outside its classes, e.g. {unknown[0]!r}")
        gold_index = np.array([index[g] for g in gold_labels], dtype=np.intp)
        probs = vec
        gold_prob = np.where(failed, 0.0, vec[np.arange(len(vec)), gold_index])
    return Arm(name, conf, correct, failed, gold_prob, probs, gold_index, rows)


def lr_input(family: Mapping[str, Arm], prefix: str) -> dict:
    """{s: (10, rows, 3)} for bootstrap_ds. s=all is draw 0, fit once and repeated for all 10 draws."""
    out = {}
    for s in GRID:
        if s == "all":
            one = family[f"{prefix}_d0_sall"].array
            out[s] = np.repeat(one[None], N_DRAWS, axis=0)
        else:
            out[s] = np.stack([family[f"{prefix}_d{r}_s{s}"].array for r in range(N_DRAWS)])
    return out


# --------------------------------------------------------------------------- tables


def _p(value):
    return None if value is None or (isinstance(value, float) and not np.isfinite(value)) else value


def arm_row(dataset: str, arm: Arm, coverage: float, val_rows: Optional[pd.DataFrame]) -> dict:
    n_failed = int(arm.failed.sum())
    base = arm.name.split("_d")[0] if arm.name.startswith(FAMILIES) else arm.name
    row = {
        "dataset": dataset,
        "arm": arm.name,
        "n": int(len(arm.conf)),
        "failures": n_failed,
        "flagged": failure_flag(dataset, n_failed),
        "acc_at_80": 100 * acc_at(arm.array[:, 0], arm.correct, arm.failed, coverage),
        "share_at_1": float(np.mean(arm.conf[~arm.failed] >= 1.0)) if (~arm.failed).any() else None,
    }
    if base not in RANK_ONLY and (~arm.failed).any():
        ece = ece_with_floor(arm.conf, arm.correct, arm.failed)
        row.update(ece=ece["ece"], ece_floor=ece["floor"], ece_n=ece["n"], ece_failed_excluded=ece["failed_excluded"])
    if base not in NO_PROPER_SCORES and arm.gold_prob is not None:
        row["nll"] = nll(arm.gold_prob, arm.failed)
        row["brier"] = brier(arm.probs, arm.gold_index, arm.failed)
    row.update(realised_coverage(arm, val_rows))
    row.update(cost_and_latency(arm.rows))
    return row


def realised_coverage(arm: Arm, val_rows: Optional[pd.DataFrame]) -> dict:
    """Threshold = the score that keeps the top 80% of the arm's unlabelled validation scores
    (their 20th percentile); realised coverage = share of rows at or above it, failed rows included."""
    if val_rows is None:
        return {"realised_coverage": None, "realised_threshold": None, "acc_at_realised": None}
    val_ok = val_rows.loc[~val_rows["failed"].to_numpy(dtype=bool), "top_prob"].to_numpy(dtype=float)
    if len(val_ok) == 0:
        return {"realised_coverage": None, "realised_threshold": None, "acc_at_realised": None}
    threshold = float(np.quantile(val_ok, 1 - VALIDATION_KEEP))
    covered = arm.failed | (arm.conf >= threshold)
    return {
        "realised_coverage": float(covered.mean()),
        "realised_threshold": threshold,
        "acc_at_realised": 100 * float(arm.correct[covered].mean()) if covered.any() else None,
    }


# Timing in prediction files is pipeline time per row (batch-amortised for local arms, concurrent for API
# arms), not latency. Latency comes from latency_probe.py and is reported separately.
PIPE_P50 = "pipeline time per row p50 s (not latency)"
PIPE_P95 = "pipeline time per row p95 s (not latency)"
PIPE_N = "pipeline time rows"


def cost_and_latency(rows: pd.DataFrame) -> dict:
    out = {"cost_usd": None, PIPE_P50: None, PIPE_P95: None, PIPE_N: 0}
    if "cost" in rows.columns:
        out["cost_usd"] = float(rows["cost"].sum())
    lat = rows["latency_s"].to_numpy(dtype=float) if "latency_s" in rows.columns else np.array([])
    if "cached" in rows.columns:
        lat = lat[~rows["cached"].to_numpy(dtype=bool)]
    lat = lat[np.isfinite(lat)]
    if len(lat):
        p50, p95 = np.percentile(lat, [50, 95])
        out.update({PIPE_P50: float(p50), PIPE_P95: float(p95), PIPE_N: int(len(lat))})
    return out


def latency_probes(runs_root: Path) -> dict:
    """The latency probes' p50 and p95 by condition, if runs/latency/probes.jsonl exists."""
    import latency_probe

    path = Path(runs_root) / "latency" / "probes.jsonl"
    if not path.exists():
        return {"status": "missing", "note": "no latency probes were run; pipeline times are not latency"}
    results = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"status": "run", **latency_probe.build_report(results)}


def risk_coverage(arm: Arm) -> list:
    return [100 * acc_at(arm.array[:, 0], arm.correct, arm.failed, c) for c in RISK_COVERAGE_POINTS]


def grid_table(ds: DsResult, flags: list) -> list:
    rows = []
    for s in GRID:
        lr = np.asarray(ds.lr_acc[s])
        rows.append(
            {
                "s": str(s),
                "d_s": ds.d[s],
                "lower": ds.lower[s],
                "upper": ds.upper[s],
                "lr_acc_mean": float(lr.mean()),
                "gap_min": ds.draw_range[s][0],
                "gap_max": ds.draw_range[s][1],
                "draws_beating_jev": ds.draws_beating_jev[s],
                "lr_passes": bool(ds.lower[s] > 0 and ds.d[s] >= 2),
                "jev_passes": bool(ds.upper[s] < 0 and ds.d[s] <= -2),
                "jev_acc": ds.jev_acc,
                "flags": ";".join(flags),
            }
        )
    return rows


def decide(ds: DsResult) -> dict:
    s_star, s_j = lr_ahead(ds), jev_ahead(ds)
    return {
        "s_star": str(s_star),
        "s_j": str(s_j),
        "verdict_row": verdict_row(s_star, s_j),
        "verdict": verdict(s_star, s_j, ds=ds),
        "equivalent_within_2": equivalent_within_2(ds),
        "isolated_lr_passes": [{"s": str(s), "label": NOT_A_CLAIM} for s in isolated_lr_passes(ds)],
    }


def coverage_notes(rows_by_coverage: Mapping[float, int], row_at_80: int) -> list:
    """Headline sentences for a verdict-row change at 70% or 90% coverage (DESIGN.md, Secondary)."""
    notes = []
    for c in HEADLINE_COVERAGES:
        row = rows_by_coverage.get(c)
        if row is not None and row != row_at_80:
            notes.append(f"At {round(100 * c)}% coverage the verdict table row is {row}, not {row_at_80} as at 80%.")
    return notes


def bank_size_table(sizes: Mapping, n_classes: int) -> dict:
    """{s: {total, per_class}} from runs/<stage>/bank_sizes.json, mean over draws."""
    out = {}
    for s in GRID:
        draws = [sizes[f"d{r}_s{s}"]["rows"] for r in ((0,) if s == "all" else range(N_DRAWS)) if f"d{r}_s{s}" in sizes]
        if not draws:
            raise ScoreRefused(f"bank_sizes.json has no entry for s={s}")
        total = float(np.mean(draws))
        out[str(s)] = {"total": total, "per_class": total / n_classes}
    return out


# --------------------------------------------------------------------------- lock and deviations


RESCORE_STAGE = "rescore"


def _cells(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def deviation_rows(path: Path) -> list:
    """Data rows of the deviations table: lines that start with "|", minus the header and its rule."""
    if not Path(path).exists():
        return []
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip().startswith("|")]
    return [ln for ln in lines[2:] if not re.fullmatch(r"\|[\s:|-]*", ln)]


def _stage_column(path: Path) -> int:
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip().startswith("|")]
    header = _cells(lines[0]) if lines else []
    if "Stage" not in header:
        raise ScoreRefused(f"{Path(path).name} has no Stage column")
    return header.index("Stage")


def check_lock(lock_path: Path, deviations_path: Path) -> list:
    """Refuse a rerun unless a deviations row about scoring was added since the last run. Returns prior runs."""
    if not Path(lock_path).exists():
        return []
    lock = json.loads(Path(lock_path).read_text())
    runs = lock.get("runs", [])
    seen = runs[-1]["deviation_rows"] if runs else 0
    new = deviation_rows(deviations_path)[seen:]
    column = _stage_column(deviations_path) if new else 0
    if not any(len(_cells(row)) > column and _cells(row)[column] == RESCORE_STAGE for row in new):
        raise ScoreRefused(
            f"{Path(lock_path).name} exists: rerunning the scorer needs a new row in {Path(deviations_path).name} "
            f"with Stage '{RESCORE_STAGE}' saying why"
        )
    return runs


def write_lock(lock_path: Path, runs: list, entry: dict) -> None:
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(lock_path).with_name(Path(lock_path).name + ".tmp")
    tmp.write_text(json.dumps({"runs": runs + [entry]}, indent=1))
    tmp.replace(lock_path)


# --------------------------------------------------------------------------- the bootstrap work


@dataclass
class Job:
    name: str
    run: Callable[..., object]  # run(n_replicates=None) gives the full result; a number gives a timing run


def project(jobs: list, b: int, clock: Callable[[], float], limit_hours: float) -> float:
    """Time each job at 100 replicates and refuse if the whole projects above the limit."""
    total = 0.0
    n = min(PROJECTION_REPLICATES, b)
    for job in jobs:
        total += guards.project_bootstrap(job.run, n, b, limit_hours, clock).projected_hours
    if total > limit_hours:
        raise guards.BootstrapTooSlow(f"the scorer's bootstrap work projects to {total:.2f} h, above the {limit_hours:g} h limit")
    return total


def _ds_job(name, lr, jev, classes, settings: Settings, coverage: float) -> Job:
    def run(n_replicates=None):
        return bootstrap_ds(lr, jev, classes, settings.b, settings.seed, n_replicates=n_replicates, coverage=coverage)

    return Job(name, run)


# --------------------------------------------------------------------------- scoring


def score(
    stage: str,
    gold: Mapping[str, pd.DataFrame],
    *,
    runs_root: Path = RUNS,
    out_dir: Path = RESULTS,
    lock_path: Path = LOCK,
    deviations_path: Path = DEVIATIONS,
    label_names: Optional[Mapping[str, list]] = None,
    val_stage: str = VAL_STAGE,
    clock: Callable[[], float] = time.monotonic,
    limit_hours: float = LIMIT_HOURS,
) -> dict:
    """The result of record: B=10,000, seed 2026, 80% coverage, written once to `out_dir` and `lock_path`."""
    if stage in (VAL_STAGE, DRYRUN, "private"):
        raise ScoreRefused(f"stage {stage!r} is not a test stage; validation goes through dry_run")
    return _score(
        stage, gold, RECORD, runs_root=runs_root, out_dir=out_dir, lock_path=lock_path, deviations_path=deviations_path,
        label_names=label_names, val_stage=val_stage, clock=clock, limit_hours=limit_hours, record=True,
    )  # fmt: skip


def permute_gold(frame: pd.DataFrame, seed: int = GOLD_SEED) -> pd.DataFrame:
    """Same rows and class counts, labels shuffled across rows, so nothing scored against it is real."""
    labels = np.random.default_rng(seed).permutation(frame["label"].to_numpy())
    return pd.DataFrame({"row_id": frame["row_id"].to_numpy(), "label": labels})


def dry_run(
    val_gold: Mapping[str, pd.DataFrame],
    *,
    runs_root: Path = RUNS,
    val_stage: str = VAL_STAGE,
    settings: Settings = RECORD,
    deviations_path: Path = DEVIATIONS,
    clock: Callable[[], float] = time.monotonic,
    limit_hours: float = LIMIT_HOURS,
    label_names: Optional[Mapping[str, list]] = None,
) -> dict:
    """End to end on validation predictions, gold permuted (seed 2026), writing only to runs/dryrun/.

    The folder is cleared first and holds its own dummy lock. Arms with no validation predictions are
    logged not run inside runs/dryrun/ only; Jev and embedding LR on Banking77 must be there.
    """
    out = Path(runs_root) / DRYRUN
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for need in (JEV, PRIMARY_FAMILY):
        names = [need] if need == JEV else [n for _, _, n in family_files(need)]
        if not all(predictions.paths(n, "banking77", val_stage, runs_root)[0].exists() for n in names):
            raise ScoreRefused(f"no validation predictions for {need} on banking77")
    logged = predictions.read_not_run(val_stage, runs_root)
    absent = {}
    for dataset in val_gold:
        for arm in expected_single_arms(dataset):
            if (arm, dataset) not in logged and not predictions.paths(arm, dataset, val_stage, runs_root)[0].exists():
                absent[(arm, dataset)] = "no validation predictions (dry run)"
        for family in FAMILIES:
            names = [n for _, _, n in family_files(family)]
            if (family, dataset) not in logged and not any(predictions.paths(n, dataset, val_stage, runs_root)[0].exists() for n in names):
                absent[(family, dataset)] = "no validation predictions (dry run)"
    (out / "absent_arms.json").write_text(json.dumps([{"arm": a, "dataset": d} for a, d in absent], indent=1))
    permuted = {dataset: permute_gold(frame) for dataset, frame in val_gold.items()}
    return _score(
        val_stage, permuted, settings, runs_root=runs_root, out_dir=out / "results", lock_path=out / "scored.lock",
        deviations_path=deviations_path, label_names=label_names, val_stage=val_stage, clock=clock, limit_hours=limit_hours,
        record=False, extra_not_run=absent,
    )  # fmt: skip


def _score(
    stage,
    gold,
    settings: Settings,
    *,
    runs_root,
    out_dir,
    lock_path,
    deviations_path,
    label_names,
    val_stage,
    clock,
    limit_hours,
    record: bool,
    extra_not_run: Optional[Mapping] = None,
) -> dict:
    if record:
        check_record(settings)
    if "banking77" not in gold:
        raise ScoreRefused("Banking77 gold is required: it alone sets the verdict")
    unknown = set(gold) - set(DATASETS)
    if unknown:
        raise ScoreRefused(f"unknown datasets {sorted(unknown)}")
    if (Path(out_dir) / "results.json").exists() and not Path(lock_path).exists():
        raise ScoreRefused(f"{Path(out_dir).name}/results.json exists without {Path(lock_path).name}; find out why before scoring")
    prior_runs = check_lock(lock_path, deviations_path)
    for d in DATASETS:
        if d not in gold and any((Path(runs_root) / stage).glob(f"*__{d}.parquet")):
            raise ScoreRefused(f"{d} has prediction files but no gold was given")

    # 1. Every arm accounted for, and every file in gold's row order. Only row ids are compared here.
    not_run = {**predictions.read_not_run(stage, runs_root), **(extra_not_run or {})}
    inputs = {d: load_arms(stage, d, runs_root, not_run) for d in DATASETS if d in gold}
    golds = {d: gold_names(gold[d], (label_names or {}).get(d)) for d in inputs}
    for d in inputs:
        check_row_order(golds[d], inputs[d])
        check_labels(golds[d], inputs[d], (label_names or {}).get(d))

    # 2. Arrays, and every input a metric will need, checked now: nothing may refuse once the lock is down.
    sizes_path = Path(runs_root) / stage / "bank_sizes.json"
    all_sizes = json.loads(sizes_path.read_text()) if sizes_path.exists() else {}
    size_tables = {}
    for d in inputs:
        if PRIMARY_FAMILY not in inputs[d].families:
            continue
        if d not in all_sizes:
            raise ScoreRefused(f"{sizes_path.name} is missing {d}, which has LR predictions; the chart needs real counts")
        size_tables[d] = bank_size_table(all_sizes[d], max(v["classes"] for v in all_sizes[d].values()))
    arms, fams, val_rows = {}, {}, {}
    for d, data in inputs.items():
        labels = golds[d]["label"].to_numpy()
        arms[d] = {name: build_arm(name, loaded, labels) for name, loaded in data.single.items()}
        fams[d] = {f: {name: build_arm(name, loaded, labels) for name, loaded in files.items()} for f, files in data.families.items()}
        for name, arm in list(arms[d].items()) + [item for fam in fams[d].values() for item in fam.items()]:
            check_arm(d, arm)
            got = predictions.read(name, d, val_stage, runs_root)
            val_rows[(d, name)] = None if got is None else got[0]
    probes = latency_probes(runs_root)

    # 3. The bootstrap work, projected from 100 replicates before any of it runs.
    jobs: list[Job] = []
    for d in inputs:
        classes = golds[d]["label"].to_numpy()
        if JEV not in arms[d]:
            continue
        jev = arms[d][JEV].array
        for f in fams[d]:
            lr = lr_input(fams[d][f], f)
            coverages = (settings.coverage,)
            if f == PRIMARY_FAMILY:  # Banking77's set the headline note; AG News's are a table only
                coverages += SECONDARY_COVERAGES
            for c in coverages:
                jobs.append(_ds_job((d, f, c), lr, jev, classes, settings, c))
    b77 = arms["banking77"]
    can_add = JEV in b77 and PRIMARY_FAMILY in fams["banking77"]
    if can_add:
        lr_s8 = lr_input(fams["banking77"][PRIMARY_FAMILY], PRIMARY_FAMILY)[8]
        claim_arms = {
            "NLI": b77[NLI].array if NLI in b77 else None,
            # His claim: embedding similarity with label descriptions at temperature 0.05, not a sensitivity row.
            "embedding similarity": b77[EMBED].array if EMBED in b77 else None,
            "small LLM": b77[SMALL_LLM].array if SMALL_LLM in b77 else None,
        }
        assert set(claim_arms) == set(CLAIM_ARMS)
        examples = b77[JEV_EXAMPLES].array if JEV_EXAMPLES in b77 else None
        b77_classes = golds["banking77"]["label"].to_numpy()

        def run_additions(n_replicates=None, s_star="none", s_j="none", ag=None):
            b = settings.b if n_replicates is None else n_replicates
            return additions(
                jev=b77[JEV].array, classes=b77_classes, lr_s8=lr_s8, his_claim_arms=claim_arms,
                jev_examples=examples, banking_s_star=s_star, banking_s_j=s_j, ag_news=ag,
                b=b, seed=settings.seed, coverage=settings.coverage,
            )  # fmt: skip

        timing_jobs = jobs + [Job("additions", run_additions)]
    else:
        timing_jobs = jobs
    projected_hours = project(timing_jobs, settings.b, clock, limit_hours)

    # 4. From here on results exist, so the lock goes down first.
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "stage": stage,
        "dry_run": not record,
        "started": started,
        "status": "started",
        "settings": asdict(settings),
        "deviation_rows": len(deviation_rows(deviations_path)),
    }
    write_lock(lock_path, prior_runs, entry)

    full = {job.name: job.run() for job in jobs}
    if record:  # a bug guard, not a refusal: check_record already held before the lock
        for (d, f, c), ds in full.items():
            if ds.b != RECORD.b or ds.n_replicates != RECORD.b or ds.partial or ds.coverage != c:
                raise RuntimeError(f"bug: {d} {f} at {c} did not run at the settings of record")
            if f == PRIMARY_FAMILY and c == settings.coverage and ds.coverage != RECORD.coverage:
                raise RuntimeError("bug: the primary contrast must be at 80% coverage")

    results = {
        "stage": stage,
        "dry_run": not record,
        "gold": "validation labels permuted with seed 2026: no real comparison" if not record else "official test",
        "settings": asdict(settings),
        "projected_bootstrap_hours": projected_hours,
        "not_run": [{"arm": a, "dataset": d, "reason": r} for d, data in inputs.items() for a, r in data.not_run.items()],
        "datasets": {},
    }
    for d, data in inputs.items():
        results["datasets"][d] = dataset_results(d, arms[d], fams[d], golds[d], full, settings, size_tables.get(d), val_rows)
    results["latency_probes"] = probes

    b77r = results["datasets"]["banking77"]
    primary = b77r.get("primary")
    if can_add and primary:
        ag = full.get(("ag_news", PRIMARY_FAMILY, settings.coverage))
        results["additions"] = run_additions(None, _s(primary["s_star"]), _s(primary["s_j"]), ag)
    else:
        results["additions"] = {"status": NOT_RUN}
    results["headline"] = headline(b77r)
    results["corrections"] = corrections(results["additions"])
    b77_flags = set(b77r.get("failure_flags", []))
    results["additions_flags"] = {
        "jev": JEV in b77_flags,
        "jev_examples": JEV_EXAMPLES in b77_flags,
        **{claim: arm in b77_flags for claim, arm in zip(CLAIM_ARMS, (NLI, EMBED, SMALL_LLM))},
    }

    published = publish_predictions(out_dir, inputs, arms, fams)
    report.write_all(results, out_dir)
    assert_no_vectors(out_dir)
    entry.update(status="complete", finished=datetime.now(timezone.utc).isoformat(timespec="seconds"), published_files=published)
    write_lock(lock_path, prior_runs, entry)
    return results


def _s(text: str):
    return text if text in ("all", "none") else int(text)


def dataset_results(d, arms, fams, gold, full, settings, size_table, val_rows) -> dict:
    out: dict = {"n_rows": int(len(gold)), "n_classes": int(gold["label"].nunique())}
    flagged = sorted(
        [name for name, arm in arms.items() if failure_flag(d, arm.failed.sum())]
        + [name for fam in fams.values() for name, arm in fam.items() if failure_flag(d, arm.failed.sum())]
    )
    out["failure_flags"] = flagged
    if size_table is not None:
        out["bank_sizes"] = size_table
    primary_key = (d, PRIMARY_FAMILY, settings.coverage)
    if primary_key in full:
        ds = full[primary_key]
        involved = [f for f in flagged if f == JEV or f.startswith(PRIMARY_FAMILY + "_d")]
        out["primary"] = {"status": "run", "coverage": settings.coverage, **decide(ds), "jev_acc": ds.jev_acc}
        out["grid"] = grid_table(ds, involved)
    else:
        out["primary"] = None
        out["primary_status"] = NOT_RUN
    if primary_key in full:
        rows = {}
        out["coverages"] = []
        for c in SECONDARY_COVERAGES:
            dsc = full[(d, PRIMARY_FAMILY, c)]
            s_star, s_j = lr_ahead(dsc), jev_ahead(dsc)
            entry = {"coverage": c, "ds_coverage": dsc.coverage, "s_star": str(s_star), "s_j": str(s_j), "jev_acc": dsc.jev_acc}
            if d == "banking77":  # the verdict-row comparison is Banking77's only; no sentence, since row 1 names 80%
                rows[c] = verdict_row(s_star, s_j)
                entry.update(verdict_row=rows[c], differs_from_80=rows[c] != out["primary"]["verdict_row"])
            out["coverages"].append(entry)
        if d == "banking77":
            out["coverage_notes"] = coverage_notes(rows, out["primary"]["verdict_row"])
    out["secondary_families"] = {}
    for f in FAMILIES:
        if f == PRIMARY_FAMILY:
            continue
        key = (d, f, settings.coverage)
        if key in full:
            ds = full[key]
            involved = [x for x in flagged if x == JEV or x.startswith(f + "_d")]
            out["secondary_families"][f] = {"s_star": str(lr_ahead(ds)), "s_j": str(jev_ahead(ds)), "label": NOT_A_CLAIM, "grid": grid_table(ds, involved)}
        else:
            out["secondary_families"][f] = {"status": NOT_RUN}

    # Per-arm tables: accuracy, failures and flags, calibration, cost and pipeline time, realised coverage.
    every = list(arms.items()) + [(name, arm) for fam in fams.values() for name, arm in fam.items()]
    out["arms"] = [arm_row(d, arm, settings.coverage, val_rows[(d, name)]) for name, arm in every]
    out["risk_coverage"] = {"coverage": list(RISK_COVERAGE_POINTS), "arms": {name: risk_coverage(arm) for name, arm in arms.items()}}
    for f, fam in fams.items():
        for s in GRID:
            draws = [fam[f"{f}_d{r}_s{s}"] for r in ((0,) if s == "all" else range(N_DRAWS))]
            out["risk_coverage"]["arms"][f"{f}_s{s}_mean"] = list(np.mean([risk_coverage(a) for a in draws], axis=0))
    out["sensitivity"] = {
        name: {"acc_at_80": 100 * acc_at(arms[name].array[:, 0], arms[name].correct, arms[name].failed, settings.coverage)}
        for name in (EMBED, *EMBED_SENSITIVITY)
        if name in arms
    }
    out["sensitivity_note"] = "Sensitivity rows only; they don't change any rule. embed_sim_raw is ranked by raw top cosine."
    return out


def headline(b77: dict) -> dict:
    primary = b77.get("primary")
    if not primary:
        return {"verdict": f"Banking77 primary contrast {NOT_RUN}", "verdict_row": None, "notes": []}
    flagged = [f for f in b77.get("failure_flags", []) if f == JEV or f.startswith(PRIMARY_FAMILY + "_d")]
    notes = list(b77.get("coverage_notes", []))
    if flagged:
        notes.append(f"Flagged for failures (31 or more): {', '.join(flagged)}.")
    return {
        "verdict": primary["verdict"],
        "verdict_row": primary["verdict_row"],
        "s_star": primary["s_star"],
        "s_j": primary["s_j"],
        "notes": notes,
    }


def corrections(adds: dict) -> list:
    """DESIGN.md, "Corrections to Aarmen's comment". Routing's correction is added by the routing test."""
    out = ['The "fixed labels" sentence is already conceded: he named zero-shot and embedding methods, which aren\'t fixed-label.']
    if adds.get("any_arm_matches_jev"):
        matched = [name for name, arm in adds["his_claim"].items() if arm.get("matches_jev")]
        out.append(f"Correction: {', '.join(matched)} matches Jev (lower limit of the 98.33% interval above -2).")
    out.append('"Most agent stacks" has no evidence here and is withdrawn or sourced.')
    return out


# --------------------------------------------------------------------------- published files


def publish_predictions(out_dir: Path, inputs, arms, fams) -> int:
    """Per arm: pred, top probability, gold probability, model, usage, latency. Never the full vector."""
    count = 0
    for d in inputs:
        folder = Path(out_dir) / "predictions" / d
        folder.mkdir(parents=True, exist_ok=True)
        every = list(arms[d].items()) + [(name, arm) for fam in fams[d].values() for name, arm in fam.items()]
        for name, arm in every:
            rows = arm.rows
            n = len(rows)
            frame = pd.DataFrame(
                {
                    "row_id": rows["row_id"].astype(str).to_numpy(),
                    "pred": rows["pred"].astype(str).to_numpy(),
                    "top_prob": rows["top_prob"].to_numpy(dtype=float),
                    "failed": arm.failed,
                    "gold_prob": arm.gold_prob if arm.gold_prob is not None else np.full(n, np.nan),
                    "model": rows["model"].astype(str).to_numpy() if "model" in rows.columns else np.full(n, model_id(name), dtype=object),
                    "input_tokens": rows["input_tokens"].to_numpy(dtype=float) if "input_tokens" in rows.columns else np.full(n, np.nan),
                    "output_tokens": rows["output_tokens"].to_numpy(dtype=float) if "output_tokens" in rows.columns else np.full(n, np.nan),
                    "cost_usd": rows["cost"].to_numpy(dtype=float) if "cost" in rows.columns else np.full(n, np.nan),
                    "latency_s": rows["latency_s"].to_numpy(dtype=float),
                }
            )
            frame.to_parquet(folder / f"{name}.parquet", index=False)
            count += 1
    return count


def assert_no_vectors(out_dir: Path) -> None:
    """Refuse if anything under `out_dir` could hold a full probability vector."""
    for path in Path(out_dir).rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in (".npz", ".npy", ".pkl", ".pickle"):
            raise PublishGuardError(f"{path.name}: array files are never published")
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            extra = set(frame.columns) - set(PUBLISHED_COLUMNS)
            if extra:
                raise PublishGuardError(f"{path.name} has columns outside the published set: {sorted(extra)}")
            for col in frame.columns:
                if frame[col].map(lambda v: isinstance(v, (list, tuple, dict, np.ndarray))).any():
                    raise PublishGuardError(f"{path.name}: column {col} holds arrays")


# --------------------------------------------------------------------------- command line


def _validation_gold() -> dict:
    """Validation labels, for the dry run only (they are permuted before use)."""
    import data

    out = {}
    for dataset in DATASETS:
        val = data.split(data.load_train(dataset), dataset).val
        out[dataset] = val[["row_id", "label"]].reset_index(drop=True)
    return out


def _test_gold() -> tuple[dict, dict]:
    """The firewall's gold frames and label names. Text frames are dropped unread."""
    import data
    import firewall

    token = firewall.freeze_token(REPO / "manifest.json")
    gold, names = {}, {}
    for dataset in DATASETS:
        _, gold[dataset] = firewall.load_test(dataset, token)
        names[dataset] = data.label_names(dataset)
    return gold, names


def _label_names() -> dict:
    """Every class name, from the committed label files (validation lacks some Banking77 classes)."""
    from labels import label_texts

    return {dataset: list(label_texts(dataset, bare=True)) for dataset in DATASETS}


def main(
    argv=None,
    *,
    runs_root: Path = RUNS,
    load_val_gold: Callable[[], dict] = _validation_gold,
    load_label_names: Callable[[], dict] = _label_names,
    settings: Settings = RECORD,
) -> int:
    p = argparse.ArgumentParser(prog="score", description="The one-shot scorer.")
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--dry-run", action="store_true", help="validation predictions, gold permuted; prints DRY RUN OK or a failure")
    what.add_argument("--stage", help="the test stage to score, once (stage1)")
    args = p.parse_args(argv)
    if args.dry_run:
        try:
            dry_run(load_val_gold(), runs_root=runs_root, settings=settings, label_names=load_label_names())
        except Exception as error:  # noqa: BLE001 - the dry run reports pass or fail and nothing else
            print(f"DRY RUN FAILED: {type(error).__name__}: {error}")
            return 1
        print("DRY RUN OK")
        return 0
    gold, names = _test_gold()
    result = score(args.stage, gold, runs_root=runs_root, label_names=names)
    print(f"scored {args.stage}: results in {RESULTS}, lock at {LOCK}")
    print(result["headline"]["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
