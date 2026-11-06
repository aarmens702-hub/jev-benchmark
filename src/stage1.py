"""Stage 1 runner: every arm's predictions for a stage, then the scorer (DESIGN.md "Arms", "Budget", "Time").

    scripts/stage1.sh --stage val --paid stage1 --region ca-central    # full dry run on validation rows
    scripts/stage1.sh --stage test --paid stage1 --region ca-central   # once, after the freeze

(The script runs `uv run python -m stage1` inside `caffeinate`, with PYTHONPATH=src.)

`--stage val` predicts the validation rows as if they were the test rows, writes to runs/val/ (gitignored)
and ends with the scorer's dry run, which prints only DRY RUN OK or a failure. Validation gold is read
only inside that dry run, where it is permuted. The pool and banks exclude validation rows, and s=all is
the pool alone, because the validation rows are the rows being predicted.

`--stage test` refuses unless the validation run passed its dry run (runs/val/dry_run_ok.json) with every arm
accounted for, and then unless `firewall.load_test` succeeds for both datasets (freeze token, manifest
with its registration, clean tree, HEAD at the freeze tag). It keeps the text frames and drops the gold frames
unread (in `_test_text`, the one place this module reaches the firewall). It records the test hashes and duplicate counts, removes pool and validation rows that share
normalized text with a test row before any bank is built (s=all included), writes bank_sizes.json, runs
every arm on text only and hands over to score.py, which loads gold itself. Everything it writes before
the scorer goes under runs/, so the tree stays clean for each resume and for the scorer's own load; the
deviations rows (test hashes, duplicates, every arm not run) are appended to deviations.md after scoring.

Order, one thing at a time:
  1. Paid, needing --paid stage1 and Aarmen's yes: latency probes first, then DESIGN's Stage 1 list in
     order: Jev Banking77; Jev AG News; Jev + examples (Banking77, bank draw 0, s=8); routing (Jev rule A,
     rule B, small LLM rule A, rule B; a 20-row smoke each at validation, the 800 rows at test); small LLM
     Banking77; small LLM AG News; Jev bare-name runs. One
     client per arm, with the grace from frozen.yaml. Before each arm, its unfinished rows times Stage 0's
     measured cost per row (with a 25% margin) is compared with what the ledger has left; if it does not
     fit, or the ledger refuses, that arm and every later one are logged not run. A rejected key or empty
     credits, a changed model, 10 failures in a row or failures above 1% halt instead: nothing more is
     marked, nothing is scored, and a rerun resumes. A routing rule change logs the routing arms not run.
  2. Free, under --max-hours, guards and checkpoints, inside one model-job lock: NLI first (it is the long
     one), then routing NLI, embedding similarity and its sensitivity rows, the LR, LR-CV and TF-IDF grids
     (draws 0-9, s=all fit once as draw 0), the routing LR arms, routing's validation-draw checks
     (validation stage only, frozen in routing/validity.json), and the local latency probes.
  3. The scorer, once every arm has a prediction file or a logged "not run".

Rerunning resumes: finished arms are skipped, finished rows are never sent again (openrouter.run_arm keeps
them), and a paid arm's prediction file is written only once the arm is complete. Nothing here prints an
accuracy or a comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yaml

import arms_local
import data
import guards
import latency_probe
import openrouter
import predict_local
import predictions
import prompts
import routing
import score
from guards import Guard, JobStopped, model_job_lock
from jev_client import EXPECTED_JEV_BUILD, JevCache, jev_decide
from labels import label_texts
from ledger import RunStop, split_torn_tail
from llm_client import llm_pick
from retrieval import Example, retrieve

log = logging.getLogger("stage1")

REPO = Path(__file__).resolve().parents[1]
# Stage 0's frozen choices. The freeze tooling moves Stage 0's outputs into a tracked stage0/ folder and adds
# stage0.FROZEN_YAML; this single constant is switched to that at the merge.
FROZEN_YAML = REPO / "stage0" / "frozen.yaml"
DRY_RUN_OK = "dry_run_ok.json"  # written in runs/val/ when the validation run's dry run passes
PAID = "stage1"  # the ledger stage for both runs, and the folder of the test run's predictions
STAGE_DIRS = {"val": "val", "test": PAID}
DATASETS = ("banking77", "ag_news")
SAFETY = 1.25  # margin on Stage 0's measured cost per row
FAILURE_PAUSE = 0.01  # VALIDATION.md: failures above 1% pause the run
EXAMPLES_DRAW, EXAMPLES_S = 0, 8
PROBE_ROWS, PROBE_WARMUP = latency_probe.DEFAULT_ROWS, latency_probe.DEFAULT_WARMUP
DEFAULT_MAX_HOURS = 12.0
# Stops that end an arm for good: it is logged not run and the next arm goes ahead.
FINAL_ARM_STOPS = {"ArmNotRun", "HiddenReasoning", "ProviderChanged"}
# Stops about the cap: this arm and every later paid arm are logged not run (DESIGN.md, Budget). A key or
# credits stop (401, 402, 403) is not one: it halts and resumes, since a key problem must not use up the run.
BUDGET_STOPS = {"BudgetExceeded"}
EXIT_OK, EXIT_FAIL, EXIT_STOPPED, EXIT_REFUSED = 0, 1, 2, 3


class Refused(Exception):
    """A condition for this run does not hold. Nothing has been sent or written."""


class PaidHalt(Exception):
    """The paid part stops here. Finished rows are kept and a rerun resumes."""


class FreeStopped(Exception):
    """A free arm stopped on --max-hours or a resource guard. A rerun resumes from the checkpoint."""


# --------------------------------------------------------------------------- the paid plan


@dataclass(frozen=True)
class PaidArm:
    kind: str  # jev, jev_examples, routing_jev, routing_llm, small_llm, jev_bare
    dataset: str
    rule: Optional[str] = None

    @property
    def routing(self) -> bool:
        return self.kind.startswith("routing")

    @property
    def uses_llm(self) -> bool:
        return self.kind in ("small_llm", "routing_llm")

    @property
    def name(self) -> str:
        """The prediction file's arm name: score.py's names, and the routing arms' own."""
        if self.kind == "routing_jev":
            return f"jev_rule_{self.rule}"
        if self.kind == "routing_llm":
            return f"small_llm_rule_{self.rule}"
        return self.kind

    @property
    def label(self) -> str:
        return f"{'routing_' if self.routing else ''}{self.name}__{self.dataset}"


def paid_plan() -> list[PaidArm]:
    """DESIGN.md, Budget, Stage 1 list, in order."""
    return [
        PaidArm("jev", "banking77"),
        PaidArm("jev", "ag_news"),
        PaidArm("jev_examples", "banking77"),
        *[PaidArm("routing_jev", "banking77", rule) for rule in routing.RULES],
        *[PaidArm("routing_llm", "banking77", rule) for rule in routing.RULES],
        PaidArm("small_llm", "banking77"),
        PaidArm("small_llm", "ag_news"),
        PaidArm("jev_bare", "banking77"),
        PaidArm("jev_bare", "ag_news"),
    ]


# --------------------------------------------------------------------------- frozen choices and Stage 0 costs


def load_frozen(path: Path) -> dict:
    """Stage 0's frozen choices. Refuses anything this runner would have to guess about."""
    path = Path(path)
    if not path.is_file():
        raise Refused(f"no {path.name}: Stage 0's frozen choices are needed")
    frozen = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if frozen.get("stage0_complete") is not True:
        raise Refused(f"{path.name} says Stage 0 is not complete")
    if (frozen.get("jev") or {}).get("build") != EXPECTED_JEV_BUILD:
        raise Refused(f"{path.name} freezes Jev build {(frozen.get('jev') or {}).get('build')!r}, the clients expect {EXPECTED_JEV_BUILD}")
    grace = (frozen.get("key_usage") or {}).get("reconcile_grace_seconds")
    if isinstance(grace, bool) or not isinstance(grace, (int, float)) or grace < 0:
        raise Refused(f"{path.name} has no usable key_usage.reconcile_grace_seconds")
    llm = frozen.get("small_llm") or {}
    if llm.get("status") == "selected":
        if not llm.get("model") or not llm.get("provider"):
            raise Refused(f"{path.name} selects a small LLM without a model and provider")
        if llm.get("reasoning_field_sent") is not False:
            raise Refused(f"{path.name}: the small LLM must be called with no reasoning field")
    elif llm.get("status") != "not run":
        raise Refused(f"{path.name}: small LLM status {llm.get('status')!r} is neither selected nor not run")
    nli = (frozen.get("nli") or {}).get("model")
    if nli not in {model for model, _ in arms_local.NLI_MODELS.values()}:
        raise Refused(f"{path.name}: NLI model {nli!r} is not one of the pinned models")
    return frozen


def _measure(stage0_dir: Path, name: str) -> Optional[tuple[float, Optional[float]]]:
    """(mean cost per sent row, mean input tokens) from a Stage 0 results file, or None."""
    path = Path(stage0_dir) / f"{name}.jsonl"
    if not path.exists():
        return None
    lines, _ = split_torn_tail(path.read_bytes())
    sent = [r for r in map(json.loads, lines) if r.get("status") in ("done", "failed") and not r.get("cached")]
    if not sent:
        return None
    tokens = [r["input_tokens"] for r in sent if r.get("input_tokens") is not None]
    return float(np.mean([r["cost"] for r in sent])), (float(np.mean(tokens)) if tokens else None)


def stage0_costs(stage0_dir: Path, llm: Optional[dict]) -> dict:
    """Projected cost per row for every paid arm, from Stage 0's measured rows (the same rules as its report).

    Bare names send less text than descriptions, so the described Jev cost bounds the bare one. The small
    LLM on AG News is scaled by Jev's input tokens, as Stage 0 projected it. Routing was not measured in
    Stage 0, so it uses the routing module's generous per-row figure; the ladder then measures it.
    """
    jev_b77, jev_ag = _measure(stage0_dir, "jev_banking77_val"), _measure(stage0_dir, "jev_ag_news_val")
    examples, bare = _measure(stage0_dir, "jev_banking77_examples"), _measure(stage0_dir, "jev_banking77_bare")
    missing = [n for n, m in (("jev_banking77_val", jev_b77), ("jev_ag_news_val", jev_ag), ("jev_banking77_examples", examples)) if m is None]
    costs = {}
    if llm is not None:
        select = _measure(stage0_dir, f"select_{llm['model'].replace('/', '_')}")
        if select is None:
            missing.append(f"select_{llm['model'].replace('/', '_')}")
        elif jev_b77 is not None and jev_ag is not None:
            ratio = jev_ag[1] / jev_b77[1] if jev_ag[1] and jev_b77[1] else 1.0
            costs[("small_llm", "banking77")] = select[0]
            costs[("small_llm", "ag_news")] = select[0] * ratio
            costs[("routing_llm", "banking77")] = max(select[0], routing.PILOT_EXPECTED_PER_ROW["small LLM"])
            costs[("probe_small_llm", "banking77")] = select[0]
    if missing:
        raise Refused(f"no Stage 0 measure in {stage0_dir} for {', '.join(missing)}; the projection needs them")
    costs.update(
        {
            ("jev", "banking77"): jev_b77[0],
            ("jev", "ag_news"): jev_ag[0],
            ("jev_examples", "banking77"): examples[0],
            ("jev_bare", "banking77"): max(jev_b77[0], bare[0] if bare else 0.0),
            ("jev_bare", "ag_news"): jev_ag[0],
            ("routing_jev", "banking77"): max(jev_ag[0], routing.PILOT_EXPECTED_PER_ROW["Jev"]),
            ("probe_jev", "banking77"): jev_b77[0],
        }
    )
    return costs


# --------------------------------------------------------------------------- runtime


def _test_text(dataset: str) -> pd.DataFrame:
    """The test text frame from the firewall; the gold frame is dropped here, unread. score.py loads gold itself."""
    import firewall

    texts, gold = firewall.load_test(dataset, score.freeze_token())  # the token is computed in one place
    del gold
    return texts


def _test_file_hash(dataset: str) -> Optional[str]:
    """SHA-256 of the pinned test file the firewall fetched, read from the local Hub cache."""
    import firewall
    import huggingface_hub

    repo_id = firewall._ALIASES.get(dataset, dataset)
    spec = firewall.PINNED_TEST_FILES[repo_id]
    path = huggingface_hub.try_to_load_from_cache(repo_id, spec["filename"], revision=spec["revision"], repo_type="dataset")
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dry_run(runs_root: Path) -> int:
    return score.main(["--dry-run"], runs_root=runs_root)


def _score_test(runs_root: Path) -> int:
    import firewall

    try:
        return score.main(["--stage", PAID], runs_root=runs_root)
    except (score.ScoreRefused, firewall.FirewallError) as error:
        print(f"scorer refused: {type(error).__name__}: {error}")
        return EXIT_FAIL


def paid_client_factory(paid: str) -> Callable[..., openrouter.Client]:
    def make_client(*, max_concurrency: int = openrouter.MAX_CONCURRENCY, reconcile_grace_seconds: float) -> openrouter.Client:
        return openrouter.build_client(paid, reconcile_grace_seconds=reconcile_grace_seconds, max_concurrency=max_concurrency)

    return make_client


@dataclass
class Runtime:
    """Everything that touches the network, the data, a model or the repo, so tests can swap in fakes.

    make_client is None without --paid: then no paid call can be made, and paid arms wait for a rerun.
    """

    make_client: Optional[Callable[..., openrouter.Client]] = None
    runs_root: Path = predictions.RUNS
    frozen_path: Path = FROZEN_YAML
    validity_path: Path = score.VALIDITY_PATH
    routing_validity: Callable = routing.run_validity
    stage0_dir: Path = predictions.RUNS / "private" / "stage0"
    deviations_path: Path = REPO / "deviations.md"
    load_split: Callable[[str], data.Split] = lambda dataset: data.split(data.load_train(dataset), dataset)
    load_test_text: Callable[[str], pd.DataFrame] = _test_text
    test_file_hash: Callable[[str], Optional[str]] = _test_file_hash
    make_encoder: Callable[[], Callable] = lambda: arms_local.load_encoder()
    make_nli_scorer: Callable[[str], Callable] = lambda key: arms_local.load_nli_scorer(key)
    lr_fitted: Callable = arms_local.lr_fitted
    lr_cv_fitted: Callable = arms_local.lr_cv_fitted
    tfidf_lr_fitted: Callable = arms_local.tfidf_lr_fitted
    clock: Callable[[], float] = time.monotonic
    timer: Callable[[], float] = time.perf_counter
    read_swap_gb: Callable[[], float] = guards.swap_used_gb
    read_free_disk_gb: Callable[..., float] = guards.free_disk_gb
    hardware: Callable[[], str] = latency_probe.hardware
    dry_run: Callable[[Path], int] = _dry_run
    score_test: Callable[[Path], int] = _score_test
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


@dataclass
class Prepared:
    target: pd.DataFrame  # (row_id, text) in gold's row order: all that predictors see
    pool: pd.DataFrame  # bank source, de-duplicated against test text at the test stage
    val: pd.DataFrame  # validation rows added back at s=all (none at the validation stage)


def text_only(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"row_id", "text"} <= set(frame.columns):
        raise ValueError(f"rows to predict need row_id and text; got {sorted(frame.columns)}")
    return frame[["row_id", "text"]].reset_index(drop=True)


def _sha256_text(target: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for row_id, text in zip(target["row_id"], target["text"]):
        h.update(json.dumps([str(row_id), text], ensure_ascii=False).encode("utf-8") + b"\n")
    return h.hexdigest()


def rows_to_send(results_path: Path, row_ids: list) -> list:
    """Rows openrouter.run_arm would still send: never finished, or failed on an outage with a pass left."""
    path = Path(results_path)
    finished, outage = {}, {}
    if path.exists():
        lines, _ = split_torn_tail(path.read_bytes())
        for r in map(json.loads, lines):
            if r.get("status") not in ("done", "failed"):
                continue
            finished[r["row_id"]] = r
            if r.get("failed") and openrouter.is_outage_reason(r.get("reason")):
                outage[r["row_id"]] = outage.get(r["row_id"], 0) + 1
    out = []
    for row_id in row_ids:
        r = finished.get(row_id)
        retriable = r is not None and r.get("failed") and openrouter.is_outage_reason(r.get("reason")) and outage.get(row_id, 0) < openrouter.OUTAGE_PASS_LIMIT
        if r is None or retriable:
            out.append(row_id)
    return out


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- the runner


class Stage1:
    def __init__(self, rt: Runtime, stage: str, *, max_hours: Optional[float] = DEFAULT_MAX_HOURS, region: str = "unknown",
                 give_up_reason: Optional[str] = None):
        if stage not in STAGE_DIRS:
            raise ValueError(f"stage must be one of {sorted(STAGE_DIRS)}")
        self.rt, self.stage = rt, stage
        self.max_hours, self.region, self.give_up_reason = max_hours, region, give_up_reason
        self.pred_stage = STAGE_DIRS[stage]
        self.routing_stage = score.routing_stage(self.pred_stage)
        # Validation needs only a smoke of each routing arm; the test run uses the full 800.
        self.routing_n = score.DRY_RUN_ROUTING_ROWS if stage == "val" else routing.N_SAMPLE
        self.root = Path(rt.runs_root)
        self.private = self.root / "private" / self.pred_stage
        self.data: dict[str, Prepared] = {}
        self.frozen: dict = {}
        self.llm: Optional[dict] = None
        self.grace: float = 0.0
        self.record: dict = {}
        self.guard: Callable[[], None] = lambda: None
        self._routing_rows: Optional[pd.DataFrame] = None
        self._raw_encoder = self._encoder = self._nli = None
        self._cache: Optional[JevCache] = None
        self._splits: dict = {}
        self._fingerprint_ok: Optional[bool] = None
        self.last_ledger = None

    # ------------------------------------------------------------------ setup

    def split(self, dataset: str) -> data.Split:
        if dataset not in self._splits:
            self._splits[dataset] = self.rt.load_split(dataset)
        return self._splits[dataset]

    def prepare(self) -> None:
        """Frozen choices, then (test stage) the firewall, then the rows to predict and the pools. Writes nothing."""
        self.frozen = load_frozen(self.rt.frozen_path)
        llm = self.frozen["small_llm"]
        self.llm = llm if llm.get("status") == "selected" else None
        self.grace = float(self.frozen["key_usage"]["reconcile_grace_seconds"])
        texts = {}
        if self.stage == "test":
            self.check_val_ready()
            for dataset in DATASETS:
                try:
                    texts[dataset] = text_only(self.rt.load_test_text(dataset))
                except Exception as error:  # noqa: BLE001 - any failure to load is a refusal
                    raise Refused(f"the test stage needs the firewall's test load to succeed: {type(error).__name__}: {error}") from None
        for dataset in DATASETS:
            sp = self.split(dataset)
            if self.stage == "val":
                # The validation rows are the rows predicted, so no bank may hold them: s=all is the pool alone.
                self.data[dataset] = Prepared(text_only(sp.val), sp.pool, sp.val.iloc[0:0])
                continue
            target = texts[dataset]
            norms = set(target["text"].map(data.normalize))
            pool, n_pool = data.drop_test_duplicates(sp.pool, norms)
            val, n_val = data.drop_test_duplicates(sp.val, norms)
            self.data[dataset] = Prepared(target, pool, val)
            self.record[dataset] = {
                "rows": int(len(target)),
                "file_sha256": self.rt.test_file_hash(dataset),
                "text_sha256": _sha256_text(target),
                "duplicates_removed": {"pool": n_pool, "validation": n_val},
            }

    def check_val_ready(self) -> None:
        """The test run needs the validation run done: its dry run passed, and every arm there accounted for."""
        if not (self.root / "val" / DRY_RUN_OK).exists():
            raise Refused("the validation run has not passed its dry run; run --stage val first")
        targets = {d: text_only(self.split(d).val) for d in DATASETS}
        rows = routing.sample_rows(targets["banking77"], score.DRY_RUN_ROUTING_ROWS)
        missing = self.unaccounted_in("val", targets, rows)
        if missing:
            raise Refused(f"the validation run is missing {len(missing)} arms (first {missing[0][0]}__{missing[0][1]}); rerun --stage val")
        validity = Path(self.rt.validity_path)
        try:
            recorded = json.loads(validity.read_text()).get("fingerprint") if validity.exists() else None
        except ValueError:
            recorded = None
        if recorded != routing.rule_fingerprint():
            raise Refused(f"{validity.name} is missing or was made under other routing rules or text; rerun --stage val and commit it")

    def write_records(self) -> None:
        """Test hashes and duplicate counts (test stage), then every bank's size, before any bank is used."""
        if self.stage == "test":
            path = self.root / self.pred_stage / "test_record.json"
            if path.exists():
                try:
                    prior = json.loads(path.read_text())["datasets"]
                    assert isinstance(prior, dict)
                except (ValueError, KeyError, TypeError, AssertionError):
                    raise Refused(f"{path} is unreadable or has no test record from this runner; find out why before going on") from None
                changed = [d for d in DATASETS if prior.get(d, {}).get("text_sha256") != self.record[d]["text_sha256"]]
                if changed:
                    raise Refused(f"test text differs from the first load for {changed}; stop and find out why")
            else:
                _atomic_json(path, {"loaded_at": self.rt.now().isoformat(timespec="seconds"), "datasets": self.record})
            for d, rec in self.record.items():
                dup = rec["duplicates_removed"]
                print(f"test {d}: {rec['rows']} rows; removed {dup['pool']} pool and {dup['validation']} validation rows sharing test text")
        sizes, short = {}, {}
        for d in DATASETS:
            for draw, s, _ in score.family_files(score.PRIMARY_FAMILY):
                bank = self.bank(d, draw, s)
                sizes.setdefault(d, {})[f"d{draw}_s{s}"] = {"rows": int(len(bank)), "classes": int(bank["label"].nunique())}
            counts = self.bank(d, 0, 35)["label"].value_counts()
            names = list(label_texts(d, bare=False))
            short[d] = {name: int(counts.get(name, 0)) for name in names if counts.get(name, 0) < 35}
        _atomic_json(self.root / self.pred_stage / predict_local.BANK_SIZES_FILE, sizes)
        _atomic_json(self.root / self.pred_stage / "short_classes.json", short)

    def bank(self, dataset: str, draw: int, s) -> pd.DataFrame:
        p = self.data[dataset]
        if s == "all":
            return data.bank(p.pool, draw, "all", val=p.val)
        return data.bank(p.pool, draw, s)

    def routing_rows(self) -> pd.DataFrame:
        """The routing sample: the first 800 Banking77 rows by routing hash (20 at validation), from text only."""
        if self._routing_rows is None:
            self._routing_rows = routing.sample_rows(self.data["banking77"].target, self.routing_n)
        return self._routing_rows

    def probe_texts(self) -> list:
        """Latency probes run on validation rows, at both stages."""
        return self.split("banking77").val["text"].head(PROBE_ROWS).tolist()

    def jev_cache(self) -> JevCache:
        if self._cache is None:
            self._cache = JevCache(self.private / "jev_cache.jsonl")
        return self._cache

    def event(self, **fields) -> None:
        path = self.root / self.pred_stage / "stage1_log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": self.rt.now().isoformat(timespec="seconds"), **fields}, default=str) + "\n")

    # ------------------------------------------------------------------ routing fingerprint

    def ensure_fingerprint(self, first_arm: str) -> bool:
        """Record the rule fingerprint before the first routing arm writes anything; later, compare against it.

        False means the rules or text changed since the first routing output: the routing test is invalid, so
        no more is spent or computed on it, and the change is logged for the scorer.
        """
        path = self.root / self.routing_stage / score.FINGERPRINT_FILE
        changes = path.with_name(score.FINGERPRINT_CHANGES)
        current = routing.rule_fingerprint()
        if not path.exists():
            _atomic_json(path, {"fingerprint": current, "first_arm": first_arm, "recorded_at": self.rt.now().isoformat(timespec="seconds")})
            self._fingerprint_ok = True
            return True
        recorded = json.loads(path.read_text())["fingerprint"]
        if changes.exists() or routing.rules_changed_since(recorded):
            if not changes.exists() or self._fingerprint_ok is None:
                with open(changes, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"recorded": recorded, "now": current, "at": self.rt.now().isoformat(timespec="seconds")}) + "\n")
                print("routing rules or text changed after the first routing output: the routing test is invalid; no more routing arms run")
            self._fingerprint_ok = False
            return False
        self._fingerprint_ok = True
        return True

    def routing_not_run(self, name: str, reason: str) -> None:
        if not predictions.paths(name, "banking77", self.routing_stage, self.root)[0].exists() and \
                (name, "banking77") not in predictions.read_not_run(self.routing_stage, self.root):  # fmt: skip
            predictions.mark_not_run(name, "banking77", self.routing_stage, reason, self.root)
            print(f"routing {name}: not run ({reason})")

    RULES_CHANGED = "routing rules or text changed after the first routing output; the routing test is invalid"

    # ------------------------------------------------------------------ paid

    def paid_stage_of(self, arm: PaidArm) -> str:
        return self.routing_stage if arm.routing else self.pred_stage

    def paid_done(self, arm: PaidArm) -> bool:
        stage = self.paid_stage_of(arm)
        return predictions.paths(arm.name, arm.dataset, stage, self.root)[0].exists() or (
            (arm.name, arm.dataset) in predictions.read_not_run(stage, self.root)
        )

    def results_path(self, arm: PaidArm) -> Path:
        return self.private / f"{arm.label}.jsonl"

    def paid_rows(self, arm: PaidArm) -> list[dict]:
        frame = self.routing_rows() if arm.routing else self.data[arm.dataset].target
        return frame.to_dict("records")

    def not_run(self, arm: PaidArm, reason: str) -> None:
        predictions.mark_not_run(arm.name, arm.dataset, self.paid_stage_of(arm), reason, self.root)
        self.event(what="not_run", arm=arm.label, reason=reason)
        print(f"{arm.label}: not run ({reason})")

    def budget_stop(self, arms: list[PaidArm], reason: str) -> None:
        print(f"halting paid arms: {reason}; the rest are marked not run")
        for arm in arms:
            if not self.paid_done(arm):
                self.not_run(arm, f"halted on the budget: {reason}")

    def new_client(self, what: str, max_concurrency: int = openrouter.MAX_CONCURRENCY) -> openrouter.Client:
        try:
            client = self.rt.make_client(max_concurrency=max_concurrency, reconcile_grace_seconds=self.grace)
        except RunStop as stop:
            raise PaidHalt(f"the client for {what} would not start: {type(stop).__name__}: {stop}") from None
        except Exception as error:  # noqa: BLE001 - a missing key, say; its text stays out of the message
            raise PaidHalt(f"the client for {what} would not start: {type(error).__name__}") from None
        self.last_ledger = client.ledger
        return client

    def decider(self, arm: PaidArm, client) -> Callable[[dict], openrouter.Decision]:
        d = arm.dataset
        if arm.kind == "routing_jev":
            return routing.jev_decider(arm.rule, client, self.jev_cache())
        if arm.kind == "routing_llm":
            return routing.llm_decider(arm.rule, self.llm["model"], self.llm["provider"], client)
        if arm.kind == "small_llm":
            options, instructions = label_texts(d, bare=False), prompts.small_llm_instructions(d)
            model, provider = self.llm["model"], self.llm["provider"]
            return lambda row: llm_pick(row["row_id"], row["text"], options, model, provider, client=client,
                                        instructions=instructions, reasoning=None)  # fmt: skip
        bare = arm.kind == "jev_bare"
        options = list(label_texts(d, bare=True)) if bare else label_texts(d, bare=False)
        instructions = prompts.jev_instructions(d)
        bank = None
        if arm.kind == "jev_examples":
            bank = [Example(text=r["text"], label=r["label"]) for r in self.bank(d, EXAMPLES_DRAW, EXAMPLES_S).to_dict("records")]
        cache = self.jev_cache()

        def decide(row):
            examples = retrieve(row["text"], bank, row_id=row["row_id"]) if bank is not None else None
            return jev_decide(row["row_id"], row["text"], options, examples=examples, client=client, cache=cache, instructions=instructions)

        return decide

    def probe_marker(self, arm: str) -> Path:
        return self.root / self.pred_stage / "latency_done" / f"{arm}.json"

    def probes_file(self) -> Path:
        """The test run's probes go where score.py reads them; the validation run keeps its own."""
        return self.root / "latency" / "probes.jsonl" if self.stage == "test" else self.root / self.pred_stage / "latency" / "probes.jsonl"

    PROBE_ORDER = (*latency_probe.API_ARMS, "network", *latency_probe.LOCAL_ARMS)

    def save_probe(self, arm: str, result: Optional[dict], status: Optional[str] = None) -> None:
        """The marker holds the result itself, written atomically; probes.jsonl is rebuilt from the markers,
        so a crash at any point leaves neither a lost nor a doubled probe."""
        if result is not None:
            result = {**result, "condition": latency_probe.condition(region=self.region, concurrency=1, hardware=self.rt.hardware()),
                      "stage": self.pred_stage}  # fmt: skip
        marker = {"status": status or (result or {}).get("status", "run"), "result": result}
        path = self.probe_marker(arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(marker, default=str))
        os.replace(tmp, path)
        lines = []
        for name in self.PROBE_ORDER:
            m = self.probe_marker(name)
            if m.exists() and (saved := json.loads(m.read_text()).get("result")) is not None:
                lines.append(json.dumps(saved, default=str))
        out = self.probes_file()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text("".join(line + "\n" for line in lines))
        os.replace(tmp, out)

    def api_probes_pending(self) -> bool:
        return any(not self.probe_marker(a).exists() for a in ("jev", "small_llm", "network"))

    def run_api_probes(self, costs: dict) -> None:
        """Serial probes on 50 validation rows, through run_arm and the ledger, then a key-only network probe."""
        rows = [{"row_id": f"latency:{i}", "text": t} for i, t in enumerate(self.probe_texts())]
        for arm in latency_probe.API_ARMS:
            if self.probe_marker(arm).exists():
                continue
            if arm == "small_llm" and self.llm is None:
                self.save_probe(arm, None, "not run: no small LLM selected")
                continue
            per_row = costs[(f"probe_{arm}", "banking77")] * SAFETY
            client = self.new_client(f"the {arm} latency probe", max_concurrency=1)
            if len(rows) * per_row > client.ledger.remaining():
                print(f"latency probe {arm}: not run, its projected cost does not fit what is left")
                self.save_probe(arm, None, "not run: does not fit")
                continue
            if arm == "jev":
                options, instructions = label_texts("banking77", bare=False), prompts.jev_instructions("banking77")

                def decide(row, client=client):
                    return jev_decide(row["row_id"], row["text"], options, client=client, cache=None, instructions=instructions)
            else:
                decide = self.decider(PaidArm("small_llm", "banking77"), client)
            result = latency_probe.probe_api(arm, rows, decide, client=client, results_path=self.private / "latency" / f"{arm}.jsonl",
                                             warmup=PROBE_WARMUP, expected_per_row=per_row, read_swap_gb=self.rt.read_swap_gb)  # fmt: skip
            if isinstance(client.stop, openrouter.KeyOrCreditsStop):  # no final marker: the probe runs again once the key is fixed
                raise PaidHalt(f"the {arm} latency probe stopped: the key was rejected or credits ran out ({client.stop}); fix it and rerun")
            print(f"latency probe {arm}: {result['status']}, {result['timed']['n']} timed requests")
            self.save_probe(arm, result)
        if not self.probe_marker("network").exists():
            client = self.new_client("the network probe", max_concurrency=1)
            result = latency_probe.network_probe(client.transport)
            self.save_probe("network", {"arm": "network", **result})

    def run_paid(self) -> None:
        plan = paid_plan()
        if self.give_up_reason:
            for arm in plan:
                if not self.paid_done(arm):
                    self.not_run(arm, f"unfinished, given up: {self.give_up_reason}")
            for probe in (*latency_probe.API_ARMS, "network"):
                if not self.probe_marker(probe).exists():
                    self.save_probe(probe, None, f"not run: given up: {self.give_up_reason}")
        todo = [arm for arm in plan if not self.paid_done(arm)]
        if not todo and not self.api_probes_pending():
            print("paid arms: every one is finished or logged not run")
            return
        if self.rt.make_client is None:
            raise PaidHalt(f"paid work is unfinished ({len(todo)} arms, latency probes {'pending' if self.api_probes_pending() else 'done'}); "
                           f"rerun with --paid {PAID} (needs Aarmen's yes)")
        costs = stage0_costs(self.rt.stage0_dir, self.llm)
        projected = sum(len(rows_to_send(self.results_path(a), [r["row_id"] for r in self.paid_rows(a)])) * costs.get((a.kind, a.dataset), 0.0)
                        for a in todo if not a.uses_llm or self.llm)  # fmt: skip
        for probe in latency_probe.API_ARMS:
            if not self.probe_marker(probe).exists() and (f"probe_{probe}", "banking77") in costs:
                projected += PROBE_ROWS * costs[(f"probe_{probe}", "banking77")]
        print(f"projected paid spend for {len(todo)} unfinished arms and the pending latency probes, at most: "
              f"US${projected * SAFETY:.4f} (Stage 0 costs plus 25%)")  # fmt: skip
        self.run_api_probes(costs)
        for i, arm in enumerate(plan):
            if not self.paid_done(arm):
                self.run_paid_arm(arm, costs, plan[i + 1:])
        if self.last_ledger is not None:
            print(f"paid arms done; ledger total US${self.last_ledger.spent():.4f}, this stage US${self.last_ledger.stage_spent():.4f}")

    def run_paid_arm(self, arm: PaidArm, costs: dict, rest: list[PaidArm]) -> None:
        if arm.uses_llm and self.llm is None:
            self.not_run(arm, "no small LLM passed selection in Stage 0")
            return
        rows = self.paid_rows(arm)
        path = self.results_path(arm)
        n_todo = len(rows_to_send(path, [r["row_id"] for r in rows]))
        per_row = costs[(arm.kind, arm.dataset)] * SAFETY
        client = self.new_client(arm.label)
        need, left = n_todo * per_row, client.ledger.remaining()
        if need > left:
            self.budget_stop([arm, *rest], f"projected US${need:.4f} for {n_todo} rows of {arm.label} is more than the US${left:.4f} left under the cap")
            return
        if arm.routing and not self.ensure_fingerprint(arm.name):
            self.not_run(arm, self.RULES_CHANGED)
            return
        decide = self.decider(arm, client)
        self.event(what="paid_start", arm=arm.label, rows=len(rows), todo=n_todo, projected=need)
        report = openrouter.run_arm(rows, decide, client=client, results_path=path, expected_per_row=per_row)
        if report.status == "complete" and report.retriable and client.stop is None:
            # Rows that failed on an outage get their second pass now (openrouter allows two in all).
            report = openrouter.run_arm(rows, decide, client=client, results_path=path, expected_per_row=per_row)
        self.event(what="paid_end", arm=arm.label, status=report.status, stop=report.stop_type, not_run=len(report.not_run))
        if report.status == "complete" and not report.not_run:
            self.write_paid(arm, rows, report.decisions)
            failed = sum(1 for d in report.decisions.values() if d.failed)
            spent = sum(d.cost for d in report.decisions.values())
            print(f"{arm.label}: {len(rows)} rows, {failed} failed, US${spent:.4f}")
            if failed / max(1, len(rows)) > FAILURE_PAUSE:
                raise PaidHalt(f"PAUSE: {arm.label} has {failed} failed rows of {len(rows)}, above 1% (VALIDATION.md). Its file is "
                               "saved and will be flagged. Log a deviations row, then rerun to continue with the next arm.")
            return
        stop = report.stop_type
        if report.status == "refused" or stop in BUDGET_STOPS:
            self.budget_stop([arm, *rest], f"{stop}: {report.reason}")
        elif report.status == "arm_not_run" or stop in FINAL_ARM_STOPS:
            self.not_run(arm, f"{stop}: {report.reason}")
        elif stop == "KeyOrCreditsStop":
            raise PaidHalt(f"{arm.label} stopped: the key was rejected or credits ran out ({report.reason}). Nothing is marked not run "
                           "and nothing is scored; fix the key or balance (never past the cap) and rerun to resume.")
        elif stop == "ModelChanged":
            raise PaidHalt(f"PAUSE: {arm.label}: the model changed mid-run (VALIDATION.md): {report.reason}. Ask Aarmen and log a "
                           "deviations row; do not rerun selectively. Finished rows are kept.")
        else:
            raise PaidHalt(f"{arm.label} stopped: {stop}: {report.reason}. Finished rows are kept and are not sent again; rerun to resume.")

    def write_paid(self, arm: PaidArm, rows: list[dict], decisions: dict) -> None:
        classes = list(routing.QUEUES) if arm.routing else list(label_texts(arm.dataset, bare=False))
        frame, vectors = predictions.decisions_frame([r["row_id"] for r in rows], decisions, classes)
        predictions.write(arm.name, arm.dataset, frame, self.paid_stage_of(arm), vectors, classes if vectors is not None else None, self.root)

    # ------------------------------------------------------------------ free

    def raw_encoder(self):
        if self._raw_encoder is None:
            self._raw_encoder = self.rt.make_encoder()
        return self._raw_encoder

    def encoder(self):
        if self._encoder is None:
            cache = self.root / "emb" / arms_local.EMBED_REVISION[:12]
            self._encoder = arms_local.memoize_encoder(self.raw_encoder(), cache_dir=cache, guard=lambda: self.guard())
        return self._encoder

    def nli_scorer(self):
        """The frozen NLI model, checked by the guard before every batch."""
        if self._nli is None:
            key = next(k for k, (model, _) in arms_local.NLI_MODELS.items() if model == self.frozen["nli"]["model"])
            self._nli = self.rt.make_nli_scorer(key)
        inner = self._nli

        def scorer(batch):
            self.guard()
            return inner(batch)

        return scorer

    def complete(self, arm: str, dataset: str, stage: str, rows: pd.DataFrame) -> bool:
        got = predictions.read(arm, dataset, stage, self.root)
        return got is not None and got[0]["row_id"].astype(str).tolist() == rows["row_id"].astype(str).tolist()

    def settled(self, arm: str, dataset: str, stage: str, rows: pd.DataFrame) -> bool:
        """Complete, or logged not run (itself or its family): either way a free step leaves it alone."""
        logged = predictions.read_not_run(stage, self.root)
        family = next((f for f in score.FAMILIES if arm.startswith(f + "_d")), None)
        return (arm, dataset) in logged or (family, dataset) in logged or self.complete(arm, dataset, stage, rows)

    def free_nli(self, dataset: str):
        target = self.data[dataset].target
        if self.settled(score.NLI, dataset, self.pred_stage, target):
            return None
        hyps = arms_local.nli_hypotheses(dataset, bare=False)
        scorer = self.nli_scorer()
        pairs = len(target) * len(hyps)
        rate = self.frozen["nli"].get("pairs_per_second")
        if rate:
            print(f"nli {dataset}: {pairs:,} pairs, about {pairs / rate / 3600:.1f} h at the measured {rate} pairs/s")
        return predict_local.run_arm(
            score.NLI, dataset, self.pred_stage, target,
            lambda texts: np.stack(list(arms_local.nli(texts, hyps, arms_local.NLI_BATCH_SIZE, scorer))),
            list(label_texts(dataset, bare=False)), chunk_size=predict_local.NLI_CHECKPOINT_ROWS, guard=self.guard, root=self.root,
            timer=self.rt.timer,
        )  # fmt: skip

    def free_embed(self, dataset: str, bare: bool):
        target = self.data[dataset].target
        base = score.EMBED_BARE if bare else score.EMBED
        names = [base] if bare else [score.EMBED, *score.EMBED_SENSITIVITY]
        if all(self.settled(n, dataset, self.pred_stage, target) for n in names):
            return None
        texts = label_texts(dataset, bare=bare)
        classes, label_list = list(texts), list(texts.values())
        encode = self.encoder()
        res = predict_local.run_arm(
            base, dataset, self.pred_stage, target, lambda t: arms_local.embed_cosine(t, label_list, encode), classes,
            to_probs=lambda cos: arms_local.softmax(cos / arms_local.EMBED_TEMPERATURE), guard=self.guard, root=self.root, timer=self.rt.timer,
        )  # fmt: skip
        if res.stopped is None and not bare:
            # Sensitivity rows come from the saved cosine scores, so they carry the primary run's timing.
            rows, cos, saved_classes = predictions.read(base, dataset, self.pred_stage, self.root)
            for suffix, view in arms_local.sensitivity_views(cos).items():
                if suffix:
                    frame = predict_local._rows_frame(rows["row_id"], view, saved_classes, lambda v: v, rows["latency_s"].to_numpy())
                    predictions.write(base + suffix, dataset, frame, self.pred_stage, view, saved_classes, self.root)
        return res

    def free_supervised(self, kind: str, dataset: str, draw: int, s):
        target = self.data[dataset].target
        name = f"{kind}_d{draw}_s{s}"
        if self.settled(name, dataset, self.pred_stage, target):
            return None
        classes = list(label_texts(dataset, bare=False))
        fitted = {}

        def fit():
            """Build the bank, embed it and fit, once and outside the timed span."""
            bank = self.bank(dataset, draw, s)
            y = bank["label"].to_numpy()
            if kind == "tfidf_lr":
                fitted["predict"] = self.rt.tfidf_lr_fitted((bank["text"].tolist(), y), classes=classes)
                return
            encode = self.encoder()
            X = encode(bank["text"].tolist())
            model = self.rt.lr_fitted((X, y), classes=classes) if kind == "lr" else self.rt.lr_cv_fitted((X, y), s, classes=classes)
            fitted["predict"] = lambda texts: model(encode(texts))

        def predict(texts):
            if "predict" not in fitted:
                fit()
            return fitted["predict"](texts)

        return predict_local.run_arm(name, dataset, self.pred_stage, target, predict, classes, chunk_size=max(len(target), 1),
                                     guard=self.guard, root=self.root, prepare=fit, timer=self.rt.timer)  # fmt: skip

    def free_routing_nli(self, rule: str):
        rows = self.routing_rows()
        name = f"nli_rule_{rule}"
        premises = pd.DataFrame({"row_id": rows["row_id"].to_numpy(), "text": [routing.nli_premise(r) for r in rows.to_dict("records")]})
        if self.settled(name, "banking77", self.routing_stage, premises):
            return None
        if not self.ensure_fingerprint(name):
            self.routing_not_run(name, self.RULES_CHANGED)
            return None
        hyps = routing.nli_hypotheses(rule)
        scorer = self.nli_scorer()
        return predict_local.run_arm(
            name, "banking77", self.routing_stage, premises,
            lambda texts: np.stack(list(arms_local.nli(texts, hyps, arms_local.NLI_BATCH_SIZE, scorer))),
            list(routing.QUEUES), chunk_size=predict_local.NLI_CHECKPOINT_ROWS, guard=self.guard, root=self.root, timer=self.rt.timer,
        )  # fmt: skip

    def _write_routing(self, name: str, rows: pd.DataFrame, pred, top, latency: float) -> None:
        frame = pd.DataFrame(
            {
                "row_id": rows["row_id"].to_numpy(),
                "pred": np.asarray(pred, dtype=object).astype(str),
                "top_prob": np.asarray(top, dtype=float),
                "failed": np.zeros(len(rows), dtype=bool),
                "latency_s": np.full(len(rows), latency, dtype=float),
            }
        )
        predictions.write(name, "banking77", frame, self.routing_stage, root=self.root)

    def free_routing_lr(self):
        """LR-A (rule A labels, never retrained) and Intent LR + code, both on the de-duplicated pool."""
        rows = self.routing_rows()
        pool = self.data["banking77"].pool
        if not self.settled("lr_a", "banking77", self.routing_stage, rows) and not self.ensure_fingerprint("lr_a"):
            self.routing_not_run("lr_a", self.RULES_CHANGED)
        elif not self.settled("lr_a", "banking77", self.routing_stage, rows):
            self.guard()
            encode = self.encoder()
            model = routing.fit_lr_a(routing.lr_a_bank(pool), encode)
            started = self.rt.timer()
            pred, top = routing.predict_lr_a(model, rows, encode)
            self._write_routing("lr_a", rows, pred, top, (self.rt.timer() - started) / max(1, len(rows)))
        names = {rule: f"intent_lr_code_rule_{rule}" for rule in routing.RULES}
        unfinished = not all(self.settled(n, "banking77", self.routing_stage, rows) for n in names.values())
        if unfinished and not self.ensure_fingerprint("intent_lr_code"):
            for name in names.values():
                self.routing_not_run(name, self.RULES_CHANGED)
        elif unfinished:
            self.guard()
            out = routing.intent_lr_code(pool, rows, self.encoder(), classes=list(label_texts("banking77", bare=False)))
            for rule, name in names.items():
                if not self.settled(name, "banking77", self.routing_stage, rows):
                    self._write_routing(name, rows, *out[rule], float("nan"))
        return None

    def routing_validity(self):
        """Validation stage only: routing.run_validity on the validation draw, frozen in routing/validity.json.

        The generator checks read validation gold (VALIDATION.md, use 6); LR-A's floor is kept as pass/fail only.
        """
        if self.stage != "val":
            return None
        path = Path(self.rt.validity_path)
        current = routing.rule_fingerprint()
        if path.exists() and json.loads(path.read_text()).get("fingerprint") == current:
            return None
        self.guard()
        stats, checks = self.rt.routing_validity(self.split("banking77"), self.encoder())
        record = {"fingerprint": current, "computed_at": self.rt.now().isoformat(timespec="seconds"),
                  "rows": int(len(self.split("banking77").val)), "stats": stats, "checks": {k: bool(v) for k, v in checks.items()}}  # fmt: skip
        _atomic_json(path, record)
        print("routing validity: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in record["checks"].items()))
        return None

    def local_probe_loader(self, arm: str) -> Callable[[], Callable[[str], object]]:
        d = "banking77"
        classes = list(label_texts(d, bare=False))

        def unit(m):
            m = np.asarray(m, dtype=np.float64)
            return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

        def load():
            # The raw encoder, not the memoized one, so a repeated text is really encoded again.
            if arm == "embed_sim":
                encode = self.raw_encoder()
                labels = unit(encode(list(label_texts(d, bare=False).values())))
                return lambda t: arms_local.softmax(unit(encode([t])) @ labels.T / arms_local.EMBED_TEMPERATURE)
            if arm == "nli":
                hyps, scorer = arms_local.nli_hypotheses(d, bare=False), self.nli_scorer()
                return lambda t: list(arms_local.nli([t], hyps, arms_local.NLI_BATCH_SIZE, scorer))
            bank = self.bank(d, EXAMPLES_DRAW, EXAMPLES_S)
            y = bank["label"].to_numpy()
            if arm == "tfidf_lr":
                model = self.rt.tfidf_lr_fitted((bank["text"].tolist(), y), classes=classes)
                return lambda t: model([t])
            encode = self.raw_encoder()
            model = self.rt.lr_fitted((encode(bank["text"].tolist()), y), classes=classes)
            return lambda t: model(encode([t]))

        return load

    def run_local_probes(self):
        texts = None
        for arm in latency_probe.LOCAL_ARMS:
            if self.probe_marker(arm).exists():
                continue
            texts = texts if texts is not None else self.probe_texts()
            self.guard()
            result = latency_probe.probe_local(arm, self.local_probe_loader(arm), texts, warmup=PROBE_WARMUP, read_swap_gb=self.rt.read_swap_gb)
            self.save_probe(arm, result)
        return None

    def free_steps(self) -> list[tuple[str, Callable]]:
        """NLI first, since it is the long job; then everything else, one model job at a time."""
        steps = [(f"nli__{d}", lambda d=d: self.free_nli(d)) for d in DATASETS]
        steps += [(f"routing nli_rule_{r}", lambda r=r: self.free_routing_nli(r)) for r in routing.RULES]
        steps += [(f"embed_sim{'_bare' if b else ''}__{d}", lambda d=d, b=b: self.free_embed(d, b)) for b in (False, True) for d in DATASETS]
        for family in score.FAMILIES:
            for d in DATASETS:
                steps += [(f"{name}__{d}", lambda f=family, d=d, r=draw, s=s: self.free_supervised(f, d, r, s))
                          for draw, s, name in score.family_files(family)]  # fmt: skip
        steps.append(("routing lr_a and intent_lr_code", self.free_routing_lr))
        steps.append(("routing validity", self.routing_validity))
        steps.append(("local latency probes", self.run_local_probes))
        return steps

    def run_free(self) -> bool:
        guard = Guard(self.max_hours, clock=self.rt.clock, read_swap_gb=self.rt.read_swap_gb,
                      read_free_disk_gb=self.rt.read_free_disk_gb, path=str(self.root))  # fmt: skip
        self.guard = guard
        try:
            for name, step in self.free_steps():
                res = step()
                if res is not None and res.stopped:
                    raise FreeStopped(f"{name}: {res.message}")
        except (FreeStopped, JobStopped) as error:
            print(f"free arms stopped ({error}); rerun to resume from the last checkpoint")
            self.event(what="free_stopped", reason=str(error))
            return False
        finally:
            self.guard = lambda: None
        print("free arms: every one is finished")
        return True

    # ------------------------------------------------------------------ finish

    def unaccounted(self) -> list[tuple[str, str]]:
        return self.unaccounted_in(self.pred_stage, {d: p.target for d, p in self.data.items()}, self.routing_rows())

    def unaccounted_in(self, stage: str, targets: dict, routing_rows: pd.DataFrame) -> list[tuple[str, str]]:
        """(arm, dataset) that the scorer expects and that has neither full predictions nor a not-run record.
        Routing files come back as (name, "routing")."""
        logged = predictions.read_not_run(stage, self.root)
        missing = []
        for d in DATASETS:
            target = targets[d]
            for arm in score.expected_single_arms(d):
                if (arm, d) not in logged and not self.complete(arm, d, stage, target):
                    missing.append((arm, d))
            for family in score.FAMILIES:
                if (family, d) in logged:
                    continue
                missing += [(n, d) for _, _, n in score.family_files(family) if not self.complete(n, d, stage, target)]
        rstage = score.routing_stage(stage)
        rlogged = predictions.read_not_run(rstage, self.root)
        for name in score.ROUTING_FILES:
            if (name, "banking77") not in rlogged and not self.complete(name, "banking77", rstage, routing_rows):
                missing.append((name, "routing"))
        return missing

    def give_up(self, missing: list[tuple[str, str]]) -> None:
        """Log every unfinished arm not run. A partial file is set aside, so the scorer never sees both."""
        families = {}
        reason = f"unfinished, given up: {self.give_up_reason}"
        for arm, d in missing:
            if d == "routing":
                for path in predictions.paths(arm, "banking77", self.routing_stage, self.root):
                    if path.exists():
                        os.replace(path, path.with_name(path.name + ".partial"))
                predictions.mark_not_run(arm, "banking77", self.routing_stage, reason, self.root)
                continue
            family = next((f for f in score.FAMILIES if arm.startswith(f + "_d")), None)
            if family is not None:
                families.setdefault((family, d), []).append(arm)
                continue
            self._set_aside(arm, d)
            predictions.mark_not_run(arm, d, self.pred_stage, f"unfinished, given up: {self.give_up_reason}", self.root)
        for (family, d), _ in families.items():
            for _, _, name in score.family_files(family):
                self._set_aside(name, d)
            predictions.mark_not_run(family, d, self.pred_stage, f"unfinished, given up: {self.give_up_reason}", self.root)

    def _set_aside(self, arm: str, dataset: str) -> None:
        for path in predictions.paths(arm, dataset, self.pred_stage, self.root):
            if path.exists():
                os.replace(path, path.with_name(path.name + ".partial"))

    def finish(self) -> int:
        missing = self.unaccounted()
        if missing and self.give_up_reason:
            self.give_up(missing)
            missing = self.unaccounted()
        if missing:
            names = ", ".join(f"{a}__{d}" for a, d in missing[:5])
            print(f"not scored: {len(missing)} arms have neither predictions nor a not-run record (first: {names}); "
                  "rerun to finish them, or pass --mark-unfinished-not-run REASON")  # fmt: skip
            return EXIT_STOPPED
        if self.stage == "val":
            marker = self.root / self.pred_stage / DRY_RUN_OK
            marker.unlink(missing_ok=True)
            code = self.rt.dry_run(self.root)
            if code == EXIT_OK:
                _atomic_json(marker, {"at": self.rt.now().isoformat(timespec="seconds")})
            return code
        code = self.rt.score_test(self.root)
        if code == EXIT_OK:  # a refused scorer leaves the tree clean, so it can be run again after a fix
            self.write_deviations()
        return code

    def write_deviations(self) -> None:
        """Test hashes, duplicate counts and every arm not run, appended once, after the scorer has loaded gold."""
        marker = self.root / self.pred_stage / "deviations_written.json"
        if marker.exists():
            return
        date = self.rt.now().date().isoformat()
        short = json.loads((self.root / self.pred_stage / "short_classes.json").read_text())
        rows = [
            ("Test files loaded and hashed: " + "; ".join(
                f"{d} {r['rows']} rows, file SHA-256 {r['file_sha256'] or 'not found in the Hub cache'}, text SHA-256 {r['text_sha256']}"
                for d, r in self.record.items()), "VALIDATION.md, Manifest: test file hashes are recorded after first load", "rule"),
            ("Pool and validation rows sharing normalized text with a test row removed before any bank: " + "; ".join(
                f"{d}: {r['duplicates_removed']['pool']} pool, {r['duplicates_removed']['validation']} validation" for d, r in self.record.items())
             + ". Classes under 35 rows at s=35: " + ("; ".join(f"{d}: " + ", ".join(f"{c} ({n})" for c, n in v.items()) for d, v in short.items() if v) or "none"),
             "VALIDATION.md, Duplicates", "rule"),
        ]  # fmt: skip
        for stage in (self.pred_stage, self.routing_stage):
            for (arm, d), reason in predictions.read_not_run(stage, self.root).items():
                rows.append((f"{arm} on {d}{' (routing)' if stage == self.routing_stage else ''} not run", reason, "pending"))
        clean = lambda s: str(s).replace("|", "/").replace("\n", " ")  # noqa: E731
        lines = [f"| {date} | {clean(what)} | {clean(why)} | {PAID} | {who} |" for what, why, who in rows]
        path = Path(self.rt.deviations_path)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        with open(path, "a", encoding="utf-8") as f:
            f.write(("" if text.endswith("\n") or not text else "\n") + "\n".join(lines) + "\n")
        _atomic_json(marker, {"rows": len(lines), "at": self.rt.now().isoformat(timespec="seconds")})
        print(f"deviations.md: {len(lines)} rows appended")

    # ------------------------------------------------------------------ all of it

    def run(self) -> int:
        try:
            self.prepare()
        except Refused as error:
            print(f"refused: {error}")
            return EXIT_REFUSED
        with model_job_lock(self.root / "model_job.lock"):
            try:
                self.write_records()
            except Refused as error:
                print(f"refused: {error}")
                return EXIT_REFUSED
            paid_ok = True
            try:
                self.run_paid()
            except Refused as error:
                print(f"refused: {error}")
                return EXIT_REFUSED
            except PaidHalt as error:
                # Free arms spend nothing, so they still run; nothing is scored until the paid part is settled.
                print(f"paid arms stopped: {error}")
                self.event(what="paid_halt", reason=str(error))
                paid_ok = False
            free_ok = self.run_free()
            if not paid_ok or (not free_ok and not self.give_up_reason):
                return EXIT_STOPPED
            return self.finish()


# --------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stage1", description="Stage 1 runner: every arm's predictions, then the scorer.")
    p.add_argument("--stage", choices=sorted(STAGE_DIRS), required=True, help="val: full dry run on validation rows; test: once, after the freeze")
    openrouter.add_paid_argument(p)
    p.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS, help="limit on the free arms in this run; a rerun resumes")
    p.add_argument("--region", default="unknown", help="where API requests leave from, for the latency probes, e.g. ca-central")
    p.add_argument("--mark-unfinished-not-run", metavar="REASON", default=None,
                   help="give up on unfinished arms: log each not run with REASON (needs a deviations row)")
    return p


def main(argv=None, runtime: Optional[Runtime] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.paid is not None and args.paid != PAID:
        parser.error(f"Stage 1 spends under --paid {PAID}")
    if runtime is None:
        runtime = Runtime(make_client=paid_client_factory(args.paid) if args.paid else None)
    elif args.paid is not None:
        parser.error("--paid cannot be combined with an injected runtime")
    runner = Stage1(runtime, args.stage, max_hours=args.max_hours, region=args.region, give_up_reason=args.mark_unfinished_not_run)
    return runner.run()


def cli() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return main()
    except guards.JobLockedError as error:
        print(f"refusing to start: {error}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(cli())
