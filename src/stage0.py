"""Stage 0: the first paid step, live checks on validation rows only (VALIDATION.md, DESIGN.md Budget).

    scripts/stage0.sh --paid stage0

Order, stopping at the first failure:
  1. Jev smoke, 20 Banking77 validation rows, then key-usage lag and a 3-row bare-name check.
  2. Small LLM selection, 50 validation rows per candidate, cheapest first; pass is 49 usable or more.
  3. Jev on every Banking77 and AG News validation row.
  4. Repeatability, 100 Banking77 rows x 3 calls.
  5. Jev + examples smoke, 20 rows.
  6. Measured tokens and projected Stage 1 spend.

Each step first checks that its projected cost fits what is left. The report holds pass/fail, failure
counts and rates, Jev's 1.0 mass and its score at the 80% cut, repeatability, tokens, costs, latency,
the confirmed field list and the selections. It never holds an accuracy, D_s or a comparison: the
bug floor is printed as PASS or FAIL and the numbers behind it are not kept. Row-level files go to
runs/private/stage0/.
"""
import argparse
import hashlib
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

import data
import prompts
import stage0_fields as fields
from jev_client import EXPECTED_JEV_BUILD, JEV_MODEL, JEV_PROVIDER, JevCache, jev_decide
from labels import label_texts
from ledger import HARD_CAP_USD, STAGE_CAPS_USD, RunStop
from llm_client import MODEL_SUFFIX, llm_pick
from openrouter import (
    DEFAULT_RECONCILE_GRACE_SECONDS,
    REPO_ROOT,
    Client,
    add_paid_argument,
    build_client,
    run_arm,
)
from retrieval import Example, retrieve

STAGE = "stage0"
CAD_PER_USD = 1.3994
CREDIT_FEE = 0.055
SMOKE_ROWS, SELECT_ROWS, REPEAT_ROWS, REPEATS, EXAMPLES_ROWS, BARE_ROWS = 20, 50, 100, 3, 20, 3
SELECT_PASS = 49
BANK_DRAW, BANK_S = 0, 8
FAILURE_RATE_LIMIT = 0.01
MASS_LIMIT = 0.75
BUG_FLOOR_JEV = 0.60
AGREEMENT_FLOOR = 0.98
CUT = 0.8
STAGE1_ROWS = {"banking77": 3080, "ag_news": 7600}
# First-contact figures per row, used only until a measure exists. The ladder replaces them after 21 rows.
FIRST_CONTACT_JEV_USD = 0.0003
EXAMPLES_COST_FACTOR = 2.0
SAFETY = 1.25
CANDIDATES = (
    # model, provider, price per M tokens in/out on 2026-11-04 (DESIGN.md), first-contact cost per row
    ("openai/gpt-oss-20b", "Darkbloom", (0.018, 0.09), 0.0001),
    ("z-ai/glm-5.3-flash", "Wafer", (0.044, 0.20), 0.0002),
)
# Candidates that fail on their own account; anything else that stops a run ends Stage 0.
CANDIDATE_STOPS = {"ArmNotRun", "HiddenReasoning", "ConsecutiveFailures"}
# 20.7 pairs/s is a real measurement: Task 4 Step 5, 2026-11-04, deberta-v3-large, batch 16, 5,000 validation pairs.
NLI = {
    "model": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c",
    "revision": "b2730f16019076bb0009481121efbe4705e0e378",
    "pairs_per_second": 20.7,
    "source": "Task 4 Step 5, 2026-11-04: batch 16, 5,000 validation pairs",
}
RATE_CHECK_EVERY = 100
USAGE_WAIT_SECONDS = 120
USAGE_POLL_SECONDS = 3
# Ruling R7: later clients wait the measured usage lag, rounded up, before believing a short report.
GRACE_CAP_SECONDS = 180
EXIT_OK, EXIT_FAIL, EXIT_STOPPED = 0, 1, 2


class Stage0Stop(Exception):
    """Stage 0 ends here. `ask` marks the stops VALIDATION.md says need Aarmen before anything else."""

    def __init__(self, step: str, reason: str, ask: bool = False):
        super().__init__(f"{step}: {reason}")
        self.step, self.reason, self.ask = step, reason, ask


@dataclass
class Runtime:
    """Everything that touches the machine, the network or the repo, so tests can swap in fakes."""

    # Called with reconcile_grace_seconds: None until step 1c has measured the lag, then that grace.
    make_client: Callable[..., Client]
    load_split: Callable[[str], data.Split] = lambda dataset: data.split(data.load_train(dataset), dataset)
    runs_root: Path = REPO_ROOT / "runs"
    report_dir: Path = REPO_ROOT
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


@dataclass
class Handle:
    client: Client
    base_reported: float
    base_confirmed: float


def sample_rows(df, n: int, prefix: str) -> list[dict]:
    """A fixed sample of validation rows: by SHA-256 of the prefix and row id. Uses no label."""
    ordered = sorted(df.to_dict("records"), key=lambda r: hashlib.sha256(f"{prefix}{r['row_id']}".encode()).hexdigest())
    return ordered[:n]


def _median(values):
    return statistics.median(values) if values else None


def _p95(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def summarize(decisions: dict) -> dict:
    """Counts, failure reasons, cost and tokens for one run. No answer content."""
    done = list(decisions.values())
    fresh = [d for d in done if not d.cached]
    ok = [d for d in done if not d.failed]
    reasons = Counter((d.reason or "unknown").split(":")[0] for d in done if d.failed)
    ins = [d.input_tokens for d in fresh if not d.failed and d.input_tokens is not None]
    outs = [d.output_tokens for d in fresh if not d.failed and d.output_tokens is not None]
    return {
        "rows": len(done),
        "failed": len(done) - len(ok),
        "reasons": dict(reasons),
        "cached": len(done) - len(fresh),
        "cost": sum(d.cost for d in fresh),
        "cost_per_row": (sum(d.cost for d in fresh) / len(fresh)) if fresh else None,
        "mean_in": statistics.fmean(ins) if ins else None,
        "mean_out": statistics.fmean(outs) if outs else None,
    }


def mass_and_cut(decisions: dict) -> dict:
    """Jev's 1.0 mass and its score at the 80% cut. Failures take the top places, as DESIGN.md ranks them."""
    scores = sorted((d.top_prob for d in decisions.values() if not d.failed), reverse=True)
    n, failures = len(decisions), len(decisions) - len(scores)
    k = math.ceil(CUT * n)
    return {
        "mass_count": sum(1 for s in scores if s >= 1 - 1e-9),
        "mass": sum(1 for s in scores if s >= 1 - 1e-9) / len(scores) if scores else None,
        "cut_score": scores[k - failures - 1] if scores and k > failures else None,
        "n": n,
    }


def repeat_stats(decisions: dict) -> dict:
    groups: dict[str, list] = {}
    for key, d in decisions.items():
        groups.setdefault(key.rsplit("#", 1)[0], []).append(d)
    complete = [g for g in groups.values() if len(g) == REPEATS and not any(d.failed for d in g)]
    agree = sum(1 for g in complete if len({d.pred for d in g}) == 1)
    ranges = [max(d.top_prob for d in g) - min(d.top_prob for d in g) for g in complete]
    return {
        "rows": len(groups),
        "complete": len(complete),
        "agreement": agree / len(complete) if complete else None,
        "median_range": _median(ranges),
    }


class Stage0:
    def __init__(self, rt: Runtime):
        self.rt = rt
        self.private = Path(rt.runs_root) / "private" / STAGE
        self.private.mkdir(parents=True, exist_ok=True)
        self.obs = fields.Observations()
        self.latency: dict[str, list] = {}
        self.steps: list[tuple[str, str, str]] = []  # name, PASS/FAIL/NOT RUN, detail
        self.facts: dict = {}
        self.findings: list[fields.Finding] = []
        self.usage: fields.UsageCheck | None = None
        self.candidates: dict = {}
        self.selected: dict | None = None
        self.selection_decided = False
        self.est_jev = FIRST_CONTACT_JEV_USD
        self.spent_before = 0.0
        self.ledger = None
        self._first_client = True
        self.grace: int | None = None  # set from the measured usage lag in step 1c
        self.cache = JevCache(self.private / "jev_cache.jsonl")
        self.splits: dict = {}

    # ------------------------------------------------------------------ plumbing

    def step(self, name: str, status: str, detail: str = "") -> None:
        self.steps.append((name, status, detail))

    def new_client(self, step: str) -> Handle:
        try:
            client = self.rt.make_client(reconcile_grace_seconds=self.grace)
        except RunStop as stop:
            raise Stage0Stop(step, f"the client would not start: {type(stop).__name__}: {stop}") from None
        client.transport = fields.Recorder(client.transport, self.obs)
        self.ledger = client.ledger
        return Handle(client, client.reported_usage(), client.ledger.confirmed())

    def fits(self, handle: Handle, step: str, rows: int, per_row: float) -> None:
        need, left = rows * per_row, handle.client.ledger.remaining()
        if need > left:
            raise Stage0Stop(step, f"projected US${need:.4f} for {rows} rows is more than the US${left:.4f} left; nothing sent")

    def gate(self, step: str) -> None:
        """Fail closed: a reply whose shape differs from the expected fixture ends Stage 0 before more spend."""
        problems = self.obs.problems()
        if problems:
            raise Stage0Stop(step, "an answer shape differs from the expected fixture: " + "; ".join(problems[:6]), ask=True)

    def run(self, handle, step, name, rows, decide, expected, *, allowed=0, row_id=lambda r: r["row_id"], first_contact=False,
            check_rate=False):
        self.fits(handle, step, len(rows), expected)
        results = self.private / f"{name}.jsonl"
        kwargs = dict(client=handle.client, results_path=results, expected_per_row=expected, row_id=row_id, allowed_arm_not_run=allowed)
        if first_contact and rows:
            first = run_arm(rows[:1], decide, **kwargs)
            self.gate(step)
            if first.status in ("stopped", "refused") and first.stop_type not in CANDIDATE_STOPS:
                return first
        if not check_rate:
            report = run_arm(rows, decide, **kwargs)
            self.gate(step)
            return report
        # Failure rate is checked after every 100 rows, so a bad arm stops early instead of at the end.
        merged: dict = {}
        report = None
        for start in range(0, len(rows), RATE_CHECK_EVERY):
            report = run_arm(rows[start:start + RATE_CHECK_EVERY], decide, **kwargs)
            self.gate(step)
            merged.update(report.decisions)
            if report.status != "complete":
                break
            failed = sum(1 for d in merged.values() if d.failed)
            if failed / len(merged) > FAILURE_RATE_LIMIT:
                self.step(step, "FAIL", f"{failed} of {len(merged)} failed so far, above 1%")
                raise Stage0Stop(step, f"failures are {failed} of {len(merged)} rows so far, above 1%. Stopped early. Pause and log.")
        if report is not None:
            report.decisions = merged
        return report

    def timed(self, step: str, fn):
        def wrapper(row):
            start = self.rt.clock()
            decision = fn(row)
            if not decision.cached:
                self.latency.setdefault(step, []).append(self.rt.clock() - start)
            return decision

        return wrapper

    def stop_if_stopped(self, step: str, report) -> None:
        if report.status == "complete":
            return
        raise Stage0Stop(step, f"{report.stop_type}: {report.reason}", ask=report.stop_type in ("ModelChanged", "ReconciliationGap", "LedgerMismatch"))

    def validation(self, dataset: str):
        if dataset not in self.splits:
            self.splits[dataset] = self.rt.load_split(dataset).val
        return self.splits[dataset]

    def jev_decider(self, handle, step, dataset, *, bare=False, bank=None, cache=True):
        options = label_texts(dataset, bare=bare)
        if bare:
            options = list(options)
        instructions = prompts.jev_instructions(dataset)

        def decide(row):
            examples = retrieve(row["text"], bank, row_id=row["row_id"]) if bank is not None else None
            return jev_decide(row["row_id"], row["text"], options, examples=examples, client=handle.client,
                              cache=self.cache if cache else None, instructions=instructions)

        return self.timed(step, decide)

    # ------------------------------------------------------------------ steps

    def smoke(self) -> None:
        step = "1 Jev smoke"
        rows = sample_rows(self.validation("banking77"), SMOKE_ROWS, "stage0:smoke:")
        handle = self.new_client(step)
        self.obs.phase = "smoke"
        report = self.run(handle, step, "jev_banking77_smoke", rows, self.jev_decider(handle, step, "banking77"),
                          self.est_jev, first_contact=True)
        summary = summarize(report.decisions)
        self.facts["smoke"] = summary
        if report.status != "complete":
            self.step(step, "FAIL", f"{report.stop_type}: {report.reason}")
            self.stop_if_stopped(step, report)
        if summary["failed"]:
            no_probs = sum(n for r, n in summary["reasons"].items() if r == "malformed")
            self.step(step, "FAIL", f"{summary['failed']} of {summary['rows']} rows gave no usable answer")
            raise Stage0Stop(step, f"Jev gave a choice with probabilities on {summary['rows'] - summary['failed']} of {summary['rows']} rows "
                             f"(malformed: {no_probs}). If probabilities are missing the primary metric does not exist: stop and ask.", ask=True)
        models = sorted({d.model for d in report.decisions.values()})
        self.facts["jev_models"] = models
        measured = summary["cost_per_row"]
        if measured:
            self.est_jev = max(measured * SAFETY, 1e-6)
        self.step(step, "PASS", f"{summary['rows']} of {summary['rows']} rows returned a choice and probabilities")
        self.usage = self.measure_usage(handle)
        if self.usage.seconds is None:
            self.step("1c key usage", "FAIL", "key usage did not show the spend within the wait")
            raise Stage0Stop(step, f"OpenRouter's key usage did not rise to match {_money(self.usage.ledger_spend)} of Jev spend within "
                             f"{USAGE_WAIT_SECONDS} s (fields 13, 15, 16). Every paid Jev run would stop on a reconciliation gap: stop and ask.", ask=True)
        self.grace = grace_from_lag(self.usage.seconds)
        self.step("1c key usage", "PASS", f"spend showed in key usage after {self.usage.seconds:.0f} s; "
                  f"later clients wait {self.grace} s before believing a short report")
        self.bare_check()

    def measure_usage(self, handle) -> fields.UsageCheck:
        spend = handle.client.ledger.confirmed() - handle.base_confirmed
        start = self.rt.clock()
        reported = 0.0
        while True:
            try:
                reported = handle.client.reported_usage() - handle.base_reported
            except Exception:
                return fields.UsageCheck(None, spend, 0.0)
            if spend > 1e-9 and reported >= 0.9 * spend - 1e-9:
                return fields.UsageCheck(self.rt.clock() - start, spend, reported)
            if self.rt.clock() - start >= USAGE_WAIT_SECONDS:
                return fields.UsageCheck(None, spend, reported)
            self.rt.sleep(USAGE_POLL_SECONDS)

    def bare_check(self) -> None:
        step = "1b bare-name schema check"
        rows = sample_rows(self.validation("banking77"), BARE_ROWS, "stage0:bare:")
        self.obs.phase = "bare"
        handle = self.new_client(step)  # one client per step: the smoke client stays with the smoke arm
        report = self.run(handle, step, "jev_banking77_bare", rows,
                          self.jev_decider(handle, step, "banking77", bare=True, cache=False), self.est_jev)
        summary = summarize(report.decisions)
        self.facts["bare"] = summary
        if report.status != "complete":
            self.step(step, "FAIL", f"{report.stop_type}: {report.reason}")
            self.stop_if_stopped(step, report)
        if summary["failed"]:
            # Bare names are not accepted: the bare-name runs are not run, and Stage 0 goes on.
            self.facts["bare_not_run"] = True
            self.step(step, "NOT RUN", f"{summary['failed']} of {summary['rows']} rows failed; the Jev bare-name runs are not run")
            return
        self.step(step, "PASS", f"{summary['rows']} rows")

    def selection(self) -> None:
        step = "2 small LLM selection"
        rows = sample_rows(self.validation("banking77"), SELECT_ROWS, "stage0:select:")
        options = label_texts("banking77", bare=False)
        instructions = prompts.small_llm_instructions("banking77")
        self.obs.phase = "selection"
        self.selection_decided = False
        for model, provider, price, first_cost in CANDIDATES:
            handle = self.new_client(step)

            def decide(row, model=model, provider=provider, handle=handle):
                return llm_pick(row["row_id"], row["text"], options, model, provider, client=handle.client, instructions=instructions)

            slug = model.replace("/", "_")
            report = self.run(handle, step, f"select_{slug}", rows, self.timed(step, decide), first_cost,
                              allowed=len(rows) - SELECT_PASS, first_contact=True)
            summary = summarize(report.decisions)
            valid = summary["rows"] - summary["failed"]
            outage = report.stop_type == "ConsecutiveFailures" and bool(summary["reasons"]) and all(
                r in ("network", "usage_unavailable") or r.startswith("http_") for r in summary["reasons"])
            entry = {"outage": outage, "requested": model, "provider": provider, "price": price, "valid": valid, "of": len(rows),
                     "summary": summary, "stop": report.stop_type, "passed": False}
            self.candidates[model] = entry
            if report.status != "complete" and report.stop_type not in CANDIDATE_STOPS:
                self.step(step, "FAIL", f"{model}: {report.stop_type}: {report.reason}")
                raise Stage0Stop(step, f"{model} on {provider}: {report.stop_type}: {report.reason}",
                                 ask=report.stop_type == "ModelChanged")
            if valid >= SELECT_PASS:
                entry["passed"] = True
                models = sorted({d.model for d in report.decisions.values() if not d.failed})
                entry["reply_models"] = models
                self.selected = entry
                self.selection_decided = True
                self.step(step, "PASS", f"{model} on {provider}: {valid} of {len(rows)} usable")
                return
        self.selection_decided = True
        tried = ", ".join(f"{m}: {c['valid']} of {c['of']}" for m, c in self.candidates.items())
        self.step(step, "NOT RUN", f"no candidate reached {SELECT_PASS} of {SELECT_ROWS} ({tried}); the small LLM arm is marked not run")

    def jev_validation(self) -> None:
        for dataset, label in (("banking77", "3a Jev Banking77 validation"), ("ag_news", "3b Jev AG News validation")):
            val = self.validation(dataset)
            rows = val.to_dict("records")
            handle = self.new_client(label)
            self.obs.phase = "full"
            expected = self.est_jev
            report = self.run(handle, label, f"jev_{dataset}_val", rows, self.jev_decider(handle, label, dataset), expected,
                              first_contact=True, check_rate=True)
            if report.status != "complete":
                self.step(label, "FAIL", f"{report.stop_type}: {report.reason}")
                self.stop_if_stopped(label, report)
            summary = summarize(report.decisions)
            summary.update(mass_and_cut(report.decisions))
            self.facts[dataset] = summary
            fresh = summary["cost_per_row"]
            if dataset == "banking77":
                correct = sum(1 for r in rows if report.decisions[r["row_id"]].pred == r["label"])
                self.facts["bug_floor_jev"] = correct / len(rows) >= BUG_FLOOR_JEV  # only the verdict is kept
                if fresh:
                    self.est_jev = max(fresh * SAFETY, 1e-6)
            rate = summary["failed"] / summary["rows"]
            if rate > FAILURE_RATE_LIMIT:
                self.step(label, "FAIL", f"{summary['failed']} of {summary['rows']} failed ({rate:.1%})")
                raise Stage0Stop(label, f"Jev failures are {summary['failed']} of {summary['rows']} ({rate:.1%}), above 1%. Pause and log.")
            if dataset == "banking77":
                if summary["mass"] is not None and summary["mass"] >= MASS_LIMIT:
                    self.step(label, "FAIL", f"1.0 mass {summary['mass']:.1%} is 75% or higher")
                    raise Stage0Stop(label, "Jev's 1.0 mass on Banking77 validation is 75% or higher: stop and re-decide coverage before the freeze.", ask=True)
                if not self.facts["bug_floor_jev"]:
                    self.step(label, "FAIL", "bug floor FAIL")
                    raise Stage0Stop(label, "the Banking77 bug floor for Jev is FAIL, which means a mapping bug. Look before spending more.")
            self.step(label, "PASS", f"{summary['rows']} rows, {summary['failed']} failed")

    def repeatability(self) -> None:
        step = "4 repeatability"
        base = sample_rows(self.validation("banking77"), REPEAT_ROWS, "stage0:repeat:")
        rows = [{**r, "key": f"{r['row_id']}#{k}"} for r in base for k in range(REPEATS)]
        handle = self.new_client(step)
        self.obs.phase = "repeat"
        decide = self.jev_decider(handle, step, "banking77", cache=False)
        report = self.run(handle, step, "jev_banking77_repeat", rows, decide, self.est_jev, row_id=lambda r: r["key"], check_rate=True)
        summary = summarize(report.decisions)
        stats = repeat_stats(report.decisions)
        self.facts["repeat"] = {**summary, **stats}
        if report.status != "complete":
            self.step(step, "FAIL", f"{report.stop_type}: {report.reason}")
            self.stop_if_stopped(step, report)
        if summary["failed"] / summary["rows"] > FAILURE_RATE_LIMIT:
            self.step(step, "FAIL", f"{summary['failed']} of {summary['rows']} calls failed")
            raise Stage0Stop(step, "repeat-call failures are above 1%. Pause and log.")
        flag = stats["agreement"] is not None and stats["agreement"] < AGREEMENT_FLOOR
        self.facts["repeat"]["disclose"] = flag or stats["agreement"] is None
        self.step(step, "PASS", "measured" + ("; below 98% agreement, disclose" if flag else ""))

    def examples_smoke(self) -> None:
        step = "5 Jev + examples smoke"
        split = self.rt.load_split("banking77")
        pool = data.bank(split.pool, BANK_DRAW, BANK_S)
        bank = [Example(text=r["text"], label=r["label"]) for r in pool.to_dict("records")]
        rows = sample_rows(split.val, EXAMPLES_ROWS, "stage0:examples:")
        handle = self.new_client(step)
        self.obs.phase = "examples"
        decide = self.jev_decider(handle, step, "banking77", bank=bank)
        report = self.run(handle, step, "jev_banking77_examples", rows, decide, self.est_jev * EXAMPLES_COST_FACTOR, first_contact=True)
        summary = summarize(report.decisions)
        counts = [d.n_examples for d in report.decisions.values() if d.n_examples is not None]
        summary.update(examples_min=min(counts) if counts else None, examples_max=max(counts) if counts else None,
                       examples_mean=statistics.fmean(counts) if counts else None)
        self.facts["examples"] = summary
        if report.status != "complete" or summary["failed"]:
            self.step(step, "FAIL", f"{summary['failed']} of {summary['rows']} failed; {report.stop_type or ''}")
            raise Stage0Stop(step, f"Jev + examples did not return {summary['rows']} of {summary['rows']}: {report.stop_type} {report.reason or ''}")
        self.step(step, "PASS", f"{summary['rows']} of {summary['rows']} rows")

    # ------------------------------------------------------------------ projection and finish

    def project_stage1(self) -> dict:
        """Test-set rows times measured cost per row. A missing measure leaves its line out and says so."""
        f = self.facts
        per = {
            "jev_banking77": (f.get("banking77") or {}).get("cost_per_row"),
            "jev_ag_news": (f.get("ag_news") or {}).get("cost_per_row"),
            "jev_examples": (f.get("examples") or {}).get("cost_per_row"),
        }
        lines = {}
        b, a = STAGE1_ROWS["banking77"], STAGE1_ROWS["ag_news"]
        if per["jev_banking77"] is not None and per["jev_ag_news"] is not None and per["jev_examples"] is not None:
            lines["Jev Banking77"] = b * per["jev_banking77"]
            lines["Jev AG News"] = a * per["jev_ag_news"]
            lines["Jev + examples Banking77"] = b * per["jev_examples"]
            # Bare names send less text than descriptions, so the described cost is an upper bound.
            lines["Jev bare-name Banking77 (upper bound)"] = b * per["jev_banking77"]
            lines["Jev bare-name AG News (upper bound)"] = a * per["jev_ag_news"]
        llm = self.selected
        if llm is not None and llm["summary"]["cost_per_row"] is not None:
            c = llm["summary"]["cost_per_row"]
            ratio = 1.0
            if per["jev_banking77"] and per["jev_ag_news"] and f["banking77"].get("mean_in") and f["ag_news"].get("mean_in"):
                ratio = f["ag_news"]["mean_in"] / f["banking77"]["mean_in"]  # AG News prompts are shorter
            lines["Small LLM Banking77"] = b * c
            lines["Small LLM AG News (scaled by Jev input tokens)"] = a * c * ratio
        subtotal = sum(lines.values())
        total = subtotal * (1 + CREDIT_FEE)
        spent = self.ledger.spent() if self.ledger is not None else 0.0
        left = HARD_CAP_USD - spent
        complete = len(lines) >= 5 and "Jev + examples Banking77" in lines
        return {"lines": lines, "subtotal": subtotal, "total_with_fee": total, "total_cad": total * CAD_PER_USD,
                "spent": spent, "left": left, "complete": complete, "fits": complete and total <= left,
                "routing_note": "routing pilot spend is not measured here and is not in this total"}

    def run_all(self) -> int:
        stop = None
        try:
            self.smoke()
            self.selection()
            self.jev_validation()
            self.repeatability()
            self.examples_smoke()
        except Stage0Stop as error:
            stop = error
            self.step(error.step, "STOPPED", error.reason)
        except RunStop as error:  # a stop raised outside run_arm, such as at client start
            stop = Stage0Stop("stage0", f"{type(error).__name__}: {error}")
            self.step("stage0", "STOPPED", stop.reason)
        projection = self.project_stage1() if stop is None or self.facts.get("banking77") else None
        self.findings = fields.evaluate(self.obs, self.usage, self.candidates)
        blocking = [f for f in self.findings if f.status == fields.MISMATCH]
        self.write_outputs(stop, projection)
        if stop is not None:
            print(f"STOPPED at {stop.step}: {stop.reason}" + ("\nThis one needs Aarmen before anything else." if stop.ask else ""))
            return EXIT_STOPPED
        failing = [name for name, status, _ in self.steps if status == "FAIL"]
        ok = not failing and not blocking and projection is not None and projection["fits"]
        print("Stage 0 " + ("PASS" if ok else "FAIL") + f"; bug floor {'PASS' if self.facts.get('bug_floor_jev') else 'FAIL'}; report written")
        return EXIT_OK if ok else EXIT_FAIL

    # ------------------------------------------------------------------ outputs

    def write_outputs(self, stop, projection) -> None:
        report_dir = Path(self.rt.report_dir)
        (report_dir / "stage0_report.md").write_text(render_report(self, stop, projection), encoding="utf-8")
        (report_dir / "frozen.yaml").write_text(yaml.safe_dump(frozen_values(self, stop), sort_keys=False), encoding="utf-8")


def grace_from_lag(seconds: float) -> int:
    """The measured lag rounded up, at least the default grace and at most GRACE_CAP_SECONDS."""
    return int(min(GRACE_CAP_SECONDS, max(DEFAULT_RECONCILE_GRACE_SECONDS, math.ceil(seconds))))


def _money(x) -> str:
    return "n/a" if x is None else f"US${x:.4f}"


def _num(x, digits=2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def frozen_values(st: Stage0, stop) -> dict:
    selected = st.selected
    llm = {"status": "not run" if st.selection_decided else "pending"}  # pending: Stage 0 stopped before selection decided
    if selected is not None:
        llm = {
            "status": "selected",
            "model": selected["requested"],
            "provider": selected["provider"],
            "provider_options": {"only": [selected["provider"]], "allow_fallbacks": False, "require_parameters": True},
            "reply_model_suffix_pattern": MODEL_SUFFIX,
            "reply_models_seen": selected.get("reply_models", []),
            "price_per_million_tokens_usd": {"input": selected["price"][0], "output": selected["price"][1], "as_of": "2026-11-04"},
        }
    models = st.facts.get("jev_models", [])
    return {
        "stage0_complete": stop is None,
        # The frozen build is the expected constant; field 1 and the model-change stop enforce that replies match it.
        "jev": {"request_model": JEV_MODEL, "provider": JEV_PROVIDER, "build": EXPECTED_JEV_BUILD, "reply_models_seen": models},
        "small_llm": llm,
        "nli": dict(NLI),
        # For Stage 1's clients: the usage lag step 1c measured, and the grace that follows from it (ruling R7).
        "key_usage": {
            "lag_seconds": None if st.usage is None or st.usage.seconds is None else round(st.usage.seconds, 1),
            "reconcile_grace_seconds": st.grace,
        },
    }


def render_report(st: Stage0, stop, projection) -> str:
    f = st.facts
    out = ["# Stage 0 report", ""]
    out.append("Validation rows only. The bug floor is shown as PASS or FAIL and no other score of an answer is kept.")
    out.append("")
    out.append("## Result")
    out.append("")
    if stop is None:
        failing = [n for n, s, _ in st.steps if s == "FAIL"]
        blocking = [x for x in st.findings if x.status == fields.MISMATCH]
        out.append("Overall: " + ("PASS" if not failing and not blocking and projection and projection["fits"] else "FAIL"))
    else:
        out.append(f"Overall: STOPPED at {stop.step}")
        out.append(f"Reason: {stop.reason}")
        if stop.ask:
            out.append("This stop needs Aarmen's decision before anything else.")
    out.append("")
    out.append("| Step | Result | Detail |")
    out.append("|---|---|---|")
    for name, status, detail in st.steps:
        out.append(f"| {name} | {status} | {detail} |")
    out.append("")
    out.append("## Checks")
    out.append("")
    out.append(f"- Bug floor, Jev on Banking77 validation: {'PASS' if f.get('bug_floor_jev') else 'FAIL' if 'bug_floor_jev' in f else 'not run'}")
    out.append(f"- Spend this stage: {_money(st.ledger.stage_spent() if st.ledger else None)} of the US${STAGE_CAPS_USD[STAGE]:.2f} limit")
    out.append("")
    out.append("## Jev")
    out.append("")
    out.append("| Set | Rows | Failed | Failure rate | 1.0 mass | Score at the 80% cut |")
    out.append("|---|---|---|---|---|---|")
    for key, label in (("smoke", "Banking77 smoke"), ("banking77", "Banking77 validation"), ("ag_news", "AG News validation")):
        s = f.get(key)
        if s:
            mass = f"{s['mass_count']} ({_pct(s['mass'])})" if "mass" in s else "n/a"
            cut = _num(s.get("cut_score")) if "cut_score" in s else "n/a"
            out.append(f"| {label} | {s['rows']} | {s['failed']} | {_pct(s['failed'] / s['rows'] if s['rows'] else None)} | {mass} | {cut} |")
    reasons = {k: v["reasons"] for k, v in f.items() if isinstance(v, dict) and v.get("reasons")}
    if reasons:
        out.append("")
        out.append("Failure reasons: " + "; ".join(f"{k}: {r}" for k, r in reasons.items()))
    out.append("")
    out.append("## Repeatability")
    out.append("")
    r = f.get("repeat")
    if r:
        out.append(f"- Rows: {r['rows']} x {REPEATS} calls; rows with all {REPEATS} answers: {r['complete']}")
        out.append(f"- Label agreement: {_pct(r['agreement'])}")
        out.append(f"- Median top-probability range: {_num(r['median_range'], 3)}")
        if r.get("disclose"):
            out.append("- Below 98% agreement: disclose, and say in the post that Jev has run noise not in the CI.")
    else:
        out.append("Not run.")
    out.append("")
    out.append("## Small LLM selection")
    out.append("")
    for model, c in st.candidates.items():
        s = c["summary"]
        out.append(f"- {model} on {c['provider']}: {c['valid']} of {c['of']} usable ({'PASS' if c['passed'] else 'FAIL'}); "
                   f"failures {s['failed']}; mean output tokens {_num(s['mean_out'], 1)}; cost {_money(s['cost'])}"
                   + ("; stopped by an outage, not judged on its answers" if c.get("outage") else f"; stopped on {c['stop']}" if c["stop"] else ""))
    if st.selected:
        out.append(f"- Selected: {st.selected['requested']} on {st.selected['provider']}")
    elif st.candidates:
        out.append("- No candidate passed. The small LLM arm is not run.")
    else:
        out.append("- Not run.")
    out.append("")
    out.append("## Tokens, cost and latency")
    out.append("")
    out.append("| Step | Calls | Mean input tokens | Mean output tokens | Cost per row | Latency median s | Latency p95 s |")
    out.append("|---|---|---|---|---|---|---|")
    for key, label, step in (
        ("smoke", "Jev smoke", "1 Jev smoke"),
        ("banking77", "Jev Banking77 validation", "3a Jev Banking77 validation"),
        ("ag_news", "Jev AG News validation", "3b Jev AG News validation"),
        ("repeat", "Jev repeatability", "4 repeatability"),
        ("examples", "Jev + examples smoke", "5 Jev + examples smoke"),
    ):
        s = f.get(key)
        if s:
            lat = st.latency.get(step, [])
            out.append(f"| {label} | {s['rows'] - s['cached']} | {_num(s['mean_in'], 0)} | {_num(s['mean_out'], 0)} | {_money(s['cost_per_row'])} | "
                       f"{_num(_median(lat))} | {_num(_p95(lat))} |")
    for model, c in st.candidates.items():
        s = c["summary"]
        out.append(f"| {model} selection | {s['rows']} | {_num(s['mean_in'], 0)} | {_num(s['mean_out'], 1)} | {_money(s['cost_per_row'])} | - | - |")
    e = f.get("examples")
    if e and e.get("examples_mean") is not None:
        out.append("")
        out.append(f"Examples retrieved per row: min {e['examples_min']}, mean {_num(e['examples_mean'], 1)}, max {e['examples_max']}.")
    out.append("")
    out.append("## Projected Stage 1 spend")
    out.append("")
    if projection and projection["lines"]:
        for name, cost in projection["lines"].items():
            out.append(f"- {name}: {_money(cost)}")
        out.append(f"- Total with the 5.5% credit fee: {_money(projection['total_with_fee'])} (CA${projection['total_cad']:.2f})")
        out.append(f"- Spent so far: {_money(projection['spent'])}; left under the US${HARD_CAP_USD} cap: {_money(projection['left'])}")
        out.append(f"- Fits under the cap: {'PASS' if projection['fits'] else 'FAIL' if projection['complete'] else 'not computed, a measure is missing'}")
        out.append(f"- {projection['routing_note']}")
    else:
        out.append("Not computed.")
    out.append("")
    out.append("## Fields to confirm (Task 5 report)")
    out.append("")
    out.append("| # | Field | Status | Note |")
    out.append("|---|---|---|---|")
    for x in st.findings:
        out.append(f"| {x.number} | {x.name} | {x.status} | {x.note} |")
    out.append("")
    return "\n".join(out)


def main(argv=None, runtime: Runtime | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_paid_argument(parser)
    parser.add_argument("--accept-key-usage-usd", type=float, default=None,
                        help="spend already on the key before the ledger began, if the first start refuses and Aarmen accepts it")
    args = parser.parse_args(argv)
    if runtime is None:
        if args.paid != STAGE:
            parser.error("Stage 0 spends real money: run it with --paid stage0, after Aarmen's yes")
        runtime = Runtime(make_client=paid_client_factory(args.paid, args.accept_key_usage_usd))
    return Stage0(runtime).run_all()


def paid_client_factory(paid: str, accept_key_usage_usd=None) -> Callable[..., Client]:
    """Clients on the real paid route. The accepted amount goes to the first start only.

    If it is refused (not an amount, or nothing to accept because the ledger already agrees), the
    client raises a RunStop before sending, and Stage 0 ends with a clean message.
    """
    accept = [accept_key_usage_usd]

    def make_client(reconcile_grace_seconds=None) -> Client:
        amount, accept[0] = accept[0], None  # only the first start can record a baseline
        grace = DEFAULT_RECONCILE_GRACE_SECONDS if reconcile_grace_seconds is None else reconcile_grace_seconds
        return build_client(paid, accept_key_usage_usd=amount, reconcile_grace_seconds=grace)

    return make_client


if __name__ == "__main__":
    sys.exit(main())
