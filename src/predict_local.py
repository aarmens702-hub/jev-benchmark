"""Run the free arms on the Mac and write prediction files.

    scripts/predict_local.sh --stage val --floor
    scripts/predict_local.sh --stage val --time-nli
    scripts/predict_local.sh --stage val --arm nli --dataset ag_news --max-hours 8

(The script sets PYTHONPATH=src and runs `uv run python -m predict_local`.)

Predictors see only the text of the rows they predict. Labels come from banks and nowhere else, and
prediction rows are reduced to (row_id, text) before any arm runs. Only the validation stage is wired
here; the test stage is run by a later stage runner after the freeze. Nothing in this module prints
or saves an accuracy: the floor check prints PASS or FAIL and nothing else.
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

import arms_local
import data
import guards
import predictions
from guards import Guard, JobStopped, model_job_lock
from labels import label_texts

log = logging.getLogger("predict_local")

NLI_CHECKPOINT_ROWS = 500
FLOOR_LR_S35 = 0.75  # VALIDATION.md, check 5: Banking77 validation LR at s=35 is at least 75% (a bug floor only)
GRID = (1, 2, 3, 5, 8, 16, 35, "all")
DRAWS = tuple(range(10))
TIMING_PAIRS_PER_DATASET = 2500  # 5,000 pairs, half from each dataset
TOTAL_NLI_PAIRS = 3080 * 77 + 7600 * 4  # the Banking77 and AG News test sets against all their labels
NLI_LIMIT_HOURS = 12.0
EXIT_OK, EXIT_FLOOR_FAIL, EXIT_STOPPED = 0, 1, 2


@dataclass
class Runtime:
    """Everything that touches the machine, the data or a model, so tests can swap in fakes."""

    runs_root: Path = predictions.RUNS
    load_split: Callable[[str], data.Split] = lambda dataset: data.split(data.load_train(dataset), dataset)
    make_encoder: Callable[[], Callable] = lambda: arms_local.load_encoder()
    make_nli_scorer: Callable[[str], Callable] = lambda key: arms_local.load_nli_scorer(key)
    clock: Callable[[], float] = time.monotonic
    timer: Callable[[], float] = time.perf_counter  # per-decision wall-clock in prediction files
    read_swap_gb: Callable[[], float] = guards.swap_used_gb
    read_free_disk_gb: Callable[..., float] = guards.free_disk_gb

    @property
    def lock_path(self) -> Path:
        return Path(self.runs_root) / "model_job.lock"

    def guard(self, max_hours: Optional[float] = None) -> Guard:
        Path(self.runs_root).mkdir(parents=True, exist_ok=True)
        return Guard(
            max_hours,
            clock=self.clock,
            read_swap_gb=self.read_swap_gb,
            read_free_disk_gb=self.read_free_disk_gb,
            path=str(self.runs_root),
        )


# --------------------------------------------------------------------------- checkpointed arm runs


@dataclass
class ArmRun:
    completed: int  # chunks finished in this call
    rows_done: int  # rows saved in total
    stopped: Optional[str] = None
    message: str = ""


def _rows_frame(row_ids, vec: np.ndarray, classes, to_probs, latency=np.nan) -> pd.DataFrame:
    ok = np.isfinite(vec).all(axis=1)
    safe = np.where(ok[:, None], vec, 0.0)
    probs = to_probs(safe)
    cls = np.array(list(classes), dtype=object)
    return pd.DataFrame(
        {
            "row_id": list(row_ids),
            "pred": np.where(ok, cls[safe.argmax(axis=1)], ""),
            "top_prob": np.where(ok, probs.max(axis=1), 0.0),
            "failed": ~ok,
            "latency_s": np.broadcast_to(np.asarray(latency, dtype=float), len(ok)).copy(),
        }
    )


def run_arm(
    arm: str,
    dataset: str,
    stage: str,
    rows: pd.DataFrame,
    predict_fn: Callable[[list], np.ndarray],
    classes,
    *,
    to_probs: Callable[[np.ndarray], np.ndarray] = lambda v: v,
    chunk_size: int = NLI_CHECKPOINT_ROWS,
    guard: Optional[Callable[[], None]] = None,
    root: Path = predictions.RUNS,
    prepare: Optional[Callable[[], None]] = None,
    timer: Callable[[], float] = time.perf_counter,
) -> ArmRun:
    """Predict `rows` in chunks, saving after each chunk, and skip rows already saved (resume).

    `predict_fn` is handed only the list of texts for the chunk being predicted and returns one score
    vector per text. Rows may carry any other columns (a label, say); they are dropped here.
    `prepare` (fitting a model, say) runs once before the first chunk and outside the timed span, so
    each row's latency_s is the chunk's prediction time divided by its rows.
    """
    rows = rows[["row_id", "text"]]
    classes = list(classes)
    saved = predictions.read(arm, dataset, stage, root)
    if saved is None:
        done_rows, done_vecs = rows.iloc[0:0][["row_id"]].assign(pred="", top_prob=0.0, failed=False, latency_s=0.0), np.zeros((0, len(classes)))
    else:
        done_rows, done_vecs, prior_classes = saved
        if prior_classes != classes or done_vecs is None:
            raise ValueError(f"{arm}: saved classes differ from this run's classes; delete the old files to start over")
    state = {"rows": done_rows, "vecs": done_vecs}
    todo = rows[~rows["row_id"].isin(set(done_rows["row_id"]))]
    chunks = (todo.iloc[i : i + chunk_size] for i in range(0, len(todo), chunk_size))

    prepared = []

    def process(chunk: pd.DataFrame) -> None:
        if prepare is not None and not prepared:
            prepare()
            prepared.append(True)
        started = timer()
        vec = np.asarray(predict_fn(chunk["text"].tolist()), dtype=np.float64)
        per_row = (timer() - started) / len(chunk)
        if vec.shape != (len(chunk), len(classes)):
            raise ValueError(f"{arm}: predictor returned shape {vec.shape}, expected {(len(chunk), len(classes))}")
        frame = _rows_frame(chunk["row_id"], vec, classes, to_probs, per_row)
        state["rows"] = pd.concat([state["rows"], frame], ignore_index=True)
        state["vecs"] = np.vstack([state["vecs"], vec])
        predictions.write(arm, dataset, state["rows"], stage, state["vecs"], classes, root)

    outcome = guards.run_with_limit(chunks, process, guard or (lambda: None))
    return ArmRun(outcome.completed, len(state["rows"]), outcome.stopped, outcome.message)


# --------------------------------------------------------------------------- banks and arms


def make_bank(sp: data.Split, dataset: str, draw: int, s, stage: str) -> pd.DataFrame:
    """The label bank for this draw and size. At the validation stage the validation rows are the rows
    being predicted, so s="all" is the pool alone; the stage runner adds them back for the test stage."""
    if s == "all":
        return data.bank(sp.pool, draw, "all", val=sp.val.iloc[0:0] if stage == "val" else sp.val)
    return data.bank(sp.pool, draw, s)


def _arm_name(kind: str, draw: int, s) -> str:
    return f"{kind}_d{draw}_s{s}"


BANK_SIZES_FILE = "bank_sizes.json"


def record_bank_size(root: Path, stage: str, dataset: str, draw: int, s, bank: pd.DataFrame) -> None:
    """Keep each bank's row and class count, so the chart can label "all" with real counts."""
    path = Path(root) / stage / BANK_SIZES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = json.loads(path.read_text()) if path.exists() else {}
    sizes.setdefault(dataset, {})[f"d{draw}_s{s}"] = {"rows": int(len(bank)), "classes": int(bank["label"].nunique())}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(sizes, indent=1, sort_keys=True))
    os.replace(tmp, path)


def run_embed_sim(rt: Runtime, dataset: str, stage: str, bare: bool, encode, guard) -> ArmRun:
    base = "embed_sim_bare" if bare else "embed_sim"
    sp = rt.load_split(dataset)
    classes = list(label_texts(dataset, bare=bare))
    label_list = list(label_texts(dataset, bare=bare).values())
    rows = sp.val[["row_id", "text"]]
    res = run_arm(
        base, dataset, stage, rows,
        lambda texts: arms_local.embed_cosine(texts, label_list, encode),
        classes,
        to_probs=lambda cos: arms_local.softmax(cos / arms_local.EMBED_TEMPERATURE),
        guard=guard, root=rt.runs_root, timer=rt.timer,
    )  # fmt: skip
    if res.stopped is None:
        # Sensitivity rows come from the saved cosine scores, not a second pass over the model, so they
        # carry the primary run's latency.
        saved_rows, cos, saved_classes = predictions.read(base, dataset, stage, rt.runs_root)
        ids = saved_rows["row_id"]
        for suffix, view in arms_local.sensitivity_views(cos).items():
            if suffix == "":
                continue
            frame = _rows_frame(ids, view, saved_classes, lambda v: v, saved_rows["latency_s"].to_numpy())
            predictions.write(base + suffix, dataset, frame, stage, view, saved_classes, rt.runs_root)
    return res


def run_supervised(rt: Runtime, kind: str, dataset: str, stage: str, draw: int, s, encode, guard) -> ArmRun:
    sp = rt.load_split(dataset)
    classes = list(label_texts(dataset, bare=False))
    rows = sp.val[["row_id", "text"]]
    fitted = {}

    def fit():
        """Build the bank, embed it and fit the model, once and outside the timed span."""
        bank = make_bank(sp, dataset, draw, s, stage)
        record_bank_size(rt.runs_root, stage, dataset, draw, s, bank)
        y = bank["label"].to_numpy()
        if kind == "tfidf_lr":
            fitted["predict"] = arms_local.tfidf_lr_fitted((bank["text"].tolist(), y), classes=classes)
            return
        X = encode(bank["text"].tolist())
        model = arms_local.lr_fitted((X, y), classes=classes) if kind == "lr" else arms_local.lr_cv_fitted((X, y), s, classes=classes)
        fitted["predict"] = lambda texts: model(encode(texts))

    def predict(texts):
        if "predict" not in fitted:
            fit()
        return fitted["predict"](texts)

    return run_arm(
        _arm_name(kind, draw, s), dataset, stage, rows, predict, classes,
        chunk_size=max(len(rows), 1), guard=guard, root=rt.runs_root, prepare=fit, timer=rt.timer,
    )  # fmt: skip


def run_nli(rt: Runtime, dataset: str, stage: str, key: str, guard) -> ArmRun:
    sp = rt.load_split(dataset)
    classes = list(label_texts(dataset, bare=False))
    hyps = arms_local.nli_hypotheses(dataset, bare=False)
    inner = rt.make_nli_scorer(key)

    def scorer(batch):
        # A 500-row chunk can take hours, so the guard runs before every batch, not just between chunks.
        # A stop here drops the partial chunk; the last saved checkpoint stays valid.
        guard()
        return inner(batch)

    arm = "nli" if key == "large" else f"nli_{key}"
    return run_arm(
        arm, dataset, stage, sp.val[["row_id", "text"]],
        lambda texts: np.stack(list(arms_local.nli(texts, hyps, arms_local.NLI_BATCH_SIZE, scorer))),
        classes, chunk_size=NLI_CHECKPOINT_ROWS, guard=guard, root=rt.runs_root, timer=rt.timer,
    )  # fmt: skip


# --------------------------------------------------------------------------- floor check


def floor_passes(pred, gold) -> bool:
    """True if validation accuracy clears the bug floor. Returns only the verdict; the value goes nowhere."""
    return np.mean(np.asarray(pred) == np.asarray(gold)) >= FLOOR_LR_S35


def run_floor(rt: Runtime) -> bool:
    """Banking77 validation, embedding LR at C=100, draw 0, s=35. Writes nothing and prints nothing."""
    sp = rt.load_split("banking77")
    encode = arms_local.memoize_encoder(rt.make_encoder())
    bank = data.bank(sp.pool, 0, 35)
    classes = sorted(set(bank["label"]))
    probs = arms_local.lr((encode(bank["text"].tolist()), bank["label"].to_numpy()), encode(sp.val["text"].tolist()), classes=classes)
    pred = np.array(classes, dtype=object)[probs.argmax(axis=1)]
    return bool(floor_passes(pred, sp.val["label"].to_numpy()))


# --------------------------------------------------------------------------- NLI timing


def time_nli(rt: Runtime, stage: str, key: str, guard) -> dict:
    """Time 5,000 (text, hypothesis) pairs, half per dataset, at the fixed batch size, on validation texts."""
    scorer = rt.make_nli_scorer(key)
    n_pairs = 0
    t0 = rt.clock()
    for dataset in ("banking77", "ag_news"):
        texts = rt.load_split(dataset).val["text"].tolist()
        hyps = arms_local.nli_hypotheses(dataset, bare=False)
        pairs = islice(((t, h) for t in texts for h in hyps), TIMING_PAIRS_PER_DATASET)
        while batch := list(islice(pairs, arms_local.NLI_BATCH_SIZE)):
            guard()
            scorer(batch)
            n_pairs += len(batch)
    seconds = rt.clock() - t0
    rate = n_pairs / seconds if seconds > 0 else float("inf")
    hours = TOTAL_NLI_PAIRS / rate / 3600.0
    return {
        "model": key,
        "model_id": arms_local.NLI_MODELS[key][0],
        "batch_size": arms_local.NLI_BATCH_SIZE,
        "pairs": n_pairs,
        "seconds": seconds,
        "pairs_per_second": rate,
        "total_pairs": TOTAL_NLI_PAIRS,
        "projected_hours": hours,
        "use_base": hours > NLI_LIMIT_HOURS,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def report_nli_timing(rt: Runtime, stage: str, t: dict) -> None:
    print(
        f"NLI timing ({t['model']}, batch {t['batch_size']}): {t['pairs_per_second']:.1f} pairs/s over {t['pairs']} pairs; "
        f"projected {t['projected_hours']:.2f} h for {t['total_pairs']:,} pairs"
    )
    base_id, base_rev = arms_local.NLI_MODELS["base"]
    if not t["use_base"]:
        print(f"12-hour rule: kept deberta-v3-{t['model']}")
    elif t["model"] == "base":
        print("12-hour rule: the base model is also projected over 12 h; stop and ask Aarmen")
    else:
        print(f"12-hour rule: use {base_id} at {base_rev}, disclosed")
    path = Path(rt.runs_root) / stage / "nli_timing.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(t) + "\n")


# --------------------------------------------------------------------------- command line


def _s_value(text: str):
    return "all" if text == "all" else int(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="predict_local", description="Free arms on the Mac (validation stage only here).")
    p.add_argument("--stage", choices=["val"], default="val", help="the test stage is run by the stage runner after the freeze")
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--floor", action="store_true", help="LR s=35 bug floor; prints PASS or FAIL only")
    what.add_argument("--time-nli", action="store_true", help="time 5,000 NLI pairs and apply the 12-hour rule")
    what.add_argument("--arm", choices=["embed_sim", "lr", "lr_cv", "tfidf_lr", "nli"])
    p.add_argument("--dataset", choices=["banking77", "ag_news"])
    p.add_argument("--bare", action="store_true", help="bare class names as label text (embed_sim only)")
    p.add_argument("--draw", type=int, default=0)
    p.add_argument("--s", type=_s_value, default=None, help="bank size: 1, 2, 3, 5, 8, 16, 35 or all")
    p.add_argument("--grid", action="store_true", help="every draw and size for lr, lr_cv or tfidf_lr")
    p.add_argument("--nli-model", choices=["large", "base"], default="large")
    p.add_argument("--max-hours", type=float, default=None)
    return p


def _grid_units():
    for s in GRID:
        for draw in ((0,) if s == "all" else DRAWS):  # at s=all every draw is identical
            yield draw, s


def main(argv=None, rt: Optional[Runtime] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rt = rt or Runtime()
    if args.arm and not args.dataset:
        parser.error("--arm needs --dataset")
    if args.bare and args.arm != "embed_sim":
        parser.error("--bare applies to embed_sim only; NLI and the supervised arms have no bare-name runs")
    supervised = args.arm in {"lr", "lr_cv", "tfidf_lr"}
    if supervised and not args.grid and args.s is None:
        parser.error(f"--arm {args.arm} needs --s (or --grid)")
    if args.grid and not supervised:
        parser.error("--grid applies to lr, lr_cv and tfidf_lr")

    with model_job_lock(rt.lock_path):
        guard = rt.guard(args.max_hours)
        if args.floor:
            ok = run_floor(rt)
            print(f"LR s=35 floor: {'PASS' if ok else 'FAIL'}")
            return EXIT_OK if ok else EXIT_FLOOR_FAIL
        try:
            if args.time_nli:
                report_nli_timing(rt, args.stage, time_nli(rt, args.stage, args.nli_model, guard))
                return EXIT_OK
            return _run_arm_command(args, rt, guard)
        except JobStopped as e:  # a guard tripped outside a chunk loop (e.g. while embedding a bank)
            print(f"stopped ({e.reason}): {e}; rerun to resume")
            return EXIT_STOPPED


def _run_arm_command(args, rt: Runtime, guard) -> int:
    results = []
    if args.arm == "nli":
        results.append((("nli" if args.nli_model == "large" else f"nli_{args.nli_model}"), run_nli(rt, args.dataset, args.stage, args.nli_model, guard)))
    elif args.arm == "embed_sim":
        encode = arms_local.memoize_encoder(rt.make_encoder(), cache_dir=Path(rt.runs_root) / "emb" / arms_local.EMBED_REVISION[:12], guard=guard)
        results.append(("embed_sim", run_embed_sim(rt, args.dataset, args.stage, args.bare, encode, guard)))
    else:
        encode = None if args.arm == "tfidf_lr" else arms_local.memoize_encoder(rt.make_encoder(), cache_dir=Path(rt.runs_root) / "emb" / arms_local.EMBED_REVISION[:12], guard=guard)
        units = _grid_units() if args.grid else [(args.draw, args.s)]
        for draw, s in units:
            res = run_supervised(rt, args.arm, args.dataset, args.stage, draw, s, encode, guard)
            results.append((_arm_name(args.arm, draw, s), res))
            if res.stopped:
                break
    for name, res in results:
        print(f"{name}: {res.rows_done} rows saved")
    for _, res in results:
        if res.stopped:
            print(f"stopped ({res.stopped}): {res.message}; rerun to resume")
            return EXIT_STOPPED
    return EXIT_OK


def cli() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        return main()
    except guards.JobLockedError as e:
        print(f"refusing to start: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(cli())
