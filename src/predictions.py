"""Prediction files: one parquet per arm and dataset under runs/<stage>/, the full vector under runs/private/.

Columns: row_id, pred, top_prob, failed, latency_s, then optionally the API arms' model, usage and
retry fields (EXTRA_COLUMNS). The private file holds each row's whole score vector and the class order,
and is never published. Writes are atomic, so a crash leaves the last complete file.

latency_s is wall-clock seconds per decision. For an API arm it is one request, retries included, with
up to 4 in flight. For a local arm it is a batch's time divided by its rows, after any fitting, so it is
an amortised figure; clean per-request latency comes from latency_probe.py.

An arm that was not run gets a line in runs/<stage>/not_run.jsonl instead of a prediction file.
"""
import json
import math
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parents[1] / "runs"
COLUMNS = ["row_id", "pred", "top_prob", "failed", "latency_s"]
# API arms add these, in this order. Anything else (a gold column, say) is refused.
EXTRA_COLUMNS = ["model", "input_tokens", "output_tokens", "cost", "cached", "attempts", "reason", "n_examples"]
NOT_RUN_FILE = "not_run.jsonl"


def _check_stage(stage: str) -> None:
    if not stage or stage == "private" or "/" in stage or stage.startswith("."):
        raise ValueError(f"bad stage {stage!r}: it names a folder under runs/, and 'private' is reserved")


def paths(arm: str, dataset: str, stage: str, root: Path = RUNS) -> tuple[Path, Path]:
    """(public parquet, private npz) for this arm and dataset."""
    _check_stage(stage)
    name = f"{arm}__{dataset}"
    return Path(root) / stage / f"{name}.parquet", Path(root) / "private" / stage / f"{name}.npz"


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def write(
    arm: str,
    dataset: str,
    rows: pd.DataFrame,
    stage: str = "val",
    vectors: Optional[np.ndarray] = None,
    classes: Optional[Sequence[str]] = None,
    root: Path = RUNS,
) -> Path:
    """Write the arm's rows (and, if given, the full per-row vectors with their class order)."""
    public, private = paths(arm, dataset, stage, root)
    columns = list(rows.columns)
    extra = columns[len(COLUMNS):]
    if columns[: len(COLUMNS)] != COLUMNS or extra != [c for c in EXTRA_COLUMNS if c in extra]:
        raise ValueError(f"columns must be {COLUMNS} then any of {EXTRA_COLUMNS} in order, got {columns}")
    if rows["row_id"].duplicated().any():
        raise ValueError("duplicate row_id in predictions")
    if vectors is not None:
        vectors = np.asarray(vectors)
        if classes is None or len(vectors) != len(rows) or vectors.shape[1] != len(classes):
            raise ValueError("vectors need `classes` and one row each")
    _atomic(public, lambda p: rows.to_parquet(p, index=False))
    if vectors is not None:

        def save(p: Path) -> None:
            with open(p, "wb") as f:  # a file object, so numpy does not append ".npz" to the temp name
                np.savez(f, vectors=vectors, classes=np.array(list(classes), dtype=str), row_id=rows["row_id"].to_numpy(dtype=str))

        _atomic(private, save)
    return public


def read(arm: str, dataset: str, stage: str = "val", root: Path = RUNS):
    """(rows, vectors or None, classes or None) for a saved arm, or None if nothing is saved."""
    public, private = paths(arm, dataset, stage, root)
    if not public.exists():
        return None
    rows = pd.read_parquet(public)
    if "latency_s" not in rows.columns:  # written before latency was recorded
        rows.insert(len(COLUMNS) - 1, "latency_s", np.nan)
    vectors = classes = None
    if private.exists():
        with np.load(private, allow_pickle=False) as z:
            vectors, classes = z["vectors"], [str(c) for c in z["classes"]]
            if list(z["row_id"]) != rows["row_id"].tolist():
                raise ValueError(f"{private.name} does not match {public.name}; delete both and rerun")
    return rows, vectors, classes


def decisions_frame(row_ids: Sequence, decisions: Mapping, classes: Sequence[str]):
    """(rows, vectors or None) for an API arm, from run_arm's decisions, in the order of `row_ids`.

    Every row must have a decision: rows left not run are the caller's to resolve first. Vectors are
    returned only if every answered row gave a probability for every class (Jev does; the small LLM
    gives only its pick's), and belong in the private file only.
    """
    classes = list(classes)
    missing = [r for r in row_ids if r not in decisions]
    if missing:
        raise ValueError(f"no decision for {len(missing)} rows, first {missing[0]!r}; mark the arm not run or finish it")
    ds = [decisions[r] for r in row_ids]
    frame = pd.DataFrame(
        {
            "row_id": list(row_ids),
            "pred": ["" if d.failed else d.pred for d in ds],
            "top_prob": [0.0 if d.failed else float(d.top_prob) for d in ds],
            "failed": [bool(d.failed) for d in ds],
            "latency_s": [np.nan if d.latency_s is None else float(d.latency_s) for d in ds],
            "model": [d.model or "" for d in ds],
            "input_tokens": pd.array([d.input_tokens for d in ds], dtype="Int64"),
            "output_tokens": pd.array([d.output_tokens for d in ds], dtype="Int64"),
            "cost": [float(d.cost) for d in ds],
            "cached": [bool(d.cached) for d in ds],
            "attempts": [int(d.attempts) for d in ds],
            "reason": [d.reason or "" for d in ds],
            "n_examples": pd.array([d.n_examples for d in ds], dtype="Int64"),
        }
    )
    full = all(d.failed or set(d.probs) == set(classes) for d in ds)
    if not full:
        return frame, None
    vectors = np.array([[math.nan] * len(classes) if d.failed else [float(d.probs[c]) for c in classes] for d in ds])
    return frame, vectors


def mark_not_run(arm: str, dataset: str, stage: str, reason: str, root: Path = RUNS) -> Path:
    """Log that an arm was not run on a dataset, with its reason. The scorer accepts this in place of a file.

    `arm` may be a family name (lr, lr_cv, tfidf_lr) to cover every draw and size of that family.
    """
    _check_stage(stage)
    if not reason or not reason.strip():
        raise ValueError("a not-run record needs a reason")
    path = Path(root) / stage / NOT_RUN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"arm": arm, "dataset": dataset, "reason": reason}) + "\n")
    return path


def read_not_run(stage: str, root: Path = RUNS) -> dict:
    """{(arm, dataset): reason} for every not-run record of the stage."""
    _check_stage(stage)
    path = Path(root) / stage / NOT_RUN_FILE
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            out[(record["arm"], record["dataset"])] = record["reason"]
    return out
