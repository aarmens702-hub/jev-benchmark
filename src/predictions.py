"""Prediction files: one parquet per arm and dataset under runs/<stage>/, the full vector under runs/private/.

Columns: row_id, pred, top_prob, failed. The private file holds each row's whole score vector and the
class order, and is never published. Writes are atomic, so a crash leaves the last complete file.
"""
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parents[1] / "runs"
COLUMNS = ["row_id", "pred", "top_prob", "failed"]


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
    if list(rows.columns) != COLUMNS:
        raise ValueError(f"columns must be exactly {COLUMNS}, got {list(rows.columns)}")
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
    vectors = classes = None
    if private.exists():
        with np.load(private, allow_pickle=False) as z:
            vectors, classes = z["vectors"], [str(c) for c in z["classes"]]
            if list(z["row_id"]) != rows["row_id"].tolist():
                raise ValueError(f"{private.name} does not match {public.name}; delete both and rerun")
    return rows, vectors, classes
