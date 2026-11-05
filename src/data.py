"""Train data, validation split, class pools and nested label banks (VALIDATION.md, Splits).

Only the named TRAIN parquet files are fetched, by `hf_hub_download` at the pinned revisions.
No test file is named, fetched or read here; the test split belongs to `firewall.py` alone.
"""
import hashlib
import json
import unicodedata
from typing import Callable, Literal, NamedTuple

import numpy as np
import pandas as pd

SEED = 20261105
TRAIN_FILE = "data/train-00000-of-00001.parquet"
PINNED = {
    "banking77": ("PolyAI/banking77", "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"),
    "ag_news": ("fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4"),
}
# Validation rows per class, from the class's size n_c in the raw train file.
# Banking77: min(10, n_c - 35), so every class keeps at least 35 pool rows. AG News: 200.
VAL_PER_CLASS: dict[str, Callable[[int], int]] = {
    "banking77": lambda n_c: max(0, min(10, n_c - 35)),
    "ag_news": lambda n_c: 200,
}


class Split(NamedTuple):
    val: pd.DataFrame
    pool: pd.DataFrame


def normalize(text: str) -> str:
    """NFKC, lowercase, whitespace collapsed, stripped."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def split_key(norm: str) -> str:
    """Row order for the validation split: SHA-256 of "seed:normalized text"."""
    return hashlib.sha256(f"{SEED}:{norm}".encode()).hexdigest()


def bank_key(draw: int, norm: str) -> str:
    """Order within a class for bank draw r: SHA-256 of "r:s-bank:normalized text"."""
    return hashlib.sha256(f"{draw}:s-bank:{norm}".encode()).hexdigest()


def _check(dataset: str) -> tuple[str, str]:
    if dataset not in PINNED:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(PINNED)}")
    return PINNED[dataset]


def _train_path(dataset: str) -> str:
    from huggingface_hub import hf_hub_download

    repo_id, revision = _check(dataset)
    return hf_hub_download(repo_id=repo_id, filename=TRAIN_FILE, repo_type="dataset", revision=revision)


def label_names(dataset: str) -> list[str]:
    """Official class names, in label-id order, from the train parquet's own schema metadata."""
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(_train_path(dataset)).schema_arrow.metadata or {}
    return json.loads(meta[b"huggingface"])["info"]["features"]["label"]["names"]


def load_train(dataset: str) -> pd.DataFrame:
    """The train split as (row_id, text, label, norm); label is the official class name.

    row_id is "<dataset>:train:<original row index>", so it is stable across runs.
    """
    path = _train_path(dataset)
    names = label_names(dataset)
    raw = pd.read_parquet(path, columns=["text", "label"])
    df = pd.DataFrame(
        {
            "row_id": [f"{dataset}:train:{i}" for i in range(len(raw))],
            "text": raw["text"].to_numpy(),
            "label": [names[i] for i in raw["label"].to_numpy()],
        }
    )
    df["norm"] = df["text"].map(normalize)
    return df


def _ordered(df: pd.DataFrame, key: Callable[[str], str]) -> pd.DataFrame:
    """Rows sorted by hash key, ties (identical text) by row_id, so the order never depends on input order."""
    keyed = df.assign(_k=df["norm"].map(key))
    # row_id strings sort lexically; ties only occur between identical texts, so any fixed rule works.
    return keyed.sort_values(["_k", "row_id"], kind="stable").drop(columns="_k").reset_index(drop=True)


def _split(df: pd.DataFrame, n_val: Callable[[int], int]) -> Split:
    ordered = _ordered(df, split_key)
    sizes = ordered["label"].value_counts()
    rank = ordered.groupby("label", sort=False).cumcount()
    take = rank < ordered["label"].map(lambda c: n_val(int(sizes[c])))
    val = ordered[take].reset_index(drop=True)
    rest = ordered[~take]
    rest = rest[~rest["norm"].isin(set(val["norm"]))]  # shares text with a validation row
    pool = rest.drop_duplicates("norm", keep="first").reset_index(drop=True)  # first in hash order kept
    return Split(val=val, pool=pool)


def split(df: pd.DataFrame, dataset: str) -> Split:
    """Validation rows (first n per class in hash order) and the class pool."""
    _check(dataset)
    return _split(df, VAL_PER_CLASS[dataset])


def bank(pool: pd.DataFrame, draw: int, s: int | Literal["all"], val: pd.DataFrame | None = None) -> pd.DataFrame:
    """Label bank for draw r and size s: the first s pool rows per class in bank-hash order.

    Banks nest: for the same draw, the bank at s is a subset of the bank at any larger s. A class with
    fewer than s rows gives what it has. s="all" is the whole pool plus the validation rows, which
    are added back because no model choice uses them.
    """
    if s == "all":
        if val is None:
            raise ValueError('s="all" adds the validation rows back, so `val` is required')
        return _ordered(pd.concat([pool, val], ignore_index=True), lambda n: bank_key(draw, n)).sort_values(
            "label", kind="stable"
        ).reset_index(drop=True)
    if isinstance(s, (int, np.integer)) and not isinstance(s, bool):
        s = int(s)  # numpy integers (from grids built with numpy) are fine
    if not isinstance(s, int) or isinstance(s, bool) or s < 1:
        raise ValueError(f"s must be a positive int or 'all', got {s!r}")
    ordered = _ordered(pool, lambda n: bank_key(draw, n))
    kept = ordered[ordered.groupby("label", sort=False).cumcount() < s]
    return kept.sort_values("label", kind="stable").reset_index(drop=True)


def drop_test_duplicates(pool: pd.DataFrame, test_norms) -> tuple[pd.DataFrame, int]:
    """After the freeze: remove pool rows whose normalized text equals a test row's. Returns (pool, count).

    Banks drawn from the returned pool take the next row in order, since order depends only on text.
    """
    test_norms = set(test_norms)
    hit = pool["norm"].isin(test_norms)
    return pool[~hit].reset_index(drop=True), int(hit.sum())
