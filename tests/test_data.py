"""Splits, nested banks and label text (VALIDATION.md, Splits and Guards).

Tests marked `data` fetch the two named TRAIN parquet files from Hugging Face (opt-in:
`uv run pytest -m data`). Nothing here reads, fetches or names a test file. The unmarked tests
use small synthetic frames and the committed label files, so they run offline.
"""
import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data
from data import SEED, Split, bank, drop_test_duplicates, load_train, normalize, split
from labels import label_texts, question

REPO = Path(__file__).resolve().parents[1]
GRID = [1, 2, 3, 5, 8, 16, 35]
DATASETS = ["banking77", "ag_news"]
BANK_SIZES = {
    "banking77": [77, 154, 231, 385, 616, 1232, 2695],
    "ag_news": [4, 8, 12, 20, 32, 64, 140],
}


def synthetic(n_classes=3, per_class=60, dupes=()):
    """A frame shaped like load_train output, with unique texts plus the given duplicated texts."""
    rows = [(f"c{c}", f"message number {i} of class {c}") for c in range(n_classes) for i in range(per_class)]
    rows += list(dupes)
    df = pd.DataFrame(rows, columns=["label", "text"])
    df.insert(0, "row_id", [f"toy:train:{i}" for i in range(len(df))])
    df["norm"] = df.text.map(normalize)
    return df[["row_id", "text", "label", "norm"]]


# --------------------------------------------------------------------------- offline unit tests


def test_normalize_is_nfkc_lowercase_collapsed_stripped():
    assert normalize("  Hello \t\n WORLD  ") == "hello world"
    assert normalize("ＡＢＣ　１２３") == "abc 123"  # full-width letters, digits and ideographic space
    assert normalize("ﬁne") == "fine"  # ligature folds under NFKC
    assert normalize("") == ""
    assert normalize("a b") == "a b"  # no-break space is whitespace
    s = "Café  Menu"
    assert normalize(s) == unicodedata.normalize("NFKC", s).lower().replace("  ", " ")


def test_split_hash_order_is_sha256_seed_colon_norm():
    assert SEED == 20261105
    assert data.split_key("hello world") == hashlib.sha256(b"20261105:hello world").hexdigest()
    assert data.bank_key(3, "hello world") == hashlib.sha256(b"3:s-bank:hello world").hexdigest()


def test_split_takes_first_rows_per_class_in_hash_order_and_removes_repeats():
    df = synthetic(dupes=[("c0", "Message number 1 of class 0"), ("c1", "message  number 2 of class 0")])
    s = data._split(df, lambda n_c: 10)
    assert s.val.label.value_counts().to_dict() == {"c0": 10, "c1": 10, "c2": 10}
    assert set(s.val.row_id).isdisjoint(s.pool.row_id)
    assert set(s.val.norm).isdisjoint(s.pool.norm)  # shared text leaves the pool
    assert s.pool.norm.is_unique
    for c in s.val.label.unique():
        want = df[df.label == c].assign(k=lambda d: d.norm.map(data.split_key)).sort_values("k").head(10)
        assert set(s.val[s.val.label == c].row_id) == set(want.row_id)


def test_split_keeps_first_in_hash_order_of_repeated_pool_text():
    df = synthetic(n_classes=1, per_class=50, dupes=[("c0", "Message number 7 of class 0")] * 2)
    s = data._split(df, lambda n_c: 0)
    kept = s.pool[s.pool.norm == "message number 7 of class 0"]
    assert len(kept) == 1
    firsts = df[df.norm == "message number 7 of class 0"].sort_values("row_id").row_id
    assert kept.row_id.iloc[0] == firsts.iloc[0]  # equal keys: lowest original row first


def test_split_is_deterministic_and_independent_of_input_order():
    df = synthetic()
    a = data._split(df, lambda n_c: 5)
    b = data._split(df.sample(frac=1, random_state=1), lambda n_c: 5)
    assert list(a.val.row_id) == list(b.val.row_id) and list(a.pool.row_id) == list(b.pool.row_id)


def test_bank_is_nested_per_class_and_excludes_validation():
    df = synthetic(per_class=80)
    sp = data._split(df, lambda n_c: 10)
    for draw in range(3):
        prev = set()
        for s in [1, 2, 3, 5, 8, 16, 35]:
            b = bank(sp.pool, draw, s)
            assert (b.groupby("label").size() == s).all() and len(b) == 3 * s
            assert set(b.row_id) >= prev and set(b.row_id).isdisjoint(sp.val.row_id)
            prev = set(b.row_id)
    assert set(bank(sp.pool, 0, 5).row_id) != set(bank(sp.pool, 1, 5).row_id)


def test_bank_order_is_sha256_draw_colon_bank_colon_norm():
    sp = data._split(synthetic(per_class=80), lambda n_c: 10)
    b = bank(sp.pool, 4, 3)
    for c, g in b.groupby("label"):
        want = sp.pool[sp.pool.label == c].assign(k=lambda d: d.norm.map(lambda n: data.bank_key(4, n)))
        assert set(g.row_id) == set(want.sort_values("k").head(3).row_id)


def test_bank_uses_what_a_small_class_has():
    df = synthetic(per_class=60)
    pool = data._split(df, lambda n_c: 0).pool
    pool = pool[~((pool.label == "c2") & (pool.row_id.map(lambda r: int(r.split(":")[-1])) % 60 >= 4))]
    b = bank(pool, 0, 35)
    assert b.groupby("label").size().to_dict() == {"c0": 35, "c1": 35, "c2": len(pool[pool.label == "c2"])}


def test_bank_accepts_numpy_integers_for_s():
    sp = data._split(synthetic(per_class=80), lambda n_c: 10)
    want = bank(sp.pool, 2, 8)
    for s in (np.int64(8), np.int32(8), np.uint8(8)):
        assert list(bank(sp.pool, 2, s).row_id) == list(want.row_id)
    for bad in (True, np.bool_(True), 2.5, np.float64(8.0), 0, np.int64(0), "7"):
        with pytest.raises(ValueError):
            bank(sp.pool, 2, bad)


def test_all_adds_validation_rows_back_on_toy_data():
    sp = data._split(synthetic(per_class=80), lambda n_c: 10)
    b = bank(sp.pool, 0, "all", val=sp.val)
    assert len(b) == len(sp.pool) + len(sp.val)
    assert set(sp.val.row_id) <= set(b.row_id) and set(sp.pool.row_id) <= set(b.row_id)
    with pytest.raises(ValueError):
        bank(sp.pool, 0, "all")  # needs the validation rows
    assert set(bank(sp.pool, 0, 35, val=sp.val).row_id).isdisjoint(sp.val.row_id)  # val only returns for "all"


def test_drop_test_duplicates_moves_to_next_row_and_counts():
    sp = data._split(synthetic(per_class=80), lambda n_c: 10)
    first = bank(sp.pool, 0, 1)
    gone = first.norm.iloc[0]
    pool2, n = drop_test_duplicates(sp.pool, {gone, "no such text"})
    assert n == 1 and len(pool2) == len(sp.pool) - 1 and gone not in set(pool2.norm)
    c = first.label.iloc[0]
    b2 = bank(pool2, 0, 1)
    assert b2[b2.label == c].row_id.iloc[0] != first.row_id.iloc[0]  # next in order
    assert drop_test_duplicates(sp.pool, set())[1] == 0


def test_label_files_offline_shape():
    b77, ag = label_texts("banking77", bare=False), label_texts("ag_news", bare=False)
    assert len(b77) == 77 and len(ag) == 4 and all(v.strip() for v in [*b77.values(), *ag.values()])
    assert list(label_texts("ag_news", bare=True).items()) == [(k, k) for k in ["World", "Sports", "Business", "Sci/Tech"]]
    assert list(label_texts("banking77", bare=True)) == list(b77) and all(k == v for k, v in label_texts("banking77", bare=True).items())
    assert list(ag) == ["World", "Sports", "Business", "Sci/Tech"]
    assert question("ag_news") == "What is this news article mainly about?"
    with pytest.raises(ValueError):
        label_texts("imdb", bare=True)


def test_third_party_notice_credits_both_repos():
    notice = (REPO / "third_party" / "NOTICE.md").read_text()
    for needle in [
        "Madheshvivekanandan/jev-vs-cosine", "911e91b1093622c605c10067b960c0aff660fc1d", "banking77_intents.v1.json",
        "zhuyansen/jev-zeroshot-vs-bert", "edbf0713583644bd3f47299fd6b104a8b2073219", "agnews.yaml", "MIT License",
        "Madhesh Vivekanandan", "Jason Zhu",
    ]:
        assert needle in notice, needle


# --------------------------------------------------------------------------- data tests (train files only)


@pytest.fixture(scope="module")
def splits():
    return {d: split(load_train(d), d) for d in DATASETS}


@pytest.mark.data
def test_load_train_shape_and_stable_row_ids():
    df = load_train("banking77")
    assert list(df.columns) == ["row_id", "text", "label", "norm"] and len(df) == 10003
    assert df.row_id.is_unique and df.row_id.iloc[0] == "banking77:train:0" and df.row_id.iloc[-1] == "banking77:train:10002"
    assert (df.norm == df.text.map(normalize)).all()
    ag = load_train("ag_news")
    assert len(ag) == 120000 and ag.row_id.iloc[5] == "ag_news:train:5"
    assert set(ag.label) == {"World", "Sports", "Business", "Sci/Tech"}
    assert df.label.nunique() == 77
    assert list(load_train("banking77").row_id) == list(df.row_id)  # same on every run


@pytest.mark.data
def test_banking77_split(splits):
    s = splits["banking77"]
    assert isinstance(s, Split)
    assert len(s.val) == 756 and s.pool.norm.is_unique and len(s.pool) == 9243
    assert s.pool.label.value_counts().min() >= 35
    vc = s.val.label.value_counts()
    assert vc.get("virtual_card_not_working") == 6 and "contactless_not_working" not in vc
    assert (vc.drop("virtual_card_not_working") == 10).all() and len(vc) == 76
    assert set(s.val.norm).isdisjoint(s.pool.norm) and set(s.val.row_id).isdisjoint(s.pool.row_id)


@pytest.mark.data
def test_ag_news_split(splits):
    s = splits["ag_news"]
    assert len(s.val) == 800 and len(s.pool) == 119052
    assert (s.val.label.value_counts() == 200).all()
    assert s.pool.norm.is_unique and set(s.val.norm).isdisjoint(s.pool.norm)


@pytest.mark.data
@pytest.mark.parametrize("dataset", DATASETS)
def test_bank_sizes_and_nesting(splits, dataset):
    # banking77 draw 0: 77, 154, 231, 385, 616, 1232, 2695. ag_news: 4, 8, 12, 20, 32, 64, 140.
    # For every draw and s, the bank at s is a subset of the bank at the next s, and no val row is in it.
    s = splits[dataset]
    val_ids = set(s.val.row_id)
    val_norms = set(s.val.norm)
    for draw in range(10):
        banks = [bank(s.pool, draw, g, val=s.val) for g in GRID]
        if draw == 0:
            assert [len(b) for b in banks] == BANK_SIZES[dataset]
        for b, g in zip(banks, GRID):
            assert len(b) == g * s.pool.label.nunique()
            # VALIDATION guard: no validation row id appears in a bank at s of 35 or less.
            assert val_ids.isdisjoint(b.row_id) and val_norms.isdisjoint(b.norm)
        for small, big in zip(banks, banks[1:]):
            assert set(small.row_id) <= set(big.row_id)


@pytest.mark.data
@pytest.mark.parametrize("dataset", DATASETS)
def test_all_adds_validation_rows_back(splits, dataset):
    s = splits[dataset]
    b = bank(s.pool, 0, "all", val=s.val)
    assert len(b) == len(s.pool) + len(s.val)
    assert set(s.val.row_id) <= set(b.row_id)
    if dataset == "banking77":
        assert abs(len(b) / 77 - 130.0) < 0.2  # DESIGN: 129.9 per class on average
    else:
        # DESIGN says 30,000 per class: that is the nominal count. 148 repeated texts leave a few fewer.
        assert len(b) == 119852 and b.label.value_counts().between(29900, 30000).all()


@pytest.mark.data
def test_splits_are_reproducible():
    a = split(load_train("banking77"), "banking77")
    b = split(load_train("banking77"), "banking77")
    assert list(a.val.row_id) == list(b.val.row_id) and list(a.pool.row_id) == list(b.pool.row_id)
    assert list(bank(a.pool, 2, 8).row_id) == list(bank(b.pool, 2, 8).row_id)


@pytest.mark.data
def test_label_files_match_dataset_labels_for_both_datasets_and_bare_names():
    for dataset in DATASETS:
        names = data.label_names(dataset)
        assert len(names) == {"banking77": 77, "ag_news": 4}[dataset]
        assert set(load_train(dataset).label) <= set(names)
        for bare in (False, True):
            assert set(label_texts(dataset, bare=bare)) == set(names)
            assert len(label_texts(dataset, bare=bare)) == len(names)
        assert set(label_texts(dataset, bare=True).values()) == set(names)  # bare text is the official name
        assert all(label_texts(dataset, bare=False)[n] != n for n in names)  # descriptions differ from names


@pytest.mark.data
def test_known_rows_map_to_label_names():
    """Pins ids to names, so an off-by-one in label mapping shows up as a failure here."""
    banking = load_train("banking77").set_index("row_id")
    assert banking.loc["banking77:train:0", "label"] == "card_arrival"
    assert data.label_names("banking77")[0] == "activate_my_card"
    assert load_train("ag_news").set_index("row_id").loc["ag_news:train:0", "label"] == "Business"
