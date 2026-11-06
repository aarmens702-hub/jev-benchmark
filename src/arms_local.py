"""The free arms that run on the Mac: embedding similarity, embedding LR (plain and CV), TF-IDF LR, zero-shot NLI.

Written for this project; nothing is copied from the reuse repositories. Every arm takes the texts to
predict and, where it learns, a label bank (the only source of labels). None of them reads a test split.
Encoders and NLI scorers are injectable, so the logic is testable with small fakes and no downloads.
"""
import logging
import os
import warnings
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple, Optional, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from guards import check_disk, free_disk_gb
from labels import label_texts

log = logging.getLogger("arms_local")

# --- pinned models (DESIGN.md, Arms and Time) ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
NLI_MODELS = {
    "large": ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c", "b2730f16019076bb0009481121efbe4705e0e378"),
    "base": ("MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c", "bddf8c5411c34ac3565e16e04384fd68b2618dda"),
}

# --- fixed settings ---
EMBED_BATCH_SIZE = 64
EMBED_TEMPERATURE = 0.05
SENSITIVITY_TEMPERATURES = (0.01, 0.2)
LR_C = 100
LR_MAX_ITER = 5000
LR_SEED = 0
CV_SEED = 0
CV_GRID = (1, 10, 100, 1000)
CV_MAX_FOLDS = 5
NLI_BATCH_SIZE = 16  # pairs per forward pass; the frozen batch size, fixed for the whole run (accepted 2026-11-04)
NLI_MAX_LEN = 256
BARE_TEMPLATE = "This text is about {}."


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- real model loaders


def load_encoder(read_free_disk_gb: Callable[..., float] = free_disk_gb) -> Callable[[Sequence[str]], np.ndarray]:
    """bge-small at its pinned revision. Disk is checked before anything is downloaded."""
    check_disk(read_free_disk_gb)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, revision=EMBED_REVISION)

    def encode(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            model.encode(
                list(texts),
                batch_size=EMBED_BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    return encode


def memoize_encoder(
    encode: Callable[[Sequence[str]], np.ndarray],
    cache_dir: Optional[Path] = None,
    guard: Optional[Callable[[], None]] = None,
    chunk: int = 4096,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Embed each distinct text once, since banks across draws and sizes overlap heavily.

    With `cache_dir`, every chunk of new embeddings is saved as its own shard file, so a stopped or
    crashed run keeps them and a rerun loads them instead of re-embedding. `guard` is called before each
    chunk, so a long embedding job can stop itself on --max-hours or a resource limit.
    """
    cache: dict[str, np.ndarray] = {}
    shards = 0
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        for f in sorted(cache_dir.glob("shard-*.npz")):
            with np.load(f, allow_pickle=False) as z:
                cache.update(zip(z["texts"].tolist(), z["vectors"]))
            shards += 1

    def cached(texts: Sequence[str]) -> np.ndarray:
        nonlocal shards
        missing = list(dict.fromkeys(t for t in texts if t not in cache))
        for i in range(0, len(missing), chunk):
            if guard is not None:
                guard()
            part = missing[i : i + chunk]
            vecs = np.asarray(encode(part), dtype=np.float32)
            cache.update(zip(part, vecs))
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                final = cache_dir / f"shard-{shards:06d}.npz"
                tmp = final.with_name(final.name + ".tmp")
                with open(tmp, "wb") as f:
                    np.savez(f, texts=np.array(part, dtype=str), vectors=vecs)
                os.replace(tmp, final)
                shards += 1
        return np.stack([cache[t] for t in texts]) if len(texts) else np.zeros((0, 0), dtype=np.float32)

    return cached


def entailment_id(label2id: dict[str, int]) -> int:
    for name, i in label2id.items():
        if name.lower() == "entailment":
            return int(i)
    raise ValueError(f"no 'entailment' label in {sorted(label2id)}")


def make_hf_scorer(tokenizer, model, device: str = "cpu") -> Callable[[Sequence[tuple[str, str]]], list[float]]:
    """Scorer over (premise, hypothesis) pairs returning the entailment logit of each pair."""
    ent = entailment_id(model.config.label2id)

    def score(pairs: Sequence[tuple[str, str]]) -> list[float]:
        import torch

        premises, hypotheses = [p for p, _ in pairs], [h for _, h in pairs]
        inputs = tokenizer(
            premises, hypotheses, truncation="only_first", max_length=NLI_MAX_LEN, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        return logits[:, ent].float().cpu().tolist()

    return score


def load_nli_scorer(key: str = "large", read_free_disk_gb: Callable[..., float] = free_disk_gb):
    """The zero-shot NLI model at its pinned revision ("large", or "base" if the 12-hour rule fires)."""
    check_disk(read_free_disk_gb)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name, revision = NLI_MODELS[key]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision)
    model.to(device)
    model.eval()
    return make_hf_scorer(tokenizer, model, device)


# --------------------------------------------------------------------------- embedding similarity


def embed_cosine(texts: Sequence[str], labels: Sequence[str], encode: Optional[Callable] = None) -> np.ndarray:
    """Cosine similarity of each text to each label text, shape (len(texts), len(labels))."""
    encode = encode or load_encoder()
    t, l = _unit(encode(list(texts))), _unit(encode(list(labels)))
    return t @ l.T


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def embed_sim(
    texts: Sequence[str], labels: Sequence[str], temperature: float = EMBED_TEMPERATURE, encode: Optional[Callable] = None
) -> np.ndarray:
    """Softmax of cosine similarity at `temperature`. Needs no labelled rows, only the label texts."""
    return softmax(embed_cosine(texts, labels, encode) / temperature)


def sensitivity_views(cos: np.ndarray) -> dict[str, np.ndarray]:
    """Primary scores and the sensitivity rows, all from the same cosine matrix.

    Keys are arm-name suffixes: "" (T=0.05), "_t0.01", "_t0.2", and "_raw" (raw cosine, ranked by top cosine).
    """
    views = {"": softmax(cos / EMBED_TEMPERATURE)}
    for t in SENSITIVITY_TEMPERATURES:
        views[f"_t{t:g}"] = softmax(cos / t)
    views["_raw"] = np.asarray(cos)
    return views


# --------------------------------------------------------------------------- logistic regression


def make_lr(c: float = LR_C, max_iter: int = LR_MAX_ITER) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=max_iter, solver="lbfgs", random_state=LR_SEED)


def fit_lr(X, y, c: float = LR_C, max_iter: int = LR_MAX_ITER, log_non_convergence: bool = True):
    """Fit one LR. Returns (model, converged). A non-converged fit is logged, never hidden."""
    model = make_lr(c, max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if not converged and log_non_convergence:
        log.warning("LogisticRegression (C=%g) did not converge within max_iter=%d", c, max_iter)
    return model, converged


def _full_columns(model: LogisticRegression, X, classes: Sequence[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    unknown = set(model.classes_) - set(index)
    if unknown:
        raise ValueError(f"bank has classes outside `classes`: {sorted(unknown)[:5]}")
    out = np.zeros((X.shape[0], len(classes)))
    out[:, [index[c] for c in model.classes_]] = model.predict_proba(X)
    return out


def lr_fitted(train, c: float = LR_C, classes: Optional[Sequence[str]] = None, max_iter: int = LR_MAX_ITER) -> Callable:
    """Fit embedding LR on the bank once; returns predict(X) -> probabilities in `classes` order."""
    X, y = train
    classes = list(classes) if classes is not None else sorted(set(y))
    model, _ = fit_lr(X, y, c, max_iter)
    return lambda test: _full_columns(model, test, classes)


def lr(train, test, c: float = LR_C, classes: Optional[Sequence[str]] = None, max_iter: int = LR_MAX_ITER) -> np.ndarray:
    """Embedding LR. train = (X, y) from the bank, test = X for rows to predict (no labels needed).

    Returns probabilities with columns in `classes` order (default: sorted bank classes).
    """
    return lr_fitted(train, c, classes, max_iter)(test)


class CVPlan(NamedTuple):
    folds: int  # 0 means no selection (C fixed)
    grid: tuple


def cv_plan(s) -> CVPlan:
    """C selection per bank size: C=100 with no CV at s=1; otherwise stratified k-fold inside the bank
    with k = min(s, 5), over C in {1, 10, 100, 1000}. (Leave-one-out was dropped as too slow.)"""
    if isinstance(s, np.integer):
        s = int(s)
    if s == "all":
        return CVPlan(CV_MAX_FOLDS, CV_GRID)
    if isinstance(s, int) and not isinstance(s, bool) and s >= 1:
        return CVPlan(0, (LR_C,)) if s == 1 else CVPlan(min(s, CV_MAX_FOLDS), CV_GRID)
    raise ValueError(f"s={s!r} must be a positive integer or 'all'")


def cv_accuracies(X, y, folds: int, grid: Iterable[float]) -> dict[float, float]:
    """Stratified k-fold accuracy inside the bank for each C. Uses the bank's own labels only."""
    y, grid = np.asarray(y), tuple(grid)
    if folds < 2:
        raise ValueError(f"need at least 2 folds, got {folds}")
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=CV_SEED).split(X, y))
    out, bad = {}, 0
    for c in grid:
        correct = 0
        for tr, te in splits:
            model, ok = fit_lr(X[tr], y[tr], c, log_non_convergence=False)
            bad += not ok
            correct += int((model.predict(X[te]) == y[te]).sum())
        out[c] = correct / len(y)
    if bad:
        log.warning("%d of %d CV fits did not converge within max_iter=%d", bad, len(splits) * len(grid), LR_MAX_ITER)
    return out


def choose_c(acc: dict[float, float]) -> float:
    """Best CV accuracy. Ties go to C=100 (the primary arm's value), then to the smaller C."""
    return max(acc, key=lambda c: (acc[c], c == LR_C, -c))


def lr_cv_fitted(train, s, classes: Optional[Sequence[str]] = None) -> Callable:
    """Choose C inside the bank, fit once; returns predict(X)."""
    X, y = train
    plan = cv_plan(s)
    c = plan.grid[0] if plan.folds == 0 else choose_c(cv_accuracies(X, y, plan.folds, plan.grid))
    return lr_fitted(train, c=c, classes=classes)


def lr_cv(train, test, s, classes: Optional[Sequence[str]] = None) -> np.ndarray:
    return lr_cv_fitted(train, s, classes)(test)


def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words=None, min_df=1)


def tfidf_lr_fitted(train, c: float = LR_C, classes: Optional[Sequence[str]] = None) -> Callable:
    """Fit TF-IDF + LR on the bank once; returns predict(texts)."""
    texts, y = train
    classes = list(classes) if classes is not None else sorted(set(y))
    vec = make_tfidf()
    model, _ = fit_lr(vec.fit_transform(texts), y, c)
    return lambda test: _full_columns(model, vec.transform(list(test)), classes)


def tfidf_lr(train, test: Sequence[str], c: float = LR_C, classes: Optional[Sequence[str]] = None) -> np.ndarray:
    """TF-IDF (word 1-2 grams) + LR. train = (texts, labels), test = texts to predict."""
    return tfidf_lr_fitted(train, c, classes)(test)


# --------------------------------------------------------------------------- zero-shot NLI


def nli_hypotheses(dataset: str, bare: bool) -> list[str]:
    """One hypothesis per class, in class order. Descriptions are used as written; bare names get a template."""
    texts = label_texts(dataset, bare=bare)
    return [BARE_TEMPLATE.format(name) for name in texts] if bare else list(texts.values())


def nli(
    texts: Sequence[str],
    hypotheses: Sequence[str],
    batch_size: int = NLI_BATCH_SIZE,
    scorer: Optional[Callable] = None,
) -> Iterator[np.ndarray]:
    """Yield, for each text in order, softmax over the entailment logits of its pairs (multi_label=False).

    Pairs stream through the scorer in fixed batches of `batch_size`, so a batch can span two texts.
    """
    scorer = scorer or load_nli_scorer()
    k = len(hypotheses)
    pairs = ((t, h) for t in texts for h in hypotheses)
    pending: list[float] = []
    while True:
        batch = list(islice(pairs, batch_size))
        if not batch:
            break
        pending.extend(scorer(batch))
        while len(pending) >= k:
            yield softmax(np.array(pending[:k]))
            del pending[:k]
