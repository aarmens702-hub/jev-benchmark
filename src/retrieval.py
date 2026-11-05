"""BM25 retrieval of labelled examples for the Jev + examples arm.

Written fresh from the recipe in DESIGN.md: BM25Okapi k1=1.5, b=0.75 on words and adjacent word
pairs; retrieve in score order, skip classes already at 4, ignore zero scores, up to 24, ties by
hash; log the count per row. Credit for the method: simonmesmith/jev-banking77-experiment. That
repo has no license, so its method is cited and none of its code was read or copied.

A bank row that equals the query would hand Jev the answer, so that raises.
"""
import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

logger = logging.getLogger("retrieval")

K1 = 1.5
B = 0.75
MAX_PER_CLASS = 4
MAX_TOTAL = 24
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Example:
    text: str
    label: str


class QueryInBank(ValueError):
    """The query's own text is in the bank. The message holds no text."""


def normalize_text(text: str) -> str:
    """VALIDATION.md's rule: NFKC, lowercase, whitespace collapsed, stripped."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def tokenize(text: str) -> list[str]:
    words = _WORD.findall(text.lower())
    return words + [f"{left} {right}" for left, right in zip(words, words[1:])]


def tie_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def retrieve(query: str, bank, *, row_id=None) -> list[Example]:
    bank = list(bank)
    normalized = normalize_text(query)
    if any(normalize_text(example.text) == normalized for example in bank):
        raise QueryInBank(f"the query on row {row_id} matches a bank row exactly")
    terms = tokenize(query)
    chosen: list[Example] = []
    if bank and terms:
        scores = BM25Okapi([tokenize(example.text) for example in bank], k1=K1, b=B).get_scores(terms)
        # The text settles exact ties, then the label, so the result never depends on bank order.
        ranked = sorted(zip(scores, bank), key=lambda pair: (-pair[0], tie_hash(pair[1].text), pair[1].label))
        per_class: dict[str, int] = {}
        for score, example in ranked:
            if score <= 0 or len(chosen) == MAX_TOTAL:
                break
            if per_class.get(example.label, 0) < MAX_PER_CLASS:
                per_class[example.label] = per_class.get(example.label, 0) + 1
                chosen.append(example)
    logger.info("retrieved row=%s count=%d", row_id, len(chosen))
    return chosen
