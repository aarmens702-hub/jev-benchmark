"""The frozen decision rules (DESIGN.md, "LR-ahead sequence", "Jev-ahead sequence", "Outcome rules").

Banking77 alone sets the verdict. `additions` never changes it. Routing claims, and the correction
they can trigger, are built with the routing test.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from metrics import GRID, N_DRAWS, DsResult, arm_accuracy, bootstrap_contrasts

MARGIN = 2

LR_SEQUENCE = ("all", 35, 16, 8, 5, 3, 2, 1)
JEV_SEQUENCE = (1, 2, 3, 5, 8, 16, 35, "all")
NOT_A_CLAIM = "not a claim"

_ROW_5_CLAIM = "No clear winner when labels are scarce."
_EQUIVALENT = "equivalent within 2 points"

# Copied from the table in DESIGN.md. tests/test_verdict.py checks each against the file.
VERDICT_ROWS = {
    1: "A light classifier with s* labels per class beats zero-label Jev at 80% coverage on this 77-way task.",
    2: "Jev leads through s_J labels per class and LR leads from s*. Jev's value on this task is limited to cold starts.",
    3: "LR leads clearly from s*. Below that, no clear difference.",
    4: "Jev leads when labels are scarce, through s_J labels per class.",
    5: f'{_ROW_5_CLAIM} If [L_s, U_s] is inside [-2, +2] at every s, say "{_EQUIVALENT}".',
}

CLAIM_LEVEL = 98.33  # Bonferroni across 3, tails of 0.835% each
CLAIM_ARMS = ("NLI", "embedding similarity", "small LLM")
MATCH_LIMIT = -2  # the arm "matches Jev" if the lower limit is above this
EXAMPLES_LEVEL = 95.0
EXAMPLES_TEXT = "Given the same 8 labels per class, Jev using them is at least as good as LR using them (one draw of labels)"
EXAMPLES_LIMIT = -2
AG_DIFFERENT_ROW = "The Banking77 result does not carry to this task."
AG_CEILING = "ceiling-limited: a 2-point lead is hard or impossible."
AG_CEILING_AT = 96
NOT_RUN = "not run"


def _require_full(ds: DsResult) -> None:
    if ds.partial:
        raise ValueError(f"a timing run of {ds.n_replicates} of {ds.b} replicates can't decide anything")


def _lr_passes(ds: DsResult, s) -> bool:
    return ds.lower[s] > 0 and ds.d[s] >= MARGIN


def _jev_passes(ds: DsResult, s) -> bool:
    return ds.upper[s] < 0 and ds.d[s] <= -MARGIN


def _last_pass_before_first_failure(ds: DsResult, sequence: tuple, passes) -> object:
    _require_full(ds)
    last = "none"
    for s in sequence:
        if not passes(ds, s):
            break
        last = s
    return last


def lr_ahead(ds: DsResult):
    """s*: the last s that passed before the first failure, walking down from s=all, or "none"."""
    return _last_pass_before_first_failure(ds, LR_SEQUENCE, _lr_passes)


def jev_ahead(ds: DsResult):
    """s_J: the last s that passed before the first failure, walking up from s=1, or "none"."""
    return _last_pass_before_first_failure(ds, JEV_SEQUENCE, _jev_passes)


def isolated_lr_passes(ds: DsResult) -> list:
    """Each s below s* that passed in isolation. Report these labelled NOT_A_CLAIM."""
    s_star = lr_ahead(ds)
    start = 0 if s_star == "none" else LR_SEQUENCE.index(s_star) + 1
    return [s for s in LR_SEQUENCE[start:] if _lr_passes(ds, s)]


def equivalent_within_2(ds: DsResult) -> bool:
    """True if [L_s, U_s] is inside [-2, +2] at every s."""
    _require_full(ds)
    return all(ds.lower[s] >= -MARGIN and ds.upper[s] <= MARGIN for s in GRID)


def _check_s(s) -> None:
    if s not in (*GRID, "none"):
        raise ValueError(f"not a grid value or 'none': {s!r}")


def verdict_row(s_star, s_j) -> int:
    """The table row that applies, 1 to 5. Exactly one row applies to every pair."""
    _check_s(s_star)
    _check_s(s_j)
    if s_star in (1, 2, 3):
        return 1
    if s_star in (5, 8, 16):
        return 3 if s_j == "none" else 2
    return 4 if s_j in (3, 5, 8, 16, 35, "all") else 5


def verdict(s_star, s_j, ds: DsResult | None = None) -> str:
    """The row's sentence with s* and s_J filled in.

    Row 5 holds an instruction as well as the verdict. Without `ds` it comes back as written. With
    `ds` the instruction is carried out: the claim, followed by the quoted phrase if every interval
    is inside [-2, +2].
    """
    row = verdict_row(s_star, s_j)
    if row == 5 and ds is not None:
        return f"{_ROW_5_CLAIM} {_EQUIVALENT}" if equivalent_within_2(ds) else _ROW_5_CLAIM
    return VERDICT_ROWS[row].replace("s*", str(s_star)).replace("s_J", str(s_j))


def _his_claim(arms: Mapping, jev, classes, b, seed, coverage) -> dict:
    if set(arms) != set(CLAIM_ARMS):
        raise ValueError(f"his_claim_arms must name exactly {CLAIM_ARMS}, using None for an arm not run")
    run = {name: arm for name, arm in arms.items() if arm is not None}
    contrasts = bootstrap_contrasts(run, jev, classes, level=CLAIM_LEVEL, b=b, seed=seed, coverage=coverage)
    out = {}
    for name in CLAIM_ARMS:
        if name not in contrasts:
            out[name] = {"status": NOT_RUN}
            continue
        c = contrasts[name]
        out[name] = {
            "status": "run",
            "C": c.estimate,
            "lower": c.lower,
            "upper": c.upper,
            "level": CLAIM_LEVEL,
            "matches_jev": c.lower > MATCH_LIMIT,
        }
    return out


def _jev_examples(jev_examples, lr_s8, classes, b, seed, coverage) -> dict:
    if jev_examples is None:
        return {"status": NOT_RUN}
    e = bootstrap_contrasts(
        {"E": jev_examples}, lr_s8[0], classes, level=EXAMPLES_LEVEL, b=b, seed=seed, coverage=coverage
    )["E"]
    lr_accs = [arm_accuracy(draw, coverage) for draw in lr_s8]
    return {
        "status": "run",
        "E": e.estimate,
        "lower": e.lower,
        "upper": e.upper,
        "level": EXAMPLES_LEVEL,
        "text": EXAMPLES_TEXT if e.lower > EXAMPLES_LIMIT else None,
        "lr_range": (min(lr_accs), max(lr_accs)),
    }


def _ag_news(ag: DsResult | None, banking_s_star, banking_s_j) -> dict:
    if ag is None:
        return {"status": NOT_RUN, "texts": [NOT_RUN]}
    s_star, s_j = lr_ahead(ag), jev_ahead(ag)
    differs = verdict_row(s_star, s_j) != verdict_row(banking_s_star, banking_s_j)
    texts = []
    if differs:
        texts.append(AG_DIFFERENT_ROW)
    if ag.jev_acc >= AG_CEILING_AT:
        texts.append(AG_CEILING)
    return {
        "status": "run",
        "s_star": s_star,
        "s_j": s_j,
        "row_differs": differs,
        "jev_acc": ag.jev_acc,
        "texts": texts,
    }


def additions(
    *,
    jev: np.ndarray,
    classes: np.ndarray,
    lr_s8: np.ndarray,
    his_claim_arms: Mapping,
    jev_examples: np.ndarray | None,
    banking_s_star,
    banking_s_j,
    ag_news: DsResult | None,
    b: int = 10_000,
    seed: int = 2026,
    coverage: float = 0.8,
) -> dict:
    """The additions in DESIGN.md's outcome rules, other than routing. These never change the verdict.

    All arrays are Banking77. `jev` and each arm in `his_claim_arms` are label-description runs.
    `his_claim_arms` names NLI, embedding similarity (temperature 0.05) and small LLM, with None for
    an arm not run. `lr_s8` is LR at s=8, (draws, rows, 2 or 3); Jev + examples is compared with its
    draw 0. `ag_news` is the AG News DsResult, or None if not run. Its "does not carry" text prints
    when its verdict table row differs from Banking77's, even if only s_J differs.
    """
    verdict_row(banking_s_star, banking_s_j)  # refuses values off the grid
    if len(lr_s8) != N_DRAWS:
        raise ValueError(f"lr_s8 must hold all {N_DRAWS} draws, got {len(lr_s8)}")
    claim = _his_claim(his_claim_arms, jev, classes, b, seed, coverage)
    return {
        "his_claim": claim,
        "any_arm_matches_jev": any(arm.get("matches_jev", False) for arm in claim.values()),
        "jev_examples": _jev_examples(jev_examples, lr_s8, classes, b, seed, coverage),
        "ag_news": _ag_news(ag_news, banking_s_star, banking_s_j),
        "prediction_on_record": banking_s_star in (1, 2, 3, 5, 8),  # true iff s* is 8 or lower
    }
