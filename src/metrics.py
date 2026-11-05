"""acc@80, the paired bootstrap and the secondary scores (DESIGN.md, "Metric").

Arm arrays have shape (rows, 2) with columns (top probability, correct), or (rows, 3) with a third
column that flags failed rows. LR arrays add a leading draws axis. A failed row is wrong and ranks
above all other rows, whatever its other columns hold.

Accuracies come back as fractions from `acc_at`. Everything in a result object is in points.
The result objects don't show values in their repr, so printing one can't leak an accuracy or a
gap during validation (VALIDATION.md).
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np

GRID = (1, 2, 3, 5, 8, 16, 35, "all")
N_DRAWS = 10
NLL_FLOOR = 0.005

# Rounding gaps to 10 decimals of a point removes float noise, which is well under 1e-12 here.
# Without it, 0.57 - 0.55 comes out below 2 points and a true 0 can come out above 0, so an exact
# boundary would be decided by rounding error rather than by the rule. A real gap would have to sit
# within 1e-10 of a boundary without being on it for this to matter, which takes many tie groups
# combining, and the difference would be a tenth of a billionth of a point.
_POINT_DECIMALS = 10
# Replicates per batch. Peak memory is a few arrays of rows * _CHUNK integers, whatever B is.
_CHUNK = 128


@dataclass(frozen=True)
class _Ranked:
    """An arm's fixed ranking. Resampling never changes the relative order of rows, so one sort will do."""

    order: np.ndarray  # row numbers, best rank first
    starts: np.ndarray  # where each tie group begins in `order`
    correct: np.ndarray  # correct flags in rank order; failed rows are False


@dataclass(frozen=True, eq=False)
class DsResult:
    """D_s per grid point, in points. `replicates` has shape (len(GRID), n_replicates)."""

    d: Mapping = field(repr=False)
    lower: Mapping = field(repr=False)
    upper: Mapping = field(repr=False)
    b: int = 10_000
    n_replicates: int = 10_000
    coverage: float = 0.8
    jev_acc: float | None = field(default=None, repr=False)
    lr_acc: Mapping | None = field(default=None, repr=False)  # s -> per-draw acc, in points
    draw_range: Mapping | None = field(default=None, repr=False)  # s -> (min, max) per-draw gap
    draws_beating_jev: Mapping | None = field(default=None, repr=False)  # s -> count of the 10
    replicates: np.ndarray | None = field(default=None, repr=False)

    @property
    def partial(self) -> bool:
        """True for a timing run. A partial result can't feed a sequence or a verdict."""
        return self.n_replicates < self.b


@dataclass(frozen=True, eq=False)
class Contrast:
    """One arm minus a base arm, in points, on the bootstrap's shared rows."""

    estimate: float = field(repr=False)
    lower: float = field(repr=False)
    upper: float = field(repr=False)
    level: float
    replicates: np.ndarray = field(repr=False)


def _kept(n: int, coverage: float) -> float:
    """The size of the covered set. Exact whole numbers stay exact, so 0.7 * 3080 is 2156.

    When coverage * n isn't whole (Banking77 validation: 756 * 0.8 = 604.8) it stays fractional,
    and the boundary row counts in proportion, the same way a tie group does.
    """
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    kept = coverage * n
    nearest = round(kept)
    return float(nearest) if abs(kept - nearest) < 1e-9 else kept


def _rank(conf: np.ndarray, correct: np.ndarray, failed: np.ndarray) -> _Ranked:
    """Failed rows first, as one group, then the others by confidence, high to low."""
    if not len(conf) == len(correct) == len(failed):
        raise ValueError("conf, correct and failed must have the same length")
    answered = np.flatnonzero(~failed)
    if np.isnan(conf[answered]).any():
        raise ValueError("a row that did not fail has no confidence")
    by_conf = answered[np.argsort(-conf[answered], kind="stable")]
    order = np.concatenate([np.flatnonzero(failed), by_conf])
    n_failed = len(order) - len(by_conf)

    # Groups: the failed rows, the first answered row, then each change of confidence.
    changes = np.flatnonzero(conf[by_conf[1:]] != conf[by_conf[:-1]]) + 1 + n_failed
    starts = np.unique(np.concatenate([[0, n_failed], changes]).astype(np.intp))
    starts = starts[starts < len(order)]  # everything failed: there is no answered group

    in_order = correct[order] & ~failed[order]
    return _Ranked(order=order, starts=starts, correct=in_order)


def _weighted_acc(arm: _Ranked, weights: np.ndarray, kept: float) -> np.ndarray:
    """acc@80 per replicate. `weights` is (rows, batch): how many times each row was drawn.

    Works on tie groups. A group's weight is its total copies in the replicate, and copies of a row
    tie with each other, so this gives the same answer as sorting the resampled rows. The group
    that straddles the cut adds c*k/g correct, as DESIGN.md says.
    """
    in_order = weights[arm.order]
    group_rows = np.add.reduceat(in_order, arm.starts, axis=0)
    group_correct = np.add.reduceat(in_order * arm.correct[:, None], arm.starts, axis=0)
    rows_so_far = np.cumsum(group_rows, axis=0)
    correct_so_far = np.cumsum(group_correct, axis=0)

    cut = (rows_so_far >= kept).argmax(axis=0)  # the first group that reaches the cut
    batch = np.arange(weights.shape[1])
    g = group_rows[cut, batch]
    c = group_correct[cut, batch]
    k = kept - (rows_so_far[cut, batch] - g)
    correct_before = correct_so_far[cut, batch] - c
    return (correct_before + c * k / g) / kept


def acc_at(conf: np.ndarray, correct: np.ndarray, failed: np.ndarray, coverage: float = 0.8) -> float:
    """Accuracy over the top `coverage` share of rows, as a fraction. Failed rows rank first and are wrong."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    failed = np.asarray(failed, dtype=bool)
    arm = _rank(conf, correct, failed)
    unit = np.ones((len(conf), 1), dtype=np.int64)
    return float(_weighted_acc(arm, unit, _kept(len(conf), coverage))[0])


def _points(fraction):
    return np.round(100 * fraction, _POINT_DECIMALS)


def _columns(arm: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a (rows, 2 or 3) array into conf, correct and failed, after checking it."""
    arm = np.asarray(arm)
    if arm.ndim != 2 or arm.shape[0] != n or arm.shape[1] not in (2, 3):
        raise ValueError(f"an arm must have shape ({n}, 2) or ({n}, 3), got {arm.shape}")
    arm = arm.astype(float, copy=False)
    failed = np.zeros(n, dtype=bool)
    if arm.shape[1] == 3:
        if not np.isin(arm[:, 2], (0.0, 1.0)).all():
            raise ValueError("the failed column must hold only 0 and 1")
        failed = arm[:, 2] != 0
    # A failed row may hold anything in the other columns; its answer doesn't exist.
    if not np.isin(arm[~failed, 1], (0.0, 1.0)).all():
        raise ValueError("the correct column must hold only 0 and 1")
    return arm[:, 0], arm[:, 1] == 1, failed


def _rank_arm(arm: np.ndarray, n: int) -> _Ranked:
    return _rank(*_columns(arm, n))


def _plain_acc(arm: _Ranked, n: int, kept: float) -> float:
    return float(_weighted_acc(arm, np.ones((n, 1), dtype=np.int64), kept)[0])


def arm_accuracy(arm: np.ndarray, coverage: float = 0.8) -> float:
    """One arm's acc@coverage on the rows as given, in points."""
    n = len(arm)
    return float(_points(_plain_acc(_rank_arm(arm, n), n, _kept(n, coverage))))


def replicate_plan(
    classes: np.ndarray, b: int, seed: int = 2026, chunk_size: int = _CHUNK
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield batches of (rows, draw indices), shapes (batch, n) and (batch, N_DRAWS).

    Step (1) of the bootstrap: each row position is filled from the rows of its own gold class, with
    replacement. Step (2): the 10 draw indices, with replacement. Each replicate consumes the same
    n + 10 uniforms in order, so a replicate doesn't depend on chunk size, and running 100
    replicates gives the first 100 of the full run. bootstrap_ds and bootstrap_contrasts share this
    plan, which is what makes the rows the same for all arms.
    """
    classes = np.asarray(classes)
    n = len(classes)
    _, inverse, counts = np.unique(classes, return_inverse=True, return_counts=True)
    members = np.argsort(inverse, kind="stable")  # row numbers grouped by class
    first_member = np.concatenate([[0], np.cumsum(counts)[:-1]])
    position_start = first_member[inverse]
    position_size = counts[inverse]

    rng = np.random.default_rng(seed)
    for done in range(0, b, chunk_size):
        batch = min(chunk_size, b - done)
        uniforms = rng.random((batch, n + N_DRAWS))
        rows = members[position_start + (uniforms[:, :n] * position_size).astype(np.intp)]
        draws = (uniforms[:, n:] * N_DRAWS).astype(np.intp)
        yield rows, draws


def _row_weights(rows: np.ndarray) -> np.ndarray:
    """(batch, n) row numbers to (n, batch) counts."""
    batch, n = rows.shape
    return np.bincount((rows * batch + np.arange(batch)[:, None]).ravel(), minlength=n * batch).reshape(n, batch)


def _interval(replicates: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval along the last axis. Level 95 gives the 2.5 and 97.5 percentiles."""
    tail = round((100 - level) / 2, 10)
    lower, upper = np.percentile(replicates, [tail, round(100 - tail, 10)], axis=-1)
    return lower, upper


def _rank_lr(lr: Mapping, n: int) -> dict:
    """One _Ranked per draw per s. At s=all, one only, since all draws must be identical there."""
    if set(lr) != set(GRID):
        raise ValueError(f"lr must hold exactly the grid {GRID}, got {sorted(map(str, lr))}")
    ranked = {}
    for s in GRID:
        draws = np.asarray(lr[s])
        if draws.ndim != 3 or draws.shape[0] != N_DRAWS:
            raise ValueError(f"lr[{s}] must have shape ({N_DRAWS}, rows, 2 or 3), got {draws.shape}")
        if s == "all":
            if not all(np.array_equal(draws[0], draws[r], equal_nan=True) for r in range(1, N_DRAWS)):
                raise ValueError("at s=all, all draws must be identical")
            ranked[s] = [_rank_arm(draws[0], n)]
        else:
            ranked[s] = [_rank_arm(draws[r], n) for r in range(N_DRAWS)]
    return ranked


def bootstrap_ds(
    lr: Mapping,
    jev: np.ndarray,
    classes: np.ndarray,
    b: int = 10_000,
    seed: int = 2026,
    *,
    n_replicates: int | None = None,
    coverage: float = 0.8,
    chunk_size: int = _CHUNK,
) -> DsResult:
    """D_s and its percentile interval (DESIGN.md, "Bootstrap").

    lr maps each s to (draws, rows, 2 or 3); jev is (rows, 2 or 3). `n_replicates` runs only the
    first replicates of the plan, so a guard can time 100 and project. The result is then marked
    partial and the sequences refuse it. Memory stays flat in B: only the final gaps are kept.
    """
    classes = np.asarray(classes)
    n = len(classes)
    kept = _kept(n, coverage)
    runs = b if n_replicates is None else n_replicates
    if not 1 <= runs <= b:
        raise ValueError(f"n_replicates must be in 1..{b}, got {n_replicates}")

    jev_arm = _rank_arm(jev, n)
    lr_arms = _rank_lr(lr, n)

    jev_acc = _plain_acc(jev_arm, n, kept)
    gaps, lr_acc, draw_range, beating = {}, {}, {}, {}
    for s in GRID:
        accs = np.array([_plain_acc(arm, n, kept) for arm in lr_arms[s]])
        accs = np.repeat(accs, N_DRAWS) if len(accs) == 1 else accs
        draw_gaps = _points(accs - jev_acc)
        gaps[s] = float(_points(accs.mean() - jev_acc))
        lr_acc[s] = _points(accs)
        draw_range[s] = (float(draw_gaps.min()), float(draw_gaps.max()))
        beating[s] = int((draw_gaps > 0).sum())

    replicates = np.empty((len(GRID), runs))
    done = 0
    for rows, draws in replicate_plan(classes, runs, seed, chunk_size):
        weights = _row_weights(rows)
        jev_replicate = _weighted_acc(jev_arm, weights, kept)
        for j, s in enumerate(GRID):
            accs = np.stack([_weighted_acc(arm, weights, kept) for arm in lr_arms[s]])
            if len(accs) == 1:  # s=all: every draw index points at the same arm
                mean = accs[0]
            else:
                mean = np.take_along_axis(accs, draws.T, axis=0).mean(axis=0)
            replicates[j, done : done + len(rows)] = _points(mean - jev_replicate)
        done += len(rows)

    lower, upper = _interval(replicates, 95.0)
    return DsResult(
        d=gaps,
        lower={s: float(lower[j]) for j, s in enumerate(GRID)},
        upper={s: float(upper[j]) for j, s in enumerate(GRID)},
        b=b,
        n_replicates=runs,
        coverage=coverage,
        jev_acc=float(_points(jev_acc)),
        lr_acc=lr_acc,
        draw_range=draw_range,
        draws_beating_jev=beating,
        replicates=replicates,
    )


def bootstrap_contrasts(
    arms: Mapping,
    base: np.ndarray,
    classes: np.ndarray,
    level: float,
    b: int = 10_000,
    seed: int = 2026,
    *,
    coverage: float = 0.8,
    chunk_size: int = _CHUNK,
) -> dict[str, Contrast]:
    """Each arm minus `base`, at the given CI level, on the same row resamples as bootstrap_ds.

    No draw resampling: these arms and `base` are one array each. The plan's draw indices go unused.
    """
    classes = np.asarray(classes)
    n = len(classes)
    if not arms:
        return {}
    kept = _kept(n, coverage)
    base_arm = _rank_arm(base, n)
    ranked = {name: _rank_arm(arm, n) for name, arm in arms.items()}
    base_acc = _plain_acc(base_arm, n, kept)

    replicates = np.empty((len(ranked), b))
    done = 0
    for rows, _ in replicate_plan(classes, b, seed, chunk_size):
        weights = _row_weights(rows)
        base_replicate = _weighted_acc(base_arm, weights, kept)
        for j, arm in enumerate(ranked.values()):
            replicates[j, done : done + len(rows)] = _points(_weighted_acc(arm, weights, kept) - base_replicate)
        done += len(rows)

    lower, upper = _interval(replicates, level)
    return {
        name: Contrast(
            estimate=float(_points(_plain_acc(arm, n, kept) - base_acc)),
            lower=float(lower[j]),
            upper=float(upper[j]),
            level=level,
            replicates=replicates[j],
        )
        for j, (name, arm) in enumerate(ranked.items())
    }


def _failed_mask(failed: np.ndarray | None, n: int) -> np.ndarray:
    return np.zeros(n, dtype=bool) if failed is None else np.asarray(failed, dtype=bool)


def nll(gold_prob: np.ndarray, failed: np.ndarray | None = None) -> float:
    """Mean negative log probability of gold, clipped to [0.005, 1]. A failed row counts as probability 0."""
    gold_prob = np.asarray(gold_prob, dtype=float)
    failed = _failed_mask(failed, len(gold_prob))
    if np.isnan(gold_prob[~failed]).any():
        raise ValueError("a row that did not fail has no gold probability")
    probs = np.where(failed, 0.0, gold_prob)
    return float(-np.log(np.clip(probs, NLL_FLOOR, 1.0)).mean())


def brier(probs: np.ndarray, gold: np.ndarray, failed: np.ndarray | None = None) -> float:
    """Multiclass Brier: mean over rows of the summed squared error, range 0 to 2.

    `probs` is (rows, classes) and `gold` holds class indices. A failed row counts as all zeros, so
    it scores 1.
    """
    probs = np.asarray(probs, dtype=float)
    gold = np.asarray(gold, dtype=np.intp)
    failed = _failed_mask(failed, len(probs))
    if np.isnan(probs[~failed]).any():
        raise ValueError("a row that did not fail has no probabilities")
    probs = np.where(failed[:, None], 0.0, probs)
    gold_prob = probs[np.arange(len(probs)), gold]
    return float(((probs**2).sum(axis=1) - 2 * gold_prob + 1).mean())


def _ece_from_bins(bin_of: np.ndarray, conf: np.ndarray, hits: np.ndarray, bins: int) -> np.ndarray:
    """ECE for each row of `hits`, shape (sims, rows). Sum over bins of |correct - conf| / n."""
    total = np.zeros(hits.shape[0])
    for i in range(bins):
        members = bin_of == i
        if members.any():
            total += np.abs(hits[:, members].sum(axis=1) - conf[members].sum())
    return total / len(conf)


def ece_with_floor(
    conf: np.ndarray,
    correct: np.ndarray,
    failed: np.ndarray | None = None,
    bins: int = 15,
    n_sim: int = 1000,
    seed: int = 2026,
) -> dict:
    """ECE and the score a perfectly calibrated arm would still get from sampling noise.

    Equal-width bins, closed on the right; 0 goes in the first bin. The floor is the mean ECE over
    `n_sim` outcome sets drawn as Bernoulli(conf). A failed row has no stated confidence, so it
    leaves this score and is counted in `failed_excluded`.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    failed = _failed_mask(failed, len(conf))
    conf, correct = conf[~failed], correct[~failed]
    if len(conf) == 0 or np.isnan(conf).any():
        raise ValueError("ECE needs at least one row and a confidence on every row that did not fail")

    # Rounding stops a value on a bin edge, such as 0.3 * 10, from landing in the next bin up.
    bin_of = np.clip(np.ceil(np.round(conf * bins, 9)).astype(np.intp) - 1, 0, bins - 1)
    ece = _ece_from_bins(bin_of, conf, correct[None].astype(float), bins)[0]

    rng = np.random.default_rng(seed)
    total = 0.0
    for done in range(0, n_sim, 100):
        batch = min(100, n_sim - done)
        hits = rng.random((batch, len(conf))) < conf
        total += _ece_from_bins(bin_of, conf, hits, bins).sum()
    return {
        "ece": float(ece),
        "floor": float(total / n_sim),
        "n": len(conf),
        "bins": bins,
        "failed_excluded": int(failed.sum()),
    }
