"""acc@80, paired bootstrap and secondary metrics. Synthetic data only; expected values are hand-worked."""
import inspect
import itertools
import math
import tracemalloc

import numpy as np
import pytest

import metrics
from metrics import GRID, acc_at, bootstrap_contrasts, bootstrap_ds, brier, ece_with_floor, nll


def arrs(conf, correct, failed=None):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    failed = np.zeros(len(conf), dtype=bool) if failed is None else np.asarray(failed, dtype=bool)
    return conf, correct, failed


def tied_arm(correct, failed=None):
    """An arm whose confidences are all tied, so acc@80 equals its overall accuracy."""
    correct = np.asarray(correct, dtype=float)
    columns = [np.full(len(correct), 0.5), correct]
    if failed is not None:
        columns.append(np.asarray(failed, dtype=float))
    return np.stack(columns, axis=-1)


def repeat_draws(arm, draws=10):
    return np.repeat(arm[None], draws, axis=0)


def lr_all_same(arm):
    """Every s and every draw holds the same arm."""
    return {s: repeat_draws(arm) for s in GRID}


# ---------------------------------------------------------------- acc_at


def test_tie_group_straddling_cut_counts_in_proportion():
    # 10 rows, keep 8. Six at 0.9 (4 correct), four tied at 0.5 (1 correct).
    # 2 of the 4 tied rows are needed: correct = 4 + 1*2/4 = 4.5, so acc = 4.5/8.
    conf, correct, failed = arrs(
        [0.9] * 6 + [0.5] * 4,
        [1, 1, 1, 1, 0, 0] + [1, 0, 0, 0],
    )
    assert acc_at(conf, correct, failed) == pytest.approx(4.5 / 8)


def test_sorts_high_to_low_regardless_of_input_order():
    # Keep 8 of 10. The two lowest-confidence rows are the correct ones, so they fall outside.
    conf = [0.01, 0.9, 0.8, 0.02, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    correct = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert acc_at(*arrs(conf, correct)) == 0.0
    # Flip them: every other row is correct, the two lowest are wrong.
    assert acc_at(*arrs(conf, 1 - np.array(correct))) == 1.0


def test_tie_rule_equals_expectation_over_all_random_tie_breaks():
    # DESIGN: "This equals the expectation over random tie-breaks". Enumerate every break.
    conf = np.array([0.9, 0.7, 0.7, 0.7, 0.7, 0.2, 0.2, 0.2, 0.1, 0.1])
    correct = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=bool)
    failed = np.zeros(10, dtype=bool)
    keep = 8
    results = []
    for tie_a in itertools.permutations(range(1, 5)):
        for tie_b in itertools.permutations(range(5, 8)):
            order = [0, *tie_a, *tie_b, 8, 9]
            results.append(correct[order[:keep]].mean())
    assert acc_at(conf, correct, failed) == pytest.approx(np.mean(results))


def test_failed_rows_rank_first_and_count_wrong():
    # 10 rows, keep 8. Two failed rows outrank everything even at the lowest score and with
    # correct=True. Then 6 non-failed rows are needed, the top four (3 correct) plus 2 of the
    # 4 tied at 0.5 (2 correct): correct = 3 + 2*2/4 = 4, so acc = 4/8.
    conf, correct, failed = arrs(
        [0.0, 0.0] + [0.9] * 4 + [0.5] * 4,
        [1, 1] + [1, 1, 1, 0] + [1, 1, 0, 0],
        [1, 1] + [0] * 8,
    )
    assert acc_at(conf, correct, failed) == pytest.approx(4 / 8)


def test_failed_rows_conf_may_be_nan():
    conf, correct, failed = arrs([np.nan, 0.9, 0.8, 0.7, 0.6], [0, 1, 1, 1, 1], [1, 0, 0, 0, 0])
    # Keep 4: the failed row (wrong) plus the top 3 (all correct): 3/4.
    assert acc_at(conf, correct, failed) == pytest.approx(3 / 4)


def test_nan_conf_on_a_row_that_did_not_fail_is_an_error():
    conf, correct, failed = arrs([np.nan, 0.9, 0.8, 0.7, 0.6], [0, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        acc_at(conf, correct, failed)


@pytest.mark.parametrize("n_failed", [8, 9, 10])
def test_failures_at_or_above_the_kept_set_give_zero(n_failed):
    # 10 rows, keep 8. All kept rows are failed rows, which are wrong.
    conf = np.full(10, 0.99)
    correct = np.ones(10, dtype=bool)
    failed = np.arange(10) < n_failed
    assert acc_at(conf, correct, failed) == 0.0


@pytest.mark.parametrize("n,kept", [(3080, 2464), (7600, 6080)])
def test_kept_set_sizes_named_in_design(n, kept):
    # The first `kept` rows are correct and the rest wrong, so acc@80 is exactly 1.
    # With one wrong row moved into the kept set it is exactly (kept-1)/kept.
    conf = np.arange(n, 0, -1) / n
    correct = np.arange(n) < kept
    failed = np.zeros(n, dtype=bool)
    assert acc_at(conf, correct, failed) == 1.0
    correct[kept - 1] = False
    assert acc_at(conf, correct, failed) == pytest.approx((kept - 1) / kept)


@pytest.mark.parametrize("coverage,kept", [(0.5, 1540), (0.7, 2156), (0.9, 2772), (1.0, 3080)])
def test_secondary_coverages_keep_exact_counts(coverage, kept):
    n = 3080
    conf = np.arange(n, 0, -1) / n
    correct = np.arange(n) < kept
    assert acc_at(conf, correct, np.zeros(n, dtype=bool), coverage=coverage) == 1.0


def test_fractional_kept_count_counts_the_boundary_row_in_proportion():
    # 5 rows, coverage 0.7: 3.5 rows are kept. Top three correct and the fourth wrong:
    # correct = 3 + 0*0.5 = 3, so acc = 3/3.5. Banking77 validation hits this (756 * 0.8 = 604.8).
    conf, correct, failed = arrs([0.9, 0.8, 0.7, 0.6, 0.5], [1, 1, 1, 0, 1])
    assert acc_at(conf, correct, failed, coverage=0.7) == pytest.approx(3 / 3.5)
    # Fourth row correct instead: correct = 3 + 1*0.5 = 3.5, so acc = 1.
    conf, correct, failed = arrs([0.9, 0.8, 0.7, 0.6, 0.5], [1, 1, 1, 1, 0])
    assert acc_at(conf, correct, failed, coverage=0.7) == pytest.approx(1.0)


@pytest.mark.parametrize("coverage", [0.0, -0.1, 1.5])
def test_coverage_out_of_range_is_an_error(coverage):
    with pytest.raises(ValueError):
        acc_at(*arrs([0.9, 0.8], [1, 0]), coverage=coverage)


def test_mismatched_lengths_are_an_error():
    with pytest.raises(ValueError):
        acc_at(np.array([0.9, 0.8]), np.array([True]), np.array([False, False]))


# ---------------------------------------------------------------- bootstrap_ds


def test_frozen_defaults_b_10000_seed_2026_grid():
    params = inspect.signature(bootstrap_ds).parameters
    assert params["b"].default == 10_000
    assert params["seed"].default == 2026
    assert GRID == (1, 2, 3, 5, 8, 16, 35, "all")
    assert params["coverage"].default == 0.8


def test_bootstrap_point_estimate_hand_worked():
    # 8 kept of 10. Jev: top 8 rows hold 4 correct, so 50%. LR at each s: 6 of its top 8, so 75%.
    # D_s = 25 points, and the same at every s.
    conf = np.arange(10, 0, -1) / 10
    jev = np.column_stack([conf, [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]])
    lr_arm = np.column_stack([conf, [1, 1, 1, 1, 1, 1, 0, 0, 1, 1]])
    ds = bootstrap_ds(lr_all_same(lr_arm), jev, np.arange(10) % 2, b=20)
    assert ds.jev_acc == pytest.approx(50.0)
    for s in GRID:
        assert ds.d[s] == pytest.approx(25.0)
        assert ds.draws_beating_jev[s] == 10
        assert ds.draw_range[s] == (pytest.approx(25.0), pytest.approx(25.0))


def test_bootstrap_resamples_within_class_and_reuses_rows_and_draws_across_s():
    # Within class: Jev is right on every row of class 0 and no other; LR on every row of classes
    # 0 and 1. All confidences tie, so acc@80 is overall accuracy, and it depends only on class
    # counts. Resampling inside classes leaves those counts fixed, so every replicate must equal
    # the point estimate. A pool-wide resample would make the interval wide.
    classes = np.repeat([0, 1, 2, 3], 25)
    jev = tied_arm(classes == 0)
    lr = lr_all_same(tied_arm(classes <= 1))
    ds = bootstrap_ds(lr, jev, classes, b=200)
    assert ds.d[8] == pytest.approx(25.0)
    assert np.all(np.abs(ds.replicates - 25.0) < 1e-9)
    assert ds.lower[8] == pytest.approx(25.0)
    assert ds.upper[8] == pytest.approx(25.0)

    # Same rows for all arms: an LR arm identical to Jev gives exactly 0 in every replicate,
    # even with row-level noise. Independent resamples per arm would not.
    rng = np.random.default_rng(1)
    noisy = np.column_stack([np.round(rng.random(100), 2), rng.random(100) < 0.6])
    ds = bootstrap_ds(lr_all_same(noisy), noisy, classes, b=100)
    assert np.all(ds.replicates == 0.0)

    # Same rows and same draw indices across s: two s values that hold the same draws must match
    # replicate by replicate, while draws differ enough that the series has real spread.
    other = np.column_stack([np.round(rng.random(100), 2), rng.random(100) < 0.6])
    draws = np.stack(
        [np.column_stack([np.round(rng.random(100), 2), rng.random(100) < 0.3 + 0.05 * r]) for r in range(10)]
    )
    lr = {s: draws for s in GRID if s != "all"}
    lr["all"] = repeat_draws(other)
    ds = bootstrap_ds(lr, noisy, classes, b=100)
    idx_1, idx_16 = GRID.index(1), GRID.index(16)
    assert np.array_equal(ds.replicates[idx_1], ds.replicates[idx_16])
    assert ds.replicates[idx_1].std() > 0


def test_bootstrap_uses_the_same_draw_indices_at_every_s():
    # 100 one-row classes: resampling rows changes nothing, so only draw resampling moves D_s.
    # At s=1, draw r holds acc (r+1)/10; at s=2, (10-r)/10. Their sum is 1.1 for every draw, so
    # with the same indices at both s, D_1 + D_2 = 110 points in every replicate (Jev is 0).
    n = 100
    classes = np.arange(n)
    jev = tied_arm(np.zeros(n))
    rising = np.stack([tied_arm(np.arange(n) < 10 * (r + 1)) for r in range(10)])
    falling = np.stack([tied_arm(np.arange(n) < 10 * (10 - r)) for r in range(10)])
    lr = {s: rising for s in GRID}
    lr[2] = falling
    lr["all"] = repeat_draws(tied_arm(np.zeros(n)))
    ds = bootstrap_ds(lr, jev, classes, b=300)
    total = ds.replicates[GRID.index(1)] + ds.replicates[GRID.index(2)]
    assert np.allclose(total, 110.0, atol=1e-9)
    # The draw-level spread is real and has the hand-worked size: draw accs are 0.1..1.0, whose
    # population variance is 0.0825, so the mean of 10 resampled draws has SE sqrt(0.00825) = 9.08 points.
    assert ds.replicates[GRID.index(1)].std() == pytest.approx(9.08, rel=0.15)
    assert ds.d[1] == pytest.approx(55.0)
    assert ds.draw_range[1] == (pytest.approx(10.0), pytest.approx(100.0))


def test_bootstrap_interval_width_matches_hand_worked_standard_error():
    # All confidences tie, so acc@80 is overall accuracy. 2 classes of 300 rows. LR right on all
    # rows, Jev right on half of each class. D = 50 points. Stratified bootstrap variance of the
    # mean: sum_c n_c * 0.25 / n^2 = 600 * 0.25 / 360000, so SE = 2.041 points. 95% half-width =
    # 1.96 * 2.041 = 4.0 points.
    classes = np.repeat([0, 1], 300)
    jev_correct = np.tile([1, 0], 300)
    ds = bootstrap_ds(lr_all_same(tied_arm(np.ones(600))), tied_arm(jev_correct), classes, b=1500)
    assert ds.d[8] == pytest.approx(50.0)
    assert ds.lower[8] == pytest.approx(46.0, abs=0.8)
    assert ds.upper[8] == pytest.approx(54.0, abs=0.8)


def test_bootstrap_interval_covers_true_gap_on_synthetic_data():
    # Ties everywhere, so acc@80 is overall accuracy with known truth: LR 0.75, Jev 0.70, gap 5 points.
    n = 400
    classes = np.arange(n) % 4
    rng = np.random.default_rng(7)
    covered = 0
    trials = 40
    for _ in range(trials):
        lr_correct = rng.random(n) < 0.75
        jev_correct = rng.random(n) < 0.70
        ds = bootstrap_ds(lr_all_same(tied_arm(lr_correct)), tied_arm(jev_correct), classes, b=200)
        covered += ds.lower[8] <= 5.0 <= ds.upper[8]
    assert covered / trials >= 0.85


def test_s_all_draws_are_identical_in_bootstrap():
    # At s=all the draws are identical, so draw resampling adds nothing there: with one-row
    # classes the interval has zero width, and the range across draws is zero.
    n = 50
    classes = np.arange(n)
    jev = tied_arm(np.arange(n) < 20)
    varied = np.stack([tied_arm(np.arange(n) < 25 + r) for r in range(10)])
    lr = {s: varied for s in GRID}
    lr["all"] = repeat_draws(tied_arm(np.arange(n) < 30))
    ds = bootstrap_ds(lr, jev, classes, b=100)
    assert ds.d["all"] == pytest.approx(20.0)
    assert np.all(np.abs(ds.replicates[GRID.index("all")] - 20.0) < 1e-9)
    assert ds.lower["all"] == pytest.approx(20.0)
    assert ds.upper["all"] == pytest.approx(20.0)
    assert ds.draw_range["all"] == (pytest.approx(20.0), pytest.approx(20.0))
    assert ds.draws_beating_jev["all"] == 10
    # The other s values do vary across draws.
    assert ds.replicates[GRID.index(1)].std() > 0


def test_s_all_with_draws_that_differ_is_an_error():
    n = 20
    classes = np.arange(n) % 2
    varied = np.stack([tied_arm(np.arange(n) < 5 + r) for r in range(10)])
    with pytest.raises(ValueError, match="all"):
        bootstrap_ds({s: varied for s in GRID}, tied_arm(np.ones(n)), classes, b=5)


def test_matches_a_naive_row_by_row_reference():
    # Reference: materialise each replicate's rows, call acc_at on every arm and draw, and average
    # over the replicate's draw indices. Ties and failures included. Must match to float noise.
    n = 90
    rng = np.random.default_rng(11)
    classes = rng.integers(0, 5, n)

    def arm(p, failed=False):
        out = np.column_stack([np.round(rng.random(n), 1), rng.random(n) < p])
        if failed:
            out = np.column_stack([out, rng.random(n) < 0.15])
        return out

    jev = arm(0.6, failed=True)
    lr = {s: np.stack([arm(0.4 + 0.05 * i) for r in range(10)]) for i, s in enumerate(GRID) if s != "all"}
    lr["all"] = repeat_draws(arm(0.8))
    b = 40
    ds = bootstrap_ds(lr, jev, classes, b=b, seed=5)

    def score(a, rows):
        failed = a[rows, 2] != 0 if a.shape[1] == 3 else np.zeros(n, dtype=bool)
        return acc_at(a[rows, 0], a[rows, 1] != 0, failed)

    done = 0
    for rows_chunk, draw_chunk in metrics.replicate_plan(classes, b, seed=5):
        for rows, draw_idx in zip(rows_chunk, draw_chunk):
            assert np.array_equal(classes[rows], classes)  # within class, position by position
            jev_acc = score(jev, rows)
            for j, s in enumerate(GRID):
                expected = 100 * (np.mean([score(lr[s][r], rows) for r in draw_idx]) - jev_acc)
                assert ds.replicates[j, done] == pytest.approx(expected, abs=1e-8)
            done += 1
    assert done == b


def test_replicates_do_not_depend_on_chunk_size_and_n_replicates_is_a_prefix():
    n = 60
    rng = np.random.default_rng(3)
    classes = np.arange(n) % 3
    jev = np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.6])
    lr = {s: np.stack([np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.7]) for _ in range(10)]) for s in GRID if s != "all"}
    lr["all"] = repeat_draws(jev)
    full = bootstrap_ds(lr, jev, classes, b=50, chunk_size=64)
    small = bootstrap_ds(lr, jev, classes, b=50, chunk_size=7)
    assert np.array_equal(full.replicates, small.replicates)

    timing = bootstrap_ds(lr, jev, classes, b=50, n_replicates=10)
    assert timing.b == 50
    assert timing.n_replicates == 10
    assert timing.partial is True
    assert full.partial is False
    assert timing.replicates.shape == (8, 10)
    assert np.array_equal(timing.replicates, full.replicates[:, :10])
    assert timing.d == full.d


def test_same_seed_repeats_and_other_seed_differs():
    n = 40
    rng = np.random.default_rng(2)
    classes = np.arange(n) % 2
    jev = np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.6])
    lr = lr_all_same(np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.7]))
    a = bootstrap_ds(lr, jev, classes, b=30, seed=2026)
    b = bootstrap_ds(lr, jev, classes, b=30, seed=2026)
    c = bootstrap_ds(lr, jev, classes, b=30, seed=2027)
    assert np.array_equal(a.replicates, b.replicates)
    assert not np.array_equal(a.replicates, c.replicates)


def test_interval_is_the_2_5_and_97_5_percentiles_of_the_replicates():
    n = 80
    rng = np.random.default_rng(4)
    classes = np.arange(n) % 2
    jev = np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.5])
    lr = lr_all_same(np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.7]))
    ds = bootstrap_ds(lr, jev, classes, b=120)
    for j, s in enumerate(GRID):
        assert ds.lower[s] == np.percentile(ds.replicates[j], 2.5)
        assert ds.upper[s] == np.percentile(ds.replicates[j], 97.5)


def test_exact_two_point_gap_is_not_lost_to_float_noise():
    # 125 rows, keep 100. LR has 57 of its kept rows correct, Jev 55. In floats 0.57 - 0.55 is
    # 0.0199999...; the rule needs D_s >= 2 to be decided on the true value, which is exactly 2.
    conf = np.arange(125, 0, -1) / 125
    jev = np.column_stack([conf, np.arange(125) < 55])
    lr = np.column_stack([conf, np.arange(125) < 57])
    ds = bootstrap_ds(lr_all_same(lr), jev, np.arange(125), b=5)
    assert ds.d[8] == 2.0
    assert ds.lower[8] == 2.0
    assert ds.upper[8] == 2.0


def test_coverage_argument_reaches_every_arm():
    # Coverage 1.0 keeps all 10: Jev 6/10, LR 8/10, D = 20. At 0.8 the correct rows inside the top
    # 8 are 4 and 6, D = 25.
    conf = np.arange(10, 0, -1) / 10
    jev = np.column_stack([conf, [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]])
    lr = lr_all_same(np.column_stack([conf, [1, 1, 1, 1, 1, 1, 0, 0, 1, 1]]))
    classes = np.arange(10)
    assert bootstrap_ds(lr, jev, classes, b=5, coverage=1.0).d[1] == pytest.approx(20.0)
    assert bootstrap_ds(lr, jev, classes, b=5, coverage=0.8).d[1] == pytest.approx(25.0)


def test_failed_jev_rows_in_the_third_column_count_against_jev():
    # 10 rows, keep 8, one-row classes. Jev: 2 failed plus its top 6 non-failed rows all correct,
    # so 6/8 = 75. LR: all 8 kept rows correct, so D = 25.
    conf = np.arange(10, 0, -1) / 10
    failed = np.arange(10) >= 8
    jev = np.column_stack([conf, np.ones(10), failed])
    lr = lr_all_same(np.column_stack([conf, np.ones(10)]))
    ds = bootstrap_ds(lr, jev, np.arange(10), b=5)
    assert ds.jev_acc == pytest.approx(75.0)
    assert ds.d[1] == pytest.approx(25.0)


def test_nan_conf_without_a_failed_column_is_an_error():
    conf = np.arange(10, 0, -1) / 10.0
    jev = np.column_stack([conf, np.ones(10)])
    jev[0, 0] = np.nan
    lr = lr_all_same(np.column_stack([conf, np.ones(10)]))
    with pytest.raises(ValueError):
        bootstrap_ds(lr, jev, np.arange(10), b=5)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda lr: lr.pop(35),  # a grid point is missing
        lambda lr: lr.__setitem__(4, lr[3]),  # an s outside the grid
        lambda lr: lr.__setitem__(1, lr[1][:9]),  # fewer than 10 draws
        lambda lr: lr.__setitem__(1, lr[1][:, :9]),  # rows don't line up
        lambda lr: lr.__setitem__(1, lr[1] * 2),  # correct column is not 0/1
    ],
)
def test_bad_inputs_are_errors(mutate):
    conf = np.arange(10, 0, -1) / 10
    arm = np.column_stack([conf, np.ones(10)])
    lr = lr_all_same(arm)
    mutate(lr)
    with pytest.raises(ValueError):
        bootstrap_ds(lr, arm, np.arange(10), b=5)


def test_memory_does_not_grow_with_b_beyond_the_replicate_matrix():
    # Holding B copies of a 200-row weight vector would add over 2.8 MB between these two runs.
    # The replicate matrix itself adds 8 * 8 B per replicate, about 115 KB (a few times that at peak).
    n = 200
    rng = np.random.default_rng(9)
    classes = np.arange(n) % 5
    jev = np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.6])
    lr = lr_all_same(np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.7]))

    def peak(b):
        tracemalloc.start()
        bootstrap_ds(lr, jev, classes, b=b)
        _, top = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return top

    small, large = peak(256), peak(2048)
    assert large - small < 1_000_000


def test_repr_does_not_print_any_accuracy_or_gap():
    # VALIDATION.md: no arm accuracy or D_s may be printed on validation data, so an accidental
    # print or a pytest failure dump of the result must not show them.
    conf = np.arange(10, 0, -1) / 10
    arm = np.column_stack([conf, np.arange(10) < 3])
    ds = bootstrap_ds(lr_all_same(arm), arm, np.arange(10), b=5)
    assert repr(ds) == "DsResult(b=5, n_replicates=5, coverage=0.8)"


# ---------------------------------------------------------------- bootstrap_contrasts


def test_contrast_point_estimate_and_replicates_use_the_same_rows_as_bootstrap_ds():
    # When every LR draw is the same arm X, D_s and the contrast X minus Jev are the same quantity,
    # so with the same seed and classes they must match replicate by replicate.
    n = 70
    rng = np.random.default_rng(6)
    classes = np.arange(n) % 3
    jev = np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.5])
    x = np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.7])
    ds = bootstrap_ds(lr_all_same(x), jev, classes, b=60, seed=2026)
    con = bootstrap_contrasts({"x": x}, jev, classes, level=95.0, b=60, seed=2026)["x"]
    assert con.estimate == ds.d[1]
    assert np.array_equal(con.replicates, ds.replicates[GRID.index(1)])
    assert con.lower == ds.lower[1]
    assert con.upper == ds.upper[1]


def test_contrast_level_sets_the_percentiles():
    n = 80
    rng = np.random.default_rng(8)
    classes = np.arange(n) % 2
    base = np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.5])
    x = np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.7])
    con = bootstrap_contrasts({"x": x}, base, classes, level=98.33, b=200)["x"]
    # 98.33% leaves (100 - 98.33) / 2 = 0.835% in each tail.
    assert con.level == 98.33
    assert con.lower == np.percentile(con.replicates, 0.835)
    assert con.upper == np.percentile(con.replicates, 99.165)


def test_contrast_hand_worked_one_row_classes():
    # One-row classes make every replicate the identity, so the interval collapses on the estimate.
    n = 100
    base = tied_arm(np.arange(n) < 50)
    x = tied_arm(np.arange(n) < 47)
    con = bootstrap_contrasts({"x": x}, base, np.arange(n), level=95.0, b=20)["x"]
    assert con.estimate == -3.0
    assert con.lower == -3.0
    assert con.upper == -3.0


def test_contrast_repr_hides_values():
    n = 10
    arm = tied_arm(np.arange(n) < 5)
    con = bootstrap_contrasts({"x": arm}, arm, np.arange(n), level=95.0, b=5)["x"]
    assert repr(con) == "Contrast(level=95.0)"


# ---------------------------------------------------------------- secondary metrics


def test_nll_clips_to_design_floor_hand_worked():
    # -ln(1.0) = 0; -ln(0.5) = 0.6931; 0.0 and 0.001 both clip to 0.005: -ln(0.005) = 5.2983.
    expected = (0 + math.log(2) + 2 * -math.log(0.005)) / 4
    assert nll(np.array([1.0, 0.5, 0.0, 0.001])) == pytest.approx(expected)


def test_nll_failed_rows_count_as_gold_probability_zero():
    expected = (math.log(2) + -math.log(0.005)) / 2
    got = nll(np.array([0.5, np.nan]), failed=np.array([False, True]))
    assert got == pytest.approx(expected)


def test_nll_nan_on_a_row_that_did_not_fail_is_an_error():
    with pytest.raises(ValueError):
        nll(np.array([0.5, np.nan]))


def test_brier_hand_worked():
    # Row 0, gold 0: (0.7-1)^2 + 0.3^2 = 0.18. Row 1, gold 0: (0.2-1)^2 + 0.8^2 = 1.28. Mean 0.73.
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    assert brier(probs, np.array([0, 0])) == pytest.approx(0.73)


def test_brier_failed_rows_score_as_all_zero_probabilities():
    # Failed row: every probability is 0, so the sum is (0-1)^2 = 1. Mean of 0.18 and 1 = 0.59.
    probs = np.array([[0.7, 0.3], [np.nan, np.nan]])
    assert brier(probs, np.array([0, 1]), failed=np.array([False, True])) == pytest.approx(0.59)


def test_ece_hand_worked():
    # bins=10. Bin (0.8, 0.9]: conf 0.85, acc 0.5, weight 0.5, adds 0.175.
    # Bin (0.5, 0.6]: conf 0.55, acc 1.0, weight 0.5, adds 0.225. ECE = 0.4.
    conf, correct, _ = arrs([0.85, 0.85, 0.55, 0.55], [1, 0, 1, 1])
    out = ece_with_floor(conf, correct, bins=10, n_sim=10)
    assert out["ece"] == pytest.approx(0.4)
    assert out["n"] == 4
    assert out["bins"] == 10


def test_ece_edges_zero_and_one_land_in_first_and_last_bins():
    out = ece_with_floor(np.array([0.0, 1.0]), np.array([False, True]), bins=2, n_sim=10)
    assert out["ece"] == pytest.approx(0.0)


def test_ece_floor_hand_worked():
    # Four rows at 0.5 in one bin. A calibrated arm gets k ~ Bin(4, 0.5) right.
    # E|k/4 - 0.5| = (2/16)*0.5 + (8/16)*0.25 = 0.1875.
    conf = np.full(4, 0.5)
    out = ece_with_floor(conf, np.array([1, 1, 0, 0], dtype=bool), bins=10, n_sim=20_000)
    assert out["ece"] == pytest.approx(0.0)
    assert out["floor"] == pytest.approx(0.1875, abs=0.005)
    # A confidence of 1.0 can never miss when calibrated, so its floor is 0.
    assert ece_with_floor(np.ones(5), np.ones(5, dtype=bool), n_sim=50)["floor"] == 0.0


def test_ece_excludes_failed_rows_and_counts_them():
    conf = np.array([0.85, 0.85, np.nan])
    correct = np.array([True, False, False])
    out = ece_with_floor(conf, correct, failed=np.array([False, False, True]), bins=10, n_sim=10)
    assert out["ece"] == pytest.approx(0.35)
    assert out["n"] == 2
    assert out["failed_excluded"] == 1


def test_ece_floor_is_seeded():
    conf = np.random.default_rng(0).random(50)
    correct = conf > 0.5
    a = ece_with_floor(conf, correct, n_sim=100)
    b = ece_with_floor(conf, correct, n_sim=100)
    assert a == b
    assert inspect.signature(ece_with_floor).parameters["seed"].default == 2026
