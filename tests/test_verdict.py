"""Sequences, verdict table and additions. Rows are checked against DESIGN.md's text."""
from pathlib import Path

import numpy as np
import pytest

from metrics import GRID, DsResult, bootstrap_contrasts
from verdict import (
    NOT_A_CLAIM,
    VERDICT_ROWS,
    additions,
    equivalent_within_2,
    isolated_lr_passes,
    jev_ahead,
    lr_ahead,
    verdict,
    verdict_row,
)

DESIGN = (Path(__file__).resolve().parents[1] / "DESIGN.md").read_text()
ALL_S = (1, 2, 3, 5, 8, 16, 35, "all", "none")


def ds_from(values: dict, b=10_000, n_replicates=None, jev_acc=None):
    """values: s -> (D_s, L_s, U_s). Unlisted s fail both sequences."""
    filled = {s: values.get(s, (0.0, -1.0, 1.0)) for s in GRID}
    return DsResult(
        d={s: v[0] for s, v in filled.items()},
        lower={s: v[1] for s, v in filled.items()},
        upper={s: v[2] for s, v in filled.items()},
        b=b,
        n_replicates=b if n_replicates is None else n_replicates,
        jev_acc=jev_acc,
    )


PASS_LR = (5.0, 1.0, 9.0)
PASS_JEV = (-5.0, -9.0, -1.0)


# ---------------------------------------------------------------- LR-ahead


def test_lr_ahead_tests_all_first_and_walks_down():
    assert lr_ahead(ds_from({"all": PASS_LR})) == "all"
    assert lr_ahead(ds_from({"all": PASS_LR, 35: PASS_LR, 16: PASS_LR})) == 16
    assert lr_ahead(ds_from({s: PASS_LR for s in GRID})) == 1


def test_lr_ahead_is_none_if_all_fails():
    assert lr_ahead(ds_from({})) == "none"


def test_lr_ahead_stops_at_first_failure_and_ignores_later_passes():
    # all and 35 pass, 16 fails, 8 and below pass. Only s*=35 counts.
    values = {s: PASS_LR for s in GRID}
    values[16] = (0.5, -1.0, 2.0)
    assert lr_ahead(ds_from(values)) == 35
    # s=all failing ends it, however many others pass.
    values = {s: PASS_LR for s in GRID}
    values["all"] = (0.5, -1.0, 2.0)
    assert lr_ahead(ds_from(values)) == "none"


def test_lr_ahead_needs_ds_at_least_2_as_well_as_lower_bound_above_0():
    # Tight interval above zero but a lead under 2 points.
    assert lr_ahead(ds_from({"all": (1.99, 1.5, 2.5)})) == "none"
    # Lead of 2 or more, but the lower bound isn't above 0.
    assert lr_ahead(ds_from({"all": (5.0, 0.0, 10.0)})) == "none"
    assert lr_ahead(ds_from({"all": (5.0, -0.1, 10.0)})) == "none"
    # Boundary: D_s = 2 exactly passes, with a lower bound just above 0.
    assert lr_ahead(ds_from({"all": (2.0, 0.001, 4.0)})) == "all"


# ---------------------------------------------------------------- Jev-ahead


def test_jev_ahead_tests_1_first_and_walks_up():
    assert jev_ahead(ds_from({1: PASS_JEV})) == 1
    assert jev_ahead(ds_from({1: PASS_JEV, 2: PASS_JEV, 3: PASS_JEV})) == 3
    assert jev_ahead(ds_from({s: PASS_JEV for s in GRID})) == "all"


def test_jev_ahead_is_none_if_1_fails():
    assert jev_ahead(ds_from({})) == "none"


def test_jev_ahead_stops_at_first_failure_and_ignores_later_passes():
    values = {s: PASS_JEV for s in GRID}
    values[3] = (0.0, -1.0, 1.0)
    assert jev_ahead(ds_from(values)) == 2
    values = {s: PASS_JEV for s in GRID}
    values[1] = (0.0, -1.0, 1.0)
    assert jev_ahead(ds_from(values)) == "none"


def test_jev_ahead_needs_ds_at_most_minus_2_as_well_as_upper_bound_below_0():
    # A 0.5 point Jev lead with a tight interval must not pass.
    assert jev_ahead(ds_from({1: (-0.5, -0.7, -0.3)})) == "none"
    assert jev_ahead(ds_from({1: (-1.99, -2.5, -1.5)})) == "none"
    # Lead of 2 or more, but the upper bound isn't below 0.
    assert jev_ahead(ds_from({1: (-5.0, -10.0, 0.0)})) == "none"
    # Boundary: D_s = -2 exactly passes.
    assert jev_ahead(ds_from({1: (-2.0, -4.0, -0.001)})) == 1


def test_timing_runs_cannot_produce_a_sequence_result():
    partial = ds_from({"all": PASS_LR}, b=10_000, n_replicates=100)
    with pytest.raises(ValueError):
        lr_ahead(partial)
    with pytest.raises(ValueError):
        jev_ahead(partial)


def test_isolated_passes_below_s_star_are_reported_as_not_a_claim():
    # s*=35; 16 fails; 5 and 2 pass in isolation.
    values = {"all": PASS_LR, 35: PASS_LR, 5: PASS_LR, 2: PASS_LR}
    assert isolated_lr_passes(ds_from(values)) == [5, 2]
    # s*=none: anything that passes is below the stopped sequence.
    assert isolated_lr_passes(ds_from({8: PASS_LR})) == [8]
    assert isolated_lr_passes(ds_from({s: PASS_LR for s in GRID})) == []
    assert NOT_A_CLAIM == "not a claim"


# ---------------------------------------------------------------- verdict table


ROW_1 = "A light classifier with s* labels per class beats zero-label Jev at 80% coverage on this 77-way task."
ROW_2 = "Jev leads through s_J labels per class and LR leads from s*. Jev's value on this task is limited to cold starts."
ROW_3 = "LR leads clearly from s*. Below that, no clear difference."
ROW_4 = "Jev leads when labels are scarce, through s_J labels per class."
ROW_5 = 'No clear winner when labels are scarce. If [L_s, U_s] is inside [-2, +2] at every s, say "equivalent within 2 points".'


def test_verdict_rows_are_verbatim_from_design():
    assert VERDICT_ROWS == {1: ROW_1, 2: ROW_2, 3: ROW_3, 4: ROW_4, 5: ROW_5}
    for row in VERDICT_ROWS.values():
        assert f"| {row} |" in DESIGN


def expected_row(s_star, s_j):
    if s_star in (1, 2, 3):
        return 1
    if s_star in (5, 8, 16):
        return 3 if s_j == "none" else 2
    return 4 if s_j in (3, 5, 8, 16, 35, "all") else 5


def test_verdict_covers_every_s_star_s_j_pair_with_exactly_one_row():
    seen = set()
    for s_star in ALL_S:
        for s_j in ALL_S:
            row = verdict_row(s_star, s_j)
            assert row == expected_row(s_star, s_j), (s_star, s_j)
            seen.add(row)
            text = verdict(s_star, s_j)
            assert text == VERDICT_ROWS[row].replace("s*", str(s_star)).replace("s_J", str(s_j))
    assert seen == {1, 2, 3, 4, 5}
    assert len(ALL_S) ** 2 == 81


@pytest.mark.parametrize(
    "s_star,s_j,text",
    [
        (1, "none", "A light classifier with 1 labels per class beats zero-label Jev at 80% coverage on this 77-way task."),
        (3, 5, "A light classifier with 3 labels per class beats zero-label Jev at 80% coverage on this 77-way task."),
        (5, 1, "Jev leads through 1 labels per class and LR leads from 5. Jev's value on this task is limited to cold starts."),
        (16, 3, "Jev leads through 3 labels per class and LR leads from 16. Jev's value on this task is limited to cold starts."),
        (8, "none", "LR leads clearly from 8. Below that, no clear difference."),
        ("none", 3, "Jev leads when labels are scarce, through 3 labels per class."),
        ("all", 8, "Jev leads when labels are scarce, through 8 labels per class."),
        (35, 16, "Jev leads when labels are scarce, through 16 labels per class."),
        ("none", "none", ROW_5),
        (35, 2, ROW_5),
        ("all", 1, ROW_5),
    ],
)
def test_verdict_text_hand_checked(s_star, s_j, text):
    assert verdict(s_star, s_j) == text


@pytest.mark.parametrize("bad", [(4, "none"), ("none", 4), (0, 1), ("All", 1), (None, 1)])
def test_values_off_the_grid_are_errors(bad):
    with pytest.raises(ValueError):
        verdict(*bad)


def test_equivalent_within_2_needs_every_interval_inside():
    inside = {s: (0.0, -2.0, 2.0) for s in GRID}
    assert equivalent_within_2(ds_from(inside)) is True
    inside[8] = (0.0, -2.01, 1.0)
    assert equivalent_within_2(ds_from(inside)) is False


def test_verdict_resolves_row_5_when_given_the_result():
    inside = ds_from({s: (0.0, -1.5, 1.5) for s in GRID})
    assert verdict("none", "none", ds=inside) == "No clear winner when labels are scarce. equivalent within 2 points"
    outside = ds_from({s: (0.0, -3.0, 1.5) for s in GRID})
    assert verdict("none", "none", ds=outside) == "No clear winner when labels are scarce."
    # Other rows ignore it.
    assert verdict(8, "none", ds=inside) == "LR leads clearly from 8. Below that, no clear difference."


# ---------------------------------------------------------------- additions


def tied(n_correct, n=100):
    return np.column_stack([np.full(n, 0.5), np.arange(n) < n_correct])


def base_kwargs(**over):
    # One-row classes, so every replicate is the identity and each interval collapses on the estimate.
    n = 100
    jev = tied(50)
    kwargs = dict(
        jev=jev,
        classes=np.arange(n),
        lr_s8=np.stack([tied(40 + r) for r in range(10)]),
        his_claim_arms={"NLI": None, "embeddings": None, "small LLM": None},
        jev_examples=None,
        banking_s_star=8,
        banking_s_j="none",
        ag_news=None,
        b=20,
    )
    kwargs.update(over)
    return kwargs


def test_his_claim_matches_jev_only_if_lower_limit_above_minus_2():
    # Jev 50%. NLI 49%: C = -1, above -2. Embeddings 48%: C = -2 exactly, not above. LLM not run.
    out = additions(**base_kwargs(his_claim_arms={"NLI": tied(49), "embeddings": tied(48), "small LLM": None}))
    claim = out["his_claim"]
    assert claim["NLI"]["status"] == "run"
    assert claim["NLI"]["C"] == -1.0
    assert claim["NLI"]["lower"] == -1.0
    assert claim["NLI"]["level"] == 98.33
    assert claim["NLI"]["matches_jev"] is True
    assert claim["embeddings"]["C"] == -2.0
    assert claim["embeddings"]["matches_jev"] is False
    assert claim["small LLM"] == {"status": "not run"}
    assert out["any_arm_matches_jev"] is True


def test_no_arm_matching_is_recorded():
    out = additions(**base_kwargs(his_claim_arms={"NLI": tied(40), "embeddings": None, "small LLM": None}))
    assert out["any_arm_matches_jev"] is False


def test_his_claim_uses_the_98_33_interval_on_the_shared_rows():
    n = 400
    rng = np.random.default_rng(12)
    classes = np.arange(n) % 8
    jev = np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.6])
    nli = np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.6])
    out = additions(
        **base_kwargs(
            jev=jev,
            classes=classes,
            lr_s8=np.stack([np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.6]) for _ in range(10)]),
            his_claim_arms={"NLI": nli, "embeddings": None, "small LLM": None},
            b=300,
        )
    )
    wide = bootstrap_contrasts({"NLI": nli}, jev, classes, level=98.33, b=300, seed=2026)["NLI"]
    narrow = bootstrap_contrasts({"NLI": nli}, jev, classes, level=95.0, b=300, seed=2026)["NLI"]
    assert out["his_claim"]["NLI"]["lower"] == wide.lower
    assert out["his_claim"]["NLI"]["upper"] == wide.upper
    assert wide.lower < narrow.lower


def test_his_claim_needs_all_three_arms_named():
    with pytest.raises(ValueError):
        additions(**base_kwargs(his_claim_arms={"NLI": None, "embeddings": None}))


JEV_EXAMPLES_TEXT = "Given the same 8 labels per class, Jev using them is at least as good as LR using them (one draw of labels)"


def test_jev_examples_text_is_verbatim_from_design():
    assert f'"{JEV_EXAMPLES_TEXT}"' in DESIGN


def test_jev_examples_compares_to_lr_draw_0_at_s8_with_95_ci():
    # LR draw 0 is 40%, later draws up to 49%. Jev + examples 39%: E = -1, above -2.
    out = additions(**base_kwargs(jev_examples=tied(39)))["jev_examples"]
    assert out["status"] == "run"
    assert out["E"] == -1.0
    assert out["lower"] == -1.0
    assert out["level"] == 95.0
    assert out["text"] == JEV_EXAMPLES_TEXT
    # LR's range over all 10 draws: 40 to 49.
    assert out["lr_range"] == (40.0, 49.0)


def test_jev_examples_no_text_when_lower_limit_is_not_above_minus_2():
    out = additions(**base_kwargs(jev_examples=tied(38)))["jev_examples"]
    assert out["E"] == -2.0
    assert out["text"] is None
    assert out["lr_range"] == (40.0, 49.0)


def test_jev_examples_interval_is_the_95_percent_one():
    n = 400
    rng = np.random.default_rng(13)
    classes = np.arange(n) % 8
    lr_s8 = np.stack([np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.6]) for _ in range(10)])
    arm = np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.6])
    out = additions(
        **base_kwargs(
            jev=np.column_stack([np.round(rng.random(n), 2), rng.random(n) < 0.6]),
            classes=classes,
            lr_s8=lr_s8,
            jev_examples=arm,
            b=300,
        )
    )["jev_examples"]
    expected = bootstrap_contrasts({"e": arm}, lr_s8[0], classes, level=95.0, b=300, seed=2026)["e"]
    assert out["lower"] == expected.lower
    assert out["upper"] == expected.upper


def test_jev_examples_not_run():
    assert additions(**base_kwargs())["jev_examples"] == {"status": "not run"}


CARRY = "The Banking77 result does not carry to this task."
CEILING = "ceiling-limited: a 2-point lead is hard or impossible."


def test_ag_news_wording_is_verbatim_from_design():
    assert f'"{CARRY}"' in DESIGN
    assert f'"{CEILING}"' in DESIGN


def test_ag_news_not_run():
    assert additions(**base_kwargs())["ag_news"] == {"status": "not run", "texts": ["not run"]}


def test_ag_news_same_bin_no_ceiling_has_no_text():
    # Banking: s*=8, s_J=none is row 3. AG News: s*=5, s_J=none is also row 3.
    values = {s: PASS_LR for s in ("all", 35, 16, 8, 5)}
    out = additions(**base_kwargs(ag_news=ds_from(values, jev_acc=90.0)))["ag_news"]
    assert out["status"] == "run"
    assert out["s_star"] == 5
    assert out["s_j"] == "none"
    assert out["bin_differs"] is False
    assert out["jev_acc"] == 90.0
    assert out["texts"] == []


def test_ag_news_different_bin_says_the_result_does_not_carry():
    # AG News s*=none, s_J=none is row 5; Banking is row 3.
    out = additions(**base_kwargs(ag_news=ds_from({}, jev_acc=90.0)))["ag_news"]
    assert out["bin_differs"] is True
    assert out["texts"] == [CARRY]


@pytest.mark.parametrize("jev_acc,capped", [(96.0, True), (97.3, True), (95.99, False)])
def test_ag_news_ceiling_at_96_or_higher(jev_acc, capped):
    out = additions(**base_kwargs(banking_s_star="none", ag_news=ds_from({}, jev_acc=jev_acc)))["ag_news"]
    assert out["bin_differs"] is False
    assert out["texts"] == ([CEILING] if capped else [])


def test_ag_news_both_texts_can_apply():
    out = additions(**base_kwargs(ag_news=ds_from({}, jev_acc=98.0)))["ag_news"]
    assert out["texts"] == [CARRY, CEILING]


@pytest.mark.parametrize("s_star,expected", [(1, True), (2, True), (3, True), (5, True), (8, True), (16, False), (35, False), ("all", False), ("none", False)])
def test_prediction_on_record_is_true_iff_s_star_is_8_or_lower(s_star, expected):
    assert additions(**base_kwargs(banking_s_star=s_star))["prediction_on_record"] is expected


def test_additions_never_touch_the_verdict():
    # Same s*, s_J with and without arms that match: the verdict function has no arm input at all,
    # and additions do not return a verdict.
    out = additions(**base_kwargs(his_claim_arms={"NLI": tied(60), "embeddings": None, "small LLM": None}))
    assert "verdict" not in out
