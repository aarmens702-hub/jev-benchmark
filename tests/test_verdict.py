"""Sequences, verdict table and additions. Rows are checked against DESIGN.md's text."""
from pathlib import Path

import numpy as np
import pytest

from metrics import GRID, DsResult, bootstrap_contrasts
from verdict import (
    CLAIM_ARMS,
    JEV_SEQUENCE,
    LR_SEQUENCE,
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


def test_sequence_order_is_the_frozen_one():
    assert LR_SEQUENCE == ("all", 35, 16, 8, 5, 3, 2, 1)
    assert JEV_SEQUENCE == (1, 2, 3, 5, 8, 16, 35, "all")


@pytest.mark.parametrize(
    "fails_at,s_star",
    [("all", "none"), (35, "all"), (16, 35), (8, 16), (5, 8), (3, 5), (2, 3), (1, 2)],
)
def test_lr_ahead_stops_at_each_position(fails_at, s_star):
    # Every other s passes, so anything past the failure would be picked up if the walk continued.
    values = {s: PASS_LR for s in GRID}
    values[fails_at] = (0.0, -1.0, 1.0)
    assert lr_ahead(ds_from(values)) == s_star


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


@pytest.mark.parametrize(
    "fails_at,s_j",
    [(1, "none"), (2, 1), (3, 2), (5, 3), (8, 5), (16, 8), (35, 16), ("all", 35)],
)
def test_jev_ahead_stops_at_each_position(fails_at, s_j):
    values = {s: PASS_JEV for s in GRID}
    values[fails_at] = (0.0, -1.0, 1.0)
    assert jev_ahead(ds_from(values)) == s_j


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


def design_table():
    """DESIGN.md's verdict table as (s* values, s_J values, verdict text), read from the file."""
    lines = DESIGN.split("Verdict, exactly one row applies:\n", 1)[1].splitlines()
    rows = []
    for line in lines[2:]:  # the header and the divider
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split(" | ")])

    def values(cell):
        if cell == "any":
            return set(ALL_S)
        if cell.endswith(" or more"):
            return set(GRID[GRID.index(int(cell.split()[0])) :])
        return {v if v in ("all", "none") else int(v) for v in cell.split(", ")}

    return [(values(a), values(b), text.strip()) for a, b, text in rows]


def test_design_table_is_read_as_five_rows():
    # So a parsing slip can't empty the check below.
    table = design_table()
    assert [(len(a), len(b)) for a, b, _ in table] == [(3, 9), (3, 8), (3, 1), (3, 6), (3, 3)]
    assert table[1][1] == {1, 2, 3, 5, 8, 16, 35, "all"}
    assert table[3][1] == {3, 5, 8, 16, 35, "all"}
    assert table[4][1] == {"none", 1, 2}
    assert [row[2] for row in table] == [ROW_1, ROW_2, ROW_3, ROW_4, ROW_5]


def test_verdict_covers_every_s_star_s_j_pair_with_exactly_one_row():
    # Expected rows come from DESIGN.md's table, not from a copy of verdict_row.
    table = design_table()
    per_row = dict.fromkeys(range(1, 6), 0)
    for s_star in ALL_S:
        for s_j in ALL_S:
            matching = [i for i, (a, b, _) in enumerate(table, start=1) if s_star in a and s_j in b]
            assert len(matching) == 1, (s_star, s_j)
            row = matching[0]
            assert verdict_row(s_star, s_j) == row, (s_star, s_j)
            filled = table[row - 1][2].replace("s*", str(s_star)).replace("s_J", str(s_j))
            assert verdict(s_star, s_j) == filled
            per_row[row] += 1
    assert per_row == {1: 27, 2: 24, 3: 3, 4: 18, 5: 9}
    assert sum(per_row.values()) == 81


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
        his_claim_arms={"NLI": None, "embedding similarity": None, "small LLM": None},
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
    out = additions(**base_kwargs(his_claim_arms={"NLI": tied(49), "embedding similarity": tied(48), "small LLM": None}))
    claim = out["his_claim"]
    assert claim["NLI"]["status"] == "run"
    assert claim["NLI"]["C"] == -1.0
    assert claim["NLI"]["lower"] == -1.0
    assert claim["NLI"]["level"] == 98.33
    assert claim["NLI"]["matches_jev"] is True
    assert claim["embedding similarity"]["C"] == -2.0
    assert claim["embedding similarity"]["matches_jev"] is False
    assert claim["small LLM"] == {"status": "not run"}
    assert out["any_arm_matches_jev"] is True


def test_no_arm_matching_is_recorded():
    out = additions(**base_kwargs(his_claim_arms={"NLI": tied(40), "embedding similarity": None, "small LLM": None}))
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
            his_claim_arms={"NLI": nli, "embedding similarity": None, "small LLM": None},
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
        additions(**base_kwargs(his_claim_arms={"NLI": None, "embedding similarity": None}))


def test_his_claim_arm_names_are_the_design_ones():
    assert CLAIM_ARMS == ("NLI", "embedding similarity", "small LLM")
    # The old loose name, which could hold any embedding arm, is refused.
    with pytest.raises(ValueError):
        additions(**base_kwargs(his_claim_arms={"NLI": None, "embeddings": None, "small LLM": None}))


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


def test_ag_news_same_row_with_different_s_star_has_no_text():
    # Banking: s*=8, s_J=none is row 3. AG News: s*=5, s_J=none is also row 3.
    values = {s: PASS_LR for s in ("all", 35, 16, 8, 5)}
    out = additions(**base_kwargs(ag_news=ds_from(values, jev_acc=90.0)))["ag_news"]
    assert out["status"] == "run"
    assert out["s_star"] == 5
    assert out["s_j"] == "none"
    assert out["row_differs"] is False
    assert out["jev_acc"] == 90.0
    assert out["texts"] == []


def test_ag_news_different_row_says_the_result_does_not_carry():
    # AG News s*=none, s_J=none is row 5; Banking is row 3.
    out = additions(**base_kwargs(ag_news=ds_from({}, jev_acc=90.0)))["ag_news"]
    assert out["row_differs"] is True
    assert out["texts"] == [CARRY]


def test_ag_news_row_differs_when_only_s_j_differs():
    # Banking: (8, none) is row 3. AG News: (8, 1) is row 2. s* is the same, so a reading of "bin"
    # that looked at s* alone would say nothing here.
    values = {s: PASS_LR for s in ("all", 35, 16, 8)}
    values[1] = PASS_JEV
    out = additions(**base_kwargs(ag_news=ds_from(values, jev_acc=90.0)))["ag_news"]
    assert out["s_star"] == 8
    assert out["s_j"] == 1
    assert out["row_differs"] is True
    assert out["texts"] == [CARRY]


def test_ag_news_same_row_when_s_star_differs_only_within_the_row():
    # Banking: (none, 1) and AG News: (35, 2) are both row 5, so nothing prints.
    values = {s: PASS_LR for s in ("all", 35)}
    values[1] = PASS_JEV
    values[2] = PASS_JEV
    out = additions(**base_kwargs(banking_s_star="none", banking_s_j=1, ag_news=ds_from(values, jev_acc=90.0)))["ag_news"]
    assert (out["s_star"], out["s_j"]) == (35, 2)
    assert out["row_differs"] is False
    assert out["texts"] == []


@pytest.mark.parametrize("jev_acc,capped", [(96.0, True), (97.3, True), (95.99, False)])
def test_ag_news_ceiling_at_96_or_higher(jev_acc, capped):
    out = additions(**base_kwargs(banking_s_star="none", ag_news=ds_from({}, jev_acc=jev_acc)))["ag_news"]
    assert out["row_differs"] is False
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
    out = additions(**base_kwargs(his_claim_arms={"NLI": tied(60), "embedding similarity": None, "small LLM": None}))
    assert "verdict" not in out
