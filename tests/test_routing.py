"""Routing rule-change test (DESIGN.md, "Routing test"). Synthetic text only; no test data, no paid calls."""
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data
import routing
from labels import label_texts
from ledger import Ledger
from openrouter import Client, PaidFlagRequired, TransportResponse

REPO = Path(__file__).resolve().parents[1]
DESIGN = (REPO / "DESIGN.md").read_text()
FIXTURES = Path(__file__).parent / "fixtures"
BUILD = "typesafe/jev-1.14-20261014"
INTENTS = sorted(label_texts("banking77", bare=True))
ONE_PER_FAMILY = {f: spec["intents"][0] for f, spec in routing.load_families().items()}


def all_states():
    names = list(routing.FIELDS)
    for values in itertools.product(*routing.FIELDS.values()):
        yield dict(zip(names, values))


def cells_frame(families=None, states=None):
    """One row per (family, state) cell, with one intent standing for each family."""
    families = families or list(ONE_PER_FAMILY)
    states = states or list(all_states())
    rows = [
        {"row_id": f"r{i}", "text": f"synthetic {i}", "label": ONE_PER_FAMILY[f], **s}
        for i, (f, s) in enumerate(itertools.product(families, states))
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ families


def test_all_77_intents_map_to_exactly_one_of_8_families():
    families = routing.load_families()
    assert len(families) == 8
    assert tuple(families) == routing.rules.FAMILIES
    listed = [intent for spec in families.values() for intent in spec["intents"]]
    assert len(listed) == len(set(listed)) == 77
    assert set(listed) == set(INTENTS)
    for name, spec in families.items():
        assert set(spec) == {"topic", "examples", "reason", "intents"}, name
        assert spec["reason"].strip() and "\n" not in spec["reason"], name
    assert all(routing.family_of(i) in families for i in INTENTS)
    with pytest.raises(KeyError):
        routing.family_of("not_an_intent")


# ------------------------------------------------------------------ rules


def test_rule_a_and_b_only_use_family_and_3_categorical_fields():
    assert len(routing.FIELDS) == 3 and all(len(v) == len(set(v)) == 3 for v in routing.FIELDS.values())
    assert len(routing.QUEUES) == len(set(routing.QUEUES)) == 8
    for name, rule in routing.RULES.items():
        assert set(rule) == set(routing.QUEUES), name  # exactly 8 queues, all used
        for queue, clauses in rule.items():
            for clause in clauses:
                assert set(clause) <= {"family", *routing.FIELDS}, (name, queue)
                for key, allowed in clause.items():
                    universe = routing.rules.FAMILIES if key == "family" else routing.FIELDS[key]
                    assert set(allowed) <= set(universe), (name, queue, key)
        # Every combination of family and state goes to exactly one queue, and each queue gets some.
        seen = set()
        for family in routing.rules.FAMILIES:
            for state in all_states():
                seen.add(routing.rules.route(rule, family, state))
        assert seen == set(routing.QUEUES), name
    # Inputs outside the family and the 3 fields are refused, not ignored.
    good = next(all_states())
    with pytest.raises(ValueError):
        routing.rules.route(routing.RULES["A"], "cash", {**good, "amount": "large"})
    with pytest.raises(ValueError):
        routing.rules.route(routing.RULES["A"], "cash", {**good, "tier": "gold"})
    with pytest.raises(ValueError):
        routing.rules.route(routing.RULES["A"], "mortgages", good)


def _hand_written(rule, family, s):
    """The rules as plain if/else, written separately from the clause data, to check what the data says."""
    if s["status"] == "frozen":
        return "amber"
    if rule == "A":
        if s["tier"] == "premium":
            return "blue"
        topics = {"card_ordering": "coral", "card_problems": "green", "payments": "indigo", "transfers": "olive",
                  "top_ups": "silver", "cash": "silver", "security": "teal", "account": "teal"}
    else:
        if s["channel"] == "phone":
            return "blue"
        topics = {"card_problems": "coral", "card_ordering": "green", "payments": "indigo", "transfers": "olive",
                  "security": "silver", "account": "silver", "top_ups": "teal", "cash": "teal"}
    return topics[family]


def test_rule_data_matches_the_hand_written_rules_on_every_cell():
    for name in ("A", "B"):
        for family in routing.rules.FAMILIES:
            for s in all_states():
                assert routing.rules.route(routing.RULES[name], family, s) == _hand_written(name, family, s)


def test_gold_is_the_rule_applied_to_the_family_of_the_gold_intent():
    s = {"tier": "personal", "status": "active", "channel": "app"}
    assert routing.gold({"label": "card_arrival", **s}, "A") == "coral"
    assert routing.gold({"label": "card_arrival", **s}, "B") == "green"
    assert routing.gold({"label": "lost_or_stolen_card", **s, "status": "frozen"}, "B") == "amber"
    frame = cells_frame()
    assert list(routing.gold_queues(frame, "A")) == [routing.gold(r, "A") for _, r in frame.iterrows()]
    with pytest.raises(ValueError):
        routing.gold({"label": "card_arrival", **s}, "C")


# ------------------------------------------------------------------ rule text

NEGATIONS = re.compile(r"\b(not|no|never|none|nor|neither|unless|without|except|nobody|nothing)\b|n't\b", re.I)
DATES = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday|today|tomorrow|yesterday|weekend|weekday|days?|weeks?|months?|"
    r"years?|hours?|minutes?|date|dated)\b",
    re.I,
)
NUMBER_WORDS = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|hundred|thousand|million|"
    r"first|second|third|half|once|twice|double|percent)\b",
    re.I,
)


def rule_texts():
    for name in ("A", "B"):
        yield from routing.queue_options(name).values()
        yield from routing.nli_hypotheses(name)
    yield routing.INSTRUCTIONS
    yield routing.LLM_INSTRUCTIONS


def test_rule_text_has_no_numbers_dates_or_double_negatives():
    texts = list(rule_texts())
    assert len(texts) == 2 * 16 + 2
    for text in texts:
        assert not re.search(r"\d", text), text
        assert not DATES.search(text), text
        assert not NUMBER_WORDS.search(text), text
        assert not NEGATIONS.search(text), text  # no negation at all, so no double negative
    # The checks themselves catch what they are for.
    assert NEGATIONS.search("send here unless the tier is not premium")
    assert DATES.search("after Monday") and NUMBER_WORDS.search("more than two")


def test_rule_text_is_generated_from_the_rule_data():
    a, b = routing.queue_options("A"), routing.queue_options("B")
    assert list(a) == list(b) == list(routing.QUEUES)
    assert a["amber"] == "Send here when the account status is frozen."
    assert a["blue"] == "Send here when the customer tier is premium and the account status is active or new."
    assert b["blue"] == "Send here when the account status is active or new and the contact channel is phone."
    topic = routing.load_families()["card_ordering"]
    assert a["coral"].startswith(f"Send here when the message is about {topic['topic']} ({topic['examples']}), ")
    assert b["green"].startswith(f"Send here when the message is about {topic['topic']} ({topic['examples']}), ")
    silver = a["silver"]
    assert "topping up the account (" in silver and ") or about cash machines and cash withdrawals (" in silver
    assert routing.nli_hypotheses("A")[0] == "The account status is frozen."
    assert len(routing.nli_hypotheses("B")) == 8


def test_arm_text_file_matches_the_code():
    assert (REPO / "routing" / "arm_text.md").read_text() == routing.render_arm_text()


def _fingerprint_root(tmp_path):
    for name in routing.FROZEN_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes((REPO / name).read_bytes())
    return tmp_path


def test_rule_fingerprint_covers_rules_families_builders_live_arm_text_and_instructions(tmp_path, monkeypatch):
    root = _fingerprint_root(tmp_path)
    before = routing.rule_fingerprint(root)
    assert before == routing.rule_fingerprint(REPO) and re.fullmatch(r"[0-9a-f]{64}", before)
    assert set(routing.FROZEN_FILES) == {"routing/rules.py", "routing/families.yaml", "src/jev_client.py", "src/llm_client.py"}
    for name in routing.FROZEN_FILES:
        path = root / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        assert routing.rule_fingerprint(root) != before, name
        path.write_bytes(original)
    assert routing.rule_fingerprint(root) == before
    # The live arm text counts, not routing/arm_text.md on disk.
    monkeypatch.setattr(routing, "render_arm_text", lambda: "changed")
    assert routing.rule_fingerprint(root) != before
    monkeypatch.undo()
    monkeypatch.setattr(routing, "INSTRUCTIONS", routing.INSTRUCTIONS + " Extra.")
    assert routing.rule_fingerprint(root) != before
    monkeypatch.undo()
    # Editing the file on disk does not matter on its own.
    assert routing.rule_fingerprint(root) == before
    assert not (root / "routing" / "arm_text.md").exists()
    assert routing.rules_changed_since(before, root) is False
    assert routing.rules_changed_since("0" * 64, root) is True
    assert routing.rules_changed_since(None, root) is False  # no arm output recorded yet


def test_families_wording_puts_cards_and_currencies_with_card_ordering_and_currency_exchange_with_payments():
    fam = routing.load_families()
    assert "which cards and currencies are supported" in fam["card_ordering"]["examples"]
    assert "holding or exchanging currencies" in fam["payments"]["examples"]
    assert "a duplicated payment" in fam["payments"]["examples"] and "twice" not in fam["payments"]["examples"]


# ------------------------------------------------------------------ sample and state


def synthetic_texts(n):
    return pd.DataFrame({"row_id": [f"t{i}" for i in range(n)], "text": [f"Synthetic  Message {i}?" for i in range(n)]})


def test_sample_order_and_state_come_from_hash_and_read_no_labels():
    texts = synthetic_texts(1000)
    out = routing.sample_rows(texts)
    assert len(out) == routing.N_SAMPLE == 800
    keys = [hashlib.sha256(("20261105:routing:" + data.normalize(t)).encode()).hexdigest() for t in texts["text"]]
    expected = [texts["row_id"][i] for i in np.argsort(keys, kind="stable")[:800]]
    assert out["row_id"].tolist() == expected
    # State comes from the same hash: the hash as an integer, read in base 3.
    for _, row in out.head(50).iterrows():
        n = int(hashlib.sha256(("20261105:routing:" + data.normalize(row["text"])).encode()).hexdigest(), 16)
        assert row["tier"] == routing.FIELDS["tier"][n % 3]
        assert row["status"] == routing.FIELDS["status"][(n // 3) % 3]
        assert row["channel"] == routing.FIELDS["channel"][(n // 9) % 3]
        assert routing.state(row["norm"]) == {k: row[k] for k in routing.FIELDS}
    # The order does not depend on the input order, and every state value occurs.
    shuffled = texts.sample(frac=1, random_state=3).reset_index(drop=True)
    assert routing.sample_rows(shuffled)["row_id"].tolist() == expected
    for field, values in routing.FIELDS.items():
        assert set(out[field]) == set(values)
    # Fewer rows than 800: all of them, in hash order (the validation draw).
    assert routing.sample_rows(texts.head(30))["row_id"].tolist() == sorted(
        texts.head(30)["row_id"], key=lambda r: keys[int(r[1:])]
    )
    # It takes text only: a frame carrying anything else, such as a label, is refused before reading.
    with pytest.raises(ValueError, match="text only"):
        routing.sample_rows(texts.assign(label="card_arrival"))
    with pytest.raises(ValueError, match="text only"):
        routing.sample_rows(texts.assign(gold=0))


def test_sample_rows_breaks_ties_between_identical_texts_by_row_id():
    texts = pd.DataFrame({"row_id": ["b", "a", "c"], "text": ["Same text", "same  TEXT", "other"]})
    out = routing.sample_rows(texts)
    same = out[out["norm"] == "same text"]["row_id"].tolist()
    assert same == ["a", "b"]


# ------------------------------------------------------------------ validity


def test_validity_flags_agreement_outside_40_60_and_blind_guess_over_70_and_lr_a_under_70():
    frame = cells_frame()  # every family with every state, once
    stats = routing.validity_stats(frame)
    assert stats["agreement"] == pytest.approx(100 * 104 / 216)
    assert stats["state_blind_A"] == pytest.approx(100 * 12 / 27)
    assert stats["text_blind_A"] == pytest.approx(100 * 144 / 216)
    assert stats["text_blind_B"] == pytest.approx(100 * 144 / 216)
    gold_a = routing.gold_queues(frame, "A")
    ok = routing.validity(frame, gold_a)
    assert ok == {"agreement_in_40_60": True, "state_blind_at_most_70": True, "text_blind_at_most_70": True,
                  "lr_a_at_least_70": True}
    assert routing.invalid_reasons(ok, rules_changed=False) == []

    # Agreement above 60%: every row frozen, so both rules send it to amber.
    frozen = cells_frame(states=[s for s in all_states() if s["status"] == "frozen"])
    assert not routing.validity(frozen, routing.gold_queues(frozen, "A"))["agreement_in_40_60"]
    # Agreement below 40%: open premium accounts on app, routed by tier under A and by topic under B.
    apart = cells_frame(states=[{"tier": "premium", "status": "active", "channel": "app"}])
    assert routing.validity_stats(apart)["agreement"] == 0
    assert not routing.validity(apart, routing.gold_queues(apart, "A"))["agreement_in_40_60"]
    # Boundaries: 40 and 60 are inside.
    assert routing.in_agreement_range(40.0) and routing.in_agreement_range(60.0)
    assert not routing.in_agreement_range(39.9999) and not routing.in_agreement_range(60.0001)

    # State-blind best guess over 70%: family alone tells the queue when every state routes by topic.
    topical = cells_frame(states=[{"tier": "personal", "status": "active", "channel": "app"}] * 3)
    checks = routing.validity(topical, routing.gold_queues(topical, "A"))
    assert not checks["state_blind_at_most_70"]
    # Text-blind best guess over 70%: state alone tells the queue when every row is frozen.
    assert not routing.validity(frozen, routing.gold_queues(frozen, "A"))["text_blind_at_most_70"]
    assert routing.blind_ok(70.0) and not routing.blind_ok(70.0001)

    # LR-A under 70% on rule A.
    wrong = gold_a.copy()
    wrong[: int(0.31 * len(wrong))] = "not_a_queue"
    assert not routing.validity(frame, wrong)["lr_a_at_least_70"]
    right = gold_a.copy()
    right[: int(0.30 * len(right))] = "not_a_queue"
    assert routing.validity(frame, right)["lr_a_at_least_70"]

    # Each failed check gives its DESIGN.md condition, word for word.
    reasons = routing.invalid_reasons(
        {"agreement_in_40_60": False, "state_blind_at_most_70": True, "text_blind_at_most_70": False,
         "lr_a_at_least_70": False},
        rules_changed=True,
    )
    assert reasons == [routing.INVALID[k] for k in ("rules_changed", "agreement", "blind", "lr_a_floor")]


def test_invalid_conditions_and_claims_are_copied_verbatim_from_design():
    routing_section = DESIGN.split("## Routing test", 1)[1].split("\n## ", 1)[0]
    invalid_line = next(line for line in routing_section.splitlines() if line.startswith("Invalid, and reported as such, if: "))
    conditions = invalid_line.removeprefix("Invalid, and reported as such, if: ").rstrip(".").split("; ")
    assert list(routing.INVALID.values()) == conditions
    assert f'- "{routing.CLAIM_FOLLOWS}" only if Jev minus Intent LR + code' in routing_section
    assert f'- "{routing.CLAIM_NOT_UNIQUE}" if small LLM or NLI minus Jev' in routing_section
    assert f'- Otherwise: "{routing.CLAIM_NONE}".' in routing_section
    assert f"- Say that {routing.ENGINEER_NOTE}" in routing_section
    assert routing.SCOPE in routing_section
    assert routing.FLOOR_NOTE in routing_section
    assert routing.MARGIN == 5 and routing.LEVEL == 95.0 and routing.N_SAMPLE == 800
    assert routing.PREFIX == "20261105:routing:" and "20261105:routing:" in DESIGN


# ------------------------------------------------------------------ claims


def arm(correct, top=None, failed=None):
    correct = np.asarray(correct, dtype=float)
    top = np.full(len(correct), 0.9) if top is None else np.asarray(top, dtype=float)
    failed = np.zeros(len(correct)) if failed is None else np.asarray(failed, dtype=float)
    return np.column_stack([top, correct, failed])


def result(arms, valid=None, rules_changed=False, n=200):
    rng = np.random.default_rng(0)
    gold_a = rng.choice(routing.QUEUES, n)
    gold_b = np.where(rng.random(n) < 0.5, gold_a, rng.choice(routing.QUEUES, n))
    checks = valid or dict.fromkeys(("agreement_in_40_60", "state_blind_at_most_70", "text_blind_at_most_70", "lr_a_at_least_70"), True)
    return routing.routing_result(gold_a, gold_b, arms, checks=checks, rules_changed=rules_changed, b=200)


def all_arms(**over):
    n = 200
    perfect = arm(np.ones(n))
    arms = {name: {"A": perfect, "B": perfect} for name in routing.ARMS}
    arms["small LLM"] = None
    arms["NLI"] = None
    arms.update(over)
    return arms


def test_claim_follows_needs_both_the_ci_against_intent_lr_code_and_the_rule_a_to_b_drop():
    n = 200
    out = result(all_arms())
    assert out["status"] == "valid"
    assert out["claims"] == [routing.CLAIM_FOLLOWS]
    assert out["jev_vs_code_B"]["lower"] > -5 and out["jev_drop"] == 0
    assert out["notes"] == [routing.ENGINEER_NOTE, routing.SCOPE, routing.FLOOR_NOTE]
    assert set(out["accuracy"]) == set(routing.ARMS) and out["accuracy"]["small LLM"] == routing.NOT_RUN

    # Jev drops 10 points from A to B, while staying close to Intent LR + code: no claim.
    jev_b = np.ones(n)
    jev_b[:20] = 0
    code_b = jev_b.copy()
    out = result(all_arms(Jev={"A": arm(np.ones(n)), "B": arm(jev_b)}, **{"Intent LR + code": {"A": arm(np.ones(n)), "B": arm(code_b)}}))
    assert out["jev_drop"] == pytest.approx(10) and out["claims"] == [routing.CLAIM_NONE]

    # Jev holds steady but trails Intent LR + code by far under B: no claim.
    weak = np.ones(n)
    weak[:60] = 0
    out = result(all_arms(Jev={"A": arm(weak), "B": arm(weak)}))
    assert out["jev_vs_code_B"]["lower"] <= -5 and out["claims"] == [routing.CLAIM_NONE]

    # Exactly a 5-point fall still counts as "5 points or less".
    five = np.ones(n)
    five[:10] = 0
    out = result(all_arms(Jev={"A": arm(np.ones(n)), "B": arm(five)}, **{"Intent LR + code": {"A": arm(np.ones(n)), "B": arm(five)}}))
    assert out["jev_drop"] == pytest.approx(5) and routing.CLAIM_FOLLOWS in out["claims"]


def test_not_unique_when_small_llm_or_nli_is_within_5_of_jev_under_rule_b():
    n = 200
    good = arm(np.ones(n))
    out = result(all_arms(**{"small LLM": {"A": good, "B": good}}))
    assert out["claims"][0] == routing.CLAIM_FOLLOWS and len(out["claims"]) == 2
    assert out["not_unique"] is True
    assert out["claims"][1].startswith(f"{routing.CLAIM_NOT_UNIQUE} ({routing.CLAIM_FOLLOWS}: shown; Jev minus Intent LR + code")
    assert "lower limit" in out["claims"][1] and "falls 0.0 points from rule A to rule B)" in out["claims"][1]
    bad = np.zeros(n)
    out = result(all_arms(NLI={"A": arm(bad), "B": arm(bad)}))
    assert out["claims"] == [routing.CLAIM_FOLLOWS] and out["not_unique"] is False
    assert "not_unique_context" not in out
    assert out["vs_jev_B"]["NLI"]["lower"] <= -5 and out["vs_jev_B"]["small LLM"] == routing.NOT_RUN

    # Not unique while Jev does not follow the rule: the first claim's result is reported as not shown.
    weak = np.ones(n)
    weak[:60] = 0
    out = result(all_arms(Jev={"A": arm(weak), "B": arm(weak)}, **{"small LLM": {"A": arm(weak), "B": arm(weak)}}))
    assert len(out["claims"]) == 1
    assert out["claims"][0].startswith(f"{routing.CLAIM_NOT_UNIQUE} ({routing.CLAIM_FOLLOWS}: not shown; Jev minus")

    # An invalid run makes no claim and carries no context either.
    checks = dict.fromkeys(("agreement_in_40_60", "state_blind_at_most_70", "text_blind_at_most_70", "lr_a_at_least_70"), True)
    out = result(all_arms(**{"small LLM": {"A": arm(weak), "B": arm(weak)}}), valid={**checks, "agreement_in_40_60": False})
    assert out["status"] == "invalid" and out["claims"] == [] and "not_unique_context" not in out


def test_failed_rows_are_wrong_and_invalid_or_missing_jev_makes_no_claim():
    n = 200
    failed = np.zeros(n)
    failed[:50] = 1
    out = result(all_arms(Jev={"A": arm(np.ones(n), failed=failed), "B": arm(np.ones(n), failed=failed)}))
    assert out["accuracy"]["Jev"]["A"]["accuracy"] == pytest.approx(75)
    assert out["claims"] == [routing.CLAIM_NONE]

    checks = dict.fromkeys(("agreement_in_40_60", "state_blind_at_most_70", "text_blind_at_most_70", "lr_a_at_least_70"), True)
    out = result(all_arms(), valid={**checks, "lr_a_at_least_70": False})
    assert out["status"] == "invalid" and out["claims"] == [] and out["invalid"] == [routing.INVALID["lr_a_floor"]]
    out = result(all_arms(), rules_changed=True)
    assert out["status"] == "invalid" and out["invalid"] == [routing.INVALID["rules_changed"]]

    out = result(all_arms(Jev=None))
    assert out["status"] == routing.NOT_RUN and out["claims"] == []


def test_agreement_floor_on_the_800_is_published():
    out = result(all_arms())
    gold_a, gold_b = out["gold_A"], out["gold_B"]
    assert out["floor"] == pytest.approx(100 * np.mean(gold_a == gold_b))


# ------------------------------------------------------------------ local arms (fake encoder)


def fake_encoder(texts):
    """A text names its intent as its last word; the vector is that intent's one-hot plus a little noise."""
    out = np.zeros((len(texts), len(INTENTS)), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i, INTENTS.index(t.split()[-1])] = 1.0
        out[i] += np.float32(int(hashlib.sha256(t.encode()).hexdigest()[:4], 16) / 65535 * 0.01)
    return out


def synthetic_pool(per_intent, intents):
    rows = [
        {"row_id": f"p{intent}{i}", "text": f"pool message {i} {intent}", "label": intent}
        for intent in intents
        for i in range(per_intent)
    ]
    df = pd.DataFrame(rows)
    df["norm"] = df["text"].map(data.normalize)
    return df


def predict_rows(intents, n_each=6):
    rows = [{"row_id": f"q{intent}{i}", "text": f"new text {i} {intent}"} for intent in intents for i in range(n_each)]
    return routing.with_state(pd.DataFrame(rows))


def test_lr_a_trains_on_35_rows_per_intent_with_hashed_state_under_rule_a_only():
    intents = list(ONE_PER_FAMILY.values())
    pool = synthetic_pool(40, intents)
    bank = routing.lr_a_bank(pool)
    assert bank.groupby("label").size().eq(35).all()
    assert bank["row_id"].tolist() == data.bank(pool, 0, 35)["row_id"].tolist()
    X, y = routing.lr_a_training_set(bank, fake_encoder)
    assert X.shape == (len(bank), len(INTENTS) + 9)
    states = routing.with_state(bank[["row_id", "text"]])
    assert list(y) == list(routing.gold_queues(states.assign(label=bank["label"].to_numpy()), "A"))
    assert (X[:, -9:].sum(axis=1) == 3).all()  # one value per field

    model = routing.fit_lr_a(bank, fake_encoder)
    rows = predict_rows(intents)
    pred, top = routing.predict_lr_a(model, rows, fake_encoder)
    gold_a = routing.gold_queues(rows.assign(label=[t.split()[-1] for t in rows["text"]]), "A")
    assert np.mean(pred == gold_a) >= 0.9
    assert ((0 < top) & (top <= 1)).all()


def test_intent_lr_code_predicts_intent_at_s8_draw_0_then_applies_each_rule():
    intents = list(ONE_PER_FAMILY.values())
    pool = synthetic_pool(12, intents)
    rows = predict_rows(intents)
    out = routing.intent_lr_code(pool, rows, fake_encoder, classes=intents)
    truth = rows.assign(label=[t.split()[-1] for t in rows["text"]])
    for rule in ("A", "B"):
        pred, top = out[rule]
        assert list(pred) == list(routing.gold_queues(truth, rule))
        assert ((0 < top) & (top <= 1)).all()
    assert list(out["A"][1]) == list(out["B"][1])  # same intent model, same confidence under both rules
    assert out["bank_rows"] == 8 * len(intents)


# ------------------------------------------------------------------ text arms (fake transports)


class FakeTransport:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def send(self, method, url, content=None):
        body = json.loads(content) if content else None
        self.calls.append(body)
        return self.handler(body, len(self.calls) - 1)


def ok(payload):
    return TransportResponse(200, {}, json.dumps(payload).encode())


def jev_reply(body, choice="blue"):
    payload = json.loads((FIXTURES / "decisions_choice_response.json").read_text())
    labels = list(body["questions"]["label"]["criteria"])
    probs = {label: round(0.2 / (len(labels) - 1), 6) for label in labels if label != choice}
    probs[choice] = 0.8
    payload["answers"]["label"].update(choice=choice, probabilities=probs)
    payload["model"] = BUILD
    return ok(payload)


def llm_reply(body, id_value=2):
    payload = json.loads((FIXTURES / "chat_logprobs_response.json").read_text())
    payload["model"] = body["model"]
    payload["provider"] = body["provider"]["only"][0]
    payload["choices"][0]["message"]["content"] = '{"id":' + str(id_value) + "}"
    payload["choices"][0]["logprobs"]["content"] = [
        {"token": '{"', "logprob": -0.001, "bytes": None, "top_logprobs": []},
        {"token": "id", "logprob": -0.001, "bytes": None, "top_logprobs": []},
        {"token": '":', "logprob": -0.001, "bytes": None, "top_logprobs": []},
        {"token": str(id_value), "logprob": -0.2, "bytes": None, "top_logprobs": []},
        {"token": "}", "logprob": -0.001, "bytes": None, "top_logprobs": []},
    ]
    payload["usage"].update(cost=0.00002, completion_tokens=5)
    payload["usage"]["completion_tokens_details"]["reasoning_tokens"] = 0
    return ok(payload)


def fake_client(tmp_path, handler, name="arm", stage=None):
    ledger = Ledger(tmp_path / f"{name}-ledger.jsonl", stage=stage)
    transport = FakeTransport(handler)
    return Client(transport, ledger, ledger.confirmed, sleep=lambda s: None), transport


ROW = {"row_id": "v1", "text": "I lost my card yesterday", "tier": "premium", "status": "active", "channel": "phone"}


def test_message_text_and_nli_premise_carry_the_state():
    msg = routing.message_text(ROW)
    assert msg == ("Message: I lost my card yesterday\nCustomer tier: premium\nAccount status: active\n"
                   "Contact channel: phone")
    premise = routing.nli_premise(ROW)
    assert premise == ("The customer tier is premium. The account status is active. The contact channel is phone. "
                       "The message says: I lost my card yesterday")


@pytest.mark.parametrize("rule", ["A", "B"])
def test_jev_adapter_sends_the_rule_text_as_options_and_the_state_with_the_message(tmp_path, rule):
    client, transport = fake_client(tmp_path, lambda body, n: jev_reply(body, "blue"))
    decision = routing.jev_decider(rule, client)(ROW)
    assert decision.pred == "blue" and not decision.failed
    body = transport.calls[0]
    question = body["questions"]["label"]
    assert question["instructions"] == routing.INSTRUCTIONS
    assert question["criteria"] == routing.queue_options(rule)  # same text; order is shuffled per row
    assert body["state"] == routing.message_text(ROW)
    assert body["model"] == "typesafe/jev-1.14"


@pytest.mark.parametrize("rule", ["A", "B"])
def test_small_llm_adapter_sends_the_rule_text_as_options(tmp_path, rule):
    client, transport = fake_client(tmp_path, lambda body, n: llm_reply(body, 2))
    decision = routing.llm_decider(rule, "openai/gpt-oss-20b", "Darkbloom", client)(ROW)
    assert decision.pred in routing.QUEUES and not decision.failed
    body = transport.calls[0]
    assert body["messages"][0]["content"] == routing.LLM_INSTRUCTIONS
    user = body["messages"][1]["content"]
    assert user.startswith("Text: " + routing.message_text(ROW))
    for queue, text in routing.queue_options(rule).items():
        assert f". {queue}: {text}" in user


def test_nli_adapter_puts_state_in_the_premise_and_rule_conditions_in_the_hypotheses():
    seen = []

    def scorer(pairs):
        seen.extend(pairs)
        # Entailment logit: high for the "frozen" hypothesis only when the premise says frozen.
        return [5.0 if ("frozen." in h and "status is frozen" in p) else 0.0 for p, h in pairs]

    rows = pd.DataFrame([{**ROW, "row_id": "a", "status": "frozen"}, {**ROW, "row_id": "b"}])
    pred, top = routing.nli_arm(rows, "A", scorer, batch_size=5)
    assert pred[0] == "amber" and top[0] > 0.9
    assert top[1] == pytest.approx(1 / 8)
    assert len(seen) == 16
    assert [h for _, h in seen[:8]] == routing.nli_hypotheses("A")
    assert all(p == routing.nli_premise(rows.iloc[0]) for p, _ in seen[:8])


def test_decisions_to_arm_array_marks_failures():
    from openrouter import Decision

    decisions = {
        "a": Decision("amber", 0.8, {}, BUILD, 0.0, False),
        "b": Decision(None, None, {}, "", 0.0, True, reason="refusal"),
    }
    rows = pd.DataFrame({"row_id": ["a", "b", "c"]})
    out = routing.decisions_to_arm(decisions, rows, np.array(["amber", "blue", "teal"]))
    assert out.tolist() == [[0.8, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]  # a row never answered is failed


# ------------------------------------------------------------------ pilot


def test_pilot_runs_20_validation_rows_on_jev_and_small_llm_under_both_rules_and_prints_no_scores(tmp_path, capsys):
    val_texts = synthetic_texts(45)
    jev, jev_t = fake_client(tmp_path, lambda body, n: jev_reply(body), "jev", stage="stage0")
    llm, llm_t = fake_client(tmp_path, lambda body, n: llm_reply(body), "llm", stage="stage0")
    results_dir = tmp_path / "runs" / "private" / "routing_pilot"
    summary = routing.run_pilot(
        val_texts, jev_client=jev, llm_client=llm, llm_model="openai/gpt-oss-20b", llm_provider="Darkbloom",
        results_dir=results_dir, jev_cache=None,
    )
    assert len(jev_t.calls) == len(llm_t.calls) == 2 * routing.PILOT_ROWS == 40
    first20 = routing.sample_rows(val_texts).head(20)
    sent = {c["state"].split("\n")[0].removeprefix("Message: ") for c in jev_t.calls}
    assert sent == set(first20["text"])
    assert set(summary) == {("Jev", "A"), ("Jev", "B"), ("small LLM", "A"), ("small LLM", "B")}
    for part in summary.values():
        assert part["status"] == "complete" and part["done"] == 20 and part["failed"] == 0
        assert set(part) == {"status", "done", "failed", "not_run", "spend_usd", "mean_input_tokens", "mean_output_tokens", "stop"}
    text = routing.format_pilot(summary)
    print(text)
    printed = capsys.readouterr().out
    assert not any(q in printed for q in routing.QUEUES)
    assert "acc" not in printed.lower() and "correct" not in printed.lower()
    # A rerun pays nothing again.
    routing.run_pilot(val_texts, jev_client=jev, llm_client=llm, llm_model="openai/gpt-oss-20b",
                      llm_provider="Darkbloom", results_dir=results_dir, jev_cache=None)
    assert len(jev_t.calls) == 40


def test_pilot_needs_ledgers_held_to_the_stage_0_cap(tmp_path):
    jev, _ = fake_client(tmp_path, lambda body, n: jev_reply(body), "jev", stage=None)
    llm, _ = fake_client(tmp_path, lambda body, n: llm_reply(body), "llm", stage="stage0")
    with pytest.raises(ValueError, match="stage0"):
        routing.run_pilot(synthetic_texts(25), jev_client=jev, llm_client=llm, llm_model="m", llm_provider="p",
                          results_dir=tmp_path / "runs" / "private" / "p", jev_cache=None)
    assert routing.STAGE0_CAP_USD == 0.50


def test_pilot_command_without_paid_flag_refuses_before_loading_anything(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not load data")

    monkeypatch.setattr(data, "load_train", boom)
    with pytest.raises(PaidFlagRequired):
        routing.main(["pilot", "--llm-model", "openai/gpt-oss-20b", "--llm-provider", "Darkbloom"])


# ------------------------------------------------------------------ bootstrap grouping, bank draw, validation draw


def test_bootstrap_resamples_within_the_gold_queue_of_rule_b_not_rule_a(monkeypatch):
    seen = []
    real = routing.bootstrap_contrasts

    def spy(arms, base, classes, **kw):
        seen.append(np.asarray(classes, dtype=object).copy())
        return real(arms, base, classes, **kw)

    monkeypatch.setattr(routing, "bootstrap_contrasts", spy)
    n = 200
    rng = np.random.default_rng(0)
    gold_a = rng.choice(routing.QUEUES, n)
    gold_b = np.where(rng.random(n) < 0.5, gold_a, rng.choice(routing.QUEUES, n))
    assert not np.array_equal(gold_a, gold_b)
    checks = dict.fromkeys(("agreement_in_40_60", "state_blind_at_most_70", "text_blind_at_most_70", "lr_a_at_least_70"), True)
    good = arm(np.ones(n))
    arms = all_arms(**{"small LLM": {"A": good, "B": good}})
    routing.routing_result(gold_a, gold_b, arms, checks=checks, rules_changed=False, b=50)
    assert len(seen) == 2  # Jev vs code, then the others vs Jev
    for classes in seen:
        assert np.array_equal(classes, gold_b) and not np.array_equal(classes, gold_a)


def test_intent_lr_code_uses_bank_draw_0_specifically(monkeypatch):
    intents = list(ONE_PER_FAMILY.values())
    pool = synthetic_pool(12, intents)
    rows = predict_rows(intents)
    draws = []
    real = routing.data.bank

    def spy(pool_, draw, s, *a, **k):
        draws.append((draw, s))
        return real(pool_, draw, s, *a, **k)

    monkeypatch.setattr(routing.data, "bank", spy)
    routing.intent_lr_code(pool, rows, fake_encoder, classes=intents)
    assert draws == [(0, 8)] and routing.DRAW == 0 and routing.INTENT_LR_S == 8
    # Draw 0 is not draw 1: the two banks differ on this pool, so the test would notice a different draw.
    assert real(pool, 0, 8)["row_id"].tolist() != real(pool, 1, 8)["row_id"].tolist()


def _split(per_intent_pool=40, per_intent_val=5):
    intents = list(ONE_PER_FAMILY.values())
    pool = synthetic_pool(per_intent_pool, intents)
    val = pd.DataFrame(
        [
            {"row_id": f"v{intent}{i}", "text": f"val message {i} {intent}", "label": intent}
            for intent in intents
            for i in range(per_intent_val)
        ]
    )
    val["norm"] = val["text"].map(data.normalize)
    return data.Split(val=val, pool=pool)


def test_validation_draw_samples_on_text_only_and_joins_the_label_afterwards(monkeypatch):
    split = _split()
    seen = []
    real = routing.sample_rows

    def spy(frame, n=routing.N_SAMPLE):
        seen.append((list(frame.columns), n))
        return real(frame, n)

    monkeypatch.setattr(routing, "sample_rows", spy)
    draw = routing.validation_draw(split)
    assert seen == [(["row_id", "text"], len(split.val))]  # no label column reaches sampling
    expected_order = real(split.val[["row_id", "text"]], len(split.val))["row_id"].tolist()
    assert draw["row_id"].tolist() == expected_order
    labels = split.val.set_index("row_id")["label"]
    assert draw["label"].tolist() == [labels[r] for r in draw["row_id"]]
    # The order is the same whatever the labels say.
    shuffled = split.val.assign(label=split.val["label"].iloc[::-1].to_numpy())
    assert routing.validation_draw(data.Split(val=shuffled, pool=split.pool))["row_id"].tolist() == expected_order
    assert all(f in draw.columns for f in routing.FIELDS)


def test_run_validity_trains_lr_a_on_the_pool_only_and_scores_the_validation_draw(monkeypatch):
    split = _split()
    banks, fitted = [], []
    real_fit = routing.fit_lr_a

    def spy(bank, encode):
        banks.append(bank)
        return real_fit(bank, encode)

    monkeypatch.setattr(routing, "fit_lr_a", spy)
    stats, checks = routing.run_validity(split, fake_encoder)
    assert len(banks) == 1
    pool_ids, val_ids = set(split.pool["row_id"]), set(split.val["row_id"])
    assert set(banks[0]["row_id"]) <= pool_ids and not set(banks[0]["row_id"]) & val_ids
    assert banks[0]["row_id"].tolist() == routing.lr_a_bank(split.pool)["row_id"].tolist()
    assert set(checks) == {"agreement_in_40_60", "state_blind_at_most_70", "text_blind_at_most_70", "lr_a_at_least_70"}
    assert all(isinstance(v, (bool, np.bool_)) for v in checks.values())
    assert stats == routing.validity_stats(routing.validation_draw(split))


# ------------------------------------------------------------------ gold pinned


PINNED_GOLD = [
    # (intent, family, state, rule A queue, rule B queue)
    ("change_pin", "card_problems", ("premium", "new", "email"), "blue", "coral"),
    ("top_up_by_bank_transfer_charge", "top_ups", ("personal", "active", "app"), "silver", "teal"),
    ("card_arrival", "card_ordering", ("business", "frozen", "phone"), "amber", "amber"),
    ("activate_my_card", "card_ordering", ("business", "active", "phone"), "coral", "blue"),
    ("verify_source_of_funds", "account", ("personal", "active", "email"), "teal", "silver"),
    ("get_physical_card", "card_ordering", ("business", "active", "email"), "coral", "green"),
    # The intents whose wording was fixed keep their family, hence their gold.
    ("supported_cards_and_currencies", "card_ordering", ("personal", "active", "app"), "coral", "green"),
    ("transaction_charged_twice", "payments", ("business", "new", "email"), "indigo", "indigo"),
    ("exchange_rate", "payments", ("premium", "active", "phone"), "blue", "blue"),
]


def test_gold_is_pinned_for_chosen_intents_and_states_so_wording_changes_cannot_move_it():
    for intent, family, (tier, status, channel), qa, qb in PINNED_GOLD:
        assert routing.family_of(intent) == family, intent
        row = {"label": intent, "tier": tier, "status": status, "channel": channel}
        frame = pd.DataFrame([row])
        assert routing.gold_queues(frame, "A")[0] == qa, intent
        assert routing.gold_queues(frame, "B")[0] == qb, intent
    # Family sizes are pinned too (12, 10, 6, 11, 11, 6, 12, 9 intents).
    sizes = {f: len(spec["intents"]) for f, spec in routing.load_families().items()}
    assert sizes == {"card_ordering": 12, "card_problems": 10, "security": 6, "top_ups": 11, "transfers": 11, "cash": 6, "payments": 12, "account": 9}
