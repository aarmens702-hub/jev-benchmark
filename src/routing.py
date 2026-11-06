"""The routing rule-change test (DESIGN.md, "Routing test"): a rule-change demonstration on synthetic
state, not a benchmark of real routing.

    uv run python src/routing.py validity        # generator checks on the validation draw
    uv run python src/routing.py write-text      # regenerate routing/arm_text.md from the rule code
    uv run python src/routing.py pilot --paid stage0 --llm-model M --llm-provider P   # controller only

Pieces:
- routing/families.yaml maps the 77 intents to 8 families. routing/rules.py holds rule A and rule B as
  data plus `route`. Every text an arm reads is generated here from those two files.
- `sample_rows` orders rows by SHA-256("20261105:routing:" + normalized text) and keeps the first 800.
  `state` reads the 3 state fields from the same hash. Neither sees a label.
- `gold(row, rule)` is the rule applied to the family of the row's gold intent and its state.
- Arms: LR-A (embeddings plus one-hot state, rule A labels, never retrained), Intent LR + code
  (intent LR at s=8, draw 0, then the rule code), and adapters for Jev, the small LLM and NLI.
- `validity` holds the generator checks; `routing_result` computes the claims and the invalid conditions.

Validation output: the generator statistics (rule agreement and blind best guesses) come from gold
alone and are printed. LR-A's floor is the only arm output allowed there, and it prints PASS or FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

import arms_local
import data
from metrics import acc_at, bootstrap_contrasts

REPO = Path(__file__).resolve().parents[1]
ROUTING_DIR = REPO / "routing"
# Files the fingerprint hashes from disk (relative to the repo root): the rule code, the family map, and the
# request builders for the text-rule arms. The arm text itself is hashed live from render_arm_text().
FROZEN_FILES = ("routing/rules.py", "routing/families.yaml", "src/jev_client.py", "src/llm_client.py")


def _load_rules():
    """routing/rules.py by file path, so it never competes with this module's name on sys.path."""
    spec = importlib.util.spec_from_file_location("routing_rules", ROUTING_DIR / "rules.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rules = _load_rules()
FIELDS: dict = rules.FIELDS
QUEUES: tuple = rules.QUEUES
RULES: dict = rules.RULES

PREFIX = "20261105:routing:"
N_SAMPLE = 800
MARGIN = 5
LEVEL = 95.0
AGREEMENT_RANGE = (40, 60)  # inclusive, in points
BLIND_LIMIT = 70  # a best guess above this is invalid
LR_A_FLOOR = 70  # LR-A below this under rule A is invalid
LR_A_S = 35
INTENT_LR_S = 8
DRAW = 0
B = 10_000
SEED = 2026
SECONDARY_COVERAGE = 0.8
PILOT_ROWS = 20
STAGE0_CAP_USD = 0.50
# Projected cost per row, so a paid run can be refused before it sends. The ladder then measures it.
# Generous for 8 short options: a Banking77 Jev call (77 options, 2,376 tokens) was about US$0.0001.
PILOT_EXPECTED_PER_ROW = {"Jev": 0.0005, "small LLM": 0.0002}

ARMS = ("Jev", "small LLM", "NLI", "LR-A", "Intent LR + code")
NOT_RUN = "not run"

# Copied from DESIGN.md, "Routing test". tests/test_routing.py checks each against the file.
CLAIM_FOLLOWS = "Jev follows a changed rule from text with no labels"
CLAIM_NOT_UNIQUE = "Not unique to Jev"
CLAIM_NONE = "no new capability shown here"
ENGINEER_NOTE = "an engineer can also get a rule change by editing code, at the cost of intent labels."
SCOPE = "rules avoid numbers, dates and double negatives, so the result says nothing about rules that need them."
FLOOR_NOTE = (
    "The share of rows where rule A and rule B agree is published as the floor: LR-A scoring near it is by "
    "construction, not a finding."
)
INVALID = {
    "rules_changed": "rules or text change after any arm output on the 800",
    "agreement": "agreement between rules is outside 40-60% on a validation draw",
    "blind": "state-blind or text-blind best guess exceeds 70% on a validation draw",
    "lr_a_floor": "LR-A scores below 70% under rule A on validation",
}

# The instructions each text-reading arm receives. Part of the frozen rule text (routing/arm_text.md).
INSTRUCTIONS = (
    "Choose the queue for this customer message. Each queue says when a message goes there, using the topic "
    "of the message and the customer tier, account status and contact channel given with it. Each message "
    "fits the condition of a single queue."
)
LLM_INSTRUCTIONS = INSTRUCTIONS + " Answer with the id of that queue."


def _points(fraction: float) -> float:
    return float(np.round(100 * fraction, 10))


def _check_rule(rule: str) -> dict:
    if rule not in RULES:
        raise ValueError(f"rule must be one of {sorted(RULES)}, got {rule!r}")
    return RULES[rule]


# --------------------------------------------------------------------------- families


@lru_cache(maxsize=1)
def _families_cached(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_families() -> dict:
    """{family: {"topic", "examples", "reason", "intents"}}, in the file's order."""
    return _families_cached(str(ROUTING_DIR / "families.yaml"))


def family_of(intent: str) -> str:
    """The family of a Banking77 intent. Raises KeyError for an unknown intent."""
    return _intent_to_family()[intent]


@lru_cache(maxsize=1)
def _intent_to_family() -> dict:
    return {intent: family for family, spec in load_families().items() for intent in spec["intents"]}


# --------------------------------------------------------------------------- sample and state


def routing_key(norm: str) -> str:
    """SHA-256("20261105:routing:" + normalized text), as hex. Sets both the sample order and the state."""
    return hashlib.sha256((PREFIX + norm).encode()).hexdigest()


def state(norm: str) -> dict:
    """The 3 state fields from the routing hash, read as an integer in base 3 (tier, then status, then channel)."""
    n = int(routing_key(norm), 16)
    out = {}
    for field, values in FIELDS.items():
        n, digit = divmod(n, len(values))
        out[field] = values[digit]
    return out


def _text_only(frame: pd.DataFrame) -> pd.DataFrame:
    extra = set(frame.columns) - {"row_id", "text"}
    if extra or not {"row_id", "text"} <= set(frame.columns):
        raise ValueError(f"routing sampling takes text only (row_id, text); got columns {sorted(frame.columns)}")
    return frame[["row_id", "text"]]


def with_state(texts: pd.DataFrame) -> pd.DataFrame:
    """(row_id, text) plus norm, the routing key and the 3 state fields. Reads no label."""
    out = _text_only(texts).copy()
    out["norm"] = out["text"].map(data.normalize)
    out["key"] = out["norm"].map(routing_key)
    states = [state(n) for n in out["norm"]]
    for field in FIELDS:
        out[field] = [s[field] for s in states]
    return out


def sample_rows(test_texts: pd.DataFrame, n: int = N_SAMPLE) -> pd.DataFrame:
    """The first n rows by routing hash (ties, which only identical texts can make, by row_id), with state.

    Takes (row_id, text) only and refuses a frame with any other column, so no label can be read here.
    """
    keyed = with_state(test_texts)
    ordered = keyed.sort_values(["key", "row_id"], kind="stable").reset_index(drop=True)
    return ordered.head(n).reset_index(drop=True)


# --------------------------------------------------------------------------- gold


def _row_state(row: Mapping) -> dict:
    return {field: row[field] for field in FIELDS}


def gold(row: Mapping, rule: str) -> str:
    """rule(family(gold intent), state). `row` needs the gold intent as "label" and the 3 state fields."""
    return rules.route(_check_rule(rule), family_of(row["label"]), _row_state(row))


def route_family(family: str, row: Mapping, rule: str) -> str:
    return rules.route(_check_rule(rule), family, _row_state(row))


def gold_queues(frame: pd.DataFrame, rule: str) -> np.ndarray:
    return np.array([gold(row, rule) for row in frame.to_dict("records")], dtype=object)


# --------------------------------------------------------------------------- validity (generator checks)


def _best_guess(groups: Sequence, queues: np.ndarray) -> float:
    """Accuracy of always guessing the most common queue within each group, in points."""
    frame = pd.DataFrame({"group": list(groups), "queue": list(queues)})
    return _points(frame.groupby(["group", "queue"]).size().groupby(level="group").max().sum() / len(frame))


def validity_stats(val_draw: pd.DataFrame) -> dict:
    """Rule agreement and the blind best guesses under each rule, in points. Gold only, no model output.

    State-blind knows the message's family but not its state. Text-blind knows the state but not the family.
    """
    gold_by_rule = {r: gold_queues(val_draw, r) for r in RULES}
    families = [family_of(i) for i in val_draw["label"]]
    states = ["|".join(values) for values in zip(*(val_draw[f] for f in FIELDS))]
    out = {"agreement": _points(np.mean(gold_by_rule["A"] == gold_by_rule["B"]))}
    for r, g in gold_by_rule.items():
        out[f"state_blind_{r}"] = _best_guess(families, g)
        out[f"text_blind_{r}"] = _best_guess(states, g)
    return out


def in_agreement_range(points: float) -> bool:
    low, high = AGREEMENT_RANGE
    return low <= points <= high


def blind_ok(points: float) -> bool:
    return points <= BLIND_LIMIT


def lr_a_floor_ok(pred, gold_a) -> bool:
    """True if LR-A's rule-A accuracy is at least 70%. Returns only the verdict; the value goes nowhere."""
    return _points(np.mean(np.asarray(pred, dtype=object) == np.asarray(gold_a, dtype=object))) >= LR_A_FLOOR


def validity(val_draw: pd.DataFrame, lr_a_pred) -> dict[str, bool]:
    """The generator checks on a validation draw. True means the check passes.

    `lr_a_pred` is LR-A's prediction for each row of `val_draw`, the only arm output the checks use.
    """
    stats = validity_stats(val_draw)
    return {
        "agreement_in_40_60": in_agreement_range(stats["agreement"]),
        "state_blind_at_most_70": all(blind_ok(stats[f"state_blind_{r}"]) for r in RULES),
        "text_blind_at_most_70": all(blind_ok(stats[f"text_blind_{r}"]) for r in RULES),
        "lr_a_at_least_70": lr_a_floor_ok(lr_a_pred, gold_queues(val_draw, "A")),
    }


def invalid_reasons(checks: Mapping[str, bool], rules_changed: bool) -> list[str]:
    """DESIGN.md's invalid conditions that hold, word for word, in DESIGN.md's order."""
    reasons = []
    if rules_changed:
        reasons.append(INVALID["rules_changed"])
    if not checks["agreement_in_40_60"]:
        reasons.append(INVALID["agreement"])
    if not (checks["state_blind_at_most_70"] and checks["text_blind_at_most_70"]):
        reasons.append(INVALID["blind"])
    if not checks["lr_a_at_least_70"]:
        reasons.append(INVALID["lr_a_floor"])
    return reasons


def rule_fingerprint(root: Path = REPO) -> str:
    """SHA-256 over the live arm text, the instructions, the rule code, the family map and the request builders.

    The arm text is `render_arm_text()` itself, not routing/arm_text.md on disk. Record it with the first arm
    output on the 800.
    """
    h = hashlib.sha256()

    def add(name: str, content: bytes) -> None:
        h.update(name.encode() + b"\0" + hashlib.sha256(content).digest())

    add("render_arm_text", render_arm_text().encode())
    add("instructions", INSTRUCTIONS.encode())
    add("llm_instructions", LLM_INSTRUCTIONS.encode())
    for name in FROZEN_FILES:
        add(name, (Path(root) / name).read_bytes())
    return h.hexdigest()


def rules_changed_since(recorded: str | None, root: Path = REPO) -> bool:
    """True if the rules or text differ from the fingerprint recorded with the first arm output. None: no output yet."""
    return recorded is not None and recorded != rule_fingerprint(root)


# --------------------------------------------------------------------------- rule text (generated from the code)


def _join(items: Sequence[str], word: str) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {word} " + items[-1]


def _clause_text(clause: Mapping) -> str:
    parts = []
    if "family" in clause:
        families = load_families()
        topics = [f"{families[f]['topic']} ({families[f]['examples']})" for f in clause["family"]]
        parts.append("the message is about " + " or about ".join(topics))
    for field in FIELDS:  # fixed order, whatever order the clause lists them in
        if field in clause:
            parts.append(f"the {rules.FIELD_NAMES[field]} is " + _join(clause[field], "or"))
    return _join(parts, "and")


def condition(rule: str, queue: str) -> str:
    """When a message goes to `queue` under `rule`, as text, generated from the clauses."""
    return "; or when ".join(_clause_text(c) for c in _check_rule(rule)[queue])


def queue_options(rule: str) -> dict[str, str]:
    """{queue: "Send here when ..."} in QUEUES order: the options Jev and the small LLM receive."""
    return {queue: f"Send here when {condition(rule, queue)}." for queue in QUEUES}


def nli_hypotheses(rule: str) -> list[str]:
    """One hypothesis per queue, in QUEUES order: the queue's condition as a sentence."""
    out = []
    for queue in QUEUES:
        text = condition(rule, queue)
        out.append(text[0].upper() + text[1:] + ".")
    return out


def message_text(row: Mapping) -> str:
    """What Jev and the small LLM get as the text: the message, then the state."""
    lines = [f"Message: {row['text']}"]
    lines += [f"{rules.FIELD_NAMES[f].capitalize()}: {row[f]}" for f in FIELDS]
    return "\n".join(lines)


def nli_premise(row: Mapping) -> str:
    """NLI premise: the state first (truncation cuts the premise from the end), then the message."""
    state_sentences = " ".join(f"The {rules.FIELD_NAMES[f]} is {row[f]}." for f in FIELDS)
    return f"{state_sentences} The message says: {row['text']}"


_EXAMPLE_ROW = {"text": "<customer message>", **{f: f"<{f}>" for f in FIELDS}}


def render_arm_text() -> str:
    """routing/arm_text.md. Generated from routing/rules.py and routing/families.yaml; a test checks the file."""
    out = [
        "# Routing test: the text each arm receives",
        "",
        "Generated by `uv run python src/routing.py write-text` from `routing/rules.py` and",
        "`routing/families.yaml`. Do not edit by hand: `tests/test_routing.py` fails if this file and the code",
        "differ. Part of the freeze.",
        "",
        "Queues: " + ", ".join(QUEUES) + ".",
        "",
        "State fields: " + "; ".join(f"{rules.FIELD_NAMES[f]}: {', '.join(v)}" for f, v in FIELDS.items()) + ".",
        "",
        "## Jev",
        "",
        "Instructions:",
        "",
        f"> {INSTRUCTIONS}",
        "",
        "Text (the `state` of the request):",
        "",
        "```",
        message_text(_EXAMPLE_ROW),
        "```",
        "",
        "Options (`criteria`): each queue with its condition under the rule, listed below. Option order is",
        "shuffled per row with seed 1105, the same order under both rules.",
        "",
        "## Small LLM",
        "",
        "System message:",
        "",
        f"> {LLM_INSTRUCTIONS}",
        "",
        "User message: `Text: ` followed by the same text as Jev, then the options as numbered lines",
        "`<id>. <queue>: <condition>`, shuffled per row with seed 1105.",
        "",
        "## NLI",
        "",
        "Premise:",
        "",
        "```",
        nli_premise(_EXAMPLE_ROW),
        "```",
        "",
        "Hypotheses: one per queue, listed below. Softmax over the entailment logits of the 8 pairs.",
        "",
    ]
    for rule in RULES:
        out += [f"## Rule {rule}", "", "Options for Jev and the small LLM:", ""]
        out += [f"- **{q}**: {text}" for q, text in queue_options(rule).items()]
        out += ["", "NLI hypotheses:", ""]
        out += [f"- **{q}**: {text}" for q, text in zip(QUEUES, nli_hypotheses(rule))]
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- local arms


def state_features(frame: pd.DataFrame) -> np.ndarray:
    """One-hot state, 9 columns: each field's values in FIELDS order."""
    cols = [(frame[f].to_numpy() == v) for f, values in FIELDS.items() for v in values]
    return np.column_stack(cols).astype(np.float32)


def lr_a_bank(pool: pd.DataFrame) -> pd.DataFrame:
    """LR-A's training rows: 35 per intent from the training pool (bank draw 0, s=35)."""
    return data.bank(pool, DRAW, LR_A_S)


def _features(frame: pd.DataFrame, encode: Callable) -> np.ndarray:
    return np.hstack([np.asarray(encode(frame["text"].tolist()), dtype=np.float32), state_features(frame)])


def lr_a_training_set(bank: pd.DataFrame, encode: Callable) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings plus one-hot hashed state, labelled with the rule A queue of each row's intent."""
    keyed = with_state(bank[["row_id", "text"]])
    keyed["label"] = bank["label"].to_numpy()
    return _features(keyed, encode), gold_queues(keyed, "A")


def fit_lr_a(bank: pd.DataFrame, encode: Callable):
    """Trained once under rule A and never retrained (so under rule B it still routes by rule A)."""
    X, y = lr_a_training_set(bank, encode)
    model, _ = arms_local.fit_lr(X, y)
    return model


def predict_lr_a(model, rows: pd.DataFrame, encode: Callable) -> tuple[np.ndarray, np.ndarray]:
    """(queue, top probability) per row. `rows` needs text and the 3 state fields, no label."""
    probs = model.predict_proba(_features(rows, encode))
    return np.asarray(model.classes_, dtype=object)[probs.argmax(axis=1)], probs.max(axis=1)


def intent_lr_code(pool: pd.DataFrame, rows: pd.DataFrame, encode: Callable, classes: Sequence[str] | None = None) -> dict:
    """Embedding LR at s=8, bank draw 0, predicts the intent; the rule code then gives the queue.

    Returns {"A": (queue, top), "B": (queue, top), "bank_rows": n}. Top probability is the intent LR's.
    """
    bank = data.bank(pool, DRAW, INTENT_LR_S)
    classes = list(classes) if classes is not None else sorted(set(pool["label"]))
    probs = arms_local.lr((encode(bank["text"].tolist()), bank["label"].to_numpy()), encode(rows["text"].tolist()), classes=classes)
    intents = np.asarray(classes, dtype=object)[probs.argmax(axis=1)]
    top = probs.max(axis=1)
    out: dict = {"bank_rows": len(bank)}
    for rule in RULES:
        queues = [route_family(family_of(i), row, rule) for i, row in zip(intents, rows.to_dict("records"))]
        out[rule] = (np.array(queues, dtype=object), top)
    return out


# --------------------------------------------------------------------------- text-rule arms


def jev_decider(rule: str, client, cache=None):
    """decide(row) for openrouter.run_arm: Jev reads the rule as its options and the state with the message."""
    from jev_client import jev_decide

    options = queue_options(rule)
    return lambda row: jev_decide(row["row_id"], message_text(row), options, client=client, cache=cache, instructions=INSTRUCTIONS)


def llm_decider(rule: str, model: str, provider: str, client):
    """decide(row) for openrouter.run_arm: the small LLM with the same text and options as Jev."""
    from llm_client import llm_pick

    options = queue_options(rule)
    return lambda row: llm_pick(row["row_id"], message_text(row), options, model, provider, client=client, instructions=LLM_INSTRUCTIONS)


def nli_arm(rows: pd.DataFrame, rule: str, scorer: Callable, batch_size: int = arms_local.NLI_BATCH_SIZE):
    """NLI with the state in the premise and the rule conditions as hypotheses. Returns (queue, top)."""
    premises = [nli_premise(row) for row in rows.to_dict("records")]
    probs = np.array(list(arms_local.nli(premises, nli_hypotheses(rule), batch_size=batch_size, scorer=scorer)))
    return np.asarray(QUEUES, dtype=object)[probs.argmax(axis=1)], probs.max(axis=1)


def decisions_to_arm(decisions: Mapping, rows: pd.DataFrame, gold_q) -> np.ndarray:
    """(top, correct, failed) per row from run_arm's decisions. A row with no decision counts as failed."""
    out = np.zeros((len(rows), 3))
    for i, (row_id, g) in enumerate(zip(rows["row_id"], gold_q)):
        d = decisions.get(row_id)
        if d is None or d.failed:
            out[i] = (0.0, 0.0, 1.0)
        else:
            out[i] = (d.top_prob, float(d.pred == g), 0.0)
    return out


def arm_array(pred, top, gold_q, failed=None) -> np.ndarray:
    """(top, correct, failed) per row for an arm that returned predictions."""
    pred = np.asarray(pred, dtype=object)
    failed = np.zeros(len(pred)) if failed is None else np.asarray(failed, dtype=float)
    return np.column_stack([np.asarray(top, dtype=float), (pred == np.asarray(gold_q, dtype=object)).astype(float), failed])


# --------------------------------------------------------------------------- claims


def _accuracies(a: np.ndarray) -> dict:
    conf, correct, failed = a[:, 0], a[:, 1] == 1, a[:, 2] == 1
    return {
        "accuracy": _points(np.mean(correct & ~failed)),
        "acc@80": _points(acc_at(conf, correct & ~failed, failed, SECONDARY_COVERAGE)),
    }


def _contrast(arms: Mapping, base: np.ndarray, classes, b: int, seed: int) -> dict:
    """Arm minus base in points on all rows, 95% CI, rows resampled within the gold queue."""
    out = bootstrap_contrasts(arms, base, classes, level=LEVEL, b=b, seed=seed, coverage=1.0)
    return {name: {"estimate": c.estimate, "lower": c.lower, "upper": c.upper, "level": LEVEL} for name, c in out.items()}


def routing_result(
    gold_a,
    gold_b,
    arms: Mapping,
    *,
    checks: Mapping[str, bool],
    rules_changed: bool,
    b: int = B,
    seed: int = SEED,
) -> dict:
    """Claims for the 800 rows (DESIGN.md, "Routing test").

    `arms` names every arm in ARMS, each {"A": array, "B": array} of (top, correct, failed) per row, or None
    if not run. `checks` comes from `validity` on the validation draw; `rules_changed` from
    `rules_changed_since`. The bootstrap resamples rows within the gold queue of the rule being compared.
    "Not unique to Jev" compares under rule B, the changed rule. An invalid run makes no claim.
    """
    if set(arms) != set(ARMS):
        raise ValueError(f"arms must name exactly {ARMS}, using None for an arm not run")
    gold_a, gold_b = np.asarray(gold_a, dtype=object), np.asarray(gold_b, dtype=object)
    out: dict = {
        "gold_A": gold_a,
        "gold_B": gold_b,
        "floor": _points(np.mean(gold_a == gold_b)),
        "accuracy": {name: NOT_RUN if a is None else {r: _accuracies(a[r]) for r in RULES} for name, a in arms.items()},
        "notes": [ENGINEER_NOTE, SCOPE, FLOOR_NOTE],
        "claims": [],
    }
    out["invalid"] = invalid_reasons(checks, rules_changed)
    if arms["Jev"] is None or arms["Intent LR + code"] is None:
        out["status"] = NOT_RUN
        return out

    jev = arms["Jev"]
    out["jev_vs_code_B"] = _contrast({"Jev": jev["B"]}, arms["Intent LR + code"]["B"], gold_b, b, seed)["Jev"]
    out["jev_drop"] = float(np.round(out["accuracy"]["Jev"]["A"]["accuracy"] - out["accuracy"]["Jev"]["B"]["accuracy"], 10))
    follows = out["jev_vs_code_B"]["lower"] > -MARGIN and out["jev_drop"] <= MARGIN

    others = {name: arms[name]["B"] for name in ("small LLM", "NLI") if arms[name] is not None}
    vs_jev = _contrast(others, jev["B"], gold_b, b, seed) if others else {}
    out["vs_jev_B"] = {name: vs_jev.get(name, NOT_RUN) for name in ("small LLM", "NLI")}
    not_unique = any(c["lower"] > -MARGIN for c in vs_jev.values())
    out["follows"], out["not_unique"] = follows, not_unique
    if not_unique:
        # "Not unique to Jev" is never reported alone: the first claim's result goes next to it.
        out["not_unique_context"] = (
            f"{CLAIM_FOLLOWS}: {'shown' if follows else 'not shown'} "
            f"(Jev minus Intent LR + code under rule B {out['jev_vs_code_B']['estimate']:.1f} points, lower limit "
            f"{out['jev_vs_code_B']['lower']:.1f}; Jev falls {out['jev_drop']:.1f} points from rule A to rule B)."
        )

    if out["invalid"]:
        out["status"] = "invalid"
        return out
    out["status"] = "valid"
    if follows:
        out["claims"].append(CLAIM_FOLLOWS)
    if not_unique:
        out["claims"].append(CLAIM_NOT_UNIQUE)
    if not out["claims"]:
        out["claims"].append(CLAIM_NONE)
    return out


# --------------------------------------------------------------------------- pilot (Stage 0)


def _require_stage0(client, name: str) -> None:
    cap = client.ledger.stage_cap_usd
    if client.ledger.stage != "stage0" or cap is None or cap > STAGE0_CAP_USD:
        raise ValueError(f"the routing pilot runs under the stage0 cap of US${STAGE0_CAP_USD}; the {name} client's ledger is not")


def run_pilot(
    val_texts: pd.DataFrame,
    *,
    jev_client,
    llm_client,
    llm_model: str,
    llm_provider: str,
    results_dir: Path,
    jev_cache=None,
) -> dict:
    """The routing pilot: the first 20 validation rows by routing hash, Jev and the small LLM, rules A and B.

    Checks schema, parsing and cost only. Takes text only, so it cannot score anything. Returns per arm and
    rule: status, rows done, failed and not run, spend and mean tokens. Results are saved under
    runs/private, so a rerun resumes without paying twice.
    """
    from openrouter import run_arm

    _require_stage0(jev_client, "Jev")
    _require_stage0(llm_client, "small LLM")
    rows = sample_rows(val_texts).head(PILOT_ROWS).to_dict("records")
    results_dir = Path(results_dir)
    summary = {}
    for arm, client in (("Jev", jev_client), ("small LLM", llm_client)):
        for rule in RULES:
            decide = (
                jev_decider(rule, client, jev_cache) if arm == "Jev" else llm_decider(rule, llm_model, llm_provider, client)
            )
            report = run_arm(
                rows,
                decide,
                client=client,
                results_path=results_dir / f"{arm.replace(' ', '_').lower()}_rule_{rule}.jsonl",
                expected_per_row=PILOT_EXPECTED_PER_ROW[arm],
            )
            done = [d for d in report.decisions.values() if not d.failed]
            summary[(arm, rule)] = {
                "status": report.status,
                "done": len(done),
                "failed": sum(d.failed for d in report.decisions.values()),
                "not_run": len(report.not_run),
                "spend_usd": float(sum(d.cost for d in report.decisions.values())),
                "mean_input_tokens": float(np.mean([d.input_tokens or 0 for d in done])) if done else None,
                "mean_output_tokens": float(np.mean([d.output_tokens or 0 for d in done])) if done else None,
                "stop": report.stop_type,
            }
    return summary


def format_pilot(summary: Mapping) -> str:
    """One line per arm and rule. Counts, spend and tokens only: no predictions, no scores."""
    lines = []
    for (arm, rule), s in summary.items():
        tokens = "n/a" if s["mean_input_tokens"] is None else f"{s['mean_input_tokens']:.0f} in / {s['mean_output_tokens']:.0f} out"
        stop = f", stopped: {s['stop']}" if s["stop"] else ""
        lines.append(
            f"{arm}, rule {rule}: {s['status']}; {s['done']} done, {s['failed']} failed, {s['not_run']} not run; "
            f"US${s['spend_usd']:.6f}; mean tokens {tokens}{stop}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- validation draw


def validation_draw(split: data.Split) -> pd.DataFrame:
    """The validation rows in routing-hash order with state, then their gold intent joined by row_id.

    Sampling sees text only; the label is attached afterwards, for the generator checks.
    """
    draw = sample_rows(split.val[["row_id", "text"]], n=len(split.val))
    labels = split.val.set_index("row_id")["label"]
    return draw.assign(label=labels.loc[draw["row_id"]].to_numpy())


def run_validity(split: data.Split, encode: Callable) -> tuple[dict, dict]:
    """Generator statistics and checks on the validation draw. LR-A is trained on the pool only."""
    draw = validation_draw(split)
    model = fit_lr_a(lr_a_bank(split.pool), encode)
    pred, _ = predict_lr_a(model, draw, encode)
    return validity_stats(draw), validity(draw, pred)


# --------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="routing", description="Routing rule-change test (DESIGN.md, Routing test).")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("validity", help="generator checks on the validation draw; LR-A prints PASS or FAIL only")
    sub.add_parser("write-text", help="regenerate routing/arm_text.md from the rule code")
    pilot = sub.add_parser("pilot", help="Stage 0 routing pilot: 20 validation rows, Jev and small LLM (paid)")
    pilot.add_argument("--paid", choices=["stage0"], default=None, help="spend real money in Stage 0; needs Aarmen's yes")
    pilot.add_argument("--llm-model", required=True)
    pilot.add_argument("--llm-provider", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "write-text":
        (ROUTING_DIR / "arm_text.md").write_text(render_arm_text())
        print("wrote routing/arm_text.md")
        return 0
    if args.command == "pilot":
        from jev_client import JevCache
        from openrouter import build_client

        # Clients first: without --paid this raises before any data is loaded.
        jev_client = build_client(args.paid)
        llm_client = build_client(args.paid)
        split = data.split(data.load_train("banking77"), "banking77")
        summary = run_pilot(
            split.val[["row_id", "text"]],
            jev_client=jev_client,
            llm_client=llm_client,
            llm_model=args.llm_model,
            llm_provider=args.llm_provider,
            results_dir=REPO / "runs" / "private" / "routing_pilot",
            jev_cache=JevCache(),
        )
        print(format_pilot(summary))
        return 0 if all(s["status"] == "complete" for s in summary.values()) else 1
    # validity
    from guards import model_job_lock

    split = data.split(data.load_train("banking77"), "banking77")
    runs = REPO / "runs"
    runs.mkdir(exist_ok=True)
    with model_job_lock(runs / "model_job.lock"):
        encode = arms_local.memoize_encoder(arms_local.load_encoder(), cache_dir=runs / "emb" / arms_local.EMBED_REVISION[:12])
        stats, checks = run_validity(split, encode)
    print(f"validation draw: {len(split.val)} rows")
    for name, value in stats.items():
        print(f"  {name}: {value:.2f}%")
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
