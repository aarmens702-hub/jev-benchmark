"""Rule A and rule B for the routing test (DESIGN.md, "Routing test"). Frozen before test data.

A rule is written as data and applied by `route`, and the text every arm reads is generated from the
same data by src/routing.py, so the code and the text cannot drift apart.

Each queue has a list of clauses. A clause maps an input to the values that send a message to that
queue: "family" to message families (routing/families.yaml), and each state field to its values. An
input left out of a clause matches any value. Clauses are conjunctions of "is one of" tests and
nothing else, so the text needs no numbers, dates or negations. Across a rule's queues the clauses
cover every combination of family and state exactly once (checked in tests/test_routing.py), so a
message never matches two queues and the order of queues does not matter.

Rule A is the routing in force. Rule B is the changed routing:
- frozen accounts still go to amber;
- blue serves phone contacts on open accounts instead of premium customers;
- coral and green swap their card topics, and silver and teal swap their topics.

This module imports nothing, so it can be loaded on its own by file path.
"""

FAMILIES = ("card_ordering", "card_problems", "security", "top_ups", "transfers", "cash", "payments", "account")

# The state fields and their values, in the order the hash assigns them (src/routing.py, `state`).
FIELDS = {
    "tier": ("personal", "premium", "business"),
    "status": ("active", "frozen", "new"),
    "channel": ("app", "phone", "email"),
}
# How each field is named in the text arms read.
FIELD_NAMES = {"tier": "customer tier", "status": "account status", "channel": "contact channel"}

QUEUES = ("amber", "blue", "coral", "green", "indigo", "olive", "silver", "teal")

_OPEN = ("active", "new")


def _topic_queue(families, **state):
    return [{"family": tuple(families), "status": _OPEN, **state}]


RULE_A = {
    "amber": [{"status": ("frozen",)}],
    "blue": [{"status": _OPEN, "tier": ("premium",)}],
    "coral": _topic_queue(["card_ordering"], tier=("personal", "business")),
    "green": _topic_queue(["card_problems"], tier=("personal", "business")),
    "indigo": _topic_queue(["payments"], tier=("personal", "business")),
    "olive": _topic_queue(["transfers"], tier=("personal", "business")),
    "silver": _topic_queue(["top_ups", "cash"], tier=("personal", "business")),
    "teal": _topic_queue(["security", "account"], tier=("personal", "business")),
}

RULE_B = {
    "amber": [{"status": ("frozen",)}],
    "blue": [{"status": _OPEN, "channel": ("phone",)}],
    "coral": _topic_queue(["card_problems"], channel=("app", "email")),
    "green": _topic_queue(["card_ordering"], channel=("app", "email")),
    "indigo": _topic_queue(["payments"], channel=("app", "email")),
    "olive": _topic_queue(["transfers"], channel=("app", "email")),
    "silver": _topic_queue(["security", "account"], channel=("app", "email")),
    "teal": _topic_queue(["top_ups", "cash"], channel=("app", "email")),
}

RULES = {"A": RULE_A, "B": RULE_B}


def _matches(clause, family, state):
    for key, allowed in clause.items():
        value = family if key == "family" else state[key]
        if value not in allowed:
            return False
    return True


def route(rule, family, state):
    """The queue for a message of this family with this state. Raises unless exactly one queue matches."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if set(state) != set(FIELDS):
        raise ValueError(f"state must have exactly the fields {sorted(FIELDS)}, got {sorted(state)}")
    for key, value in state.items():
        if value not in FIELDS[key]:
            raise ValueError(f"{key}={value!r} is not one of {FIELDS[key]}")
    hits = [queue for queue, clauses in rule.items() if any(_matches(c, family, state) for c in clauses)]
    if len(hits) != 1:
        raise ValueError(f"family {family!r} with state {state} matches {hits}; a rule must match exactly one queue")
    return hits[0]
