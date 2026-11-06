# Routing rule tuning log

DESIGN.md allows rule text to be tuned only to meet the generator checks on a validation draw, using
no arm output except LR-A's rule-A floor. Every change is logged here, before the freeze. After any arm
output on the 800 test rows, no change is allowed: the run would be invalid.

Validation draw: the 756 Banking77 validation rows (VALIDATION.md, Splits), put in routing-hash order
by `sample_rows` with state from the same hash, then joined to their gold intents. Command:
`uv run python src/routing.py validity`.

| Date | Change | Why | Checks after |
|---|---|---|---|
| 2026-11-05 | First version: 8 families from intent names; rule A (frozen to amber, premium on open accounts to blue, topics to the other six queues); rule B (frozen to amber, phone on open accounts to blue, coral and green swap card topics, silver and teal swap topics). | Initial design. | Not yet run. |
| 2026-11-05 | Family text split into a short topic and examples in brackets, so a queue that takes two families reads unambiguously. "Out of date" reworded as "slow to update". | Text checks in tests/test_routing.py: "date" is on the date-word list. Made before any validity run; no data was used. | Not yet run. |
| 2026-11-05 | None. First validity run: agreement 49.60%; state-blind 44.31% (A), 44.84% (B); text-blind 68.52% (A), 69.05% (B); LR-A rule-A floor PASS. | All checks pass, so nothing was tuned. | All PASS. |

Text-blind best guess is close to the 70% limit, because frozen accounts and the blue condition route
more than half the rows by state alone. It passes on the validation draw, which is where DESIGN.md
applies the check. Rules were not tuned to widen the margin, since tuning is allowed only to meet the checks.
