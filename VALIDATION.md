# Validation protocol (in force for the whole build)

Any agent or session working on this project follows these rules. If a rule blocks the work, stop and ask Aarmen. Do not work around it.

## Splits
Normalized text = NFKC, lowercase, whitespace collapsed, stripped. Row order = SHA-256 of "seed:normalized text".

- **Banking77:** seed 20261105. Validation takes the first min(10, n_c - 35) rows per class: 756 rows (75 classes x 10, `virtual_card_not_working` 6, `contactless_not_working` 0). Pool = 9,247 rows, minus any row sharing normalized text with a validation row, then repeated text removed within the pool (first in hash order kept). Checked on the real train file 2026-11-04: 756 validation rows, 9,243 unique pool rows, at least 35 per class.
- **AG News:** same seed, 200 per class = 800 rows. Pool = 119,200, same text rule.
- **Bank draws:** r=0..9, order by SHA-256("r:s-bank:normalized text") within class. s rows = first s, so draws are nested.
- **s=all:** adds validation rows back, since no model choice uses them.
- **Duplicates:** after the freeze, remove pool rows matching a test row's normalized text and move to the next in order. If a class has fewer than s rows, use what it has and record it. Only `contactless_not_working` is at this limit.
- **Other seeds:** option order 1105, bootstrap 2026.

## Validation data may be used for
1. Schema, parsing and retry checks.
2. Token, cost and latency measurement.
3. Failure rates.
4. Jev's 1.0 mass and the score at the 80% cut.
5. Repeatability: 100 rows, 3 calls each, label agreement and median top-probability range. Below 98% agreement, disclose and say in the post that Jev has run noise not in the CI.
6. Routing generator checks, which need no model output except LR-A's rule-A floor.
7. Label-mapping sanity, printed as pass/fail only.

## Validation data may not be used for
- Choosing or editing descriptions, prompts, C, temperature, retrieval settings, the small LLM, grid, coverage, margin or bins.
- Computing or viewing any D_s or arm-vs-arm comparison.
- Printing arm accuracy values.

A failed check may lead to a **bug fix only**, logged with a reason.

## Guards
- **Fetch by file.** All data arrives by `hf_hub_download` of the named train parquet at the pinned revision. No `load_dataset`.
- **`src/firewall.py`** is the only module that may load test. `load_test(token)` raises unless the token equals SHA-256 of the frozen `manifest.json`, the manifest holds a registration (the public GitHub repo URL and the tagged commit), the git tree is clean, and HEAD is at the freeze tag.
- **`tests/test_firewall.py`** runs in CI and fails if:
  - any test parquet exists under `data/` or the HF cache before the freeze;
  - `split="test"`, `test-0000` or `load_dataset(` appears outside `firewall.py`;
  - `load_test` without a token does not raise;
  - any `.cache/` or prior results file is in the repo;
  - any validation row id appears in a bank at s of 35 or less;
  - `predict*.py` can import the gold file.
- **Reuse.** Copy named files only at pinned SHAs, never clone into the project. The Jev cache starts empty and is keyed on model build.
- **Manifest.** SHA-256 for code, lock file, descriptions, prompts, rule text, seeds, thresholds and train parquet. Test file hashes and post-freeze duplicate counts are recorded after first load.
- **One-shot scorer.** Test text and gold are split on load. Predictors see text only. `score.py` runs only when every arm has a prediction file or a logged "not run". It writes `scored.lock`, and rerunning needs a deviations row.
- **Pilot ledger.** Append-only log of every paid call and row id.

## Checks that must pass before the freeze
1. All firewall tests pass.
2. Jev through OpenRouter returns choice plus probabilities on 20 of 20 smoke rows, with `model` logged.
3. Jev failures on validation are 1% or less.
4. Jev's 1.0 mass on Banking77 validation is below 75%.
5. Banking77 validation overall accuracy is at least 60% for Jev and at least 75% for LR at s=35. Bug floors only, reported as pass/fail. Lower means a mapping bug.
6. Small LLM selection is complete, or the arm is marked not run.
7. Repeatability is measured.
8. The NLI timing rule in DESIGN.md is resolved.
9. Routing generator checks pass.
10. A dry run on validation goes end to end through the scorer, using a dummy lock, with gold permuted (seed 2026) so its output carries no real comparison. It writes only to a gitignored folder and prints pass or fail.
11. Projected spend, using measured tokens, is under the remaining cap.
12. The small LLM's provider is pinned with no fallbacks, and its price is recorded.
13. DESIGN.md matches the manifest.

## What stops the project
- **Stop:** Jev can't return usable top probabilities via OpenRouter. The primary metric would not exist. Use the direct API only by a logged decision, before the freeze.
- **Stop and re-decide coverage before the freeze, logged:** Jev's 1.0 mass on Banking77 validation is 75% or higher.
- **Stop and disclose:** any test text or gold is read before the freeze. The study can no longer be called pre-registered for that dataset.
- **Pause:** `model` changes mid-run, or failures pass 1%. Log in the deviations table. Do not rerun selectively.
- **Halt spending:** projected spend passes the cap. Mark the rest as not run.
- **Never:** change anything frozen after Stage 1 starts without a deviations row and a statement in the post.

## Still unknown until Stage 0
- Jev 1.14's tie mass and AG News behaviour. All tie evidence is from 1.13 on a free route.
- Whether OpenRouter's Jev route returns the full probability output.
- Number of test duplicates.
- Whether third parties' claims about untuned descriptions and C are true. Their history can't confirm it.
