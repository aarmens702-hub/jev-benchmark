# Registration: How many labels is Jev worth?

This is the pre-registration for a small benchmark of Jev, a decision model served through OpenRouter, against cheap classifiers that need no retraining or only a few labels.

## How this registration works

- The record is this public GitHub repository, https://github.com/aarmens702-hub/jev-benchmark, at the tag `freeze-v1`. The repo was private while it was built and became public at the freeze. There was no OSF registration (see Deviations).
- `manifest.json` at that tag records the repo URL and holds the SHA-256 of every frozen input file: code, tests, lock file, label text, prompts, routing rules, Stage 0's outputs, these documents, and the pinned train files. The routing rule fingerprint is recorded too. Seeds, thresholds and model ids are recorded as values, not hashed; the files that hold them are hashed. `scripts/freeze.sh --verify` recomputes everything and lists any difference.
- The tag records the commit. A commit cannot contain its own SHA, so the manifest names the repo and the public tag names the commit.
- The test loader (`src/firewall.py`) refuses to load test data unless all of these hold: the token equals the SHA-256 of `manifest.json` at the tag; HEAD is the tagged commit; the tagged manifest names this repo, has no pending hash and verifies against the tree, the code's settings and the train files; the tree is clean, with nothing hidden from git; and the public repo, read without credentials, shows the tag and `main` at that same commit. The last check only passes once the repo is public.
- The rules below are quoted from DESIGN.md and VALIDATION.md at the same tag. If anything here reads differently, those two files govern.
- Registration prevents later changes. It does not make the design blind: the analyst read prior results before setting it (see Disclosure).

## Disclosure

The analyst read prior Banking77 and AG News results before registering, including Jev at about 83% on a test sample, LR matching it near 5 labels per class, and about half of answers at confidence 1.0 (Jev 1.13, tallied from a third party's cached outputs on a mixed sample, with no text or gold). The 80% coverage, grid, bins and prediction were set after reading them. Registration prevents later changes; it does not make the design blind. Inherited choices used labels this study does not count: C=100 (1,500 training messages, 19.5 per class) and the retrieved-24 recipe (770 messages, 10 per class, on a larger bank). Jev's training data is not disclosed, so memorisation of public datasets cannot be ruled out.

AI help was used to build the code and to draft this document.

## Why

A senior ML engineer replied on LinkedIn to Aarmen's comment on a post about Jev. This study answers him with a public, pre-registered test, and publishes whatever it shows.

His claim, verbatim: "zeroshot NLI, embedding based classifiers and constrained output models already support dynamic classification without task specific retraining." His request, verbatim: "The more interesting comparison would be Jev versus optimized lightweight classifiers, measuring cost, latency, calibration, and accuracy at equivalent automation coverage. What fundamentally new capability does Jev introduce beyond those existing approaches?"

Aarmen's comment, verbatim: "I think calling it a "rebranded classifier" is a bit of an overstatement. A traditional classifier is trained on fixed labels for one task. Jev gets the options when you call it, so one model can route requests or filter out irrelevant docs without being retrained. It also gives you a confidence score, so you only send the uncertain cases up to the LLM. Most agent stacks today send every routing decision to an LLM, paying for a full reasoning call to do what is essentially a classification. Jev is built as a decision model that can make the same call for a lot less and at lower latency."

## Question

At equal coverage, how many labels per class does a light classifier need to beat Jev, when Jev gets no labels and when Jev gets the same labels?

## Primary metric

**acc@80.** Sort an arm's n rows by top probability, high to low, and keep the first 0.8n (2,464 on Banking77, 6,080 on AG News). A tie group straddling the cut counts in proportion: if k of its g rows are needed and c of the g are correct, add c*k/g correct. acc@80 = correct/0.8n. This equals the expectation over random tie-breaks, so no tie seed is used.

## Primary contrast

D_s = mean over draws r=0..9 of acc@80(embedding LR, draw r, s) minus acc@80(zero-label Jev), in points. Grid: 1, 2, 3, 5, 8, 16, 35, all. A draw is one seeded nested ordering of the training pool. At s=all, all draws are identical.

**Bootstrap.** B=10,000, seed 2026. In each replicate: (1) resample test rows with replacement within gold class, same rows for all arms; (2) resample the 10 draw indices with replacement, same indices at every s; (3) re-sort and recompute every arm's acc@80; (4) compute D_s. [L_s, U_s] = 2.5 and 97.5 percentiles. Percentiles use numpy's default linear interpolation. Gaps are rounded to 10 decimals of a point before any comparison, so an exact boundary is decided by the rule and not by float noise.

## Sequences

**LR-ahead sequence.** Test s = all, 35, 16, 8, 5, 3, 2, 1 in that order. s passes if L_s > 0 and D_s >= 2. Stop at the first failure. s* = last s that passed, or "none" if s=all fails.

**Jev-ahead sequence.** Test s = 1, 2, 3, 5, 8, 16, 35, all in that order. s passes if U_s < 0 and D_s <= -2. Stop at the first failure. s_J = last s that passed, or "none".

Both sequences use the same 2-point margin, so neither side has an easier path to a claim. Each is a fixed-sequence procedure with family-wise one-sided error at most 2.5%. A dip breaks the run, so a non-monotone result cannot produce a low crossover. Report any s below s* that passed in isolation, labelled as not a claim. For each s, also report the range across draws and how many of the 10 beat Jev.

## Verdict (Banking77 only)

Banking77 alone sets the verdict. Margin is 2 points of acc@80.

Verdict, exactly one row applies:

| s* | s_J | Verdict |
|---|---|---|
| 1, 2, 3 | any | A light classifier with s* labels per class beats zero-label Jev at 80% coverage on this 77-way task. |
| 5, 8, 16 | 1 or more | Jev leads through s_J labels per class and LR leads from s*. Jev's value on this task is limited to cold starts. |
| 5, 8, 16 | none | LR leads clearly from s*. Below that, no clear difference. |
| 35, all, none | 3 or more | Jev leads when labels are scarce, through s_J labels per class. |
| 35, all, none | none, 1, 2 | No clear winner when labels are scarce. If [L_s, U_s] is inside [-2, +2] at every s, say "equivalent within 2 points". |

"Inside [-2, +2]" includes the endpoints.

## Additions, which never change the verdict

- **His claim.** For each of NLI, embedding similarity (label descriptions, temperature 0.05) and small LLM, C = arm minus Jev (label descriptions), 98.33% CI (Bonferroni across 3, tails of 0.835% each). The arm "matches Jev" if the lower limit is above -2. Arms not run are reported as not run.
- **Jev + examples.** E = Jev + examples minus LR, both on bank draw 0, s=8. Same-rows bootstrap, 95% CI. If the lower limit is above -2: "Given the same 8 labels per class, Jev using them is at least as good as LR using them (one draw of labels)". Show LR's range over all 10 draws.
- **AG News.** Report s* and s_J by the same rules, as a check on an easy 4-way topic task. If its verdict table row differs from Banking77's: "The Banking77 result does not carry to this task." If Jev's AG News acc@80 is 96 or higher: "ceiling-limited: a 2-point lead is hard or impossible." If not run: "not run".
- **Routing.** Claims below.
- **Corrections to Aarmen's comment.** The "fixed labels" sentence is already conceded: he named zero-shot and embedding methods, which aren't fixed-label. The reply says so whatever the data shows. Also state a correction if any arm "matches Jev" or the routing test shows "Not unique to Jev". "Most agent stacks" has no evidence here and is withdrawn or sourced.

## Prediction on record

The prediction is that embedding LR beats zero-label Jev by s=8 on Banking77.

**Prediction on record.** True if and only if s* is 8 or lower on Banking77.

## Secondary measures

Risk-coverage curve. s* and s_J at 50, 70, 90 and 100% coverage; if the Banking77 verdict row changes at 70 or 90, the headline says so, and AG News s* and s_J at those coverages are reported as a table with no headline effect. Realised coverage uses each arm's threshold at the 20th percentile of its unlabelled validation scores (keeping the top 80%). Realised coverage on test at that threshold. ECE with noise floor (15 equal-width bins closed on the right, with 0 in the first bin; the floor is the mean ECE over 1,000 outcome sets drawn as Bernoulli(confidence), seed 2026; a failed row counts as gold probability 0 in NLL and as all-zero probabilities in Brier, and is left out of ECE, with the number left out reported), NLL (clip all probabilities to [0.005, 1], since Jev rounds to 2 decimals and returns 0.0), Brier, cost and latency (no headline speed ratio). Chart labels "all" with real counts (129.9 per class on average on Banking77, imbalanced; 119,852 total on AG News, about 29,960 per class) and has a total-labels axis.

## Data and splits

- **Primary, sets the verdict:** Banking77 official test, 3,080 rows. `PolyAI/banking77` at `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`.
- **Secondary check, not a replication:** AG News, `fancyzhx/ag_news` at `eb185aade064a813bc0b7f42de02595523103ca4`, official test 7,600 rows. It tests whether the pattern appears on an easy 4-way topic task and cannot confirm or overturn the Banking77 verdict. Same grid, same C=100 (not validated for AG News; disclose). No Jev + examples. No claim on out-of-scope messages.
- Label text: Banking77 `prompts/banking77_intents.v1.json` from `Madheshvivekanandan/jev-vs-cosine` at `911e91b1093622c605c10067b960c0aff660fc1d`. AG News `labels/agnews.yaml` and its question from `zhuyansen/jev-zeroshot-vs-bert` at `edbf0713583644bd3f47299fd6b104a8b2073219`. AG News descriptions are read as the literal text in the file, because a YAML flow mapping cuts each at its first comma; the upstream repo's loader does truncate them, so the post discloses this. Bare names: official Banking77 names; World, Sports, Business, Sci/Tech.
- All data arrives by `hf_hub_download` of the named train parquet at the pinned revision. Test files are not on disk until the freeze.

Splits, from VALIDATION.md:

Normalized text = NFKC, lowercase, whitespace collapsed, stripped. Row order = SHA-256 of "seed:normalized text".

- **Banking77:** seed 20261105. Validation takes the first min(10, n_c - 35) rows per class: 756 rows (75 classes x 10, `virtual_card_not_working` 6, `contactless_not_working` 0). Pool = 9,247 rows, minus any row sharing normalized text with a validation row, then repeated text removed within the pool (first in hash order kept). Checked on the real train file 2026-11-04: 756 validation rows, 9,243 unique pool rows, at least 35 per class.
- **AG News:** same seed, 200 per class = 800 rows. Pool = 119,200, same text rule.
- **Bank draws:** r=0..9, order by SHA-256("r:s-bank:normalized text") within class. s rows = first s, so draws are nested.
- **s=all:** adds validation rows back, since no model choice uses them.
- **Duplicates:** after the freeze, remove pool rows matching a test row's normalized text and move to the next in order. If a class has fewer than s rows, use what it has and record it. Only `contactless_not_working` is at this limit.
- **Other seeds:** option order 1105, bootstrap 2026.

## Arms

| Arm | Exact | Where |
|---|---|---|
| Jev | `typesafe/jev-1.14` via OpenRouter, provider TypeSafe, build `jev-1.14-20261014` | API |
| Jev + examples | Banking77 only, bank draw 0, s=8 | API |
| Zero-shot NLI | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c` at `b2730f16019076bb0009481121efbe4705e0e378` | Mac |
| Embedding similarity | `BAAI/bge-small-en-v1.5` at `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | Mac |
| Embedding LR (primary) | Those embeddings, C=100, draws 0-9 | Mac |
| Embedding LR-CV (secondary) | C from {1,10,100,1000}: stratified k-fold inside the bank with k = min(s, 5), C=100 at s=1, ties toward 100 and then the smaller C. No outside labels | Mac |
| TF-IDF + LR | Word 1-2 grams, lowercase, no stop list, min_df=1, C=100, max_iter=5000 | Mac |
| Small LLM | `mistralai/mistral-nemo` on provider DekaLLM, selected in Stage 0; integer option ids, enum schema, logprobs, `require_parameters: true` | API |

The small LLM is frozen as: model `mistralai/mistral-nemo`; provider pinned with `only: [DekaLLM]`, `allow_fallbacks: false` and `require_parameters: true`; no reasoning field sent; `max_tokens` 16; temperature 0; price US$0.018 input and US$0.030 output per million tokens on 2026-11-05. NLI stays on the large model: 5,000 timed validation pairs ran at 20.7 pairs per second, which projects to about 3.6 hours, inside the 12-hour rule.

**Scores.**

- Jev: returned probability of its choice.
- NLI: HF zero-shot semantics, multi_label=False, softmax across entailment logits, max_len 256, batch size 16. With descriptions, the hypothesis is the description sentence as written. With bare names, "This text is about {name}."
- Embedding similarity: softmax of cosine similarity at temperature 0.05, fixed with no labels here. The value was used with bge-m3 in prior work, not bge-small, so also report acc@80 at temperatures 0.01 and 0.2 and ranked by raw top cosine. These are sensitivity rows and don't change any rule.
- LR: predict_proba.
- Small LLM: top probability = exp(sum of token logprobs along the chosen id). If a token has no logprob, or the provider cannot return logprobs under the schema, the arm is "not run". Its NLL and Brier are not reported.

**Fixed settings.** Jev and small LLM option order randomized per row, seed 1105. Jev + examples: BM25Okapi k1=1.5, b=0.75 on words and adjacent word pairs; retrieve in score order, skip classes already at 4, ignore zero scores, up to 24, ties by hash; log the count per row. Require Jev's response `model` to match the 1.14 build seen in Stage 0 (`jev-1.14-20261014` at endpoint level); a change pauses the run.

**Failures.** A row with no valid answer after retries is wrong and is kept in the covered set, ranked above all other rows. Report failures per arm. 31 or more failures on Banking77, or 76 or more on AG News (about 1%), flags that arm in every table.

**Label accounting.** No choice for any arm may use validation or test labels. C, temperature, prompts, descriptions and retrieval settings are fixed.

**Bare-name runs.** Only Jev and embedding similarity, to show how much the label descriptions matter. No bare-name runs for NLI or the small LLM, for time and cost.

**Cut.** SetFit, fine-tuned transformer, open Jev-like models, cascades, CLINC150.

## Routing test

A rule-change demonstration on synthetic state, not a benchmark of real routing.

- **Frozen before test data:** map of 77 intents to 8 families, written from intent names only; 3 state fields with 3 values each; rule A and rule B as code; exactly 8 queues; the text each arm receives; the generator below. The manifest records the routing rule fingerprint.
- **Gold** = rule(family(gold intent), state). Intent labels are required.
- **Sample:** after the freeze, order official test rows by SHA-256("20261105:routing:" + normalized text) and take the first 800. State values come from the same hash. No labels are read during sampling.
- **Arms:** Jev; small LLM; NLI with state in the premise and rule conditions in the hypothesis; LR-A (embeddings plus one-hot state, trained on 35 rows per intent from the training pool with hashed state, labelled under rule A, never retrained); Intent LR + code (embedding LR at s=8, bank draw 0, predicts intent, then the rule code is applied).
- **Metric:** accuracy on all 800 rows under rule A and under rule B. acc@80 is secondary. Margin 5 points. Bootstrap as above, 95% CI. The share of rows where rule A and rule B agree is published as the floor: LR-A scoring near it is by construction, not a finding.

**Claims** ("Not unique to Jev" is computed under rule B and reported next to the result of the first claim; "Otherwise" means neither claim holds; the bootstrap resamples within the gold queue of the rule compared):

- "Jev follows a changed rule from text with no labels" only if Jev minus Intent LR + code has lower limit above -5 under rule B and Jev's accuracy falls by 5 points or less from rule A to rule B.
- "Not unique to Jev" if small LLM or NLI minus Jev has lower limit above -5.
- Otherwise: "no new capability shown here".
- Say that an engineer can also get a rule change by editing code, at the cost of intent labels.

**Validation draw results, recorded before the freeze:** rule agreement 49.6%; state-blind best guess 44.3% and 44.8%; text-blind best guess 68.5% and 69.1% in sample (about 63% cross-fitted); LR-A rule-A floor PASS. About 55% of rows are routed by state alone, so this test mostly measures whether an arm follows a text rule over categorical state, with topic understanding tested on under half the rows; Intent LR + code has a small edge on Banking77's confusable intents. The post says both. Queue names are neutral colours, so no arm can route from a name's meaning (compare Sun et al. 2026, arXiv 2609.26758, on decision models following option names).

**Scope:** rules avoid numbers, dates and double negatives, so the result says nothing about rules that need them. Say why in the post.

**Invalid, and reported as such, if:** rules or text change after any arm output on the 800; agreement between rules is outside 40-60% on a validation draw; state-blind or text-blind best guess exceeds 70% on a validation draw; LR-A scores below 70% under rule A on validation.

**Build order:** routing is built last. If it comes out invalid or isn't ready, it is reported that way and does not hold up the Banking77 result.

## Caveats and invalid conditions

Caveats stated now, before any test data:

- Disclosure above: the design was set after reading prior results, and inherited choices used labels this study does not count.
- Jev's training data is not disclosed, so memorisation of public datasets cannot be ruled out.
- AG News is a check on an easy 4-way task, not a replication. C=100 was not validated for AG News.
- AG News descriptions are the literal text in the file, which differs from what the upstream loader reads.
- The embedding temperature of 0.05 comes from prior work with a different embedder; sensitivity rows are reported and change no rule.
- Repeatability on validation was 98.0% label agreement over 3 calls on 100 rows, right at the 98% line, so no run-noise disclosure is required, but Jev's run-to-run noise is not in the confidence intervals.
- The routing test is mostly decided by state, as described above, and says nothing about rules with numbers, dates or negation, or about real traffic.
- Full 77-way Jev probability vectors are not published, so full multiclass Brier can't be recomputed from public files.

From VALIDATION.md, "What stops the project":

- **Stop:** Jev can't return usable top probabilities via OpenRouter. The primary metric would not exist. Use the direct API only by a logged decision, before the freeze.
- **Stop and re-decide coverage before the freeze, logged:** Jev's 1.0 mass on Banking77 validation is 75% or higher.
- **Stop and disclose:** any test text or gold is read before the freeze. The study can no longer be called pre-registered for that dataset.
- **Pause:** `model` changes mid-run, or failures pass 1%. Log in the deviations table. Do not rerun selectively.
- **Halt spending:** projected spend passes the cap. Mark the rest as not run.
- **Never:** change anything frozen after Stage 1 starts without a deviations row and a statement in the post.

**Test firewall.** See VALIDATION.md. Test files are not on disk until the freeze. All predictions are written before any gold label is read. Once the test stage starts, stopping cannot depend on results, and every skipped run goes in the deviations table with its reason. `src/firewall.py` is the only module that may load test data, under the conditions listed at the top of this document. The scorer runs once and writes `scored.lock`; rerunning it needs a new deviations row.

## Budget

Hard cap CA$10 (about US$7.15 at 1.3994), everything included, enforced by a limit on the key and in code. Stage 0, before the freeze and on validation rows only, was capped at US$0.50 and spent US$0.146. A full run of the Stage 1 pipeline on validation rows only, ending in the blind dry run, cost US$0.27 more before the freeze, by the ledger (see deviations.md). Projected Stage 1 spend from measured tokens is US$1.60 with the 5.5% credit fee (about CA$2.23), routing excluded.

- **Stage 1, after freeze, approved all at once, in order:** Jev Banking77; Jev AG News; Jev + examples; routing; small LLM Banking77; small LLM AG News; Jev bare-name runs.

Stop if projected spend passes the cap and mark the rest as not run. Gold is not read until the stage ends.

## Stage 0 result (2026-11-05)

PASS, validation only. Jev 1.14 build jev-1.14-20261014: 0 failures on 756 Banking77 and 800 AG News rows; confidence 1.0 on 41.1% and 67.1%; 98.0% label agreement over 3 calls on 100 rows. Small LLM selected: mistralai/mistral-nemo on DekaLLM (49 of 50 usable, about 10 output tokens, no reasoning). Usage posts in about 24 s; reconcile grace 120 s. Stage 0 spend US$0.146; projected Stage 1 US$1.60 with fee.

Full detail is in `stage0/stage0_report.md` and `stage0/frozen.yaml`, both hashed in the manifest. No accuracy value from validation is kept except the bug floor, shown as PASS.

## Deviations so far

All four were decided and written into DESIGN.md before the freeze, before any test data was on disk. They are rows in `deviations.md` with Stage "pre-freeze". A fifth row records the pre-freeze validation run of the full pipeline, which used no test data.

1. **Small LLM candidates replaced.** The first candidates, openai/gpt-oss-20b on Darkbloom and z-ai/glm-5.3-flash on Wafer, returned HTTP 400 "Reasoning is mandatory for this endpoint and cannot be disabled" in Stage 0, so neither could run with reasoning off. They were replaced by mistralai/mistral-nemo on DekaLLM, mistralai/mistral-small-3.2-24b-instruct on Parasail and meta-llama/llama-3.1-8b-instruct on CoreWeave, tried cheapest first with no reasoning field. Selection used no labels and picked mistralai/mistral-nemo on DekaLLM.
2. **LR-CV folds.** Embedding LR-CV picks C by stratified k-fold inside the bank with k = min(s, 5), not leave-one-out, because leave-one-out could take many hours on the Mac. It is a secondary arm, and it uses no outside labels either way.
3. **AG News descriptions read literally.** The descriptions are the literal text in the file, because a YAML flow mapping cuts each at its first comma. The upstream loader does cut them, so this study's AG News label text differs from the upstream repo's.
4. **Registration by public GitHub repo instead of OSF.** Aarmen's decision on 2026-11-05. The registration is this repo made public at the tag `freeze-v1`, with the manifest at the tag recording the repo URL and the public tag recording the commit.

Any later change to a frozen input gets a row in `deviations.md` and a statement in the post.

## Reproducing it

- **Result of record:** the single frozen Stage 1 run. It is not re-run on paid arms; that would double spend and past the cap at 3x.
- **Public replication:** one command recomputes all tables from published predictions (predicted label, top probability, gold probability, model id, usage). A second command re-runs paid arms with the reader's own key. Full multiclass Brier can't be recomputed from public files, because 77-way Jev vectors aren't published; the post says so.
