# How many labels is Jev worth? (design v2, 2026-11-04)

Status: revised after a builder check and a validation review. Nothing built, run or paid for. Validation rules are in [VALIDATION.md](VALIDATION.md) and stay in force for the whole build.

## Why
A senior ML engineer replied to Aarmen's comment on a LinkedIn post about Jev, asking for this comparison. Answer him with a public, pre-registered test, and publish whatever it shows. Reply after results unless Aarmen changes that.

His claim, verbatim: "zeroshot NLI, embedding based classifiers and constrained output models already support dynamic classification without task specific retraining." His request, verbatim: "The more interesting comparison would be Jev versus optimized lightweight classifiers, measuring cost, latency, calibration, and accuracy at equivalent automation coverage. What fundamentally new capability does Jev introduce beyond those existing approaches?"

Aarmen's comment, verbatim: "I think calling it a "rebranded classifier" is a bit of an overstatement. A traditional classifier is trained on fixed labels for one task. Jev gets the options when you call it, so one model can route requests or filter out irrelevant docs without being retrained. It also gives you a confidence score, so you only send the uncertain cases up to the LLM. Most agent stacks today send every routing decision to an LLM, paying for a full reasoning call to do what is essentially a classification. Jev is built as a decision model that can make the same call for a lot less and at lower latency."

## Question
At equal coverage, how many labels per class does a light classifier need to beat Jev, when Jev gets no labels and when Jev gets the same labels?

## Metric
**acc@80.** Sort an arm's n rows by top probability, high to low, and keep the first 0.8n (2,464 on Banking77, 6,080 on AG News). A tie group straddling the cut counts in proportion: if k of its g rows are needed and c of the g are correct, add c*k/g correct. acc@80 = correct/0.8n. This equals the expectation over random tie-breaks, so no tie seed is used.

**Primary contrast.** D_s = mean over draws r=0..9 of acc@80(embedding LR, draw r, s) minus acc@80(zero-label Jev), in points. Grid: 1, 2, 3, 5, 8, 16, 35, all. A draw is one seeded nested ordering of the training pool. At s=all, all draws are identical.

**Bootstrap.** B=10,000, seed 2026. In each replicate: (1) resample test rows with replacement within gold class, same rows for all arms; (2) resample the 10 draw indices with replacement, same indices at every s; (3) re-sort and recompute every arm's acc@80; (4) compute D_s. [L_s, U_s] = 2.5 and 97.5 percentiles. Percentiles use numpy's default linear interpolation. Gaps are rounded to 10 decimals of a point before any comparison, so an exact boundary is decided by the rule and not by float noise.

**LR-ahead sequence.** Test s = all, 35, 16, 8, 5, 3, 2, 1 in that order. s passes if L_s > 0 and D_s >= 2. Stop at the first failure. s* = last s that passed, or "none" if s=all fails.

**Jev-ahead sequence.** Test s = 1, 2, 3, 5, 8, 16, 35, all in that order. s passes if U_s < 0 and D_s <= -2. Stop at the first failure. s_J = last s that passed, or "none".

Both sequences use the same 2-point margin, so neither side has an easier path to a claim. Each is a fixed-sequence procedure with family-wise one-sided error at most 2.5%. A dip breaks the run, so a non-monotone result cannot produce a low crossover. Report any s below s* that passed in isolation, labelled as not a claim. For each s, also report the range across draws and how many of the 10 beat Jev.

**Secondary.** Risk-coverage curve. s* and s_J at 50, 70, 90 and 100% coverage; if the Banking77 verdict row changes at 70 or 90, the headline says so, and AG News s* and s_J at those coverages are reported as a table with no headline effect. Realised coverage uses each arm's threshold at the 20th percentile of its unlabelled validation scores (keeping the top 80%). Realised coverage when each arm's threshold is the 80% quantile of its unlabelled validation scores. ECE with noise floor (15 equal-width bins closed on the right, with 0 in the first bin; the floor is the mean ECE over 1,000 outcome sets drawn as Bernoulli(confidence), seed 2026; a failed row counts as gold probability 0 in NLL and as all-zero probabilities in Brier, and is left out of ECE, with the number left out reported), NLL (clip all probabilities to [0.005, 1], since Jev rounds to 2 decimals and returns 0.0), Brier, cost and latency (no headline speed ratio). Chart labels "all" with real counts (129.9 per class on average on Banking77, imbalanced; 119,852 total on AG News, about 29,960 per class) and has a total-labels axis.

## Data
- **Primary, sets the verdict:** Banking77 official test, 3,080 rows. `PolyAI/banking77` at `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`.
- **Secondary check, not a replication:** AG News, `fancyzhx/ag_news` at `eb185aade064a813bc0b7f42de02595523103ca4`, official test 7,600 rows. It tests whether the pattern appears on an easy 4-way topic task and cannot confirm or overturn the Banking77 verdict. Same grid, same C=100 (not validated for AG News; disclose). No Jev + examples. No claim on out-of-scope messages.
- Label text: Banking77 `prompts/banking77_intents.v1.json` from `Madheshvivekanandan/jev-vs-cosine` at `911e91b1093622c605c10067b960c0aff660fc1d`. AG News `labels/agnews.yaml` and its question from `zhuyansen/jev-zeroshot-vs-bert` at `edbf0713583644bd3f47299fd6b104a8b2073219`. AG News descriptions are read as the literal text in the file, because a YAML flow mapping cuts each at its first comma; the upstream repo's loader does truncate them, so the post discloses this. Bare names: official Banking77 names; World, Sports, Business, Sci/Tech.
- Splits and duplicates: see VALIDATION.md.

## Arms
| Arm | Exact | Where |
|---|---|---|
| Jev | `typesafe/jev-1.14` via OpenRouter | API |
| Jev + examples | Banking77 only, bank draw 0, s=8 | API |
| Zero-shot NLI | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c` at `b2730f16019076bb0009481121efbe4705e0e378` | Mac |
| Embedding similarity | `BAAI/bge-small-en-v1.5` at `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | Mac |
| Embedding LR (primary) | Those embeddings, C=100, draws 0-9 | Mac |
| Embedding LR-CV (secondary) | C from {1,10,100,1000}: stratified k-fold inside the bank with k = min(s, 5), C=100 at s=1, ties toward 100 and then the smaller C. No outside labels | Mac |
| TF-IDF + LR | Word 1-2 grams, lowercase, no stop list, min_df=1, C=100, max_iter=5000 | Mac |
| Small LLM | First passing of `mistralai/mistral-nemo`, `mistralai/mistral-small-3.2-24b-instruct`, `meta-llama/llama-3.1-8b-instruct` (see selection); integer option ids, enum schema, logprobs, `require_parameters: true` | API |

**Scores.**
- Jev: returned probability of its choice.
- NLI: HF zero-shot semantics, multi_label=False, softmax across entailment logits, max_len 256, batch size 16. With descriptions, the hypothesis is the description sentence as written. With bare names, "This text is about {name}."
- Embedding similarity: softmax of cosine similarity at temperature 0.05, fixed with no labels here. The value was used with bge-m3 in prior work, not bge-small, so also report acc@80 at temperatures 0.01 and 0.2 and ranked by raw top cosine. These are sensitivity rows and don't change any rule.
- LR: predict_proba.
- Small LLM: top probability = exp(sum of token logprobs along the chosen id). If a token has no logprob, or the provider cannot return logprobs under the schema, the arm is "not run". Its NLL and Brier are not reported.

**Fixed settings.** Jev and small LLM option order randomized per row, seed 1105. Jev + examples: BM25Okapi k1=1.5, b=0.75 on words and adjacent word pairs; retrieve in score order, skip classes already at 4, ignore zero scores, up to 24, ties by hash; log the count per row. Require Jev's response `model` to match the 1.14 build seen in Stage 0 (`jev-1.14-20261014` at endpoint level); a change pauses the run.

**Failures.** A row with no valid answer after retries is wrong and is kept in the covered set, ranked above all other rows. Report failures per arm. 31 or more failures on Banking77, or 76 or more on AG News (about 1%), flags that arm in every table.

**Small LLM selection uses no labels.** Candidates are model-and-provider pairs with no built-in reasoning that list logprobs, top_logprobs and structured outputs, tried cheapest first: mistralai/mistral-nemo on DekaLLM ($0.018/$0.030 per M on 2026-11-05), then mistralai/mistral-small-3.2-24b-instruct on Parasail ($0.09/$0.30), then meta-llama/llama-3.1-8b-instruct on CoreWeave ($0.22/$0.22). Deviation, logged before the freeze: Stage 0 on 2026-11-05 found that gpt-oss-20b and glm-5.3-flash return HTTP 400 "Reasoning is mandatory for this endpoint and cannot be disabled", so neither can run with reasoning off; requests to the new candidates send no reasoning field. Call each with one named provider and `allow_fallbacks: false`, so a run can't mix providers or precision levels. On 50 validation rows, a candidate passes if it returns valid schema output with usable logprobs for 49 or more. Pick the first that passes and freeze the model, provider and any tag. If none passes, the arm is not run, and the post says his "constrained output models" claim went untested. Accuracy is not computed.

**Label accounting.** No choice for any arm may use validation or test labels. C, temperature, prompts, descriptions and retrieval settings are fixed.

**Bare-name runs.** Only Jev and embedding similarity, to show how much the label descriptions matter. No bare-name runs for NLI or the small LLM, for time and cost.

**Cut.** SetFit, fine-tuned transformer, open Jev-like models, cascades, CLINC150.

## Outcome rules (frozen before test data)
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

Additions, which never change the verdict:
- **His claim.** For each of NLI, embedding similarity (label descriptions, temperature 0.05) and small LLM, C = arm minus Jev (label descriptions), 98.33% CI (Bonferroni across 3, tails of 0.835% each). The arm "matches Jev" if the lower limit is above -2. Arms not run are reported as not run.
- **Jev + examples.** E = Jev + examples minus LR, both on bank draw 0, s=8. Same-rows bootstrap, 95% CI. If the lower limit is above -2: "Given the same 8 labels per class, Jev using them is at least as good as LR using them (one draw of labels)". Show LR's range over all 10 draws.
- **AG News.** Report s* and s_J by the same rules, as a check on an easy 4-way topic task. If its verdict table row differs from Banking77's: "The Banking77 result does not carry to this task." If Jev's AG News acc@80 is 96 or higher: "ceiling-limited: a 2-point lead is hard or impossible." If not run: "not run".
- **Routing.** Claims below.
- **Prediction on record.** True if and only if s* is 8 or lower on Banking77.
- **Corrections to Aarmen's comment.** The "fixed labels" sentence is already conceded: he named zero-shot and embedding methods, which aren't fixed-label. The reply says so whatever the data shows. Also state a correction if any arm "matches Jev" or the routing test shows "Not unique to Jev". "Most agent stacks" has no evidence here and is withdrawn or sourced.

## Routing test (a rule-change demonstration on synthetic state, not a benchmark of real routing)
Frozen before test data: map of 77 intents to 8 families, written from intent names only; 3 state fields with 3 values each; rule A and rule B as code; exactly 8 queues; the text each arm receives; the generator below.

Gold = rule(family(gold intent), state). Intent labels are required.

Sample: after the freeze, order official test rows by SHA-256("20261105:routing:" + normalized text) and take the first 800. State values come from the same hash. No labels are read during sampling.

Arms: Jev; small LLM; NLI with state in the premise and rule conditions in the hypothesis; LR-A (embeddings plus one-hot state, trained on 35 rows per intent from the training pool with hashed state, labelled under rule A, never retrained); Intent LR + code (embedding LR at s=8, bank draw 0, predicts intent, then the rule code is applied).

Metric: accuracy on all 800 rows under rule A and under rule B. acc@80 is secondary. Margin 5 points. Bootstrap as above, 95% CI. The share of rows where rule A and rule B agree is published as the floor: LR-A scoring near it is by construction, not a finding.

Claims ("Not unique to Jev" is computed under rule B and reported next to the result of the first claim; "Otherwise" means neither claim holds; the bootstrap resamples within the gold queue of the rule compared):
- "Jev follows a changed rule from text with no labels" only if Jev minus Intent LR + code has lower limit above -5 under rule B and Jev's accuracy falls by 5 points or less from rule A to rule B.
- "Not unique to Jev" if small LLM or NLI minus Jev has lower limit above -5.
- Otherwise: "no new capability shown here".
- Say that an engineer can also get a rule change by editing code, at the cost of intent labels.

Validation draw results, recorded before the freeze: rule agreement 49.6%; state-blind best guess 44.3% and 44.8%; text-blind best guess 68.5% and 69.1% in sample (about 63% cross-fitted); LR-A rule-A floor PASS. About 55% of rows are routed by state alone, so this test mostly measures whether an arm follows a text rule over categorical state, with topic understanding tested on under half the rows; Intent LR + code has a small edge on Banking77's confusable intents. The post says both. Queue names are neutral colours, so no arm can route from a name's meaning (compare Sun et al. 2026, arXiv 2609.26758, on decision models following option names).

Scope: rules avoid numbers, dates and double negatives, so the result says nothing about rules that need them. Say why in the post.

Invalid, and reported as such, if: rules or text change after any arm output on the 800; agreement between rules is outside 40-60% on a validation draw; state-blind or text-blind best guess exceeds 70% on a validation draw; LR-A scores below 70% under rule A on validation.

Build order: routing is built last. If it comes out invalid or isn't ready, it is reported that way and does not hold up the Banking77 result.

## Freeze
Code, lock file, model ids and pins, label text, prompts, seeds, grid, coverage, tie and failure rules, bootstrap code, outcome rules, routing rules. Register on OSF with the git tag, SHA-256 manifest and the frozen archive, or make the repo public at that moment. Needs Aarmen's yes. Keep a deviations table.

**Disclosure, in the registration and the post:** the analyst read prior Banking77 and AG News results before registering, including Jev at about 83% on a test sample, LR matching it near 5 labels per class, and about half of answers at confidence 1.0 (Jev 1.13, tallied from a third party's cached outputs on a mixed sample, with no text or gold). The 80% coverage, grid, bins and prediction were set after reading them. Registration prevents later changes; it does not make the design blind. Inherited choices used labels this study does not count: C=100 (1,500 training messages, 19.5 per class) and the retrieved-24 recipe (770 messages, 10 per class, on a larger bank). Jev's training data is not disclosed, so memorisation of public datasets cannot be ruled out.

**Test firewall.** See VALIDATION.md. Test files are not on disk until the freeze. All predictions are written before any gold label is read. Once the test stage starts, stopping cannot depend on results, and every skipped run goes in the deviations table with its reason.

## Reuse
MIT, copy named files at pinned SHAs with notice and credit, never clone into the project, test each piece before the freeze:
- Madheshvivekanandan/jev-vs-cosine at `911e91b`: label descriptions, LR, embedder, Jev client. Its committed `.cache/jev_decisions.jsonl` must not be copied or published.
- zhuyansen/jev-zeroshot-vs-bert at `edbf071`: NLI and embedding arms, curves, metrics, AG News labels.
- ickma2311/jev-baselines-eval at `3b828cb68992c00835b3f126a3d13534da875277`: latency scripts.
- tapesymbolstate/jev-calibration-audit at `daab9e2c2d5d5683bf07f3482deb653c26856219` (`jujumilk3` redirects here): calibration metrics reviewed as prior work; none copied, ECE, NLL and Brier are in `src/metrics.py`.

No license, method and citation only, and do not clone: simonmesmith/jev-banking77-experiment (it holds test labels and predictions), 4esv/jev-eval, mAliAytekin/banking77.

## Budget
Hard cap CA$10 (about US$7.15 at 1.3994), everything included, enforced by a limit on the key. Explicit yes before each stage.

Credits: US$5 bought and key limit set by Aarmen on 2026-11-04. Top up only if Stage 0's measured numbers require it, and never past US$7.15.

Estimate from assumed tokens (one measured figure, 2,376 per Banking77 Jev call on 1.13), before the bare-name cut: Jev runs US$1.07, pilots at most US$0.50, small LLM US$0.40 (gpt-oss on Darkbloom) to US$0.96 (GLM on Wafer). Total with the 5.5% fee: about CA$2.90 to CA$3.75. Recompute from measured tokens at the end of Stage 0.

- **Stage 0, before freeze, validation rows only, at most US$0.50:** Jev smoke test, small LLM selection, Jev on validation, repeatability, routing pilot.
- **Stage 1, after freeze, approved all at once, in order:** Jev Banking77; Jev AG News; Jev + examples; routing; small LLM Banking77; small LLM AG News; Jev bare-name runs.

Stop if projected spend passes the cap and mark the rest as not run. Gold is not read until the stage ends.

## Time
Computer time estimated at 3-7 hours (267,560 NLI pairs, plus routing). Time 5,000 pairs on the Mac on validation rows at the frozen batch size, half from each dataset. If the projection exceeds 12 hours, use `MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c` at `bddf8c5411c34ac3565e16e04384fd68b2618dda`, disclosed.

## Reproducing it
- **Result of record:** the single frozen Stage 1 run. It is not re-run on paid arms; that would double spend and past the cap at 3x.
- **Aarmen's check:** from a clean copy, re-run all free arms and the scorer, recompute every table from saved predictions, and spot-check 100 rows of Jev on validation data.
- **Public replication:** one command recomputes all tables from published predictions (predicted label, top probability, gold probability, model id, usage). A second command re-runs paid arms with the reader's own key. Full multiclass Brier can't be recomputed from public files, because 77-way Jev vectors aren't published; the post says so.

## Rules
- Terms: never train a model on Jev outputs (MCA 2.3(b)). MCA 14.1 is broad, so link pricing and docs rather than copying them.
- Personal account, no employer data, and the author's employer is not discussed. Disclose AI help. Aarmen must be able to explain every line.
- Aarmen's yes needed for: each paid stage, OSF registration, public repo, each post and reply.
- Public text follows the author's writing rules.

## Stage 0 result (2026-11-05)
PASS, validation only. Jev 1.14 build jev-1.14-20261014: 0 failures on 756 Banking77 and 800 AG News rows; confidence 1.0 on 41.1% and 67.1%; 98.0% label agreement over 3 calls on 100 rows. Small LLM selected: mistralai/mistral-nemo on DekaLLM (49 of 50 usable, about 10 output tokens, no reasoning). Usage posts in about 24 s; reconcile grace 120 s. Stage 0 spend US$0.146; projected Stage 1 US$1.60 with fee. Full detail in stage0_report.md and frozen.yaml.

## Needed from Aarmen
The OpenRouter key is in the macOS Keychain under service `jev-benchmark-openrouter`, entered by Aarmen. Claude never sees or prints the value; code reads it at run time. An OSF account, created by him, before the freeze. The original post's article is useful background for the post but doesn't block the freeze.

## Deliverables
Public repo, OSF registration, about 800-word blog post on GitHub Pages with the gap-vs-labels chart, LinkedIn post, reply to him.
