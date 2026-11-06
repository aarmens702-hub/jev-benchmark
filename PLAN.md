# How Many Labels Is Jev Worth? Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the frozen benchmark in DESIGN.md and carry it from training data, through the Stage 0 gate and registration, to one scored test run and a publishable result.

**Architecture:** A small Python repo. Every arm writes a prediction file. A firewall is the only way to read test data, and a one-shot scorer is the only thing that reads gold. Free arms run on the Mac; paid arms go through one budgeted OpenRouter client.

**Tech Stack:** Python 3.12, uv with a lock file, numpy, pandas, pyarrow, scikit-learn, rank-bm25, sentence-transformers, transformers, torch (MPS), huggingface_hub, httpx, pyyaml, matplotlib, pytest.

**Spec:** `DESIGN.md` and `VALIDATION.md`. Executors read both. They override this plan where they conflict.

## Global Constraints
- Repo: `jev-benchmark`, private until Aarmen says yes.
- No test text or gold before the freeze. Test files aren't on disk until then. All data comes by `hf_hub_download` of named files at pinned revisions. Never `load_dataset`.
- Validation may not produce any printed arm accuracy, D_s or arm-vs-arm comparison. A failed check allows a bug fix only, logged with a reason.
- The API key comes from the macOS Keychain, service `jev-benchmark-openrouter`, read with `security` at run time. Never print, log, commit or pass it as a command-line argument.
- Aarmen's explicit yes is needed before: each paid stage, OSF registration, making the repo public, each post and reply.
- Hard budget US$7.15 in total across all stages, enforced in code. Stage 0 is at most US$0.50. Current balance is US$5, and the key's own limit is US$10, so the code's limit is the one that protects the cap.
- Copy reused code as named files at pinned SHAs into `third_party/` with license notices. Never clone. Never copy `.cache/` or result files. Code from unlicensed repos is not copied.
- Never train any model on Jev outputs. Full 77-way Jev vectors stay in gitignored `runs/private/`.
- No Claude attribution or co-author lines in commits, PRs or issues. Commit with Aarmen's GitHub noreply email.
- Public text follows the author's writing rules (plain words, no em dashes, every claim measured).
- Seeds: split 20261105, option order 1105, bootstrap 2026, routing sample prefix `20261105:routing:`.

**Spend guards (all tested in Task 5):**
- Build agents never run a paid command or read the key. Only the controller runs paid stages, after Aarmen's yes for that stage, with `--paid <stage>`. Without that flag, clients can only use a fake transport.
- Tests never spend. `read_key` raises under pytest, and network is blocked except for marked `data` tests that fetch named train files from Hugging Face.
- Every paid run climbs a ladder: 1 row, then 20, then a measured projection. It continues only if the projected total stays inside the cap and the stage limit.
- Before each paid arm, compare projected cost with the remaining budget and refuse if it doesn't fit. Count retries as spend.
- Reconcile the ledger with OpenRouter's reported usage every 100 calls. A gap over 10% stops the run.
- Stop after 10 consecutive failures. At most 3 retries per row, and at most 4 concurrent requests.
- Small LLM: `max_tokens` 16 and reasoning off, with the cost taken from reported usage. If it spends more than 16 output tokens per call on average, stop, because hidden reasoning tokens are being billed.
- Backstops outside the code: the US$5 balance and no saved payment method.

**Compute guards (tested in Tasks 1 and 4):**
- `pytest` has a 60-second timeout per test, and the default run takes under 2 minutes. Full-data and model tests are opt-in markers.
- Build agents never run full arms, the full bootstrap or NLI beyond the timing sample.
- Every long job has `--max-hours`, checkpoints, resumes, and stops itself on the limit. Fixed batch size, one model job at a time.
- Before a long job, run a small sample and project the time. NLI's limit is the 12-hour rule. Bootstrap times 100 replicates first and stops if projected above 1 hour.
- Stop a Mac job if swap grows by more than 2 GB or free disk drops below 5 GB. Check disk before model downloads, and download only the pinned models.
- Overnight jobs run under `caffeinate` so the Mac can't sleep mid-run, and they stop when finished.
- No training beyond logistic regression, capped at `max_iter=5000`, with a non-converged run logged.

## Review Focus
1. API returns a label not in the options, an empty body, or a malformed payload: the row retries, then is logged as a failure and scored wrong; nothing crashes the run.
2. A 429, a 529, or the Mac sleeping or crashing overnight: a rerun resumes from saved rows without paying twice or changing finished rows.
3. Key missing or rejected, or credits run out mid-run: a clean stop with a clear message, finished rows kept, the rest marked not run, and no key in the message.
4. OpenRouter's "about 32,000 bytes" long-context mode: a request over 30,000 bytes refuses to send, because the mode is documented as less accurate.
5. Reused code carrying stale answers: the Jev cache starts empty and is keyed on model build, so a cached answer from another build is never served.

---

### Task 1: Repo, lock file and test firewall

**Files:** Create `pyproject.toml`, `uv.lock`, `.gitignore`, `README.md`, `DESIGN.md`, `VALIDATION.md`, `scripts/` (copied from the planning folder), `src/firewall.py`, `tests/test_firewall.py`, `deviations.md`.

**Interfaces:**
- Produces: `firewall.load_test(dataset: str, token: str) -> tuple[pd.DataFrame, pd.DataFrame]` returning (row_id, text) and (row_id, label) as separate frames; `firewall.freeze_token(manifest_path: Path) -> str`; `firewall.FirewallError`.

- [ ] **Step 1: Write the tests.** `tests/test_firewall.py` fails if: a `test-*.parquet` or a matching snapshot exists under `data/` or the HF cache before the freeze; `split="test"`, `test-0000` or `load_dataset(` appears in any tracked file outside `firewall.py`, this test and the docs; `load_test` without a valid token doesn't raise `FirewallError`; a `.cache/` or prior results file is tracked; any `predict*.py` imports a gold reader. Also `test_load_test_rejects_dirty_tree`, `test_load_test_rejects_head_not_at_freeze_tag` and `test_load_test_rejects_manifest_without_osf_id`, using a temp git repo.
- [ ] **Step 2: Run them.** `uv run pytest tests/test_firewall.py -v`. Expected: FAIL, module missing.
- [ ] **Step 3: Implement `src/firewall.py`.** Token = SHA-256 of `manifest.json`. All four conditions in VALIDATION.md must hold before any download.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit.** `git init`, set the noreply email, commit `chore: repo, lock file and test firewall`.

### Task 2: Data, splits and banks

**Files:** Create `src/data.py`, `src/labels.py`, `tests/test_data.py`, `third_party/` label files, `third_party/NOTICE.md`.

**Interfaces:**
- Consumes: pinned revisions in DESIGN.md.
- Produces: `normalize(text: str) -> str`; `load_train(dataset: str) -> pd.DataFrame` (row_id, text, label, norm); `split(df, dataset) -> Split(val: DataFrame, pool: DataFrame)`; `bank(pool, draw: int, s: int | Literal["all"], val=None) -> pd.DataFrame`; `label_texts(dataset: str, bare: bool) -> dict[str, str]`.

- [ ] **Step 1: Write the tests** (marked `data`, train files only):
```python
def test_banking77_split():
    s = split(load_train("banking77"), "banking77")
    assert len(s.val) == 756 and s.pool.norm.is_unique and len(s.pool) == 9243
    assert s.pool.label.value_counts().min() >= 35

def test_ag_news_split():
    s = split(load_train("ag_news"), "ag_news")
    assert len(s.val) == 800 and len(s.pool) == 119052

def test_bank_sizes_and_nesting():
    # banking77 draw 0: 77, 154, 231, 385, 616, 1232, 2695. ag_news: 4, 8, 12, 20, 32, 64, 140.
    # For every draw and s, the bank at s is a subset of the bank at the next s, and no val row is in it.

def test_all_adds_validation_rows_back(): ...
def test_label_files_match_dataset_labels_for_both_datasets_and_bare_names(): ...
```
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Hash order and rules exactly as in VALIDATION.md. Post-freeze test-duplicate removal is `drop_test_duplicates(pool, test_norms) -> (pool, count)`, moving to the next row in order. Copy the pinned label files.
- [ ] **Step 4: Run them.** Expected: PASS. `tests/test_firewall.py` still PASS.
- [ ] **Step 5: Commit.** `feat: splits, nested banks and label text`.

### Task 3: Metrics, bootstrap and verdict

**Files:** Create `src/metrics.py`, `src/verdict.py`, `tests/test_metrics.py`, `tests/test_verdict.py`.

**Interfaces:**
- Produces: `acc_at(conf: np.ndarray, correct: np.ndarray, failed: np.ndarray, coverage: float = 0.8) -> float`; `bootstrap_ds(lr: dict[s, np.ndarray[draws, rows, 2]], jev, classes, b=10_000, seed=2026) -> DsResult`; `lr_ahead(ds) -> s | "none"`; `jev_ahead(ds) -> s | "none"`; `verdict(s_star, s_j) -> str`; `additions(...) -> dict`; `ece_with_floor`, `nll`, `brier`.

- [ ] **Step 1: Write the tests.**
```python
def test_tie_group_straddling_cut_counts_in_proportion():
    # 10 rows, keep 8. Six at 0.9 (4 correct), four tied at 0.5 (1 correct).
    # 2 of the 4 tied rows are needed: correct = 4 + 1*2/4 = 4.5, so acc = 4.5/8.
    assert acc_at(conf, correct, failed) == pytest.approx(4.5 / 8)

def test_failed_rows_rank_first_and_count_wrong(): ...
def test_lr_ahead_stops_at_first_failure_and_ignores_later_passes(): ...
def test_lr_ahead_needs_ds_at_least_2_as_well_as_lower_bound_above_0(): ...
def test_jev_ahead_needs_ds_at_most_minus_2_as_well_as_upper_bound_below_0():
    # A 0.5 point Jev lead with a tight interval must not pass.
def test_verdict_covers_every_s_star_s_j_pair_with_exactly_one_row(): ...
def test_bootstrap_resamples_within_class_and_reuses_rows_and_draws_across_s(): ...
def test_bootstrap_interval_covers_true_gap_on_synthetic_data(): ...
def test_s_all_draws_are_identical_in_bootstrap(): ...
```
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Verdict strings and additions copied verbatim from DESIGN.md. NLL clips to [0.005, 1].
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit.** `feat: acc@80, paired bootstrap and frozen verdict rules`.
- [ ] **Step 6: Validation guard review.** A fresh agent checks `metrics.py` and `verdict.py` line by line against DESIGN.md, using synthetic data only. A mistake here could quietly change the verdict.

### Task 4: Free arms

**Files:** Create `src/arms_local.py`, `src/predictions.py`, `src/predict_local.py`, `tests/test_arms_local.py`.

**Interfaces:**
- Consumes: `data.bank`, `data.label_texts`.
- Produces: `predictions.write(arm, dataset, rows)`, one parquet per arm under `runs/<stage>/`, columns `row_id, pred, top_prob, failed`, with the full vector in `runs/private/`. Arms: `embed_sim(texts, labels, temperature)`, `lr(train, test, c=100)`, `lr_cv(train, test)`, `tfidf_lr(train, test)`, `nli(texts, hypotheses, batch_size) -> iterator`.

- [ ] **Step 1: Write the tests** using toy data: `test_lr_uses_c_100_and_fixed_seeded_solver`, `test_lr_cv_uses_loo_for_s_2_to_16_5fold_for_35_and_all_c100_at_s1`, `test_embed_sim_temperature_changes_confidence_not_pick`, `test_nli_uses_description_as_hypothesis_and_template_for_bare_names`, `test_nli_multi_label_false_softmax_across_entailment`, `test_predict_resumes_from_saved_rows_without_recomputing`, `test_predictors_cannot_read_gold`, `test_long_job_stops_itself_at_max_hours_and_keeps_checkpoint`, `test_job_stops_when_swap_grows_over_2gb_or_disk_under_5gb` (with fake readings), `test_second_model_job_refuses_to_start_while_one_runs`, `test_bootstrap_projects_from_100_replicates_and_stops_above_1_hour`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Pinned model revisions. NLI checkpoints every 500 rows. Copy from zhuyansen and jev-vs-cosine where useful, with notices. Every arm also accepts rows with no labels.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Bug floor and timing, on validation.** `uv run python -m predict_local --stage val --floor` prints only `LR s=35 floor: PASS` or `FAIL`. `--time-nli` times 5,000 pairs (half per dataset) and prints pairs per second and projected hours. Apply the 12-hour rule in DESIGN.md and log it.
- [ ] **Step 6: Commit.** `feat: free arms with checkpoints and NLI timing`.

### Task 5: Budgeted API clients (no network in tests)

**Files:** Create `src/secrets_mac.py`, `src/ledger.py`, `src/openrouter.py`, `src/jev_client.py`, `src/llm_client.py`, `src/retrieval.py`, `tests/test_clients.py`.

**Interfaces:**
- Produces: `read_key() -> str`; `Ledger(path, cap_usd=7.15, stage_cap_usd)`, with `.charge(cost, meta)`, `.projected(n_rows, per_row) -> float`, raising `BudgetExceeded`; `jev_decide(row_id, text, options, order_seed=1105, examples=None) -> Decision(pred, top_prob, probs, model, cost, failed)`; `llm_pick(row_id, text, options, model, provider) -> Decision`; `retrieve(query, bank) -> list[Example]`.

- [ ] **Step 1: Read current docs.** Confirm the request and response shape for `POST https://openrouter.ai/api/alpha/decisions`, and `provider.only`, `allow_fallbacks`, `require_parameters` and logprobs. Write the shapes into `tests/fixtures/`.
- [ ] **Step 2: Write the tests** with a fake HTTP transport:
```python
def test_no_paid_flag_means_fake_transport_only(): ...
def test_read_key_raises_under_pytest(): ...
def test_ladder_runs_1_then_20_then_projects_before_continuing(): ...
def test_arm_refused_when_projected_cost_exceeds_remaining_budget(): ...
def test_retries_are_charged_to_the_ledger(): ...
def test_ledger_gap_over_10_percent_with_reported_usage_stops_the_run(): ...
def test_10_consecutive_failures_stop_the_run(): ...
def test_endless_529s_stop_at_3_retries_per_row_and_never_loop(): ...
def test_concurrency_never_exceeds_4(): ...
def test_llm_sends_max_tokens_16_and_reasoning_off(): ...
def test_llm_mean_output_above_16_tokens_stops_the_run(): ...
def test_key_never_appears_in_repr_logs_exceptions_or_ledger(): ...
def test_ledger_refuses_call_past_cap_before_sending(): ...
def test_key_rejected_or_credits_out_stops_cleanly_and_keeps_rows(): ...
def test_label_not_in_options_retries_then_marks_failed(): ...
def test_429_and_529_back_off_and_retry_and_do_not_count_as_failures_unless_final(): ...
def test_resume_does_not_pay_twice(): ...
def test_cache_is_empty_at_start_and_keyed_on_model_build(): ...
def test_model_change_mid_run_pauses(): ...
def test_option_order_is_randomised_per_row_with_seed_1105_and_mapped_back(): ...
def test_top_prob_from_exp_sum_of_token_logprobs_along_chosen_id(): ...
def test_missing_token_logprob_marks_arm_not_run(): ...
def test_provider_pinned_with_no_fallbacks_and_require_parameters(): ...
def test_request_over_30000_bytes_refuses_to_send(): ...
def test_retrieval_max_4_per_class_up_to_24_ignores_zero_scores_ties_by_hash_logs_count(): ...
```
- [ ] **Step 3: Run them.** Expected: FAIL.
- [ ] **Step 4: Implement.** BM25Okapi k1=1.5, b=0.75 on words and adjacent word pairs, written fresh from the README recipe with credit. Read the key by subprocess; never put it in an exception or a URL.
- [ ] **Step 5: Run them.** Expected: PASS.
- [ ] **Step 6: Commit.** `feat: budgeted Jev and LLM clients with a pinned provider`.

### Task 6: Stage 0 (paid, needs Aarmen's yes)

**Files:** Create `src/stage0.py`, `stage0_report.md`, `tests/test_stage0.py`.

**Interfaces:**
- Consumes: Tasks 2, 4 and 5.
- Produces: `stage0_report.md` with pass/fail per check; `frozen.yaml` values for the small LLM, its provider, the Jev build and the NLI model.

- [ ] **Step 1: Write the test.** `test_stage0_report_contains_no_accuracy_or_comparison_values` runs the report with fakes and asserts only allowed fields appear. `test_stage0_stops_at_stage_cap_of_0_50`.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement** in this order, stopping on failure: 20-row Jev smoke; 50-row selection per candidate, cheapest first; Jev on Banking77 and AG News validation; 100 rows x 3 repeatability; measured tokens and projected spend. Bug floor prints `PASS` or `FAIL`.
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Ask Aarmen.** State the projected cost in CA$. Wait for a clear yes.
- [ ] **Step 6: Run Stage 0.** `uv run python -m stage0`. Expected: report written, spend under US$0.50. Check real spend with `scripts/check_key.sh`.
- [ ] **Step 7: Apply the stop rules.** No usable probabilities, or a 1.0 mass at or above 75%: stop and ask. Other failures: bug fix only, logged.
- [ ] **Step 8: Commit.** `chore: Stage 0 report and frozen selections`. Row-level files stay in `runs/private/`.

### Task 7: One-shot scorer and blind dry run

**Files:** Create `src/score.py`, `src/report.py`, `tests/test_score.py`.

**Interfaces:**
- Consumes: `metrics`, `verdict`, prediction files.
- Produces: `score(stage, gold) -> writes results/*.json, results/*.csv, results/chart.png, scored.lock`; published predictions with `pred, top_prob, gold_prob, model, usage`.

- [ ] **Step 0: Add latency.** Clients and local arms record wall-clock seconds per decision in prediction files, with `test_latency_is_recorded_for_every_arm`. Add `src/latency_probe.py`, adapted from ickma's scripts: serial requests after warm-up, cold start and model load kept apart, a minimal-request network probe, and memory swap noted. Report p50 and p95 by condition (hardware, region, date, concurrency), with no headline speed ratio. Probes run on validation rows.
- [ ] **Step 1: Write the tests.** `test_refuses_when_an_arm_has_neither_predictions_nor_not_run`; `test_refuses_rerun_without_deviation_row`; `test_gold_read_only_inside_score`; `test_published_files_never_contain_full_vectors`; `test_chart_labels_all_with_real_counts_and_has_total_labels_axis`; `test_dry_run_permutes_gold_and_writes_to_dryrun_only`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** The dry run uses validation predictions with gold permuted (seed 2026), so its output carries no real comparison. It writes only to gitignored `runs/dryrun/`, and prints `DRY RUN OK`. All tables and rows in DESIGN.md's secondary list and additions are computed here, including embedding temperature rows and coverage at 50, 70, 90 and 100%.
- [ ] **Step 4: Run them, then the dry run.** Expected: PASS and `DRY RUN OK`.
- [ ] **Step 5: Commit.** `feat: one-shot scorer and blind dry run`.

### Task 8: Routing test (built last, can't hold up the main result)

**Files:** Create `src/routing.py`, `routing/families.yaml`, `routing/rules.py`, `routing/arm_text.md`, `tests/test_routing.py`.

**Interfaces:**
- Consumes: `data`, `arms_local`, clients.
- Produces: `gold(row, rule) -> queue`; `sample_rows(test_texts) -> first 800 by hash`; `state(norm) -> dict`; `validity(val_draw) -> dict[str, bool]`; arms `lr_a`, `intent_lr_code`, and text-rule arms.

- [ ] **Step 1: Write the tests.** `test_all_77_intents_map_to_exactly_one_of_8_families`; `test_rule_a_and_b_only_use_family_and_3_categorical_fields`; `test_rule_text_has_no_numbers_dates_or_double_negatives`; `test_sample_order_and_state_come_from_hash_and_read_no_labels`; `test_validity_flags_agreement_outside_40_60_and_blind_guess_over_70_and_lr_a_under_70`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Write the family map from intent names only. Tune rule text only to meet the validity checks, on a validation draw, with no arm output except LR-A's floor. Claims copied verbatim from DESIGN.md.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit.** `feat: routing rule-change test`.

### Task 9: Pre-freeze review, registration and freeze (needs Aarmen's yes)

**Files:** Create `src/freeze.py`, `manifest.json`, `registration.md`, `tests/test_freeze.py`.

**Interfaces:**
- Produces: `manifest.json`, git tag `freeze-v1`, and an OSF id in the manifest.

- [ ] **Step 1: Write and pass the test.** `test_manifest_hashes_code_lock_labels_prompts_rules_seeds_thresholds_and_train_parquet`; `test_manifest_detects_any_change`.
- [ ] **Step 2: Independent review.** A fresh validation-guard agent checks the repo against VALIDATION.md and DESIGN.md, including all 13 pre-freeze checks. Fix only with a logged reason, and repeat until clean.
- [ ] **Step 3: Write `registration.md`** with the design, prediction, outcome rules and disclosure from DESIGN.md.
- [ ] **Step 4: Ask Aarmen** to review `registration.md`, create an OSF account, and choose between uploading the frozen archive or making the repo public. Wait for a clear yes. He files the registration; Claude doesn't create accounts or enter credentials.
- [ ] **Step 5: Tag.** Add the OSF id, commit, tag `freeze-v1`. Expected: `git status` clean, and `load_test` works only with the token.

### Task 10: Stage 1, one run (paid, needs Aarmen's yes)

**Files:** Create `src/stage1.py`, `runs/stage1/`, `results/`.

**Interfaces:**
- Consumes: `firewall.load_test` and everything above.
- Produces: all prediction files, then `results/` and `scored.lock`.

- [ ] **Step 1: Ask Aarmen.** State spend so far and the projected total in CA$, approving all of Stage 1 at once. Wait for a clear yes.
- [ ] **Step 2: Load test data.** Record test hashes and duplicate counts in `deviations.md`. Rebuild banks with `drop_test_duplicates`.
- [ ] **Step 3: Run paid arms in the frozen order,** text only. Expected: stops only at the cap, with the rest marked not run.
- [ ] **Step 4: Run free arms overnight,** text only. Expected: all prediction files written, resuming after any interruption.
- [ ] **Step 5: Run the scorer once.** Expected: `results/` and `scored.lock`.
- [ ] **Step 6: Commit.** `results: frozen run`. Report the verdict row to Aarmen exactly as the rules state it, with every deviation.

### Task 11: Aarmen's check

- [ ] **Step 1: Write `make replicate`.** It recomputes all tables from published predictions; a second target re-runs paid arms with the reader's own key.
- [ ] **Step 2: Aarmen re-runs** free arms and the scorer from a clean copy, and spot-checks 100 Jev rows on validation data. Expected: free-arm numbers within seed noise, tables identical from saved predictions. Any gap goes in `deviations.md`.
- [ ] **Step 3: Walkthrough.** Aarmen reads the code until he can explain every line. Commit fixes to docs only; analysis changes need a deviation row.

### Task 12: Publish (each item needs Aarmen's yes)

**Files:** Create `docs/index.md`, `docs/chart.png`, `drafts/linkedin-post.md`, `drafts/reply.md`.

- [ ] **Step 1: Draft** an 800-word post, two labeled variants of the LinkedIn post and of the reply, with AI help disclosed and the correction from DESIGN.md. Every number is cited to `results/`. Aarmen rewrites them in his own words.
- [ ] **Step 2: Fresh numbers check.** An agent checks every number in the drafts against `results/`. Expected: no mismatch.
- [ ] **Step 3: Ask Aarmen** for a separate yes on making the repo public, on the blog, on the post and on the reply. He posts them.
- [ ] **Step 4: Update the author's private notes** with the outcome and links.
