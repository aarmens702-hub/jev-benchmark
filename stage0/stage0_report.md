# Stage 0 report

Validation rows only. The bug floor is shown as PASS or FAIL and no other score of an answer is kept.

## Result

Overall: PASS

| Step | Result | Detail |
|---|---|---|
| 1 Jev smoke | PASS | 20 of 20 rows returned a choice and probabilities |
| 1c key usage | PASS | spend showed in key usage after 24 s; later clients wait 120 s before believing a short report |
| 1b bare-name schema check | PASS | 3 rows |
| 2 small LLM selection | PASS | mistralai/mistral-nemo on DekaLLM: 49 of 50 usable |
| 3a Jev Banking77 validation | PASS | 756 rows, 0 failed |
| 3b Jev AG News validation | PASS | 800 rows, 0 failed |
| 4 repeatability | PASS | measured |
| 5 Jev + examples smoke | PASS | 20 of 20 rows |

## Checks

- Bug floor, Jev on Banking77 validation: PASS
- Spend this stage: US$0.1463 of the US$0.50 limit

## Jev

| Set | Rows | Failed | Failure rate | 1.0 mass | Score at the 80% cut |
|---|---|---|---|---|---|
| Banking77 smoke | 20 | 0 | 0.0% | n/a | n/a |
| Banking77 validation | 756 | 0 | 0.0% | 311 (41.1%) | 0.77 |
| AG News validation | 800 | 0 | 0.0% | 537 (67.1%) | 0.96 |

## Repeatability

- Rows: 100 x 3 calls; rows with all 3 answers: 100
- Label agreement: 98.0%
- Median top-probability range: 0.010

## Small LLM selection

- mistralai/mistral-nemo on DekaLLM: 49 of 50 usable (PASS); failures 1; mean output tokens 10.1; cost US$0.0025
- Selected: mistralai/mistral-nemo on DekaLLM

## Tokens, cost and latency

| Step | Calls | Mean input tokens | Mean output tokens | Cost per row | Latency median s | Latency p95 s |
|---|---|---|---|---|---|---|
| Jev smoke | 20 | 2371 | 828 | US$0.0001 | 0.18 | 0.21 |
| Jev Banking77 validation | 736 | 2369 | 828 | US$0.0001 | 0.17 | 0.22 |
| Jev AG News validation | 800 | 417 | 47 | US$0.0000 | 0.14 | 0.20 |
| Jev repeatability | 100 | 2369 | 828 | US$0.0001 | 0.16 | 0.21 |
| Jev + examples smoke | 20 | 3178 | 828 | US$0.0001 | 0.19 | 0.23 |
| mistralai/mistral-nemo selection | 50 | 1743 | 10.1 | US$0.0001 | - | - |

Examples retrieved per row: min 24, mean 24.0, max 24.

## Projected Stage 1 spend

- Jev Banking77: US$0.3065
- Jev AG News: US$0.1331
- Jev + examples Banking77: US$0.4111
- Jev bare-name Banking77 (upper bound): US$0.3065
- Jev bare-name AG News (upper bound): US$0.1331
- Small LLM Banking77: US$0.1550
- Small LLM AG News (scaled by Jev input tokens): US$0.0673
- Total with the 5.5% credit fee: US$1.5958 (CA$2.23)
- Spent so far: US$0.1463; left under the US$7.15 cap: US$7.0037
- Fits under the cap: PASS
- routing pilot spend is not measured here and is not in this total

## Fields to confirm (Task 5 report)

| # | Field | Status | Note |
|---|---|---|---|
| 1 | Jev model string is the expected 1.14 build | confirmed | build jev-1.14-20261014 |
| 2 | Jev probabilities cover exactly the options | confirmed | keys equal the options on every reply |
| 3 | Jev choice is the top option | confirmed | choice is within rounding of the top |
| 4 | usage.cost is present on every reply | confirmed | present |
| 5 | token counts are present on every reply | confirmed | present as integers |
| 6 | provider.only spelling is accepted on both surfaces | confirmed | both surfaces answered HTTP 200 under the pin (request-side plus HTTP 200 only) |
| 7 | provider is echoed on replies | confirmed | every reply names the pinned provider: {'jev': ['TypeSafe'], 'chat': ['DekaLLM']} |
| 8 | Jev reads criteria in the order sent | not tested | the docs are silent and a test needs extra paid calls; the order sent is shuffled per row |
| 9 | Jev accepts the provider pin with require_parameters | confirmed | Jev answered HTTP 200 with require_parameters true (request-side plus HTTP 200 only) |
| 10 | logprobs come back under json_schema | confirmed | logprobs seen from: ['mistralai/mistral-nemo'] |
| 11 | requests without a reasoning field are accepted | confirmed | accepted |
| 12 | reasoning tokens stay zero | confirmed | null or zero on every reply |
| 13 | key usage reflects spend | confirmed | usage rose with spend |
| 14 | usage.cost matches what the key is billed | confirmed | reported usage was 1.00 of summed usage.cost |
| 15 | Decisions spend appears in key usage | confirmed | Decisions spend appeared in key usage |
| 16 | seconds for key usage to post | confirmed | 24 s |
| 17 | state accepts the examples object | confirmed | accepted: HTTP 200 (request-side plus HTTP 200 only) |
| 18 | null criteria accepted for bare names | confirmed | accepted: HTTP 200 (request-side plus HTTP 200 only) |
| 19 | 77 and 151 choices accepted | unconfirmed | 77 accepted; 151 not tested, no 151-way task is in scope |
| 20 | refusals, 400s and 404s are billed | not tested | no 400 or 404 was seen |
| 21 | small LLM reply model matches the suffix pattern | confirmed | mistralai/mistral-nemo: replied ['mistralai/mistral-nemo'], pattern (?:-\d{8})? |
| 22 | Jev reply model starts with typesafe/ | confirmed | starts with typesafe/ |
