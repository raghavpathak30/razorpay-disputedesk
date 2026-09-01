# DECISIONS.md

Append-only. Never rewrite an entry. If something turns out wrong, add a new dated
entry that corrects it and leave the original visible.

This file exists for two reasons. It stops the same argument being had twice across
sessions, and at the end it is most of the architecture document the submission
requires.

Format:

```
## YYYY-MM-DD — <short title>

**Decision:** what was decided.
**Why:** the reasoning, including the option that was rejected.
**Status:** DECIDED | SUPERSEDED by <date> | REVERSED by <date>
```

For measurements, use this instead:

```
## YYYY-MM-DD — <what was measured>

**Result:** the number, with the split, the seed count, and the units.
**How:** the command or script that produced it, and the artifact it wrote.
**Caveats:** what this number does not show.
**Status:** CONFIRMED-RAN | ESTIMATE | UNVERIFIED
```

Never let a number appear in the README or the pitch video unless it has an entry
here marked CONFIRMED-RAN.

---

## 2026-08-31 — Track and problem selection

**Decision:** Razorpay Buildathon Track 02 (AI Risk Manager). One loss class:
fraud-reason-code chargebacks. Build a dispute triage and evidence-filing agent.
**Why:** Track 02's stated bar is measured precision and recall on a held-out set
with an explicit false-positive cost, which is the strongest thing to demonstrate.
Rejected: reusing prior mule-account detection work, because it is tabular ML with
no genuine agent component and would score poorly on AI Judgment.
**Status:** DECIDED

## 2026-08-31 — LLM authority boundary

**Decision:** The LLM drafts the explanation letter and normalises free-text order
context. It does not decide contest vs accept, does not map reason codes to
evidence types, and does not compute money.
**Why:** The judging criteria penalise forcing an LLM where a deterministic system
is better. The reason-code map is published by card networks and is a lookup. The
decision is an expected-value calculation. Both are worse with a language model in
the loop, and the boundary is itself the most defensible architectural claim in the
submission.
**Status:** DECIDED

## 2026-08-31 — Synthetic data over real data

**Decision:** Fully synthetic dispute dataset, generated from a written causal
story, with no anchoring to IEEE-CIS or any public transaction dataset.
**Why:** No public labelled dispute dataset exists. Anchoring to IEEE-CIS buys
realism that cannot be defended under panel questioning in the time available, and
costs a day. A documented generative process with a stated accuracy ceiling is more
defensible than a half-real dataset with an unclear provenance story.
**Caveats:** Every number produced by this project is conditional on the generator
being a reasonable model of reality. This must be said explicitly in the README and
in the pitch video rather than left for a judge to discover.
**Status:** DECIDED

## 2026-08-31 — Python version

**Decision:** requires-python >=3.11, no upper cap. CI matrix runs 3.11 and 3.13.
**Why:** Nothing in the stack requires 3.11 specifically. Development happens on
3.13; gating only on 3.11 in CI would let divergence surface late. Testing both
also makes the Phase 5 reproduce-from-clean-clone gate meaningful.
**Status:** DECIDED

## 2026-08-31 — Generator calibration

**Decision:** E[p] ≈ 0.25, p band [0.02, 0.75], modes ~0.08 / ~0.39.
Guessed oracle PR-AUC 0.30–0.36 against a prevalence baseline of 0.25.
**Why:** The first draft implied E[p] ≈ 0.46, which claims nearly half of
fraud-coded chargebacks are winnable. Published representment win rates for
fraud reason codes are commonly cited near 17% as industry estimates. Our
label is "winnable if contested optimally with correct evidence", which
justifies sitting above the observed rate but not at double it.
**Caveats:** 0.30–0.36 is a guess from a two-point idealization, not a
derivation. Phase 2 computes the real ceiling; a different value is a
corrected guess, not a defect.
**Status:** DECIDED

**Result:** ... TF-IDF + logistic regression on customer_communication_log
recovers true_fraud at AUC 0.6371 (5-fold CV) — real signal, not a leak, and
the baseline for Phase 3's LLM extraction to beat.

## 2026-08-31 — Phase 2 model quality, measured

**Result:** LightGBM (`disputedesk/model`, `ModelConfig` defaults) on
`disputedesk/features`' feature set, temporal holdout only, median and IQR
(25th/75th percentile) across 20 seeds (0–19), n_rows=15000 per seed:
- model PR-AUC (average precision): **0.3522** (IQR 0.3448–0.3696)
- prevalence baseline: **0.2377** (IQR 0.2331–0.2422)
- oracle Bayes ceiling, closed-form sweep (GENERATOR.md §5): **0.4572** (IQR
  0.4527–0.4585)
- precision / recall, at threshold = train-split label prevalence (median
  0.2543): **0.3537** / **0.6954**
- calibration error (expected calibration error, 10 equal-width bins):
  **0.0270** (IQR 0.0208–0.0314) — well calibrated
- permutation importance (holdout, average-precision scoring) ranks
  `ip_geo_billing_distance_km` and `prior_order_count` well above every other
  feature; LightGBM's gain importance, by contrast, ranks the pure-noise
  `checkout_hour_of_day` control 5th on a single-seed check (seed 42) — above
  seven real features — confirming GENERATOR.md/PHASES.md's warning against
  ever reporting gain importance as a headline figure.

**How:** `python -m eval.run_harness --n-seeds 20 --n-rows 15000`. Writes
`data/eval/per_seed_do_not_report_individually.csv` (raw per-seed rows, not a
headline number on their own) and `data/eval/headline_median_iqr.csv` (the
median/IQR table above).

**Caveats:** Precision/recall use a placeholder threshold (train-split label
prevalence), not a real policy decision — the policy engine (SPEC.md §4) does
not exist yet (Phase 3). This is not the operating point the eventual system
will use, only a documented stand-in so Phase 2 has *some* precision/recall
to report, as PHASES.md's gate requires. Cost-weighted rupee metrics and the
two policy baselines are explicitly out of scope for this measurement
(Phase 3 needs `representment_cost`, not decided yet).

This corrects GENERATOR.md §5's guessed oracle PR-AUC of 0.30–0.36 (open
parameter 9): the real ceiling on generated data is measurably higher, at
~0.46. The *direction* GENERATOR.md predicted held (model clears prevalence
by a wide margin; oracle exceeds the trained model) — only the guessed
magnitude of the ceiling was low, which GENERATOR.md's own §5 text
anticipated as a live possibility ("a different value is a corrected guess,
not a defect").

**Status:** CONFIRMED-RAN

## 2026-08-31 — Oracle closed-form vs. single-draw AP, reconciled

A Phase 1 sanity check measured `average_precision_score(y_true, p_true)` on
the seed-42 holdout at 0.4335. The Phase 2 entry above reports the
closed-form `oracle_pr_auc(p_true)` at median 0.4572 (IQR 0.4527–0.4585
across 20 seeds) — 0.4335 falls outside that IQR, which needed explaining
before either number could be trusted.

**Result:** on the seed-42, n=15000 holdout (n=3545 rows) used by both
checks: closed-form `oracle_pr_auc` = **0.4556**. GENERATOR.md §5's own
specified cross-check — draw many replicate `Bernoulli(p)` label samples from
the same `p` vector and confirm the closed form matches their mean, not the
rough two-point idealization a prior version of this project's tests used as
a stand-in — was implemented and run: mean of 500 replicate
`average_precision_score` draws = **0.4569** (std 0.0173, standard error of
the mean 0.0008). The closed form and the replicate mean agree within 1.7
standard errors — statistically indistinguishable. Phase 1's single value,
0.4335, sits 1.35 standard deviations below that mean — ordinary sampling
variance for a single realized draw at this holdout size, not a sign the two
formulas measure different things.

**How:** `pytest tests/test_eval_oracle_replicate_check.py -v`
(`test_closed_form_equals_the_mean_of_replicate_label_draws` and
`test_a_single_realized_draw_is_within_a_few_standard_deviations_of_the_mean`),
implementing the replicate-sample check GENERATOR.md §5 specifies but a
prior version of `tests/test_eval_oracle.py` had substituted with the
document's rough two-point illustration instead — agreement with that
illustration was not a valid stand-in for this check, since it was the same
reasoning that produced the 0.30–0.36 ceiling guess the measurement above
overturned.

**Caveats:** This settles that the closed form is the correct thing to
report (it is the expectation; a single realized draw is one noisy sample of
it, with std ≈0.017 at this holdout size) — it does not mean any individual
future single-seed check should be expected to land close to the closed
form. Precision/recall/PR-AUC checks that use one realized draw (e.g. a
quick sanity check on one seed) should expect ~1–2 point swings from
sampling noise alone and should not be treated as headline numbers
(CLAUDE.md invariant 3) for exactly this reason.

**Status:** CONFIRMED-RAN

## 2026-08-31 — representment_cost_inr sensitivity sweep, measured

A sensitivity analysis reported *alongside* the configured
`representment_cost_inr=400.0` (`disputedesk/policy/config.py`), not a
retune - the configured value was not changed to produce this result, and
this entry does not propose changing it.

**Result:** `low_confidence_band=(0.45, 0.55)`, seeds 0–19, n_rows=15000 per
seed held fixed; `representment_cost_inr` swept from 0 to 10,000. Escalated
disputes scored under `escalate_mode="naive_contest"` throughout (the fair
mode against baseline A - see the entry above). Policy vs. baseline A
(contest everything), rupees recovered per 1,000 disputes, median across 20
seeds:

| cost | policy | baseline A | policy − baseline A |
|---|---|---|---|
| 0 | 2,101,093 | 2,101,093 | 0 |
| 100 | 2,000,929 | 2,001,093 | −163 |
| 250 | 1,850,879 | 1,851,093 | −214 |
| 300 | 1,805,301 | 1,801,093 | +4,209 |
| **400 (configured)** | **1,714,015** | **1,701,093** | **+12,923** |
| 600 | 1,545,496 | 1,501,093 | +44,403 |
| 900 | 1,326,262 | 1,201,093 | +125,170 |
| 1,000 | 1,253,163 | 1,101,093 | +152,071 |
| 2,000 | 770,639 | 101,093 | +669,547 |
| 2,500 | 614,829 | −398,908 | +1,013,737 |
| 6,000 | −4,978 | −3,898,908 | +3,893,929 |

**Where the advantage disappears:** for `representment_cost_inr` roughly in
[0, 290], the median advantage is small (magnitude under ~2,200 out of a
~2,000,000 base, i.e. well under 0.1%) and flips sign between adjacent swept
values (−219 at 50, −163 at 100, +117 at 120, −145 at 140, +536 at 150, ...,
−214 at 250, +4,209 at 300) - individual near-threshold disputes flipping
which side of `decide()`'s cutoff they land on as cost changes in small
steps, not a real, seed-robust difference. In this band the policy is
**statistically indistinguishable from baseline A**, not reliably ahead of
it. Below cost≈120 the two are effectively identical by construction: near
cost=0, `expected_value = p_win * amount - cost ≈ p_win * amount`, which is
positive for nearly every dispute, so the policy contests almost everything
baseline A already contests everything.

**Where it becomes substantial:** the advantage grows monotonically and
robustly for cost ≥ ~300, crossing 10% of baseline A's own recovered total
between cost=800 (6.6%) and cost=900 (10.4%) - roughly **2.25× the
configured value**. By cost≈2,000 baseline A's own recovered rate has fallen
to near zero while the policy still recovers 770,639; by cost≈2,500 and
above, baseline A goes net *negative* (contesting everything is actively
destroying value) while the policy stays solidly positive - the clearest
qualitative divergence in the sweep, driven by the policy's selectivity
(declining to contest disputes whose `P(win) * amount` no longer clears the
now-larger cost) versus baseline A paying the fee on every dispute
regardless of merit.

**At the configured value (400):** the advantage (+12,923, ≈0.75% of
baseline A) sits just above the noisy crossover band, not yet in the
"substantial" regime - consistent with the 2026-08-31 "Phase 3 cost-weighted
business metrics" entry's `naive_contest` result above.

**How:** `python -m eval.run_cost_sensitivity --n-seeds 20 --n-rows 15000
--costs 0 50 100 110 120 130 140 150 160 170 180 190 200 250 300 400 500 600
700 800 900 1000 1200 1500 2000 2500 3000 4000 5000 6000 8000 10000`. Writes
`data/eval/cost_sensitivity_per_seed_do_not_report_individually.csv` and
`data/eval/cost_sensitivity_median_iqr.csv`. `eval/cost_sensitivity.py`
reuses each seed's `eval.harness.run_seed_pipeline` predictions once (`P(win)`
does not depend on `representment_cost_inr`), so all 33 swept costs come
from the same 20 trained models, not 33×20 retrains.

**Caveats:** the crossover band's exact edges (which specific cost values
flip sign) are themselves noise and should not be over-read - what's robust
is the qualitative shape (near-parity below ~300, growing and monotonic
above it), not any single swept value's sign. This sweep used the model
trained under the configured `low_confidence_band=(0.45, 0.55)`; it does not
sweep the band itself, which was explicitly out of scope for this analysis.

**Status:** CONFIRMED-RAN

## 2026-08-31 — Phase 3 cost-weighted business metrics, measured

**Result:** `PolicyConfig` defaults (`representment_cost_inr=400.0`,
`low_confidence_band=(0.45, 0.55)` — unchanged from their original values),
temporal holdout only, median and IQR across 20 seeds (0–19), n_rows=15000
per seed (median n_test=3609.5):

Rupees recovered per 1,000 disputes, by what an escalated dispute is
credited (`eval.business_metrics.EscalateMode` — see that module's
docstring):

| escalate_mode | policy (median) | beats baseline A (contest everything, median 1,701,092) | beats baseline B (accept everything, 0) |
|---|---|---|---|
| `zero` (no outcome claimed) | 1,559,504 | **NO** | YES |
| `oracle` (human always right) | 1,727,152 | YES | YES |
| `naive_contest` (human contests every escalated case, same as baseline A already assumes for those rows) | 1,714,015 | **YES** | YES |

`escalated_amount_share_of_holdout` (median): **0.0433** — 4.33% of total
holdout rupees sit in the ~199.5-of-3609.5 (≈5.5% by count) escalated
disputes; the escalated share of *count* and of *money* are close, not
wildly disproportionate, at this `low_confidence_band` width.

**How:** `python -m eval.run_business_harness --n-seeds 20 --n-rows 15000`.
Writes `data/eval/business_per_seed_do_not_report_individually.csv` and
`data/eval/business_headline_median_iqr.csv`.

**Caveats — this correction matters more than the numbers themselves:** an
earlier run of this same command only reported the `zero`-mode number and
concluded "the policy loses to contest-everything." That conclusion was
wrong, or at least not established, and was caught before being recorded
here: crediting an escalated dispute 0 rupees assumes the human who reviews
it takes no action and recovers nothing, which is not what "escalate to a
human" means (SPEC.md §4's own framing is that the band exists so the system
has an honest "I don't know" path, not a claim that escalated disputes are
worthless) — it structurally penalizes the policy for having an abstention
path that contest-everything doesn't have. `naive_contest` is the fair
comparison against baseline A specifically, because it scores the escalated
rows exactly the way baseline A already scores them (baseline A contests
every dispute, escalated or not); under that scoring, the policy **beats
both baselines**. `oracle` is a deliberately optimistic upper bound (it uses
the true label as foreknowledge no real reviewer has) and is reported for
context, not as a claim about real human accuracy. The one case that would
be a genuine loss — not beating baseline A under `naive_contest` — did not
occur. `representment_cost_inr` and `low_confidence_band` were not changed
to produce this result (CLAUDE.md: do not adjust a metric definition or a
policy parameter to make a number win — this correction changed how an
*already-escalated* row is scored for reporting purposes, not the policy's
own contest/accept/escalate decision, which is identical across all three
rows of the table above).

**Status:** CONFIRMED-RAN

## 2026-08-31 — LLM provider: Groq

**Decision:** The real `LLMClient` implementation (`disputedesk/evidence/llm.py`)
calls Groq's OpenAI-compatible chat completions endpoint
(`https://api.groq.com/openai/v1/chat/completions`), configured with model
`openai/gpt-oss-20b`. Endpoint, model, and API key are all read from
`disputedesk.config.get_settings()` (`LLM_API_URL`, `LLM_MODEL`,
`LLM_API_KEY` in `.env`/`.env.example`) - not hardcoded as class constants,
per the Phase 0 rule that `config.py` is the only place that reads
`os.environ`. Supersedes the earlier `AnthropicHttpLLMClient` from the
2026-08-31 Phase 3 session, which was never wired to a real key or called
live.

**Why (provider):** Groq was chosen by the requester; not re-litigated here.

**Why (model - `openai/gpt-oss-20b`, not a frontier model):** SPEC.md §2
scopes the LLM to exactly two jobs - drafting a short `explanation_letter`
and normalising messy free text into typed fields (`NormalizedCommunicationLog`,
`ExplanationLetterOutput` in `disputedesk/evidence/schemas.py`). Neither task
needs multi-step reasoning, long-context synthesis, or broad world knowledge;
both are short, schema-constrained, single-turn completions where a small
model is plausibly sufficient. Reaching for the largest available model here
would be exactly the "forcing an LLM/unnecessary tech" failure mode the AI
Judgment criterion penalises (echoing the same reasoning already recorded in
the "LLM authority boundary" entry above, applied to model *size* instead of
LLM *scope*) - the smallest model that plausibly clears the bar is the
defensible choice, not the largest one available.

Verified against `console.groq.com/docs/models` and
`console.groq.com/docs/deprecations` on 2026-08-31 (today), not recalled from
training data (CLAUDE.md: do not invent a citation):
- `llama-3.1-8b-instant` (8B, the obvious "smallest" candidate) is
  deprecated for free/developer-tier usage, with a shutdown date of
  2026-08-16 - already past as of this decision. Groq's own deprecation
  notice recommends migrating to `openai/gpt-oss-20b`.
- `openai/gpt-oss-20b` is listed under "Production Models" (not "Preview")
  on the current models page: 131,072-token context, 20B total parameters
  but only 3.6B active per forward pass (mixture-of-experts), with both JSON
  Object Mode and JSON Schema Mode support - directly useful for
  `evidence/`'s schema-constrained outputs. It is the smaller of the two
  production GPT-OSS models (the other is 120B) and the model Groq itself
  points free/developer users toward as the small-model default going
  forward.
- Not chosen: `llama-3.3-70b-versatile` (70B, production) and
  `qwen/qwen3-32b` (32B, availability/tier unclear at verification time) -
  both larger than `gpt-oss-20b` with no identified requirement in these two
  tasks that `gpt-oss-20b` couldn't plausibly meet; `gemma2-9b-it` was
  returned by one lookup as "still active" and by another as carrying an
  October-2025 deprecation date already in the past relative to today - the
  two sources disagreed, so it was not selected without a resolving,
  citable source.

**Caveats:** "Plausibly handles" both tasks is a judgment based on the
model's stated capabilities (JSON mode, instruction-following at this size
class), not yet an empirical eval on this project's own prompts - the one
real call in this session (see the entry immediately below) checks that the
API contract works, not output quality at scale. If `evidence/`'s repair-then-
fallback behavior (SPEC.md §7) fires often in real use, that is a signal to
revisit this choice, not a reason to silently swap models without a new
dated entry here.

**Status:** DECIDED

## 2026-08-31 — Groq live call, verified

**Result:** `GroqHttpLLMClient` called against the real
`https://api.groq.com/openai/v1/chat/completions` endpoint with
`LLM_MODEL=openai/gpt-oss-20b`, using an already-exported `GROQ_API_KEY`
from the shell environment (never written to a file or committed). Two live
requests: `client.complete(prompt)` directly, then a second completion of
the same prompt via `call_llm_and_validate(...)` (a fresh, non-deterministic
call, not a reuse of the first) using the real
`normalize_comms_log_v1` prompt against a sample customer message. Both
requests returned HTTP 200; `response.raise_for_status()` did not raise;
`body["choices"][0]["message"]["content"]` was present and non-empty on
both. The first completion's JSON parsed and validated against
`NormalizedCommunicationLog` on the **first attempt** - no repair call was
needed - confirming the auth header, endpoint path, request body shape, and
response-parsing code (`disputedesk/evidence/llm.py`,
`disputedesk/evidence/validated_call.py`) all work end to end against the
real provider, not just against `FakeLLMClient`.

Sample output (first completion, verbatim):
```json
{
  "claims_unauthorized_transaction": true,
  "mentions_prior_bank_contact": true,
  "mentions_shared_card_access": false,
  "mentions_travel": false,
  "tone": "polite",
  "is_substantive": true,
  "summary": "The customer claims an unknown charge, has already contacted their bank, and requests a refund."
}
```

**How:** ad hoc interactive session, not a committed script - constructed
`Settings` from environment variables directly (`LLM_API_KEY` set from the
shell's `GROQ_API_KEY`, `LLM_API_URL`/`LLM_MODEL` set explicitly,
`RAZORPAY_KEY_ID`/`RAZORPAY_KEY_SECRET`/`DATABASE_URL` set to dummy values
since `Settings` requires them all), then called `GroqHttpLLMClient()`
directly. No `.env` file was created; the key was never printed or logged.

**Caveats:** this confirms the API *contract* (auth, endpoint, request/response
shape, one successful schema validation) works, not output quality across
the range of real dispute messages this system will see, and not the repair
path (SPEC.md §7) against a real malformed response - `FakeLLMClient`
already covers that deterministically in `tests/test_evidence_*.py`, and
should keep doing so; this was a one-time connectivity/contract check, not a
replacement for those tests.

**Status:** CONFIRMED-RAN

## 2026-09-01 — LLM normalisation quality vs. TF-IDF baseline, measured

**The LLM normalisation does not beat the TF-IDF baseline. It loses, clearly.**

**Result:** `eval.llm_normalization_quality`, live Groq API
(`openai/gpt-oss-20b`), n=60 synthetic disputes (seed=0, `GeneratorConfig()`
defaults), `customer_communication_log` -> `NormalizedCommunicationLog` via
the real `normalize_communication_log` (prompt
`disputedesk/evidence/prompts/normalize_comms_log_v1.txt`, unmodified).
7 typed fields (`eval.llm_normalization_quality.FEATURE_COLUMNS` -
`claims_unauthorized_transaction`, `mentions_prior_bank_contact`,
`mentions_shared_card_access`, `mentions_travel`, `is_substantive`,
`tone_polite`, `tone_terse`; the free-text `summary` field excluded, since
the comparison is about the *typed* fields specifically) fed to a logistic
regression under 5-fold stratified CV, ROC AUC per fold - the same
downstream-classifier methodology as the recorded TF-IDF baseline, so the
comparison isolates the feature-extraction step:

- true_fraud prevalence in the sample: 0.317 (19 of 60)
- `human_review_required` rate: **0.0** - every one of the 60 completions
  validated against `NormalizedCommunicationLog` on the first LLM call, no
  repair or fallback needed anywhere in the sample. The extraction is
  reliable in *format*; this measurement is about whether it is predictive
  in *content*.
- fold AUCs: [0.5938, 0.3594, 0.3281, 0.2500, 0.5741]
- **mean AUC: 0.4211 (std 0.1378)** - below 0.5 (worse than a coin flip) on
  3 of 5 folds
- **TF-IDF + logistic regression baseline (recorded, 2026-08-31 "Generator
  calibration" entry, 5-fold CV): 0.6371**
- **verdict: LLM normalisation does NOT beat the baseline** - 0.4211 vs.
  0.6371, a wide margin, not a close call decided by noise alone (though see
  Caveats on the sample size).

**Why (diagnostic, not a fix - nothing was changed based on this):**
per-class means of the 7 typed fields are nearly identical between
`true_fraud=True` and `true_fraud=False`:

| field | true_fraud=False | true_fraud=True |
|---|---|---|
| claims_unauthorized_transaction | 0.976 | 0.947 |
| mentions_prior_bank_contact | 0.195 | 0.211 |
| mentions_shared_card_access | 0.146 | 0.053 |
| mentions_travel | 0.146 | 0.105 |
| is_substantive | 0.976 | 0.947 |
| tone_polite | 0.805 | 0.684 |
| tone_terse | 0.098 | 0.105 |

Two fields (`claims_unauthorized_transaction`, `is_substantive`) sit near
1.0 for both classes - almost every generated message reads as a real,
substantive unauthorized-charge claim regardless of `true_fraud`, so a
boolean extraction of "did they claim unauthorized use" has almost no
variance left to correlate with anything. This is consistent with
`disputedesk/generator/comms.py`'s own documented design (session-2 fix,
GENERATOR.md §3): the four opening/claim phrase pools are shared across both
classes with deliberately mild weight differences ("max ratio 1.5:1") so no
fixed string is class-exclusive - the differentiating signal is a subtle
*frequency tilt* across near-synonymous phrasings, exactly the kind of
signal a bag-of-words TF-IDF vectorizer preserves as continuous per-token
weights and a coarse yes/no LLM extraction collapses away. `tone_polite`
carries the largest per-class gap (0.805 vs. 0.684) but is still a weak,
noisy single feature at this sample size.

**How:** `python -m eval.run_llm_normalization_quality --n-rows 60 --seed 0
--sleep-seconds 7.0 --out-dir data/eval`, with `LLM_API_KEY` set from an
already-exported `GROQ_API_KEY` (never written to a file). Writes
`data/eval/llm_normalization_quality_sample.csv`. An earlier attempt at
n=80/sleep=4.0 hit `HTTP 429` (the per-minute token budget for
`openai/gpt-oss-20b`'s free tier is consumed faster than a naive
prompt-length estimate suggested - a live probe showed a single
normalisation call costs ~540 total tokens including ~150 *reasoning*
tokens the visible completion doesn't show, and the provider's rate-limit
bucket accounting drains faster than that alone would imply). Fixed by
adding 429-aware retry with backoff and slower pacing to
`eval/llm_normalization_quality.py` and `eval/run_llm_normalization_quality.py`
(tested with `tests/test_eval_llm_normalization_quality.py`'s
`_FlakyThenValidLLMClient`, an in-memory fake - no network in any test) -
this changed request pacing and retry behaviour only, never the prompt,
schema, feature encoding, or CV methodology, all of which were written and
frozen (per this module's own docstring) before either run.

**Sample size, why 60:** free-tier limits for `openai/gpt-oss-20b`
(retrieved from `console.groq.com/docs/rate-limits` on 2026-08-31): 30 RPM /
1,000 RPD / 8,000 TPM / 200,000 TPD. At the observed ~540-830 tokens
consumed per call (bucket accounting vs. reported `usage.total_tokens`
disagree somewhat, likely because the provider reserves against requested
`max_tokens` rather than actual usage), 60 requests with no repairs used
roughly 32,000-50,000 tokens total - comfortably inside the daily 200,000
budget and the 60-of-1,000 daily request budget, with 7-second pacing
keeping the per-minute rate well under the 8,000 TPM ceiling.

**Caveats:** n=60 (19 positive) is small - the 5-fold CV folds hold only
~12 rows (~4 positive) each, and the fold-to-fold spread (0.25 to 0.59) is
wide; a different seed would very plausibly move the mean AUC by more than
a few points. This was a deliberate one-shot, free-tier-bounded sample, not
a multi-seed measurement (CLAUDE.md invariant 3 governs *headline* numbers
about the system's own performance; this is a single diagnostic comparison
against an already-recorded baseline, following the same "out-of-band
sanity check, one run" precedent the TF-IDF number itself was produced
under). The gap between 0.4211 and 0.6371, however, is large enough (and
the diagnostic table above explains a real structural reason for it) that
it should not be read as "too close to call" - it is a genuine result, not
noise dressed up as one. Per this session's explicit instruction, no prompt,
schema, or feature-encoding change was made in response to this number, and
none should be made without a new dated entry here explaining why.

**Status:** CONFIRMED-RAN

## 2026-09-01 — Phase 4: Razorpay client, audit log, webhook, failure recovery

**Decision:** `disputedesk/client/razorpay.py` implements `accept()`
(`POST /v1/disputes/{id}/accept`) and `contest()` (`PATCH
/v1/disputes/{id}/contest`), HTTP Basic Auth (`key_id:key_secret`).
Verified against Razorpay's own documentation on 2026-09-01, not recalled
from training data: `https://razorpay.com/docs/api/authentication/`
(Basic Auth scheme, confirmed verbatim), `https://razorpay.com/docs/api/disputes/accept/`
(method, empty body, `status` moves to `"lost"`, irreversible),
`https://razorpay.com/docs/api/disputes/contest/` (method, full field list
including per-evidence-type document-id lists and the `others` array
shape, `action: "draft"|"submit"`, full example response), and
`https://razorpay.com/docs/api/disputes/fetch-all/` (dispute entity field
list, matching `DisputeRecord`). "Test mode" is not a separate base URL -
it is determined by which key pair is configured - so
`razorpay_api_base_url` in `disputedesk/config.py` defaults to the one real
host either way.

**Why (no document-upload pipeline):** the real `contest()` API expects
evidence as pre-uploaded document ids (`doc_...`) per evidence type. This
project never built a document-upload/file-storage path (not listed in
SPEC.md or PHASES.md) - fabricating document ids that reference no real
file would be worse than omitting them. `contest()` submits the drafted
`explanation_letter` text as `summary` (truncated to the API's documented
1000-character limit) with `action="submit"`; `required_evidence_types`
(already computed by `evidence/reason_code_map.py`) are recorded in the
audit log for a human reviewer, not attached as files. This is a real,
load-bearing gap between this system and a production filing, stated here
rather than left for a judge to discover.

**Why (webhook envelope is an assumption, not a citation):** two lookups
for Razorpay's specific dispute-webhook JSON envelope 404'd
(`/docs/webhooks/payloads/disputes/`, `/docs/webhooks/payloads/`). The
`disputedesk/api/schemas.py` envelope (`event` / `payload.dispute.entity`)
follows Razorpay's well-known general webhook shape used elsewhere on its
platform, not a page confirmed for disputes specifically - flagged in that
module's own docstring, following the same assumption-vs-citation
convention `evidence/reason_code_map.py` already established. What the
route actually gates on is `status == "open"` (a `Literal["open"]` on the
schema), the one field PHASES.md Phase 4 names directly ("receives an
`open` dispute event") and the one verified against the real dispute
entity. The order-context fields (`avs_match` through
`checkout_hour_of_day`) are this system's own join, not part of Razorpay's
webhook at all (SPEC.md §1 step 1) - assumed to arrive pre-joined, shaped
like `DisputeRecord`'s own fields, since no order-lookup system exists in
this project (out of scope).

**Decision (audit log - two append-only tables, not one mutable row):**
`disputedesk/audit/models.py` has `decisions` (one row per dispute, UNIQUE
`dispute_id`, written *before* the Razorpay call - model_version, features,
p_win, policy_branch, expected_value, prompt_version, validation_result,
human_review_required) and `api_outcomes` (at most one row per dispute,
UNIQUE `dispute_id`, written after the API call finishes - action, outcome,
response/error). `disputedesk/audit/log.py` exposes only insert functions;
no update or delete path exists anywhere in the codebase.

**Why:** CLAUDE.md requires the decision persisted *before* the API call
and the log append-only with *no update path*. A single mutable row can't
satisfy both at once - filling in an API response after the call would be
an UPDATE. Two insert-only tables, joined by `dispute_id` in
`get_audit_trail`, get both properties literally rather than by convention.
Each table's own `dispute_id` UNIQUE constraint is the idempotency gate
PHASES.md Phase 4 item 2 asks the database (not application logic) to
enforce - proven directly in `tests/test_audit_log.py` by inserting a
duplicate row that bypasses `record_decision` entirely and asserting the
database itself raises `IntegrityError`.

**Decision (shared retry/backoff):** `disputedesk/retry.py`'s
`call_with_backoff` - exponential backoff on `httpx.TimeoutException` or a
429 (honoring `Retry-After`), any other exception propagates immediately -
is used by both `disputedesk/client/razorpay.py` (SPEC.md §7 failure path
1) and `disputedesk/evidence/llm.py`'s `GroqHttpLLMClient` (the free-tier
429s the 2026-09-01 LLM-quality measurement above first hit). One helper,
tested once (`tests/test_retry.py`), rather than each script/client
re-implementing backoff - which is what PHASES.md Phase 4 item 7 asked to
fix for the Groq path specifically.
`tests/test_client_razorpay.py::test_timeout_then_success_files_exactly_once`
proves this end to end against `RazorpayHttpClient` (via
`httpx.MockTransport`, no real socket): a timeout followed by a successful
retry results in exactly one successful response, not two filings.

**Decision (no persisted model artifact):** `disputedesk/model/registry.py`
trains one model in memory per process (`lru_cache`), fixed seed 42, from
the same generate -> temporal-split -> train pipeline `eval/harness.py`
already uses - not a new training path. `MODEL_VERSION` names the seed and
config, recorded on every audit row. Building a real model-registry/
persistence layer (joblib artifact, versioned storage) was out of scope for
this phase (not listed in SPEC.md or PHASES.md) and would be new scope this
session was not asked for; this is flagged as a real limitation, not
silently treated as production-ready.

**Caveats:** no test in this session makes a network call (`httpx.MockTransport`
and `FakeLLMClient`/`FakeRazorpayClient` throughout, per CLAUDE.md). The
demo script (`disputedesk/cli/demo.py`) is fully self-contained (fakes
only, in-memory DB, in-memory model) so it runs from a clean clone with no
`.env`; it demonstrates the timeout-retry mechanism by exercising
`call_with_backoff` directly against a simulated `httpx.ConnectTimeout`
rather than literally severing a network connection mid-run, since the
demo has no live network call to sever by design.

**Status:** DECIDED

## 2026-09-01 — Phase 5 freeze: refactor, security review, docs

**Decision/correction:** the 2026-08-31 "Python version" entry above states
"CI matrix runs 3.11 and 3.13" - `.github/workflows/ci.yml` as it actually
exists only runs a single `python-version: "3.11"` job, not a matrix. Left
uncorrected in the code during this freeze (`.github/workflows/ci.yml` is
not one of the files this session's refactor/security scope covers), noted
here per CLAUDE.md's rule against letting a recorded claim silently stay
wrong. The dev `.venv` this project has actually been developed and tested
under is 3.13, so both versions are exercised locally even though CI itself
only checks one.

**Refactor pass, what was removed:** two genuinely dead code paths, found
by checking every non-test function/class definition against the rest of
the codebase for any reference outside its own definition -
`disputedesk.audit.db.session_scope` (a context-manager convenience never
adopted by any caller - the webhook, the demo script, and every test all
construct a session via `make_session_factory` directly instead) and
`eval.business_metrics.BusinessOutcome` (a dataclass defined and never
instantiated anywhere). No file exceeded CLAUDE.md's 300-line limit (max
was 277, `disputedesk/cli/demo.py`); three functions exceeded the 50-line
limit (`disputedesk.audit.log.record_decision`,
`eval.business_metrics.build_business_row`,
`eval.run_business_harness.main`) and were brought under it - the first by
trimming its docstring (its length was documentation, not branching logic),
the other two by extracting a genuinely separable piece each
(`_cost_and_escalate_summary`, `_print_report`) rather than restructuring
working logic. `eval.business_metrics`'s per-seed regression test
(`tests/test_eval_business_harness_regression.py`) and the full suite both
still pass unchanged after the split, confirming the extraction didn't
change any computed value.

**Security review, findings and fixes:** `disputedesk/api/schemas.py`'s
`DisputeEntity` validated most fields (`amount`, `prior_order_count`, etc.)
but left `id`, `payment_id`, `reason_code`, and `customer_communication_log`
as unconstrained `str` - PHASES.md Phase 4 item 5 and this session's brief
both ask for validation on every field, not just `status`. Three real gaps,
fixed:
1. `id`/`payment_id` are interpolated directly into the Razorpay API
   request path (`disputedesk/client/razorpay.py`'s
   `f"/disputes/{dispute_id}/..."`, string interpolation, not a URL-safe
   join) - an unconstrained value could contain `/`, `?`, or other
   URL-structuring characters and alter the request target. Fixed with a
   `pattern=r"^[A-Za-z0-9_]{1,64}$"` constraint matching the charset
   Razorpay's own ids actually use.
2. `reason_code` had no constraint, but
   `disputedesk/evidence/reason_code_map.py::required_evidence_types`
   raises `KeyError` on anything outside the four known codes - a
   `CONTEST`-bound dispute with an unrecognized code would have surfaced as
   an unhandled 500, not a 422 rejected at the boundary. Fixed by
   constraining `reason_code` to `Literal[*REASON_CODES]` (the same four
   codes `disputedesk/features/build.py` already treats as the known
   vocabulary).
3. `customer_communication_log` had no length bound, letting an oversized
   payload inflate LLM token cost/latency without limit. Fixed with
   `max_length=5000` - a cost/DoS bound, not a correctness one.

Confirmed clean, not fixed because nothing was broken: no secret in any
tracked file or in git history (`git log --all -p` grepped for
key-shaped strings); `os.environ`/`os.getenv` read only inside
`disputedesk/config.py` (grepped across the whole tree); unvalidated LLM
output never reaches the Razorpay client at any point -
`disputedesk/api/pipeline.py` only ever reads
`packet.explanation_letter.letter_text`, a field that exists only after
`disputedesk/evidence/validated_call.py`'s schema validation or the
deterministic template fallback, never a raw completion.

**Known gap, deliberately not fixed in this freeze:** the webhook has no
signature/HMAC verification (`X-Razorpay-Signature` or equivalent) - real
Razorpay webhook documentation for the dispute-event signature scheme could
not be located (same "two lookups 404'd" gap the 2026-09-01 "Phase 4" entry
above already notes for the payload envelope itself). Adding verification
now would be new scope this freeze's brief explicitly excludes ("no new
features" - PHASES.md Phase 5), not a "fix what's broken" correction to
existing behavior, so it's recorded here as a stated limitation
(`ARCHITECTURE.md`'s "Known gaps" section) rather than built.

**Demo fixes (two, explicitly permitted this freeze):**
`disputedesk/cli/demo.py` previously replayed one fixture dispute under
different `id`s for every step, so every step's displayed `p_win` was the
identical 0.2658 - a viewer had no way to see the model responding to
input. Added `WEAK_EVIDENCE_EVENT`, a deliberately opposite feature profile
(weak AVS/CVV, unrecognized device, no delivery proof, filed fast, no order
history, low amount), scored at `p_win=0.0756` and decision=`accept` by the
same demo model that scores the original fixture at `p_win=0.2658` and
decision=`contest` - shown as a new step 3, contrasted explicitly against
step 1. Also silenced the `StarletteDeprecationWarning` that otherwise
prints above the demo's output (starlette's `httpx2`-not-installed warning,
harmless but visible in a pitch-video recording) by filtering that specific
warning class around the one import that triggers it, not by installing a
new dependency or broadening the filter to other warnings.

**How (verification):** `ruff check .` and `pytest` (190 tests, up from 184
- four new regression tests for the schema constraints above, in
`tests/test_api_schemas.py`) both pass after every change in this entry.
`python -m disputedesk.cli.demo` re-run and its full output re-read after
the demo fixes.

**Status:** DECIDED

## 2026-09-01 — Demo: third failure path (429 + Retry-After), real RazorpayHttpClient

**Decision:** Added a third demo step (`disputedesk/cli/demo.py`'s
`demo_failure_path_429_retry_after`, printed as "7. Failure path 3") showing
a real HTTP 429 with a `Retry-After` header recovering through the existing
`disputedesk.retry.call_with_backoff` helper - no new retry logic. Unlike
the existing timeout demo (step 6, which calls `call_with_backoff` directly
against a synthetic flaky function), this step runs the real
`RazorpayHttpClient` through the real webhook route, so it demonstrates
actual recovery: dispute filed, `api_outcomes` row written `success`. The
429 is injected as a transport-layer fixture (`httpx.MockTransport`, routed
in via a temporary `mock.patch("httpx.request", ...)` - the same pattern
`tests/test_client_razorpay.py` already uses), never by monkeypatching
`call_with_backoff`'s internals.

**Seam added (smallest found; `call_with_backoff` itself needed none - it
already takes `sleep_fn`):** `RazorpayHttpClient.__init__` gained a
`sleep_fn: Callable[[float], None] = time.sleep` parameter, threaded
straight into its existing `call_with_backoff(...)` call. Real callers never
pass it (production behaviour unchanged); the demo uses it to print the
honoured `Retry-After` value `call_with_backoff` actually received and then
sleep a compressed, labelled fraction of it, so the run stays fast and the
printed values stay real and deterministic (no wall-clock timestamps, no
jitter). Covered by
`tests/test_client_razorpay.py::test_429_retry_after_reaches_injected_sleep_fn`.

**Test coverage checked before writing anything new:**
`tests/test_retry.py::test_429_honors_retry_after_header_over_computed_delay`
already asserted the generic helper honours `Retry-After` over its computed
backoff schedule - it existed before this session, so no duplicate was
added there. The one new test above is narrower: it proves the new
`sleep_fn` seam on `RazorpayHttpClient` actually receives that honoured
value, which nothing previously tested (the existing
`test_429_is_retried_and_then_succeeds` only asserted the retry occurred at
all, using `retry-after: "0"`, not any specific honoured value).

**Known gap, worsened by this change, not fixed in this session:**
`disputedesk/cli/demo.py` was already over CLAUDE.md's 300-line module
limit before this session (337 lines - a pre-existing violation from the
"Demo: add a second, visibly different dispute fixture" commit, after the
Phase 5 freeze's own refactor pass had it at 277). This session's addition
brings it to 440 lines; two new top-level helpers
(`_make_429_then_200_handler`, `_compressed_sleep_fn`, `_print_closing_banner`)
were extracted specifically to keep every individual *function* under the
50-line limit, but the file-level 300-line split was explicitly out of this
session's scope ("do not refactor anything else"). Left as a stated
limitation for a future refactor pass, not silently ignored.

**Status:** DECIDED

## 2026-09-01 — Demo split: deterministic Segment A vs. non-reproducible LLM Segment B

**Decision/correction:** `disputedesk/cli/demo.py` previously implied its
entire run was safe to treat as reproducible (no network, no secrets,
fakes throughout - see the 2026-08-31 "Phase 4" entry's own caveat: "the
demo script is fully self-contained (fakes only ...)"). That was true
*then*, because every LLM call in the demo went through `FakeLLMClient`
with fixed canned responses. It is no longer true for the whole script: the
demo now has a second half that calls a real `GroqHttpLLMClient`. Reporting
the full demo's stdout as byte-identical across cold clones would now be
false, because live LLM completions are not bit-reproducible - the same
prompt against the same model can (and, per this session's own test run,
did: one of the two sampled disputes below hit the existing repair-then-
fallback path on one run and might not on another) return different text
call to call. **The claim is corrected here, not silently narrowed: it now
applies to Segment A only.**

Split into two printed segments:
- **Segment A - Deterministic.** Every existing demo step (webhook
  end-to-end, idempotency, second-dispute contrast, malformed-payload
  rejection, both `SPEC.md` §7 failure paths, and the 429/Retry-After path
  from the entry above) plus nothing new. No LLM call anywhere in this
  segment - `FakeLLMClient` with fixed responses throughout, same as
  before. `python -m disputedesk.cli.demo --deterministic-only` (new flag)
  runs only this segment, for whatever reproducibility check needs a clean
  byte-identical run - confirmed by running it twice against a stripped
  environment (`env -i`, no `.env`, no exported secrets) and diffing:
  identical both times.
- **Segment B - LLM output (not reproducible).** New. Prints the real
  drafted `explanation_letter` for exactly two disputes, chosen from
  *existing* fixtures (no new fixture data needed) to differ on both axes
  the brief asked for: `CONTEST_WORTHY_EVENT` (`MC_4837`, full required
  evidence set available - AVS, CVV, device, and delivery all confirmed) and
  `WEAK_EVIDENCE_EVENT` (`VISA_83`, a documented gap - none of those four
  signals present). Per dispute, prints the reason code, the deterministic
  reason-code map's required evidence types, which of those this specific
  dispute's context actually supports, and the full letter text
  (unmodified, unmodified `letter_text` field - no truncation). Uses the
  real `GroqHttpLLMClient` and the existing, unmodified
  `assemble_evidence_packet` / `draft_explanation_letter` code path and
  prompt - printing only, nothing in `evidence/` changed. Skips with a
  clear message (not a crash) if `get_settings()` can't construct a real
  `Settings` object (missing `.env`).

**New demo-only logic, not part of production `evidence/`:** which required
evidence types a specific dispute's own context actually supports
(`_available_evidence_types` in `disputedesk/cli/demo.py`) is new judgment
this session added purely for Segment B's print output. The production
evidence assembler (`disputedesk/evidence/assembler.py`) does not gate
assembly on availability today - it assembles the full required set
unconditionally for every contest decision (a stated gap already recorded
in `ARCHITECTURE.md`'s "Known gaps": no document-upload pipeline, no
per-type file-existence check). This demo-only helper does not change that;
it only judges, for display, which of the required types this project's own
existing order-context signals (AVS/CVV match, device fingerprint,
delivery confirmation) would plausibly support.

**Bug fixed as a necessary correctness fix, not scope creep:** the 429 demo
step (entry above) used `os.environ.setdefault(...)` to inject placeholder
Razorpay/LLM credentials, which - because it never had anything to revert
to and `pydantic-settings` prefers a real env var over a real `.env` file -
permanently overwrote the process environment with those placeholders for
the rest of the run. This would have silently broken Segment B's real
`GroqHttpLLMClient()` (it would pick up the placeholder `LLM_API_KEY` and
fail, or worse, appear to work against a non-existent
`https://example.test/llm`). Fixed by scoping that injection to
`mock.patch.dict(os.environ, ..., clear=False)` (restores the exact prior
environment on exit) and clearing `get_settings`'s `lru_cache` both after
that block and (already) before it, so a real `.env`'s values reach
Segment B unpoisoned. Covered implicitly: `pytest` still passes with no
`.env` present (Segment A/production tests never call `get_settings()`
through this path), and a manual run with real Groq credentials exported
confirmed Segment B receives them correctly.

**Docs updated to match:** `README.md`'s "Running the demo from a clean
clone" section now describes both segments separately, states the
byte-identical claim only for Segment A, and documents `.env`/network as a
Segment-B-only requirement (previously the whole demo was described as
needing neither). `disputedesk/cli/demo.py`'s own module docstring updated
the same way. No existing `SPEC.md`, `PHASES.md`, or checklist text was
found claiming demo-wide reproducibility to begin with (checked directly -
`SPEC.md` §9's checklist item 12 is an unchecked box with no evidence
citation attached to it in-repo); the correction target was this project's
own prior framing in `DECISIONS.md` and `README.md`, both addressed above.

**Why the split, not a fix to make Segment B "reproducible" instead:**
pinning the LLM's output (e.g., temperature 0, or re-using a cached
completion) would not make live provider output bit-reproducible across
machines/time in the way this project's other reproducibility claims mean
it (same model version, same weights, same sampling implementation, no
provider-side drift) - and CLAUDE.md is explicit that claiming
reproducibility that isn't real is worse than admitting the LLM segment
isn't. Scoping the claim to Segment A, rather than manufacturing a false
appearance of determinism for Segment B, is the honest fix.

**Status:** DECIDED

## 2026-09-01 — Letter-drafting validation reliability, weak vs. full evidence, measured

**The weak-evidence dispute falls back to the deterministic template far
more often than the full-evidence one, and every single failure - on both
disputes, all 40 runs - is the completion getting truncated before the JSON
closes, never a Pydantic schema-constraint violation.**

**Result:** `draft_explanation_letter` (`disputedesk/evidence/draft_letter.py`,
unmodified - prompt `explanation_letter_v1`, unmodified), called directly
(not through `assemble_evidence_packet`) against the real Groq API
(`openai/gpt-oss-20b`), 20 independent attempts each for the two Segment B
demo fixtures (`disputedesk/cli/demo.py`'s `CONTEST_WORTHY_EVENT` / `MC_4837`,
full required evidence available, and `WEAK_EVIDENCE_EVENT` / `VISA_83`, a
documented gap - no AVS/CVV/device/delivery signal), same fixed
`NormalizedCommunicationLog` reused across all 20 attempts per dispute
(obtained once per dispute via one real `normalize_communication_log` call,
so the only thing varying between the 20 attempts is the live model's own
non-determinism, not the input):

| | full_evidence (MC_4837) | weak_evidence (VISA_83) |
|---|---|---|
| first draft passed validation | 17/20 (85%) | 8/20 (40%) |
| repair attempted | 3/20 (15%) | 12/20 (60%) |
| repair succeeded (of attempted) | 0/3 (0%) | 2/12 (17%) |
| **final path = template_fallback** | **3/20 (15%)** | **10/20 (50%)** |

Side by side: **full_evidence fails 15% of the time; weak_evidence fails
50% of the time** - roughly 3.3x, and material by any reasonable reading.

**How:** `LLM_API_KEY=$GROQ_API_KEY LLM_API_URL=https://api.groq.com/openai/v1/chat/completions
LLM_MODEL=openai/gpt-oss-20b python -m eval.run_llm_letter_validation_reliability
--n-runs 20 --sleep-seconds 8.0 --out-dir data/eval` (`RAZORPAY_KEY_ID`/
`RAZORPAY_KEY_SECRET`/`DATABASE_URL` set to unused dummy values only because
`get_settings()` validates one `Settings` object, not per-feature). New
files this session, measurement-only, no production code touched:
`eval/llm_letter_validation_reliability.py` (pure record-building logic,
unit-tested with `FakeLLMClient` in
`tests/test_eval_llm_letter_validation_reliability.py` - 6 new tests, no
network), `eval/run_llm_letter_validation_reliability.py` (the live-API
entry point). Writes `data/eval/llm_letter_validation_reliability.csv` (one
row per run) and `data/eval/llm_letter_validation_reliability_raw_failures.json`
(every failing raw completion plus its exact error, for the diagnostic
below). `pytest` (197 tests, up from 191) and `ruff check .` both pass.

**Which validation rule is rejecting them: none - this is not a schema
problem.** Across all 40 runs (both disputes), every recorded error is a
`json.JSONDecodeError` (`_validation_error_text` in
`eval/llm_letter_validation_reliability.py` reuses
`disputedesk/evidence/validated_call.py`'s own `_parse` helper, so this is
the exact error production validation would raise, not a re-derived
approximation). Not one `pydantic.ValidationError` was observed on either
dispute - no field ever violated a schema constraint (`letter_text`'s
50-4000 character bound, `cites_evidence_types`'s shape, or `extra="forbid"`)
in any of the 40 runs. Three representative raw failing completions
(`weak_evidence`, all `json.JSONDecodeError`, verbatim from
`llm_letter_validation_reliability_raw_failures.json`):

1. **run 0, first attempt - empty completion.** `raw_response == ""`.
   Error: `JSONDecodeError('Expecting value: line 1 column 1 (char 0)')`.
   The model returned zero visible content.
2. **run 4, first attempt - truncated after ~140 characters, mid-sentence,
   before the JSON string even closes:**
   ```
   {"letter_text":"To the Visa Network,\n\nWe acknowledge receipt of the chargeback notification (Reason Code VISA_83) for transaction of INR 220
   ```
   Error: `JSONDecodeError('Unterminated string starting at: line 1 column 16 (char 15)')`
   (character 15 is the `"` that opens the `letter_text` value - the string
   itself, and therefore the whole JSON object, never closes).
3. **run 1, first attempt - truncated after 2,896 characters, well into a
   substantive multi-paragraph letter:**
   ```
   {"letter_text":"Subject: Merchant Narrative Supporting Dispute of Transaction INR 2200.00 – VISA_83\n\nDear Visa Dispute Team,\n\n[... ~2.7k more characters of real letter content ...]g details match the cardholder's records, the service was delivered successfully, and the customer had the opportunity to contest the purchase before the chargeback was lodged. We respectfully request
   ```
   Error: `JSONDecodeError('Unterminated string starting at: line 1 column 16 (char 15)')`
   (same error class as #2 - the parser reports the same starting position
   for any unterminated string regardless of how much text came after it).

All three - and every other failure in the sample - fit one root cause:
`disputedesk/evidence/llm.py`'s `GroqHttpLLMClient.complete()` sends a fixed
`"max_tokens": 1024` on every request (shared with `normalize_comms.py`'s
much shorter typed-field extraction task). The model sometimes spends most
or all of that budget on Groq's hidden "reasoning" tokens before emitting
any visible content at all (example 1: empty), sometimes emits a few dozen
visible tokens of letter text before running out (example 2), and sometimes
writes most of a real letter but still runs out before the closing quote
and brace (example 3). The repair call (`repair_addendum_v1` prompt) does
not fix this - it is the same completion request with the same
`max_tokens=1024`, so it fails the same way for the same reason (10 of 12
weak-evidence repairs, and all 3 full-evidence repairs, failed; see the CSV
for the two repairs that did succeed).

**Why weak_evidence fails more - a hypothesis, not confirmed by this
measurement:** `GroqHttpLLMClient.complete()` does not surface token usage
(`disputedesk/evidence/llm.py` reads only
`body["choices"][0]["message"]["content"]`), so this measurement has no
direct evidence of how many hidden reasoning tokens each call actually
spent - only the visible-content length and where it got cut off. The
weak-evidence prompt asks the model to write a confident contest letter
from a context with no AVS/CVV/device/delivery signal at all (`disputedesk/evidence/prompts/explanation_letter_v1.txt`
still instructs a single coherent narrative regardless of how much genuine
evidence exists), which plausibly invites more hedging/reasoning before or
during the visible letter text than the full-evidence case, where the
narrative is simpler to write - consistent with weak_evidence's failures
skewing more toward the empty-completion pattern (5 of 12 first-attempt
failures) than full_evidence's (0 of 3). This is a plausible explanation
for the gap, not a proven one; confirming it would need the raw token-usage
field Groq's response body carries but this client doesn't currently read.

**Caveats:** n=20 per dispute, one run, one seed of live-model sampling -
not a multi-seed measurement (CLAUDE.md invariant 3 governs headline claims
about this *system's* own performance; this is a diagnostic measurement of
a third-party model's behavior on two fixed prompts, following the same
one-shot precedent the 2026-09-01 "LLM normalisation quality" entry's n=60
measurement used). A different run could show different absolute rates:
`json.JSONDecodeError`'s prevalence and the total absence of any
`ValidationError` are the load-bearing, more-likely-to-replicate finding
here, not the exact 15%/50% split. Per this session's explicit instruction,
nothing was fixed in response to this - `max_tokens=1024` in
`disputedesk/evidence/llm.py` was not touched, and no prompt, schema, or
retry-count was changed. Any future change made in response to this finding
needs its own dated entry here explaining what changed and why.

**Status:** CONFIRMED-RAN

## 2026-09-01 — Letter-drafting reliability, re-measured after the fix

**Fixed. Both fixtures now pass 20/20, zero repairs needed on either. The
weak-evidence gap did not survive the fix - it did not partially close, it
closed completely (50% -> 0%, 15% -> 0%). The truncation bug fully explains
the prior gap; the earlier "weak-evidence prompts more hedging" hypothesis
is not needed and not supported by this result (see the correction at the
end of this entry).**

**1. Parameter name and whether Groq was honouring it:** confirmed live,
2026-09-01, before touching any code: a bare request with `"max_tokens": 50`
against `openai/gpt-oss-20b` returned `usage.completion_tokens: 50` and
`finish_reason: "length"` - Groq **was honouring** the deprecated `max_tokens`
alias, not silently ignoring it. The bug was never the parameter name; it
was the *value*, combined with a fact the old code had no visibility into
(see #2). `disputedesk/evidence/llm.py` now sends `max_completion_tokens`
(console.groq.com/docs/reasoning, verified 2026-09-01: the documented
parameter for `openai/gpt-oss-20b`), not `max_tokens`, on principle - both
work, but only one is documented and safe to depend on going forward.

**2. Usage logging - the instrument, added before anything else changed:**
`GroqHttpLLMClient._record_usage` now reads `usage.prompt_tokens`,
`usage.completion_tokens`, and `usage.completion_tokens_details.reasoning_tokens`
from every response, appends a record to a new `self.usage_log` list, and
logs it via `logging.getLogger(__name__).info(...)`. Confirmed live
(2026-09-01) that `reasoning_tokens` **counts against the same
`completion_tokens` budget as visible output**, not a separate one - the
`max_tokens=50` probe above returned `reasoning_tokens: 48` of those 50,
leaving 2 for visible content (0 characters emitted). This is the root
cause the 2026-09-01 "Letter-drafting validation reliability" entry
diagnosed structurally (every failure a `JSONDecodeError` from truncation,
never a schema `ValidationError`) but couldn't see numerically, because the
client discarded `usage` entirely before this fix.

**3. Completion budget, raised from arithmetic:** `GroqHttpLLMClient.MAX_COMPLETION_TOKENS = 1512`
(`disputedesk/evidence/llm.py`, replacing the old flat `1024`), derived and
commented in the code as: `ExplanationLetterOutput.letter_text`'s own
`max_length=4000` chars / ~3.6 measured chars-per-token on this model
(live: 2804 chars/581 tokens and 1678 chars/349 tokens, both at
`reasoning_effort="low"`) ≈ 1112 visible tokens, + ~100 tokens of JSON
scaffolding (keys, braces, up to 5 `cites_evidence_types` strings), + ~300
tokens reserved for reasoning (measured 4-5 at `reasoning_effort="low"`,
but budgeted wide since a caller can override to `"medium"`/`"high"`
without touching this constant) = 1512.

**4. `reasoning_effort`, added as a configurable parameter, defaulting to
`"low"`:** `GroqHttpLLMClient.__init__(..., reasoning_effort: str = "low")`,
sent as `"reasoning_effort"` in every request body. Justified in the
constructor's own docstring: both of this client's two SPEC.md §2 jobs -
drafting a letter from already-decided, already-supplied facts, and
normalising free text into typed booleans - are prose generation and
extraction, not reasoning tasks; the contest/accept decision that *would*
need reasoning is made upstream, deterministically, before the LLM is ever
called (`disputedesk/policy/`, unchanged, still never sees a prompt).
Confirmed live (2026-09-01, same two prompts): `reasoning_effort="low"` cut
`reasoning_tokens` from 300-700 (provider default, unset) to 4-5 per call,
with `finish_reason="stop"` (a complete, non-truncated response) both times.

**Result - the re-measurement, unchanged harness:** `eval/run_llm_letter_validation_reliability.py`
re-run exactly as the 2026-09-01 "Letter-drafting validation reliability"
entry ran it (`eval/llm_letter_validation_reliability.py`'s
`run_letter_reliability_sample`/`run_one_draft_attempt`/`failure_rate` -
the actual classification logic - untouched), 20 letter-drafting attempts
per fixture, `reasoning_effort="low"` (this client's new default):

| | full_evidence (MC_4837) - before → after | weak_evidence (VISA_83) - before → after |
|---|---|---|
| first draft passed validation | 17/20 (85%) → **20/20 (100%)** | 8/20 (40%) → **20/20 (100%)** |
| repair attempted | 3/20 (15%) → **0/20 (0%)** | 12/20 (60%) → **0/20 (0%)** |
| repair succeeded (of attempted) | 0/3 (0%) → n/a (none attempted) | 2/12 (17%) → n/a (none attempted) |
| **final path = template_fallback** | **15% → 0%** | **50% → 0%** |

Side by side, before → after: **full_evidence 15% → 0%; weak_evidence
50% → 0%.** The 3.3x gap is gone, not narrowed.

**Median token usage per letter-drafting call** (from
`GroqHttpLLMClient.usage_log`, one fresh client instance per fixture so
each fixture's log is isolated; the one `normalize_communication_log` call
per fixture is excluded - this is calls to the drafting step specifically,
n=20 each since zero repairs means exactly one call per run):

| | full_evidence (MC_4837) | weak_evidence (VISA_83) |
|---|---|---|
| median completion_tokens | 412 | 478 |
| median reasoning_tokens | 9 | 6 |

Both comfortably under the 1512 ceiling, and - notably - weak_evidence's
median `reasoning_tokens` (6) is not higher than full_evidence's (9) at
`reasoning_effort="low"`. This is the direct evidence against the hedging
hypothesis: see the correction below.

**How:** `LLM_API_KEY=$GROQ_API_KEY LLM_API_URL=https://api.groq.com/openai/v1/chat/completions
LLM_MODEL=openai/gpt-oss-20b python -m eval.run_llm_letter_validation_reliability
--n-runs 20 --sleep-seconds 8.0 --reasoning-effort low --sample-letters 2
--out-dir data/eval`. `eval/run_llm_letter_validation_reliability.py` gained
per-fixture usage tracking and a `_print_sample_letters` step this session
(the classification harness itself did not change - see above); it now
takes `--reasoning-effort` so a future comparison run doesn't need a code
edit. The pre-fix CSV/JSON were preserved as
`data/eval/llm_letter_validation_reliability_before_fix.csv` and
`..._raw_failures_before_fix.json` before this run overwrote the originals.
New tests: `tests/test_evidence_llm_groq.py` (7 tests, `httpx.MockTransport`,
no network) asserting the request body sends `max_completion_tokens` not
`max_tokens`, `reasoning_effort` defaults to `"low"` and is overridable, and
`usage_log` records `prompt_tokens`/`completion_tokens`/`reasoning_tokens`
correctly (including when the field is absent). `pytest` (204 tests, up
from 197) and `ruff check .` both pass.

**Letter-quality check, two full weak-evidence letters at
`reasoning_effort="low"` (printed verbatim by the run above, not excerpted
here to keep this entry a reasonable length - see the run's stdout / re-run
with `--sample-letters 2` to reproduce): both are complete, coherent,
correctly cite the required evidence types by name (`access_activity_log`,
`billing_proof`, `proof_of_service`, `customer_communication`), address the
correct reason code and amount, and end with `finish_reason="stop"` (no
truncation). **Judgment: quality did not drop; no `reasoning_effort="medium"`
re-run was run.** One real, minor flaw worth recording rather than
smoothing over: the first sample letter (run 0) contains two sign-off
blocks back to back ("Sincerely, Merchant Services Team" immediately
followed by "Best regards, [Merchant Name]") - a small coherence artifact,
not a validation failure (the schema has no rule against it) and not
something this session fixed, since fixing prompt content was out of scope
here. This is a judgment call made in this session, not confirmed by the
user viewing the letters directly - the raw text remains in this run's
stdout for independent review, and a `--reasoning-effort medium` comparison
run is one command away if that judgment is disputed.

**Correction to the 2026-09-01 "Letter-drafting validation reliability"
entry's hedging hypothesis:** that entry proposed, explicitly flagged as
"a hypothesis, not confirmed," that weak_evidence's higher failure rate
might reflect the model spending more hidden reasoning tokens hedging a
weaker case. This measurement did not confirm it and does not need it: the
gap fully closes under `reasoning_effort="low"` for both fixtures alike,
and at low effort weak_evidence's median `reasoning_tokens` (6) is not
higher than full_evidence's (9) - if extra hedging were a real, separate
effect on top of the truncation bug, some residual gap or reasoning-token
asymmetry would be expected to survive fixing the budget, and none did.
The original entry is left visible and uncorrected in place, per this
file's own append-only rule; this entry is the correction.

**Caveats:** n=20 per fixture, one run at `reasoning_effort="low"`, same
one-shot precedent as the entry this corrects (CLAUDE.md invariant 3 governs
headline claims about this *system's* own performance, not a diagnostic
measurement of a third-party model's behavior on two fixed prompts). A 0%
failure rate at n=20 is consistent with a true rate as high as roughly 14%
at typical confidence levels - "fixed" here means "the specific failure
mode that produced 15%/50% did not recur in 40 attempts," not "provably
zero forever." `disputedesk/evidence/prompts/explanation_letter_v1.txt` and
`disputedesk/evidence/draft_letter.py`'s logic were not touched - only
`disputedesk/evidence/llm.py`'s request parameters and usage handling.

**Status:** CONFIRMED-RAN

## 2026-09-01 — Policy precision/recall added to the cost sweep; Phase 2 threshold coincidence checked per-seed

Read-only investigation (a prior session, same day) traced the contest/accept
predicate end to end and found the Phase 2 precision/recall headline
(`eval/harness.py`, threshold = train-split label mean, median 0.2543) was
never consumed by the policy engine - it's a Phase-2-era placeholder,
unrelated to `decide()`'s expected-value rule. This entry adds the policy's
own precision/recall to the existing `representment_cost_inr` sweep
(`eval/cost_sensitivity.py`) so the real operating numbers can be reported
alongside the rupee curve from the 2026-08-31 "representment_cost_inr
sensitivity sweep" entry above, and records a per-seed check of whether the
Phase 2 number's resemblance to the configured-cost precision is real or
coincidental.

**1. Policy precision/recall, added to the sweep.** `eval/cost_sensitivity.py`
now computes, for every (seed, cost) pair already in the sweep: CONTEST as
the positive prediction, `won_if_contested` as the label, `precision_score`/
`recall_score` (`sklearn.metrics`, `zero_division=0`) against the same
`decide_batch` decisions already used for the rupee numbers - no retraining,
no new predictions, same `run_seed_pipeline` output reused per seed as
before. `summarize_sweep` reports median/q25/q75 across the 20 seeds, same
convention as every other headline in this project.

**2. ESCALATE convention, stated and enforced, not just documented.**
ESCALATE rows are folded in as a *positive* prediction, alongside CONTEST.
This matches `ESCALATE_MODE = "naive_contest"`, already used for the rupee
recovery numbers, where an escalated row is credited exactly as if it had
been contested - so precision/recall and the rupee numbers describe the same
underlying set of "effectively contested" rows, not two different
definitions side by side in one table. This is enforced, not just narrated:
`eval/cost_sensitivity._predicted_positive` raises `NotImplementedError` if
`ESCALATE_MODE` is ever changed away from `"naive_contest"` without this
function being updated to match.

**Result**, seeds 0-19, n_rows=15000 per seed, `low_confidence_band=(0.45,
0.55)` held fixed (same run as the 2026-08-31 sweep entry):

| cost | precision (median, IQR) | recall (median, IQR) | policy advantage vs. baseline A (median, INR/1,000) |
|---|---|---|---|
| 0 | 0.2377 (0.2331-0.2422) | 1.0000 (1.0000-1.0000) | 0 |
| 50 | 0.2378 (0.2333-0.2423) | 1.0000 (0.9985-1.0000) | -219 |
| 100 | 0.2387 (0.2340-0.2429) | 0.9965 (0.9917-0.9977) | -163 |
| 200 | 0.2432 (0.2378-0.2467) | 0.9768 (0.9753-0.9812) | +1,772 |
| 300 | 0.2479 (0.2422-0.2524) | 0.9506 (0.9459-0.9578) | +4,209 |
| **400 (configured default)** | **0.2543 (0.2476-0.2569)** | **0.9155 (0.9111-0.9232)** | **+12,923** |
| 600 | 0.2641 (0.2571-0.2675) | 0.8446 (0.8301-0.8513) | +44,403 |
| 800 | 0.2724 (0.2672-0.2775) | 0.7726 (0.7655-0.7812) | +86,000 |
| 1,000 | 0.2791 (0.2700-0.2849) | 0.7036 (0.6957-0.7126) | +152,071 |
| 1,500 | 0.2915 (0.2848-0.2979) | 0.5613 (0.5513-0.5715) | +390,402 |
| 2,000 | 0.2991 (0.2949-0.3080) | 0.4521 (0.4380-0.4744) | +669,547 |
| 3,000 | 0.3142 (0.3074-0.3278) | 0.3123 (0.2999-0.3443) | +1,384,613 |
| 4,000 | 0.3233 (0.3137-0.3375) | 0.2339 (0.2207-0.2558) | +2,168,360 |
| 6,000 | 0.3369 (0.3264-0.3644) | 0.1540 (0.1408-0.1759) | +3,893,929 |
| 8,000 | 0.3467 (0.3346-0.3780) | 0.1252 (0.1100-0.1369) | +5,722,304 |
| 10,000 | 0.3599 (0.3435-0.3868) | 0.1069 (0.0981-0.1192) | +7,573,243 |

At cost=0, precision (0.2377) equals the test-split label prevalence exactly
- expected by construction, since recall=1.0 there means every row is
effectively contested, so precision degenerates to prevalence. Precision
rises and recall falls monotonically as cost rises, matching `decide()`'s
mechanics: a higher cost raises the per-row breakeven `p_win` required to
contest (`expected_value = p_win * amount - cost > 0`), so the policy
contests a narrower, higher-precision, lower-recall slice as cost grows.
₹400 is the configured default, not "the" operating point - the table above,
not any single row of it, is the result; the cost-sensitivity framing from
the 2026-08-31 entry (near-parity with baseline A below ≈290, growing and
monotonic advantage above ≈300) applies here exactly as it does to the
rupee numbers.

**3. Per-seed coincidence check.** The configured-cost (400) precision
median, 0.2543, is visually identical to the unrelated Phase 2 placeholder
threshold's median (also 0.2543, train-split label mean). Checked whether
this is the same quantity (a plumbing bug: precision reading train-split
data instead of holdout decisions) or two different statistics whose medians
merely coincide. For each of the 20 seeds: A = that seed's train-split label
mean (`SeedResult.threshold`, `eval/harness.py`), B = that seed's policy
precision at `representment_cost_inr=400` from this sweep. **A ≠ B at every
one of the 20 seeds** (no seed has A−B = 0; signs vary, e.g. seed 15:
A−B=+0.02091, seed 17: A−B=−0.01861) - **they agree only at the median, not
per-seed.** Spread of A−B across seeds: min=-0.018614, max=+0.020911,
mean=+0.001208, population stdev=0.008088. Conclusion: no plumbing bug - A
and B are computed from different splits by different code paths (A from
`train_df` before any model/policy runs; B from the holdout `test_df`'s
`decide_batch` decisions); their medians sitting close is coincidental, both
being statistics that land near this generator's overall label prevalence,
not evidence of shared computation. Not investigated further since the
result was negative (no bug found); reproducible via the seed loop in this
session's transcript (`run_seed_pipeline` + `_predicted_positive` +
`precision_score`, `representment_cost_inr=400`, `n_rows=15000`, seeds
0-19).

**How:** `python -m eval.run_cost_sensitivity --n-seeds 20 --n-rows 15000`
(uses the module's own `DEFAULT_COSTS`, a superset of the rows above).
Writes the same two files as the 2026-08-31 sweep entry; the precision/recall
columns are new columns on the same per-seed and summary CSVs, not a
separate artifact.

**Status:** CONFIRMED-RAN

## 2026-09-01 — Visa reason code rename: `VISA_83` -> `VISA_10_4`

**Rename, not a re-derivation.** Visa retired standalone reason code 83 in
2018 under Visa Claims Resolution (VCR), reclassifying card-not-present
fraud into the 10.x dispute conditions. The current equivalent for this
generator's CNP-fraud scenario is condition 10.4, "Other Fraud – Card-Absent
Environment." User-verified against multiple chargeback references outside
this repo, 2026-09-01. `VISA_83` renamed to `VISA_10_4` everywhere it
appeared as a live value: `disputedesk/generator/config.py`,
`disputedesk/features/build.py` (`REASON_CODES`, same tuple position, index
2 - the ordinal encoding a trained model sees is unchanged),
`disputedesk/evidence/reason_code_map.py`, `disputedesk/cli/demo.py`'s
`WEAK_EVIDENCE_EVENT` fixture and Segment B print label,
`eval/run_llm_letter_validation_reliability.py`'s fixture label, `README.md`,
and the affected tests (`tests/test_evidence_draft_letter.py`,
`tests/test_features_build.py`). Not touched: this file's own prior dated
entries (append-only - they correctly describe runs that used `VISA_83` at
the time) and gitignored generated artifacts under `data/`, which regenerate
naturally.

**Sanity check against the repo's own cited source - it disagrees.**
`GENERATOR.md` §8 cites Razorpay's published chargeback reference
(`https://cdn.razorpay.com/files/chargeback_codes.pdf`) for the four
`reason_code` values. Re-fetched 2026-09-01: it still lists, verbatim,
`VISA | 83 | Fraud-Card Absent Environment | Fraud`. Razorpay's own reference
has not been updated to the post-VCR 10.x scheme. `GENERATOR.md`'s table row
is left exactly as published (a citation should say what its source says,
not what we wish it said) with a new dated revision note next to it
explaining that the system's live `reason_code` value has diverged from that
citation on purpose, in favor of the currently-correct Visa condition over a
stale published one.

**Evidence-map check under 10.4 (checklist item from this session):**
`REQUIRED_EVIDENCE_BY_REASON_CODE["VISA_10_4"]` still maps to the same
`_CNP_FRAUD_EVIDENCE` tuple as the other three codes - `billing_proof`
(AVS/CVV), `access_activity_log` (device/IP), `proof_of_service` (delivery),
`customer_communication`, `explanation_letter`. **Gap found, not fixed:**
neither `SPEC.md` §3's fixed evidence-object vocabulary
(`shipping_proof`, `billing_proof`, `cancellation_proof`,
`customer_communication`, `proof_of_service`, `explanation_letter`,
`refund_confirmation`, `access_activity_log`, `refund_cancellation_policy`,
`terms_and_conditions`, `others`) nor the order-context data model
(`disputedesk/evidence/context.py`'s `DisputeContext`, or `SPEC.md` §3's
"Order context" field list) has a slot for 3-D Secure / Visa Secure
authentication data at all - `billing_proof` already stands in for "the
charge was authenticated as this cardholder" generally (AVS/CVV), not 3DS
liability-shift data specifically. This is a real gap for condition 10.4,
which VCR designed 3DS/Visa Secure authentication evidence to address
directly, but it is a data-model gap (no field anywhere upstream to source
it from), not a one-line fix to this map. Deliberately not added - doing so
would tell the letter-drafting LLM authentication evidence is available that
the system does not actually have. Flagged for a future session; needs a new
`DisputeContext`/order-context field before the map can honestly claim it.

**Deterministic demo, re-verified stable at the new value.** Ran
`python -m disputedesk.cli.demo --deterministic-only` twice after the
rename; `diff` between the two runs is empty. This confirms Segment A is
reproducible at its new value, not that it matches a pre-rename capture (none
was taken - CLAUDE.md's revision-history convention doesn't require
recording an exact stdout snapshot pre-change to compare against, only that
the property being claimed, reproducibility, still holds after). Structurally,
this rename could not have changed any *number* Segment A prints: `VISA_10_4`
occupies the same tuple position (index 2) in `REASON_CODES` as `VISA_83`
did, so the ordinal integer the model actually sees for the weak-evidence
dispute (`disp_demo_weak_001`, `p_win=0.0756`, decision=`accept`) is bit-for-
bit unchanged; Segment A's stdout never prints the literal reason-code string
in the first place (only `dispute_id`, `p_win`, `decision`,
`expected_value_inr`), so no printed text differs either.

**Letter-drafting reliability, re-measured for `weak_evidence` only (this
fixture's reason code goes into the letter prompt as input, so the prior
20-run result no longer describes what's actually shipped).** Same harness,
unchanged (`eval.llm_letter_validation_reliability.run_letter_reliability_sample`
via `eval/run_llm_letter_validation_reliability.py`'s own `_run_for_event`,
called directly for this one fixture only - `full_evidence`/`MC_4837` and the
prompt template were not touched, per instruction), live Groq,
`reasoning_effort="low"`, n=20:

| | before (`VISA_83`, 2026-09-01 "re-measured after the fix" entry) | after (`VISA_10_4`, this entry) |
|---|---|---|
| first draft passed validation | 20/20 (100%) | 20/20 (100%) |
| repair attempted | 0/20 | 0/20 |
| final path = template_fallback | 0/20 (0%) | 0/20 (0%) |

**Result: unchanged.** 20/20 first-draft-valid, 0 repairs, 0 template
fallbacks - identical to the pre-rename measurement. Expected: the reason
code is one short token substituted into the same prompt template
(`disputedesk/evidence/prompts/explanation_letter_v1.txt`, not touched), and
nothing about `VISA_10_4` vs `VISA_83` changes prompt length, JSON shape, or
completion-token budget in a way that would move a truncation-driven failure
mode (the actual root cause identified in the 2026-09-01 "Letter-drafting
reliability, re-measured after the fix" entry). Not investigated further
since the result was the expected null result, not a surprise.

**Caveats:** n=20, one run at `reasoning_effort="low"`, same one-shot
precedent as every other letter-drafting reliability measurement in this
file (CLAUDE.md invariant 3 governs headline claims about this *system's*
own performance, not a diagnostic measurement of a third-party model's
behavior on one fixed prompt per fixture). A 0% failure rate at n=20 is
consistent with a true rate as high as roughly 14% at typical confidence
levels. `full_evidence` (`MC_4837`) was not re-run - it is unaffected by this
rename and its prior 20/20 result still describes the shipped fixture
unchanged.

**How:** ad hoc script calling
`eval.run_llm_letter_validation_reliability._run_for_event(WEAK_EVIDENCE_EVENT,
"low", 20, 8.0)` directly (the same function the CLI's `main()` calls per
fixture) - not a modification to the harness or the CLI script, just a
narrower invocation for one fixture. Full test suite (207 tests) and `ruff
check`/`ruff format --check` pass after the rename.

**Status:** CONFIRMED-RAN

## 2026-09-02 — ESCALATE rate added to the cost sweep; naive_contest's zero-human-cost assumption sized as a sensitivity

Read-only measurement plus a small addition to the existing
`representment_cost_inr` sweep output (`eval/cost_sensitivity.py`) - no
change to `disputedesk/policy/engine.py` or to `ESCALATE_MODE`.

**1. ESCALATE rate, added as a column.** `eval/cost_sensitivity.sweep_seed`
now also records `policy_escalate_rate` (fraction of the holdout where
`decide()` returns `Decision.ESCALATE`) per (seed, cost); `summarize_sweep`
reports its median/q25/q75 alongside precision, recall, and the rupee
numbers, same 20-seed convention as everything else. Full table, seeds
0-19, n_rows=15000 per seed:

| cost | precision (median, IQR) | recall (median, IQR) | escalate rate (median, IQR) | policy advantage vs. baseline A (median, INR/1,000) |
|---|---|---|---|---|
| 0 | 0.2377 (0.2331-0.2422) | 1.0000 (1.0000-1.0000) | 0.0562 (0.0485-0.0624) | 0 |
| 50 | 0.2378 (0.2333-0.2423) | 1.0000 (0.9985-1.0000) | 0.0562 (0.0485-0.0624) | -219 |
| 100 | 0.2387 (0.2340-0.2429) | 0.9965 (0.9917-0.9977) | 0.0562 (0.0485-0.0624) | -163 |
| 200 | 0.2432 (0.2378-0.2467) | 0.9768 (0.9753-0.9812) | 0.0562 (0.0485-0.0624) | +1,772 |
| 300 | 0.2479 (0.2422-0.2524) | 0.9506 (0.9459-0.9578) | 0.0562 (0.0485-0.0624) | +4,209 |
| **400 (configured default)** | **0.2543 (0.2476-0.2569)** | **0.9155 (0.9111-0.9232)** | **0.0562 (0.0485-0.0624)** | **+12,923** |
| 600 | 0.2641 (0.2571-0.2675) | 0.8446 (0.8301-0.8513) | 0.0562 (0.0485-0.0624) | +44,403 |
| 800 | 0.2724 (0.2672-0.2775) | 0.7726 (0.7655-0.7812) | 0.0562 (0.0485-0.0624) | +86,000 |
| 1,000 | 0.2791 (0.2700-0.2849) | 0.7036 (0.6957-0.7126) | 0.0562 (0.0485-0.0624) | +152,071 |
| 1,500 | 0.2915 (0.2848-0.2979) | 0.5613 (0.5513-0.5715) | 0.0562 (0.0485-0.0624) | +390,402 |
| 2,000 | 0.2991 (0.2949-0.3080) | 0.4521 (0.4380-0.4744) | 0.0562 (0.0485-0.0624) | +669,547 |
| 3,000 | 0.3142 (0.3074-0.3278) | 0.3123 (0.2999-0.3443) | 0.0562 (0.0485-0.0624) | +1,384,613 |
| 4,000 | 0.3233 (0.3137-0.3375) | 0.2339 (0.2207-0.2558) | 0.0562 (0.0485-0.0624) | +2,168,360 |
| 6,000 | 0.3369 (0.3264-0.3644) | 0.1540 (0.1408-0.1759) | 0.0562 (0.0485-0.0624) | +3,893,929 |
| 8,000 | 0.3467 (0.3346-0.3780) | 0.1252 (0.1100-0.1369) | 0.0562 (0.0485-0.0624) | +5,722,304 |
| 10,000 | 0.3599 (0.3435-0.3868) | 0.1069 (0.0981-0.1192) | 0.0562 (0.0485-0.0624) | +7,573,243 |

**The escalate rate is measured constant across every swept cost, confirming
a structural fact rather than assuming it.** `decide()`'s low-confidence
check (`low <= p_win <= high`) runs before the cost-dependent
`expected_value`/CONTEST-vs-ACCEPT branch and never reads `cost` or `amount`
(`disputedesk/policy/engine.py`) - which rows escalate is fixed by `p_win`
and `low_confidence_band` alone, both held fixed across this sweep, so the
rate cannot move with cost by construction. Measured median 0.0562
(IQR 0.0485-0.0624) at all 16 swept values, verified per-seed
(`tests/test_eval_cost_sensitivity.py::test_escalate_rate_is_invariant_to_cost_per_seed`)
rather than taken on faith.

**2. Sensitivity: how much of the advantage figure is the
zero-human-cost assumption load-bearing on?** `ESCALATE_MODE="naive_contest"`
credits every escalated row as contested at the *true* win rate
(`won_if_contested`, not a simulated coin flip) and at zero cost beyond the
`representment_cost_inr` already charged for the filing itself - i.e., a
human reviewer's time costs nothing extra. Bounded sensitivity (not a new
headline, `ESCALATE_MODE` and `disputedesk/policy/` untouched): if a human
reviewer instead costs one *additional* `representment_cost_inr` per
escalated-and-contested row (same win rate, same filing cost, plus this one
new cost), every escalated row's credited rupees drops by exactly
`representment_cost_inr`, regardless of win/loss. Aggregated per 1,000
disputes, the overstatement this introduces into the reported
`policy_advantage_median` is `escalate_rate_median x representment_cost_inr x
1000`:

| cost | escalate rate (median) | overstatement (INR/1,000) | reported advantage (INR/1,000) | overstatement / advantage | advantage after this adjustment |
|---|---|---|---|---|---|
| **400 (configured default)** | 0.056189 | **22,475** | 12,923 | **174%** | **-9,553** |
| 600 | 0.056189 | 33,713 | 44,403 | 76% | +10,690 |
| 1,000 | 0.056189 | 56,189 | 152,071 | 37% | +95,882 |
| 2,000 | 0.056189 | 112,377 | 669,547 | 17% | +557,169 |
| 4,000 | 0.056189 | 224,754 | 2,168,360 | 10% | +1,943,606 |
| 10,000 | 0.056189 | 561,886 | 7,573,243 | 7% | +7,011,357 |

**At the configured default (400), the assumption is load-bearing, not
negligible: the overstatement (22,475/1,000) is larger than the entire
reported advantage (12,923/1,000).** Under this one alternative assumption
- human review costs as much as a representment filing, changing nothing
else - the sign of the headline comparison flips: the policy would trail
baseline A by roughly 9,553/1,000 at cost=400, not lead it by 12,923/1,000.
This is not a claim that the alternative assumption is correct (no real
human-review cost was measured or is being proposed here - `PolicyConfig`
gained no new parameter), only that the currently-reported advantage at the
configured cost is thin enough (a ~0.75% margin over baseline A, per the
2026-08-31 sweep entry) that it does not survive this specific, plausible
alternative pricing of escalation. **The ratio shrinks fast as cost rises**
(174% at 400, down to 7% at 10,000) because the overstatement term grows
only linearly in cost while the advantage itself grows faster - so this
assumption matters most exactly where the headline number already sits on
its thinnest margin (near the configured default), and is immaterial in the
high-cost regime where the advantage is already large and robust.

**How:** same 20-seed, 16-cost sweep as the 2026-09-01 "Policy
precision/recall added to the cost sweep" entry
(`python -m eval.run_cost_sensitivity --n-seeds 20 --n-rows 15000`); the
escalate-rate column and the sensitivity table above are read off the same
per-seed/summary CSVs, not a separate run. Full test suite (209 tests,
including the two new escalate-rate checks in
`tests/test_eval_cost_sensitivity.py`) and `ruff check`/`ruff format --check`
pass (three pre-existing, unrelated formatting drifts in files this session
did not touch are left as found).

**Status:** CONFIRMED-RAN

---

## 2026-09-02 — Demo reproducibility check moved from description to enforcement

**Decision:** The Segment A byte-identical claim (`README.md`'s "Running the
demo from a clean clone" section, and the manual "ran it twice, diffed clean"
verifications in the 2026-09-01 and prior DECISIONS.md entries) was never
scripted anywhere - not in CI, not as a Makefile target, not as any checked-in
script. It existed only as prose describing the property and as ad hoc manual
runs recorded after the fact. That check also had a real gap even when done by
hand: `diff` on two empty files, or two files from crashed runs, returns
success - nothing upstream of the `diff` call asserted the demo actually ran
and printed something. Added a CI step (`.github/workflows/ci.yml`, "Demo
reproducibility (Segment A)") that runs `python -m disputedesk.cli.demo
--deterministic-only` twice, fails if either run exits non-zero, fails if
either captured file is empty, and only then diffs the two files.
**Why:** "reproducible" was a claim resting on a check that ran only when a
session happened to run it by hand, and even then couldn't distinguish two
successful identical runs from two crashed/empty ones. A property this project
reports as evidence (CLAUDE.md: "no metric is ever reported... nothing unbuilt
is described as built") needs the check enforced at commit time like every
other headline claim here, not re-verified manually each time someone asks.
**Status:** DECIDED
