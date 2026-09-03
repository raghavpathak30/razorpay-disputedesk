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

---

## 2026-09-02 — CORRECTION: the fallback letter was being submitted to the card network

**Old claim:** `disputedesk/client/razorpay.py`'s module docstring, and
DECISIONS.md's 2026-09-01 letter-drafting entries, described SPEC.md §7 failure
path 2 as working: "invalid LLM output ... falls back to a deterministic
template letter and flags the record for human review. The system degrades, it
does not crash." The demo printed exactly that, and the test
`test_llm_failure_degrades_to_template_and_flags_human_review` asserted it,
ending with `assert result.api_outcome.outcome == "success"` and the comment
"Degrades, does not crash: the (template) letter still gets filed."

**New claim:** it degraded, and then filed the degraded artifact. The template
letter's own last sentence reads "This letter was generated by a deterministic
template because automated drafting was unavailable; it has not been reviewed
by a person yet." That text was sent to Razorpay's contest endpoint as
`summary` with `action="submit"` — i.e. the system told the card network, in
writing, that its evidence had not been reviewed, and submitted it anyway. The
`human_review_required` flag was written to the audit row *after* the filing
decision was already made, and nothing read it.

**Why the old claim was wrong:** the review flag and the letter were two
separate values travelling side by side (`LetterResult.letter` and
`LetterResult.human_review_required`). By the time the letter reached the API
client it had been flattened to a bare `str` (`_EvidenceOutcome.letter_text`)
and the flag had been left behind in the audit row. Nothing in the type system
connected them, so "flagged for review" and "filed" were not mutually
exclusive states — they were simply two unrelated facts that happened to both
be true. Every test asserted the flag was set; none asserted the letter was
withheld, and the one test that looked at filing asserted the *wrong*
behaviour and locked it in.

**Fix:** `disputedesk/evidence/letter.py` — a `DraftedLetter` carrying a
required `provenance` field in `{model, fallback, low_confidence}`, assigned on
the branch that constructs it. `require_submittable()` is the only door to
`action="submit"` and raises `LetterNotSubmittableError` on anything that is
not `model`. It is called inside `RazorpayHttpClient.contest()` *and*
`FakeRazorpayClient.contest()`, before the request body is built, so a future
caller reaching the client directly still cannot file an unreviewed letter;
`contest()`'s parameter is typed `DraftedLetter`, not `str`, so provenance
cannot be dropped on the way. `disputedesk/api/pipeline.py` withholds ahead of
that gate and records an `api_outcomes` row with `outcome="withheld_for_review"`.

**Deliberately not done:** a withheld contest is not silently re-decided as
accept. Accepting is irreversible (Razorpay's accept endpoint moves the dispute
to "lost"), and a letter-drafting failure is no evidence at all about whether
the dispute is winnable. The `policy_branch` on the decision row still reads
`contest`, because that is what the policy engine decided — CLAUDE.md
invariant 4 means an LLM failure must not become a policy decision.

**Cost of the fix, stated plainly:** the system now files strictly fewer
disputes than it did. Every dispute whose letter drafting fails twice becomes a
human's problem instead of an automated filing. That is the correct behaviour
and it is also a throughput reduction; the rate at which it happens is the
letter-drafting failure rate measured in the 2026-09-01 "Letter-drafting
reliability, re-measured after the fix" entry — which, per the next entry
below, is now itself stale.

**Status:** DECIDED. Pinned by `tests/test_evidence_letter_provenance.py`
(11 tests, incl. a construct-a-fallback-and-call-the-submit-path test asserting
both the raise and that no network call was made).

---

## 2026-09-02 — CORRECTION: letters were drafted to 4,000 characters and submitted at 1,000

**Old claim:** `ExplanationLetterOutput.letter_text` was bounded at 4,000
characters, `GroqHttpLLMClient.MAX_COMPLETION_TOKENS = 1512` was derived from
that 4,000-character ceiling ("the budget must be able to reach that
ceiling"), and the drafting prompt asked the model for "50-4000 characters".
The demo prints the drafted letter under the heading "live Groq completion,
verbatim, no truncation".

**New claim:** Razorpay's contest endpoint documents `summary` as "It can have
a maximum length of 1000 characters"
(https://razorpay.com/docs/api/disputes/contest/, re-verified 2026-09-02), and
`RazorpayHttpClient.contest()` closed the gap with `summary[:1000]`. A letter
drafted anywhere near the ceiling the prompt asked for lost up to three
quarters of its body on the wire, silently, with no error and no audit trace.
The demo's "no truncation" heading was true of what it *printed* and false of
what was *submitted*.

**Why the old claim was wrong:** the same number existed twice, in two modules,
with two different values — 4,000 in `evidence/schemas.py` and 1,000 in
`client/razorpay.py` — and the second one was applied with a slice rather than
a check, so the disagreement could never surface as a failure. The prompt was
given the wrong one.

**Fix:** `NETWORK_SUMMARY_MAX_CHARS = 1000` is now defined once, in
`disputedesk/evidence/letter.py`, with the source URL and verification date on
it. The output schema, the drafting prompt, and the request body all read it
from there. Over-limit text fails validation at construction rather than being
truncated: an over-long model draft fails `ExplanationLetterOutput`, triggers
the one repair attempt, and — if that also fails — degrades to the
deterministic template, which the previous entry's gate then withholds from
submission. The one path allowed to shorten text
(`fallback_text_and_provenance`, for the template only) returns
`provenance = low_confidence`, which is equally unsubmittable.

**Prompt version:** new file `explanation_letter_v2.txt` rather than an edit to
v1, per `disputedesk/evidence/prompts.py`'s own rule. v2 states the limit as a
hard constraint ("a letter longer than 1000 characters is rejected outright and
the dispute goes to a human queue instead of being filed") rather than as the
"50-4000 characters" range v1 quoted.

**Measurement this invalidates — flagged, not quietly carried forward:** the
2026-09-01 "Letter-drafting reliability, re-measured after the fix" entry
measured first-draft validation pass rate against the *4,000*-character schema
and the v1 prompt. Both have changed. That number no longer describes the
system and must be re-measured against v2 before it is quoted again. It cannot
be re-measured in this session: no `LLM_API_KEY` is configured in this
environment (there is no `.env`), and CLAUDE.md forbids a test making a network
call, so the run is a manual one-off for whoever holds the key. Two directions
it could move, both plausible, neither assumed: a tighter, clearly-stated
budget may raise the pass rate (the model is no longer being invited to write
past the limit), or it may lower it (1,000 characters is a harder target to hit
while still clearing the 50-character floor and staying coherent). The old
number stays visible in its own entry, marked stale by this one.

**`MAX_COMPLETION_TOKENS` left unchanged at 1512:** the derivation comment in
`disputedesk/evidence/llm.py` now over-provisions rather than under-provisions,
which is the safe direction — a budget wider than the schema ceiling cannot
truncate a valid letter. Re-deriving it downward would be a token-cost
optimisation, not a correctness fix, and is not in this remediation's scope.

**Status:** DECIDED. Pinned by
`tests/test_evidence_letter_provenance.py::test_an_over_limit_model_body_routes_to_review_and_is_never_truncated`
(a stub returning a 3,000-character body twice) and
`test_the_http_client_submits_a_model_letter_verbatim` (a 1,000-character body
arriving on the wire whole).

---

## 2026-09-02 — Observed while verifying the contest endpoint: `action="submit"` needs a document id

**Observation, not a fix:** while re-reading
https://razorpay.com/docs/api/disputes/contest/ to source the 1,000-character
`summary` limit, the same page states that at least one document id must be
provided across the evidence fields when `action="submit"` is used. This
project has no document-upload pipeline (ARCHITECTURE.md "Known gaps"), so
`contest()` sends `summary` and no document ids — meaning a real submit against
the live API would likely be rejected by Razorpay, not merely incomplete.

**Why this is recorded and not fixed:** building a file-storage/upload path is
new scope, not a defect in what exists, and this remediation pass is explicitly
scoped to the eleven defects found by the verification pass. Recording it here
so the gap reads as known rather than undiscovered. ARCHITECTURE.md's "No
document-upload pipeline" bullet is updated to say this outright.

**Status:** UNVERIFIED — not tested against the live API (no key configured in
this environment). Sourced from Razorpay's documentation only.

---

## 2026-09-02 — CORRECTION: the webhook 422'd reason codes Razorpay itself publishes

**Old claim:** `disputedesk/api/schemas.py` constrained `reason_code` to
`Literal[*REASON_CODES]` — the four codes with an evidence strategy — with the
comment "which would otherwise surface as an unhandled 500 for a
`contest`-bound dispute instead of a 422 rejected here, at the boundary, per
this phase's 'validate every field, not just status' review".
`tests/test_api_schemas.py::test_unrecognized_reason_code_is_rejected` asserted
it. The 2026-09-01 "Visa reason code rename" entry and its commit message
(`b5770a1`, "VISA_83 -> VISA_10_4 with legacy wire alias") described a legacy
wire alias; **no such alias was ever implemented** — `grep` for `VISA_83`
across `disputedesk/` found it only inside a comment.

**New claim:** a dispute arriving with `VISA_83` — the code Razorpay's own
published chargeback reference lists for card-absent fraud, and the exact code
`GENERATOR.md` §8 cites as this project's Visa source — was rejected with HTTP
422 and nothing else happened. No decision row, no queue entry, no audit trace.
A real chargeback with a real response deadline disappeared at the boundary.

**Why the old claim was wrong:** it treated "malformed payload" and "reason
code this system has no strategy for" as the same condition. They are not, and
their correct outcomes are opposites. A malformed payload should be rejected
loudly because there is nothing to act on. An unrecognised code is a
well-formed dispute the system merely cannot handle *automatically* — the one
thing it must never do is drop it. Choosing 422 optimised for "no unhandled
500" over "no lost dispute", which is the wrong trade for the actual product.

**Fix:**

1. `data/reference/razorpay_chargeback_codes.csv` — Razorpay's published list,
   all 71 rows across Mastercard/Visa/Amex, transcribed from
   `https://cdn.razorpay.com/files/chargeback_codes.pdf` (retrieved 2026-09-02,
   via `pdftotext -layout`). URL, retrieval date, and method are in the file's
   own header comment. Transcribed verbatim including the source's typos.
2. `disputedesk/evidence/published_reason_codes.py` — loads that fixture; the
   accepted set is now a question about a data file with a source on it rather
   than about a hand-typed enum.
3. `reason_code` on the webhook is now `str` with a length bound only — no
   value check and no character-class pattern, because a *malformed* code must
   also reach the fallback rather than 422. Safe here specifically:
   `reason_code` is never interpolated into a request path (only `id` and
   `payment_id` are, and those keep their strict pattern), reaches SQLite only
   as a bound parameter, and is consumed by an ordinal encoder that tolerates
   unseen values and a dict lookup that now fails closed.
4. Unrecognised codes route to a documented fallback: HTTP 200, a full audit
   row tagged `validation_result="reason_code_unrecognised"`,
   `human_review_required=True`, and an `api_outcomes` row with
   `outcome="withheld_for_review"`. Nothing is filed in either direction.
5. `LEGACY_WIRE_ALIASES = {"VISA_83": "VISA_10_4"}` in `reason_code_map.py` —
   the alias the earlier commit message claimed, now actually implemented and
   applied inside `required_evidence_types`, so a payload carrying the
   published-but-retired code lands on the same evidence strategy rather than
   in the review queue.

**Deliberately not filed, not even as accept:** an unrecognised code withholds
*both* directions, including an ACCEPT the policy engine reached. Accepting is
irreversible, and this system does not know what it would be accepting.

**A test was reversed, not deleted:**
`test_unrecognized_reason_code_is_rejected` asserted the defect. It is now
`test_an_unrecognized_reason_code_is_accepted_at_the_boundary`, with the
reversal and its reason written into the test's own docstring so a reader of
that file sees the history without coming here.

**What this does not fix:** the system still has an evidence strategy for
exactly four codes. Every other published code — 67 of the 71 — is queued for a
person. That is the correct behaviour for a project scoped to one loss class
(SPEC.md: "Exactly one. Nothing else."), and it is also an honest statement
that this system automates a narrow slice of a merchant's dispute volume. The
change is that the other 67 now arrive in a queue instead of vanishing.

**Status:** DECIDED. Pinned by
`tests/test_evidence_published_reason_codes.py` — 82 tests, including one
parametrised over all 71 published codes asserting HTTP 200 for every one, and
separate tests that a malformed code and an out-of-scope published code both
reach the fallback rather than a 422.

---

## 2026-09-02 — CORRECTION: the audit log was not append-only

**Old claim:** `disputedesk/audit/models.py`: "Two tables, both append-only by
construction: `disputedesk/audit/log.py` — the only module that writes to
either — exposes insert-only functions and no update or delete path exists
anywhere in the codebase." `disputedesk/audit/log.py`: "nothing calls `UPDATE`
or `DELETE` anywhere in this module, by construction, not by convention
alone." README and ARCHITECTURE.md both list "append-only audit log" as built.

**New claim:** the *codebase* had no update or delete path. The *database* had
no objection to one. The verification pass opened an ordinary session, mutated
a `DecisionRecord`, committed, then deleted a row — both succeeded. Every claim
above was a claim about the callers that existed on 2026-09-01, restated as if
it were a property of the store. The one place the project got this right
already — idempotency, enforced by a UNIQUE constraint precisely because
PHASES.md said "enforce this in the database, not in application logic" — is
the standard this failed to meet, in the same file.

**Why the old claim was wrong:** "by construction" was doing work the
construction did not do. An audit log's whole value is that its contents can be
trusted by someone who does not trust the application, and every guarantee that
lives only in application code is unavailable to exactly that reader. The
wording ("not by convention alone") shows the distinction was understood and
then not acted on.

**Fix, two independent mechanisms answering two different threats:**

1. **Triggers.** `install_append_only_guards` (`disputedesk/audit/db.py`)
   creates `BEFORE UPDATE` and `BEFORE DELETE` triggers on both tables, each
   `RAISE(ABORT, ...)`. Called from `init_db`, idempotent, so every process
   start re-asserts them. This stops the ordinary path — a bug, a stray script,
   a future code path.
2. **Hash chain.** Every row stores `prev_hash` (its predecessor's `row_hash`)
   and `row_hash` (SHA-256 over `prev_hash` plus the row's business fields,
   `features_json` included). `verify_chain()`
   (`disputedesk/audit/chain.py`) walks both tables and checks both links.
   This catches an edit made by something that *could* drop the triggers.

**What the chain does not do, stated rather than left to be assumed:** it is
tamper-*evident*, not tamper-proof. An actor who can rewrite every row from the
edit point forward, in order, produces a self-consistent chain. There is no
off-box anchor — the head hash is not published anywhere — so nothing outside
the database would notice. What the chain buys is that a tamper stops being a
single silent `UPDATE` and becomes a full-suffix rewrite, and that any copy of
any single `row_hash` taken off-box would catch even that. Recording the limit
because a hash chain is exactly the kind of feature that invites the stronger
claim.

**A guarantee that got weaker, flagged first:** `disputedesk/audit/db.py`'s
docstring previously said "nothing here is SQLite-specific except the
`check_same_thread` connect arg", and CLAUDE.md's stack note says Postgres
should be "a connection-string change, not a rewrite". Trigger DDL is
dialect-specific and only the SQLite form is written and tested here, so
`install_append_only_guards` **raises** `AppendOnlyGuardsUnavailableError` on
any other dialect. A Postgres deployment now needs the connection string *plus*
the equivalent DDL (or `REVOKE UPDATE, DELETE` on the application role, which
is the better Postgres answer). Failing closed was chosen over shipping
untested Postgres DDL and describing it as working — CLAUDE.md invariant 6.
ARCHITECTURE.md's "Known gaps" says this outright.

**One incidental correctness fix found while building this:** `created_at` was
written tz-aware (`datetime.now(UTC)`) into a plain `DateTime` column, which
SQLite stores without the offset and reads back naive. Harmless before, since
nothing compared the two; fatal to a hash chain, which would commit to
`...+00:00` at insert and recompute over `...` at verification, breaking the
chain on rows nobody had touched. `now_utc()` now returns tz-naive UTC —
storing what the database can actually store, rather than special-casing the
hash function around a round-trip that was always lossy.

**A race the chain introduces, and how it is closed:** two concurrent inserts
could read the same chain tail and fork it. `prev_hash` is UNIQUE, so a fork is
a constraint violation rather than a silent branch; the loser re-reads the tail
and retries (`_insert_chained`, bounded at 5 attempts, then
`ChainContentionError` with nothing recorded and nothing filed). Distinguishing
a `dispute_id` collision (idempotency working — return the existing row) from a
`prev_hash` collision (retry) is done by looking up the dispute, not by parsing
driver error text.

**Status:** DECIDED. Pinned by `tests/test_audit_append_only.py` — 12 tests:
ordinary-session UPDATE and DELETE on both tables all raise, a raw-SQL UPDATE
raises, and four chain-tamper tests that drop the triggers through a privileged
connection and assert `verify_chain()` fails — including the hard case where the
attacker recomputes the edited row's own `row_hash` to be self-consistent.

---

## 2026-09-02 — Cost sweep assumptions made explicit: human review excluded, submission success assumed

**Decision:** the cost sweep continues to score **auto-filed disputes only**.
The `withheld_for_review` path Phase 0 added is excluded rather than modelled,
and both of the sweep's load-bearing assumptions are now named constants and
prose in `eval/cost_sensitivity.py` instead of being implicit.

**Why exclude rather than model:** modelling it needs two inputs and only one
is measurable.

- The **reason-code** component of the withheld rate is exactly **0%** on
  every dataset the sweep scores: the generator emits only the four codes with
  an evidence strategy. Asserted, not assumed, by
  `tests/test_eval_sweep_assumptions.py::test_the_generator_never_produces_a_reason_code_that_would_be_withheld`.
- The **letter-drafting** component is **currently unmeasured**. Its only
  empirical input was the 2026-09-01 letter-drafting reliability run, which
  the Phase 0 remediation invalidated by replacing both the output schema
  (4,000 → 1,000 characters) and the prompt (v1 → v2), and which cannot be
  re-measured here (no `LLM_API_KEY` configured).

Choosing a withheld rate with no measurement behind it would put an invented
number into a headline, which is worse than an explicit exclusion. The
rejected alternative was to pick a plausible-looking rate and a human-review
cost and present the result as modelled.

**The exclusion is not neutral, and the direction is stated:** it can only
*overstate* the policy's advantage, never understate it. A withheld dispute is
credited a filing that did not happen and charged nothing for the review that
did. `break_even_human_review_cost_inr` turns that into a checkable number:
at the configured ₹400, the paired advantage of +11,210/1,000 is cancelled if
each human-touched dispute costs about **₹200** to review. That is computed at
the measured ESCALATE rate (5.62%) alone, so it is an upper bound — any
non-zero withheld rate lowers it. For scale, `representment_cost_inr` already
budgets ₹150 of analyst time per *contested* dispute.

**Second assumption, and it is currently false:** the sweep credits recovered
value to every filed contest, i.e. assumes each is accepted for review.
Razorpay's contest endpoint documents that `action="submit"` requires at least
one document id (verified 2026-09-02), and this project sends none because it
has no document-upload pipeline. **A contest filed by this system today would
very likely be rejected by the live API.** Recorded as
`SWEEP_ASSUMES_EVERY_SUBMISSION_IS_ACCEPTED = False` with a test that keeps it
false until the upload path exists. The *comparison* against baseline A is
unaffected (baseline A files through the same client and inherits the same
gap); the absolute rupee totals are contingent on a component that does not
exist, and the README now says so.

**Reproduce:** `python -m eval.run_cost_sensitivity --n-seeds 20 --n-rows 15000`
(the `break_even_review_cost_inr` column); `pytest tests/test_eval_sweep_assumptions.py`.
**Status:** DECIDED.

---

## 2026-09-02 — CORRECTION: the cost sweep used an unpaired estimator on a paired design

**Old claim:** the sweep's headline was `policy_advantage_median =
median(policy) − median(baseline_a)`, per cost. From it the README concluded:
"**Below ≈290**, the policy is **statistically indistinguishable from baseline
A** — the median advantage is under 0.1% of the ~2,000,000 base and its sign
flips between adjacent swept values (noise from individual near-threshold
disputes crossing `decide()`'s cutoff, not a real effect)." The configured-cost
advantage was reported as **+12,923 INR/1,000, ≈0.75% over baseline A**.

**New claim:** every seed scores both arms on the identical generated dataset,
the identical temporal split, and the identical trained model. That is a paired
design. The estimator is now the mean of per-seed differences, with a 95%
percentile bootstrap over seeds and a sign-test count reported alongside.
Under it:

| Cost (₹) | Old (diff of medians) | New (paired mean) | 95% CI | Seeds + |
|---:|---:|---:|---|---:|
| 0 | 0 | 0 | 0 to 0 | 0/20 |
| 50 | −219 | **−131** | −284 to −6 | 12/20 |
| 100 | −163 | **−257** | −563 to +53 | 10/20 |
| 200 | +1,772 | **+1,040** | +289 to +1,787 | 14/20 |
| 300 | +4,209 | **+4,184** | +2,673 to +5,717 | 18/20 |
| **400** | **+12,923** | **+11,210** | **+8,508 to +13,633** | **19/20** |
| 600 | +44,403 | +42,923 | +38,006 to +47,564 | 20/20 |
| 800 | +86,000 | **+95,731** | +88,717 to +102,444 | 20/20 |
| 1,000 | +152,071 | +162,600 | +154,130 to +170,686 | 20/20 |
| 2,000 | +669,547 | +685,586 | +671,494 to +700,821 | 20/20 |
| 4,000 | +2,168,360 | +2,180,044 | +2,151,701 to +2,210,984 | 20/20 |
| 10,000 | +7,573,243 | +7,555,900 | +7,501,981 to +7,607,412 | 20/20 |

**The headline got worse: +12,923 → +11,210, and ≈0.75% → ≈0.66% over
baseline A.** That is the new number.

**Why the old one was wrong:** a difference of medians answers a question
nobody asked — the two medians can come from different seeds, so the statistic
is not an estimate of the paired difference at all. It also carried no
interval, which is what let a per-point sign change be *asserted* to be noise
rather than tested. Seed-to-seed variation here is large and shared (it moves
both arms together), which is exactly the variance pairing removes.

**Conclusions that did not survive, and the sentence that was deleted:** the
"below ≈290 … sign flips … not a real effect" sentence is deleted from the
README rather than softened. The paired data contradicts it in both
directions:

- **₹200 and ₹250 are measurable positives**, not noise (+1,040, CI +289 to
  +1,787, 14/20; and +1,690, CI +291 to +3,018, 14/20). The 14/20 count at the
  low end matches what the verification pass reported.
- **₹50 is a measurable *negative*** (−131, CI −284 to −6, excludes zero) —
  while 12 of 20 seeds are positive. Both are true and both are now reported:
  when the policy loses at this cost it loses far harder than it wins, the
  seven negative seeds running to −1,167 against a best positive seed of +120.
  A majority of seeds improving is not the same as expected value improving,
  and reporting only one of the two statistics would hide that.
- Only **₹0, ₹100 and ₹150** are genuinely indistinguishable from baseline A.

**The finding, in the form the brief asked for:** *no measurable advantage
over "contest everything" at or below ₹150 per representment; the advantage
becomes measurable at **₹200** and grows monotonically above it.* The
configured ₹400 sits above that threshold, but not far above it.

**Grid refined to locate the threshold:** ₹150, ₹250 and ₹350 were added to
the swept costs. The old grid jumped 100 → 200 → 300 and could not resolve
where the transition actually is; asserting "≈290" from it was not supported
by the points measured.

**Reproduce:** `python -m eval.run_cost_sensitivity --n-seeds 20 --n-rows 15000`.
Regression-pinned at CI scale (8 seeds × 5,000 rows, exact values to 4dp) by
`tests/test_eval_cost_sweep_regression.py` — whose own numbers are explicitly
**not results**: at that scale the ₹200 advantage is negative, the opposite of
the headline, which is why that file says so in its first paragraph.
**Status:** CONFIRMED-RAN (20 seeds × 15,000 rows, seeds 0–19, 2026-09-02).

---

## 2026-09-02 — CORRECTION: the TF-IDF baseline had no code, and was measured at a different n than the arm it was compared against

**Old claim:** README — "**We measured whether the LLM actually adds predictive
value here, and it does not.**" TF-IDF + logistic regression: **AUC 0.6371**.
LLM normalisation: **mean AUC 0.4211** (n=60, seed 0). "The gap is wide enough,
and structurally explained enough, to be a real result." Elsewhere the gap was
characterised as a wide margin rather than a close call decided by noise.

**New claim:** on the identical 60 items, with identical cross-validation
folds, paired:

| Arm | AUC (pooled out-of-fold) | Per-fold mean | vs. chance, 95% CI |
|---|---:|---:|---|
| TF-IDF + logistic regression | 0.5392 | 0.5104 | +0.0392 (−0.1349 to +0.2081) — not distinguishable |
| LLM typed fields | 0.3768 | 0.4211 | −0.1232 (−0.2735 to +0.0304) — not distinguishable |

**Paired difference +0.1624, 95% paired bootstrap CI −0.0648 to +0.3858 —
includes zero.** The direction survives; the claim that it is beyond noise
does not.

**Why the old claim was wrong — three separate failures, worst first:**

1. **The baseline had no implementation anywhere in the repository.** 0.6371
   appeared exactly once, as a bare `**Result:**` line appended to the
   2026-08-31 "Generator calibration" *decision* entry — not in the measurement
   format this file mandates, with no n, no seed, no command, and no
   CONFIRMED-RAN status. This file's own header says: "Never let a number
   appear in the README or the pitch video unless it has an entry here marked
   CONFIRMED-RAN." That rule was written in this repository and then broken by
   the single most rhetorically load-bearing number in the README.
2. **The two arms were measured at different sample sizes, and the difference
   was attributed to method.** Re-implementing the baseline shows what 0.6371
   almost certainly was: a large-sample measurement. The same implementation
   scores **0.6479 at n=3,000** — near-reproducing the recorded figure — and
   **0.5104 at n=60**. So roughly half of the apparent 0.216 gap was sample
   size, not extraction method. This is the substantive error; the missing code
   is what allowed it to go unnoticed.
3. **Nothing was paired and nothing had an interval**, so "not a close call
   decided by noise" asserted a statistical property that no computed quantity
   in the comparison could support.

**What is now committed:** `eval/tfidf_baseline.py` (the baseline as a
first-class module, with the vectorizer fit **inside each training fold** —
fitting it on the full corpus would leak test-fold vocabulary and idf weights
and inflate precisely the number the LLM is judged against; a shuffled-label
control guards this), `eval/extraction_comparison.py` (paired out-of-fold
scoring and a paired bootstrap over items), and
`data/reference/llm_normalization_arm_n60_seed0.csv` — the recorded LLM run
itself, which previously lived only in gitignored `data/eval/` and so was
reproducible by nobody. The runner refuses to report anything if the
regenerated items no longer match the recorded arm row for row.

**The LLM arm reproduces exactly** (0.4210648… per-fold mean), which places the
error entirely on the baseline side of the comparison, not the LLM side.

**Not raised, and why:** the brief said to raise n if cheap. It is not — the
LLM arm needs a live API key and ~1,000 Groq calls, and no key is configured
in this environment. The TF-IDF arm's own n=60-vs-n=3,000 spread suggests
n ≈ 1,000 is roughly where this comparison becomes readable. Command:
`python -m eval.run_llm_normalization_quality --n-rows 1000 --seed 0`, then
commit the recording alongside the n=60 one.

**What this does not change:** the LLM's narrow role in this system was never
justified by this measurement, and is not weakened by its correction. The
policy engine is a pure function of `P(win)` and `amount`, the reason-code
mapping is a published lookup table, and no LLM output does arithmetic on
money (SPEC.md §2). That argument is architectural and stands without any AUC
number at all. The README now says so explicitly, so a reader is not left
thinking the boundary rested on this table.

**Reproduce:** `python -m eval.run_extraction_comparison` (no API key, no
network). Exact values pinned by
`tests/test_eval_extraction_comparison_regression.py`.
**Status:** CONFIRMED-RAN (2026-09-02).

---

## 2026-09-02 — Mutation spot-check on the Phase 0 guards

**Result:** three of the Phase 0 tests were checked by breaking the code they
cover and confirming the test fails, then restoring. One per defect.

| Defect | Mutation | Result |
|---|---|---|
| 0.1 letter provenance | `require_submittable`'s check made unreachable (`if False`) | 5 of 11 tests in `test_evidence_letter_provenance.py` fail, including the "no network call was made" test and the fake-client parity test |
| 0.3 reason codes | pipeline's `is_supported_reason_code` gate made unreachable | 69 of 82 tests in `test_evidence_published_reason_codes.py` fail (unsupported codes reach `required_evidence_types` and 500) |
| 0.4 append-only | `DELETE` dropped from the installed trigger verbs | both delete tests fail; the four update tests correctly still pass |

**Why this was worth five minutes:** every defect in this remediation is a
test that passed for the wrong reason or an assertion that pinned the defect
in place. A guard that cannot fail is the exact failure mode being corrected,
so the guards themselves needed the same treatment.

**Caveat:** three mutations is a spot-check, not a mutation-testing run. It
establishes that these three guards are load-bearing; it says nothing about
coverage of the rest of the suite.
**Status:** CONFIRMED-RAN (2026-09-02).

---

## 2026-09-02 — WITHDRAWN: letter-drafting reliability (v1 prompt, 4,000-char schema)

**Withdraws:** the 2026-09-01 "Letter-drafting reliability, re-measured after
the fix" entry's headline result — both demo fixtures passing 20/20 with zero
repairs needed.

**Why:** it was measured against `explanation_letter_v1` and a
4,000-character `ExplanationLetterOutput` ceiling. The Phase 0 remediation
(2026-09-02) replaced both — the prompt is now `explanation_letter_v2` and the
ceiling is the card network's real 1,000 characters. The number therefore
describes a system that no longer exists, and re-running this repository at
any commit after 2026-09-02 cannot reproduce it. A number that cannot
reproduce is worse than a missing one.

**Scope of the withdrawal — deliberately narrow.** Only the pass-rate figure
is withdrawn. Everything in that entry about the *root cause* still holds and
is not affected: that `usage.completion_tokens_details.reasoning_tokens`
counts against the same completion budget as visible output, and that Groq
was honouring the deprecated `max_tokens` alias rather than ignoring it. Both
were verified live against the provider, and neither depends on the prompt or
the schema.

**Not a README change:** the figure was never quoted in `README.md` or
`ARCHITECTURE.md` — checked by grep across both. It lived only here. So the
withdrawal costs the submission nothing, which is exactly why it was taken
rather than left standing as "stale".

**To restore:** re-measure against v2 and append a fresh entry.

    export LLM_API_KEY=...   # Groq
    export LLM_API_URL=https://api.groq.com/openai/v1/chat/completions
    export LLM_MODEL=openai/gpt-oss-20b
    python -m eval.run_llm_letter_validation_reliability

Two directions are plausible and neither is predicted here: a tighter,
explicitly-stated character budget may raise the first-draft pass rate, or
1,000 characters may prove a harder target to hit while still clearing the
50-character floor.

**Status:** WITHDRAWN.

---

## 2026-09-02 — CORRECTION: the leakage guard could not fire on the worst possible leak

**Old claim:** CLAUDE.md and PHASES.md both list the leakage guard as a
first-class test asserting no feature column is derivable from the label, with
"a deliberately leaky control case fails it". `tests/test_generator_leakage_guard.py`
implemented it as a single Pearson threshold, `|r| > 0.9`, over numeric columns.

**New claim:** that guard passes a frame containing an exact copy of the
generator's `p` **and** a perfect string leak, simultaneously. Two independent
reasons:

1. **Correlation against a binary label has a ceiling set by the label's own
   noise.** `won_if_contested ~ Bernoulli(p)`, so a verbatim copy of `p` — the
   most total leak there is — correlates only ~0.36 with it. The threshold was
   0.9. No threshold that also admits legitimate features could have fired.
   The existing "control" test passed only because it used `p` itself as the
   *target*, where the correlation is 1.0 by construction — it never exercised
   the case that mattered.
2. **Non-numeric columns were skipped entirely**, so a column of literal
   `"recovered"`/`"written off"` strings was never examined.

**Fix — three independent guards (`eval/leakage.py`), each catching what the
others miss:**

- **(a) Provenance.** Set equality against `DISPUTE_FRAME_COLUMNS`, a frozen
  allowlist typed out by hand (deriving it from `DisputeRecord.model_fields`
  would let adding a latent to the model silently widen the allowlist to admit
  it — a test asserts the two agree). Plus a value-hash of every feature column
  against every latent column. No statistics, so it holds however noisy the
  label is, and it is the only guard that catches a copied latent carrying no
  label signal at all. Also asserted at the generator boundary itself.
- **(b) Discrimination ceiling.** Bayes AUC — what ranking on true `p` achieves
  — against each feature's univariate AUC, flagging at ≥98% of the Bayes lift.
- **(c) Categorical.** Normalised mutual information plus a per-level purity
  check (any level with ≥30 rows at a label rate of exactly 0 or 1).

**Measured margins, so the thresholds are checkable rather than asserted**
(n=5,000, seed 11):

Bayes AUC **0.7397** (lift 0.2397); flag threshold at 98% = lift 0.2349.

| Feature | AUC | Lift | % of ceiling |
|---|---:|---:|---:|
| ip_geo_billing_distance_km | 0.3326 | 0.1674 | **69.9%** |
| prior_order_count | 0.6252 | 0.1252 | 52.2% |
| avs_match | 0.6186 | 0.1186 | 49.5% |
| cvv_match | 0.6163 | 0.1163 | 48.5% |
| device_fingerprint_known | 0.6067 | 0.1067 | 44.5% |
| days_between_purchase_and_dispute | 0.5691 | 0.0691 | 28.8% |
| amount | 0.4454 | 0.0546 | 22.8% |
| checkout_hour_of_day (noise control) | 0.4921 | 0.0079 | 3.3% |
| purchase_ts | 0.4922 | 0.0078 | 3.3% |
| delivery_confirmed | 0.5063 | 0.0063 | 2.6% |
| filed_at / respond_by | 0.4951 | 0.0049 | 2.1% |
| prior_dispute_count | 0.4973 | 0.0027 | 1.1% |

A copy of `p` sits at 100.0% of the ceiling; so does `-p` and any monotone
transform (AUC is rank-based).

**The margin is thinner than an earlier draft of this comment claimed.** That
draft said the strongest legitimate feature reached "roughly a third of the
ceiling" — written before measuring, and wrong by a factor of two. It is 69.9%.
So 98% has about 28 points of headroom: real, but not enormous, and the tests
now assert it directly (`< 0.85`) so a generator change that narrows it fails
loudly instead of quietly making the constant arbitrary.

**Two design corrections found by the controls themselves, both worth
recording because both were my errors caught by the fixtures rather than by
review:**

1. **The shuffled-label control failed the first implementation.** With the
   label shuffled the Bayes AUC collapses to 0.4972 — a lift of 0.0028 — so
   every feature's residual sampling noise trivially exceeded 98% of it and
   seven legitimate features were flagged. A ratio against a denominator
   indistinguishable from zero is not a measurement. Guard (b) now additionally
   requires an absolute lift ≥ 0.02, which can only ever *suppress* a flag; a
   leak under a shuffled label is still caught by guard (a)'s hash check, which
   needs no signal at all. That division of labour is the argument for having
   three guards rather than one.
2. **Timestamps were being checked by the wrong guard.** As object columns
   `purchase_ts`/`filed_at`/`respond_by` scored NMI 0.1201 — not a leak, an
   artefact of a near-unique column's own cardinality — which would have eaten
   most of the NMI threshold's headroom. They are ordered quantities, so they
   now go through guard (b) instead, where they score 2–3% of the ceiling. The
   remaining categoricals score ≤ 0.000463 against a 0.30 threshold: three
   orders of magnitude of headroom.

**Deliberate exclusions, named rather than silent:** `id`/`payment_id` (unique
per row, assigned from the row index, and excluded from the model's feature set
anyway) and `customer_communication_log` (not a model input — `features/build.py`
excludes it explicitly — and *designed* to carry `true_fraud` signal per
GENERATOR.md §3, so a leak check would flag the generator working as documented;
that signal is measured openly by `eval/extraction_comparison.py` instead). All
three are still covered by guards (a) and (b).

**Reproduce:** `pytest tests/test_generator_leakage_guard.py`.
**Status:** DECIDED.

---

## 2026-09-02 — Hand-rolled batch AUC fuzzed against sklearn; no discrepancy

**Result:** `eval.auc.auc_batch` agrees with `sklearn.metrics.roc_auc_score` to
**1e-12 absolute** across ~18,000 generated comparisons — 600 parametrised
cases of 30 rows each, spanning n ∈ {3, 5, 12, 37, 60, 200}, base rates
0.05–0.95, and 1/2/3/10/1000 distinct score values (1 = every item tied,
2–3 = the heavy-tie shape a bootstrap resample produces) — plus explicit edge
cases: all-identical scores, n=1, n=2 both ordered and tied, all-positive,
all-negative, a reversed perfect ranking, ties straddling the decision
boundary, and boolean-vs-integer label dtypes.

**No discrepancy was found, so no Phase 1 interval changes.** Every confidence
interval reported in Phase 1 stands as published.

**Single-class behaviour, defined rather than discovered:** returns NaN, where
sklearn raises. Deliberate — a bootstrap resample that happens to draw one
class is an ordinary event, not an error, and `paired_auc_difference` drops
those draws. A raise would push per-row exception handling into the hot loop
this function exists to keep fast.

**Also done:** extracted from `eval/extraction_comparison.py` to `eval/auc.py`
so the rebuilt leakage guard can use the same fuzzed implementation rather than
a second copy.

**Reproduce:** `pytest tests/test_eval_auc_batch_property.py` (635 tests).
**Status:** CONFIRMED-RAN (2026-09-02).

---

## 2026-09-02 — CORRECTION: a business-harness assertion computed its expected value from the function under test

**Old claim:** `test_build_business_row_is_internally_consistent_on_a_hand_built_set`
verified the rupee accounting on a hand-built four-dispute case.

**New claim:** its baseline-A assertion computed the expected value by calling
`contest_everything_recovered` — the same production function it was checking:

    baseline_a = contest_everything_recovered(won, amount, COST).sum() / 4 * 1000.0
    assert row["baseline_a_contest_everything_recovered_per_1000_inr"] == baseline_a

That asserts only that `build_business_row` calls that function. **Verified by
mutation:** making `contest_everything_recovered` ignore the representment cost
entirely (pass `0.0` instead of the real cost) left the test passing.

**Fix:** every expected value in that test is now a literal worked out on paper
from SPEC.md §4 and §6, with the four-row derivation written into the docstring
so a reader can check the arithmetic without running anything. The same
mutation now fails the test. A second test was split out for the case the
original could not distinguish — `naive_contest` and `oracle` scoring diverge
only when an escalated dispute would have *lost*, and the original fixture's
escalated row happened to be a win.

**One assertion reviewed and *kept*, against first impressions:**
`test_false_positive_cost_matches_the_fixed_representment_cost_times_count`
round-trips the per-1,000 figure back to a rupee total, which reads
tautological. It is not — mutation-tested by doubling the per-FP cost, which
makes it fail. It is weak (it re-derives from the same relationship), so a
genuinely independent test was added alongside rather than replacing it:
`test_false_positive_and_negative_accounting_against_an_independent_derivation`
recomputes FP and FN in plain numpy from SPEC.md's closed form, never calling
`decide`, `decide_batch`, `false_positive_cost` or `false_negative_cost`.

**Reproduce:** `pytest tests/test_eval_business_metrics.py tests/test_eval_business_harness_regression.py`.
**Status:** DECIDED.

---

## 2026-09-02 — CORRECTION: the oracle single-draw test never tested the historical value, which no longer reproduces anyway

**Old claim:** `test_a_single_realized_draw_is_within_a_few_standard_deviations_of_the_mean`
described itself as confirming that "Phase 1's seed-42
`average_precision_score(y_true, p_true)` = 0.4335" is ordinary sampling noise
around the closed-form oracle value.

**New claim, two separate problems:**

1. **The test drew a fresh sample instead.** It called
   `np.random.default_rng(1)` to generate a *new* Bernoulli draw and checked
   that. The historical value appeared only in the docstring and was never
   under test. The test's name and docstring described work it did not do.
2. **0.4335 no longer reproduces.** Running the current generator at seed 42
   gives **0.4305**. The generator changed after that measurement
   (GENERATOR.md revision 2: `amount` became a weakly causal draw on
   `true_fraud`, and a noise feature was added), so the recorded figure
   describes a dataset this repository no longer produces.

**Fix:** split into a golden-fixture regression asserting the reproducible
historical value (0.4304927827841146, at the frozen seed, drawing nothing) and
a distributional sanity test that draws a fresh sample and says so in its
docstring. A third test pins the 0.4335-vs-0.4305 gap so it cannot be quietly
re-asserted, and a fourth checks the claim the old docstring made but never
tested — that the *historical* value, not a fresh draw, sits within ordinary
sampling noise of the closed form. It does.

**The gap changes no conclusion.** 0.0030 on a quantity whose replicate
standard deviation is far larger; the single-draw-vs-closed-form reasoning
holds at either value. It is recorded rather than silently updated because a
number quoted in a docstring that cannot be reproduced is the defect class this
whole remediation exists for.

**Reproduce:** `pytest tests/test_eval_oracle_replicate_check.py`.
**Status:** DECIDED.

---

## 2026-09-02 — CORRECTION: demo step 6 narrated a timeout it did not induce

**Old claim:** the demo's step 6 printed "Failure path 1: the Razorpay API
times out, then recovers" and "(exercising the exact
disputedesk.retry.call_with_backoff both disputedesk/client/razorpay.py and
disputedesk/evidence/llm.py use)".

**New claim:** the Razorpay API was not involved. The step called
`call_with_backoff` directly on a local closure that incremented a counter and
raised `httpx.ConnectTimeout` on its first call. No client, no request, no
route — the printed narration described a failure of a component the step never
touched. Step 7 (the 429 path), added later, did it properly at the transport
layer; step 6 was never brought up to match.

**Fix:** step 6 now injects a real `httpx.ReadTimeout` from an
`httpx.MockTransport` underneath the real `RazorpayHttpClient`, reached over
the real webhook route — the same shape as step 7. The client builds a real
request, hands it to `httpx`, and gets a real timeout back. It files its own
dispute (`disp_demo_timeout_recover`) and prints its own audit row, so
"filed exactly once after a retry" is now visible in the log rather than
asserted.

**A second piece of false narration found while fixing it:** the shared
`_compressed_sleep_fn` printed "the honoured Retry-After value - matches the
header above" on *every* backoff. A timeout carries no `Retry-After` header;
step 6's 0.5s comes from exponential backoff. It is now a factory taking the
reason as a parameter, and each step states its own truthfully.

**Reproduce:** `python -m disputedesk.cli.demo --deterministic-only`.
**Status:** DECIDED.

---

## 2026-09-02 — The ₹50 loss tail, measured

**Result:** at `representment_cost_inr = 50` the paired mean advantage is
negative while a majority of seeds are positive. The per-seed distribution
(20 seeds × 15,000 rows, seeds 0–19):

| Cost (₹) | Mean | Seeds +/− | Worst seed | Best seed | Spread | Mean loss ÷ mean gain |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | −130.6 | 12 / 7 | −1,166.6 | +119.7 | 1,286.3 | **7.65×** |
| 100 | −257.2 | 10 / 10 | −1,538.5 | +816.5 | 2,355.1 | 2.52× |
| 400 (configured) | +11,210.3 | 19 / 1 | −5,394.4 | +20,597.2 | 25,992 | 0.45× |

(Counts do not sum to 20 at ₹50: one seed's difference is exactly zero — the
policy made identical decisions to baseline A on it.)

**In its own voice:** at ₹50 the policy wins slightly more often than it loses
and loses **7.65 times harder** when it does. That is why the mean and the sign
count disagree, and why reporting either alone would mislead — in opposite
directions depending on which. The asymmetry is a low-cost phenomenon: the
ratio falls monotonically with cost and is below 1 by the configured ₹400,
where the policy wins both more often and bigger.

**Mechanism, offered as an explanation and not a measurement:** at a near-zero
cost almost every dispute clears the `p_win * amount > cost` bar, so the policy
and baseline A agree on nearly every row. The few rows they disagree on are
ones the policy declined to contest — and on a seed where several of those
turn out to have been winnable, the policy forgoes the full `amount` while its
wins are worth only the ₹50 fee it avoided. Small upside, large downside, by
construction of the cost model at that cost.

**Reproduce:** `python -m eval.run_cost_sensitivity --n-seeds 20 --n-rows 15000`
(the "per-seed advantage distribution" block). Pinned at CI scale by
`tests/test_eval_loss_tail.py`.
**Status:** CONFIRMED-RAN (2026-09-02).
## 2026-09-02 — Phase 5 freeze reopened for one AI surface, and re-frozen

**Decision:** the Phase 5 freeze declared on 2026-09-01 is reopened to add the
grounding gate (`disputedesk/evidence/grounding.py`), and only that. The README
now reads "frozen except this surface, re-frozen on 2026-09-02" rather than
withdrawing the freeze.

**Why reopen at all.** The submission is judged on AI Judgment, and the
system's LLM surface was two text-shaping jobs at the end of a deterministic
pipeline. The gate is the one integration point where a model does something a
deterministic function cannot: deciding whether a drafted letter asserts a fact
the record has no field for. `docs/AI-SURFACE.md` records the three candidates
considered, the four killed, and why this one ranked first on measurability and
invariant risk rather than on judgment depth.

**Why only this one.** The tail cost of reopening — the security review pass
over a new LLM output on a submission path, the README's component table, and
Phase 5's three cold runs — is paid once for one integration and roughly twice
for two. `docs/AI-SURFACE.md` §2.2's narrative-consistency candidate was
declined on model-authority grounds and stays declined; it is recorded in the
README's "What we did not build" section rather than deferred silently.

**What did not change.** The policy engine is untouched: `decide()` is still a
pure function of `p_win` and `amount`, `disputedesk/policy/` still imports
nothing from `disputedesk/evidence/`, and
`tests/test_grounding_gate_pipeline.py::TestPolicyEngineCannotReachTheGate`
asserts that as a property of the source rather than a convention.

**Status:** DECIDED.

---

## 2026-09-02 — Grounding gate: a drafted letter must trace to the record before it can be filed

**Decision:** the `explanation_letter` is graded against the seven-field
dispute record before it can be submitted. A letter carrying an assertion the
record contradicts, or an assertion the record has no field for, is withheld as
`LetterProvenance.FAILED_GROUNDING` and queued for a person.

**What the gate is for.** Schema validation checks shape and length; it cannot
check whether a letter is *true*. Two failure classes, and only the second
justifies a model:

- **Contradiction** — the letter asserts something a record field denies.
  Enumerable, and `eval/grounding_baseline.py` (a field-matcher) handles it.
- **Unrecorded assertion** — the letter asserts a fact no field covers at all:
  a tracking number, a signature, a phone call. A deterministic checker
  validates the fields it enumerates; it cannot enumerate what the model
  invented, because that set is neither finite nor known in advance.

**Where it sits.** Furthest from the policy boundary of anything in
`docs/AI-SURFACE.md`: it reads our own model's output against our own record,
never the customer's narrative, never `p_win`. It runs inside
`assemble_evidence_packet`, after the policy decision is made and persisted.

**One-directional, and tested in both directions.** It can move a letter from
`MODEL` to `FAILED_GROUNDING`. It cannot move one toward submission: a letter
arriving non-`MODEL` is returned untouched with no LLM call spent, and no path
in the module constructs a letter with `MODEL` provenance. The policy branch on
the audit row is unchanged by a withhold — `policy_branch` still reads
`contest`; what changed is that the packet was not fit to file, recorded
separately as `validation_result="grounding_gate_withheld"`.

**Fails closed, every mode tested.** Grader raises or times out; malformed JSON
twice; schema violation; an invented field name; extra keys; an empty
assertions list. All six land on `FAILED_GROUNDING`. The empty-list case is
deliberate: an empty verdict means the grader found nothing to check, which is
not evidence the letter is clean, and treating it as a pass would make "return
nothing" the cheapest way to wave anything through.

**Security.** The grader reads model output drafted from
`customer_communication_log`, which is attacker-influenced. The letter is
delimited and labelled as data in the prompt, and
`tests/test_evidence_grounding_security.py` pins that a fully-complying grader
still cannot produce a submittable letter: the injection's most direct ask
(empty list, mark it grounded) is refused by the schema, a format hijack is
unparseable and fails closed, and the gate has no mechanism to write back to
the record. What is *not* claimed is that a live model resists the injection —
that is measured, not asserted.

**Measured:** not yet. See the two entries below.

**Reproduce:** `pytest tests/test_evidence_grounding.py
tests/test_evidence_grounding_security.py tests/test_grounding_gate_pipeline.py`
(no API key, no network).
**Status:** DECIDED (behaviour); measurement UNVERIFIED — see below.

---

## 2026-09-02 — Policy branch rates measured, and what the gate's false-flag rate costs

**Result:** at `PolicyConfig` defaults (`representment_cost_inr=400.0`,
`low_confidence_band=(0.45, 0.55)`), seeds 0–19, n_rows=15,000 per seed, on the
temporal holdout:

| branch | median | IQR |
|---|---:|---|
| CONTEST | **0.8052** | 0.7985–0.8186 |
| ESCALATE | **0.0562** | 0.0485–0.0624 |
| ACCEPT | 0.1363 | 0.1316–0.1407 |

The ESCALATE figure reproduces the 5.62% already recorded in the 2026-09-02
"Cost sweep assumptions" entry, which is the cross-check that says this
measurement and that one describe the same run.

**Why the CONTEST rate was needed.** The gate can only withhold letters that
were going to be *filed*. An escalated dispute never reaches it (no evidence is
assembled on that branch) and an accepted one has no letter. So the gate's
false-flag rate applies to the CONTEST share, not to the whole population:

    human_touched_rate = escalate_rate + contest_rate × gate_false_flag_rate

**The finding, and it is bad news for the gate.** Feeding that into the
break-even the cost sweep already computes (`eval/review_cost.py`):

| gate false-flag rate | human-touched rate | break-even review cost (INR) |
|---:|---:|---:|
| 0.00 | 0.0562 | 200 |
| 0.02 | 0.0723 | 155 |
| 0.05 | 0.0965 | 116 |
| 0.10 | 0.1367 | 82 |
| 0.20 | 0.2172 | 52 |
| 0.50 | 0.4588 | 24 |

Read the other way — the more useful direction while the rate is unmeasured —
**at the ₹150 of analyst time `disputedesk/policy/config.py` already budgets
per contested dispute, the gate's false-flag rate must stay below 2.3% or it
cancels the policy's entire measured advantage over baseline A.** At ₹100 a
review the budget is 6.9%; at ₹200 the ESCALATE rate alone already exhausts it
and there is no room at all.

This is a tighter budget than the gate was proposed against, and it is the
first thing a reader should know about it. It does not depend on the gate's
measured performance — it is arithmetic over numbers already recorded — so it
stands whether or not the API-key run ever happens.

**Still an upper bound, for the same reason as before:** the letter-drafting
component of the withheld rate remains unmeasured (its only empirical input was
invalidated by the v1→v2 prompt and schema change), so it is excluded from the
denominator, and a component excluded from the denominator can only make the
true break-even lower.

**Reproduce:** `pytest tests/test_eval_review_cost.py` pins the arithmetic and
the ₹200 and 2.3% figures. The branch rates come from
`eval.cost_sensitivity.run_seed_pipeline` + `decide_batch` over seeds 0–19 at
n_rows=15000 — the same pipeline `python -m eval.run_cost_sensitivity
--n-seeds 20 --n-rows 15000` runs.
**Status:** CONFIRMED-RAN (2026-09-02, 20 seeds × 15,000 rows).

---

## 2026-09-02 — Grounding gate: NOT MEASURED, and why

**Result: none. There is no gate performance number, and none is claimed.**

The brief this was built to asked for a measured number with an interval rather
than a recorded command, on the basis that a key run would happen first. No
`LLM_API_KEY` is configured in this environment and no `.env` exists, so the
gate arm could not be run. Recording an unmeasured capability behind a point
estimate is exactly the failure the 2026-09-02 TF-IDF correction is about, so
nothing is reported.

**What was built and is reproducible without a key:**

- the gate, its schema, and every failure mode (`pytest tests/test_evidence_grounding*.py`);
- the deterministic baseline (`eval/grounding_baseline.py`);
- the corpus construction and its composition table (`eval/grounding_corpus.py`);
- the interval estimators — Wilson for rates, exact McNemar plus a paired
  bootstrap for arm differences (`eval/grounding_stats.py`);
- the cost-model consequence, in full (entry above).

**One corpus fact that is measured, and caps the claim.** The baseline's
shape-based detector catches **6 of the 12** Class B insertion templates:
`signature`, `tracking`, `phone_call`, `email_open`, `date`, `ip_login`. It
misses the other six — `loyalty`, `no_prior_disputes`, `order_contents`,
`refund_offer`, `terms_accepted`, `warehouse` — which are plain declarative
claims with no distinctive shape. So on this corpus the baseline's Class B
ceiling is 50% by construction of the template set, and a gate scoring near 50%
would be no better than regexes. A reader is entitled to say the six templates
were chosen to be invisible to the baseline; the templates are committed
verbatim and this split is published so that judgment is available to them
rather than hidden. That is the ceiling on any claim built from this corpus.

**The exact commands for the run, in order:**

```
python -m eval.run_grounding_draft --n-letters 120 --seed 0   # needs LLM_API_KEY
python -m eval.run_grounding_eval --seed 0                     # needs LLM_API_KEY
python -m eval.run_grounding_eval --baseline-only              # no key needed
```

The first commits `data/reference/grounding_letters_seed0.csv` so everything
downstream reproduces without a key, as the TF-IDF correction required of the
LLM arm.

**n and power, decided before running:** 120 drafts yields ~120 clean, ~120
Class B and up to 120 Class A items. At n=120 a Wilson interval on a false-flag
rate near 0.05 is about 0.02–0.11 — readable, and narrow enough to place the
rate against the 2.3% budget above. It is *not* narrow enough to distinguish
2.3% from 5%, which is worth saying plainly: if the measured rate lands in that
region, the honest report is that the gate's cost cannot be resolved at this n.

**Status:** UNVERIFIED (not run — no API key in this environment).

---

## 2026-09-03 — Grounding-gate corpus resized before the key run; Class B framing corrected

**This supersedes the "n and power" paragraph of the entry above.** That figure — n=120 — is
kept visible above rather than edited in place; this entry is why it is not the number the key
run should use.

**The power problem, worked through properly.** The entry above eyeballed n=120 as "narrow
enough to place the rate against the 2.3% budget." Computed with the same `eval.grounding_stats.wilson`
the gate's own report uses, the 95% upper bound at **zero** observed false flags is:

| n (clean letters) | Wilson upper bound at 0 flags | clears the 2.3% budget |
|---:|---:|---|
| 120 | 3.10% | no — already past budget at zero observed flags |
| 130 | 2.87% | no |
| 150 | 2.50% | no |
| 200 | 1.88% | yes, with room |
| 250 | 1.51% | yes, comfortably |

n=120 was worse than it looked: even a perfectly clean run (0/120 flagged) produces an interval
whose upper bound sits *above* the 2.3% budget, so the eval could not have cleared the gate even
in the best case it could observe. That is a sizing defect, not a result — it says nothing about
the gate's actual rate, only that the instrument could not have resolved it favourably.

**Decision: draft 250 letters, not 120.** `eval/run_grounding_draft.py --n-letters` default
raised 120 → 250. Every drafted letter with `provenance="model"` yields exactly one clean corpus
item (`eval.grounding_corpus.build_corpus`), so 250 drafts (allowing for the rare fallback-drafted
letter being excluded — measured near 0% failure rate at the current prompt per the 2026-09-01
"Letter-drafting reliability, re-measured" entry) puts the clean arm at 250, giving a 1.51% upper
bound if the observed rate is zero — comfortably clear of 2.3%. Class A and Class B ride along at
the same n (up to 250 each) since they are drawn from the same drafted set; their detection rates
are illustrative and do not need this level of power (see the reframe below for Class B
specifically).

**Cost consequence, stated plainly.** `docs/AI-SURFACE.md`'s original design estimated ~360 live
calls for the grounding-gate measurement (n≈120, one corpus). At n=250: drafting is 2 calls per
letter (normalise + draft) = 500 calls; grading the corpus is one call per item, and a fully
mentions-all-fields corpus is 3×250 = 750 items = 750 calls. **≈1,250 live calls total**, roughly
3.5× the original estimate. This is a direct, arithmetic consequence of the power requirement
above, not scope creep — the original n could not have produced a resolvable answer to the
question the gate exists to answer.

**Reporting is now placed against the budget explicitly, not left for a reader to compute.**
`eval.review_cost` gained `MEASURED_ESCALATE_RATE`, `MEASURED_CONTEST_RATE`,
`MEASURED_ADVANTAGE_PER_1000_INR` (the same figures as the entry above, named so this comparison
and that one cannot silently drift apart), `ANALYST_TIME_BUDGET_INR = 150.0` (the analyst-time
component already named in `disputedesk/policy/config.py`'s `REPRESENTMENT_COST_INR` comment —
not a new figure invented for this comparison), and `budget_verdict(observed: Rate) -> str`,
which reports one of three outcomes from the interval, not the point estimate alone: **clears**
(the whole interval sits under budget), **misses — not economically viable at this operating
point** (the whole interval sits over budget), or **straddles — not resolved at this n** (neither).
`eval.grounding_eval.report()` now prints this line immediately under the false-flag rate.

**If the measured rate misses the budget, that is the reported finding, in those words.** A gate
that costs more in review time than the policy recovers is not economically viable at the current
review-cost estimate, and that is a publishable result on its own — it does not weaken the
arithmetic in the 2026-09-02 "Policy branch rates measured" entry, it confirms that entry's
budget was tight for a reason.

**Class B reframed: lead with the structure, not the 6/12 margin.** The prior entry's framing —
"the baseline catches 6 of 12 templates" — invites exactly the objection made against it: a reader
can always say the other six were chosen to be invisible to the baseline, and no amount of adding
templates fixes that, because the same objection applies to any finite set. The argument that
does not have that weakness is structural, and it was always the real one: **a regex baseline
detects lexical shape** — a tracking-number pattern, a signature keyword, a date format. **A
plain declarative claim** ("the customer is enrolled in our loyalty programme") **has no
distinguishing shape to detect**, by definition of what a regex can represent. The baseline's
ceiling on that half of Class B is therefore set by the expressive limits of pattern matching,
not by which twelve templates happen to be committed. The 6/12 split stays published exactly as
it was — it is real, measured, and worth keeping — but it now illustrates the structural point
rather than carrying the argument itself.

**Explicit disclaimer, going in wherever Class B's numbers are reported:** the measured Class B
margin between the gate and the baseline is construction-dependent — a function of which twelve
templates this corpus happens to contain — and should be read as *an illustration* of the
structural argument above, not as independent evidence for it. Class A stays as designed:
near-parity between the gate and the baseline there is the *expected* result (both arms are
built to catch an enumerable, recorded-field flip), and is reported as a control, not a finding.

**The exact commands for the run, in order — 250, not 120:**

```
python -m eval.run_grounding_draft --n-letters 250 --seed 0   # needs LLM_API_KEY
python -m eval.run_grounding_eval --seed 0                     # needs LLM_API_KEY
python -m eval.run_grounding_eval --baseline-only               # no key needed
```

**Reproduce:** `pytest tests/test_eval_review_cost.py` pins `false_flag_budget()` at 2.30% and
`budget_verdict`'s three-way classification; the power table above is reproducible with
`python -c "from eval.grounding_stats import wilson; [print(n, wilson(0, n, '').ci_high) for n in (120,130,150,200,250,300)]"`.

**Status:** DECIDED (corpus size and reporting format). Measurement itself is still UNVERIFIED —
same reason as the entry above: no `LLM_API_KEY` in this environment.

---

## 2026-09-03 — Five grounding-gate files landed inside an unrelated commit; not repaired by rewriting history

**What happened.** A second session, working on the leakage guard (Phase 2, unrelated to the
grounding gate), committed `0ea4a4d` (`fix(2.0-2.5): rebuild the leakage guard; remove tests that
could not fail`, 2026-09-02T21:41:18+05:30) while five files from this session's still-uncommitted
grounding-gate work were present on disk mid-edit. `git add -A`-shaped staging in that commit swept
them in:

- `disputedesk/evidence/grounding.py`
- `disputedesk/evidence/letter.py` (the `FAILED_GROUNDING` provenance addition)
- `disputedesk/evidence/prompts/grounding_gate_v1.txt`
- `tests/test_evidence_grounding.py`
- `tests/test_evidence_grounding_security.py`

This session's own commit, `1e72158` (`feat: grounding gate - a letter must trace to the record
before it can be filed`, 2026-09-03T00:59:14+05:30), then landed the rest of the same feature
(`disputedesk/api/pipeline.py`, `disputedesk/evidence/assembler.py`, `eval/grounding_*.py`,
`eval/review_cost.py`, the remaining tests) on top, with the five swept files already present
from `0ea4a4d` and therefore showing as unmodified in that second commit's diff. The result: one
feature's implementation is split across two commits with unrelated commit messages, neither of
which names the other half.

**Why history was not rewritten.** Rebasing or amending `0ea4a4d` to remove the swept files would
touch a commit another session's own history and messages describe, and CLAUDE.md's git protocol
reserves rebase/amend for the author's own not-yet-shared work. The tree is correct and green as
committed; the defect is entirely in the historical record being hard to read, not in any file's
content.

**Verified after both commits landed:** `pytest` — 1127 passed; `ruff check .` — all checks
passed; `ruff format --check` — no diffs; `python -m disputedesk.cli.demo --deterministic-only`
run twice, byte-identical output (the same check CI's "Demo reproducibility (Segment A)" step
runs).

**Decision, going forward: no concurrent sessions against this tree.** This is the second
collision inside one calendar day — the first split the grounding gate across two commits: this
session, working alone, found `ARCHITECTURE.md` modified on disk three minutes after `1e72158`
landed (an uncommitted TF-IDF-figure sync, correct and consistent with the already-published
number, left in place rather than reverted or claimed) with two other `claude` processes visible
on the machine, and found `DECISIONS.md` itself modified on disk mid-edit while writing the
corpus-resize entry above. A half-applied edit sitting inside an unrelated commit is exactly the
failure mode `git blame` and a commit message both stop working for, and it is expensive to find
once the session that could explain it has ended.

**Reproduce:** `git show --stat 0ea4a4d | grep -E 'grounding|letter\.py|prompts/'` shows the five
swept files; `git log --oneline -5` shows the two commits and everything between them.
**Status:** RECORDED (2026-09-03).

---

---

## 2026-09-03 — Stale-number audit (Phase 2 addendum, item A)

Systematic sweep: every numeric claim in `DECISIONS.md`, `README.md`,
`ARCHITECTURE.md`, `GENERATOR.md`, and Python docstrings that predates the
2026-09-01/02 validation work, re-run against current code. Triggered by the
2026-09-02 oracle-test correction's claim that 0.4335 "no longer reproduces...
the generator changed under it (GENERATOR.md revision 2)" - which turned out
itself to be wrong, and is the first correction below.

### Finding zero, load-bearing on everything after it: there was no revision-2 code transition

**Old claim** (2026-09-02, the oracle golden-fixture entry): "0.4335 no longer
reproduces. Running the current generator at seed 42 gives 0.4305 - the
generator changed after that measurement (GENERATOR.md revision 2: the
`amount` draw became weakly causal on `true_fraud`, and a noise feature was
added), so the old figure describes a dataset this repository no longer
produces."

**New claim:** false. `git log --oneline -- disputedesk/generator/` shows
exactly one commit that ever wrote this code (`960ced1`, "Phase 1: synthetic
dispute generator per GENERATOR.md") and two later commits that touched it
(`b5770a1`: a string rename with no numeric effect, verified below; `0ea4a4d`:
this session's own leakage-guard assertions, which read already-computed
columns and change nothing upstream of them). `GENERATOR.md`'s "revision 2"
language - the weakly-causal `amount` draw, the `checkout_hour_of_day` noise
feature, all of it - was already fully present in the generator code at
`960ced1`, the very first commit. There was never a code transition for a
later measurement to have been taken before.

**Proof, not assertion:** checked out `960ced1` and `b5770a1` into separate
worktrees and ran `generate_dataset(15000, seed=42, GeneratorConfig())` at
each. Both produce `average_precision_score(y_true, p_true) =
0.4304927827841146` - identical to HEAD, to the full float. The number this
repository's generator has *always* produced at that seed is 0.4305, not
0.4335, at every commit that has ever existed.

**What this means for 0.4335:** it was never computed from any code this
repository has committed. The entry that recorded it (2026-08-31, "Oracle
closed-form vs. single-draw AP, reconciled") describes it as "a Phase 1
sanity check" - consistent with an informal, uncommitted script run before
the generator's final parameters were locked in, not a drift. This is a
different and arguably worse failure mode than the one my earlier correction
described: not "a number that was right and later went stale," but "a number
that was never right and nothing was watching for it." Everything else in
that same entry - the closed-form oracle value (0.4556, reproduces to
0.45566...), the 500-replicate mean (0.4569, reproduces to 0.45685...), its
std (0.0173) and standard error (0.0008) - reproduces exactly, which is what
makes "the generator changed" an implausible explanation on its own terms: if
it had, those would have moved too.

**Correcting my own correction:** the 2026-09-02 entry's diagnosis was wrong
and is itself corrected here, append-only, per this project's own rule for
exactly this situation.

### Every seed-pinned claim in GENERATOR.md reproduces exactly; the unseeded ones do not, and cannot be checked

GENERATOR.md's revision notes cite several "out-of-band sanity check"
measurements from the generator's design process, some pinned to
`n=15,000, seed=42` and some to a larger `n` with no seed recorded:

| claim (GENERATOR.md location) | recorded | today, same seed | delta | reproduces? |
|---|---:|---:|---:|---|
| `AUC(days_between_purchase_and_dispute, true_fraud)`, n=15,000, seed=42 (§1 L5) | 0.3504 | 0.350375 | 0.00002 | **yes, exact** |
| same, n=300,000, seed unrecorded | 0.3507 | 0.3496–0.3527 across seeds 0,1,7,11,42 | up to 0.0011 | not checkable - no seed recorded |
| `AUC(amount, true_fraud)`, n=15,000, seed=42 (§3) | 0.6082 | 0.608249 | 0.00005 | **yes, exact** |
| same, n=200,000, seed unrecorded | 0.5998 | 0.5978 at seed=42 | 0.0020 | not checkable - no seed recorded |
| `reason_code` misclassification rate, n=15,000, seed=42 (§1 L6) | 0.0971 | 0.097067 | 0.00003 | **yes, exact** |

Every claim that names its seed reproduces to the fourth decimal place from
current code. The two that do not name a seed sit close to, but not exactly
on, the current value at every seed tried - consistent with genuine run-to-run
sampling variance at those larger `n` (the θ-ratio derivation these sections
document targets a *population* AUC; any single large-`n` draw lands near it,
not on it), not with a code change, since the seed-42 companion measurement at
the same design point reproduces exactly in every case. Recorded as a
documentation gap (the seed for a large-`n` verification run should have been
written down) rather than a contradiction - there is nothing to withdraw,
because there is no committed way to check what seed those two numbers used.
**Status:** DECIDED (gap noted, not fixed - would require re-deriving which
seed, if any, produced each number, which is not recoverable from the repo).

### Correction: `ARCHITECTURE.md` still asserted the withdrawn LLM-vs-TF-IDF claim

**Old claim:** `ARCHITECTURE.md`'s LLM-boundary section, unedited since
2026-09-01: "We measured whether the LLM adds predictive value, and it does
not," citing AUC 0.4211 against "a TF-IDF + logistic-regression baseline's
0.6371 on the same task."

**New claim:** `README.md` was corrected on 2026-09-02 (the TF-IDF baseline
entry) - `ARCHITECTURE.md` was not, and stood alongside it making the
opposite claim to the one the README now makes. Fixed to match: the paired
comparison (identical 60 items, identical CV folds) gives a difference of
+0.1624 with a 95% CI of −0.0648 to +0.3858, which includes zero. "The LLM
does not add predictive value" is corrected to "at the sample size available,
this cannot be shown either way," with a pointer to the README's fuller
table. The LLM's narrow role is re-stated as resting on the architectural
boundary (policy engine reads only `p_win`/`amount`; reason-code mapping is a
lookup table; no LLM output does arithmetic), not on this measurement -
stated explicitly so a reader does not need this number to trust that
boundary.

**Reproduce:** `python -m eval.run_extraction_comparison`. **Status:**
DECIDED.

### Correction: the LLM normalisation-quality run script would have reproduced its own defect

**Old claim:** none stated as such, but `eval/run_llm_normalization_quality.py`
- the script `README.md`'s own "what would settle it" note tells a future
reader to run, to re-measure the LLM arm at a larger `n` once an API key is
available - imported `TFIDF_BASELINE_AUC = 0.6371` from
`eval.llm_normalization_quality` and printed `beats baseline: YES/NO` against
it on every run.

**New claim:** that is the exact large-sample-baseline-vs-small-sample-LLM-run
comparison the 2026-09-02 TF-IDF correction fixed. Following this project's
own README instruction to re-run this script would have silently reproduced
the defect the correction exists to prevent. Fixed:
`eval.llm_normalization_quality.TFIDF_BASELINE_AUC` is removed (grep confirms
it had exactly two importers, both in this pair of files); the run script now
computes the TF-IDF baseline fresh, on the same generator items and the same
seed as the LLM arm it just collected, via a new
`paired_comparison_against_tfidf` function, and reports the paired bootstrap
comparison instead of a bare boolean. `paired_comparison_against_tfidf`
refuses (raises) if the passed sample's `true_fraud` column does not match a
fresh regeneration at the given `n_rows`/`seed` - the same pairing safeguard
`eval/run_extraction_comparison.py` uses for the committed n=60 fixture,
applied here to a fresh, not-yet-committed sample.

**Reproduce:** `pytest tests/test_eval_run_llm_normalization_quality_comparison.py`
(pure logic, no network - the live run itself needs `LLM_API_KEY`, unavailable
in this environment). **Status:** DECIDED.

### Correction: the human-review-cost sensitivity table used the pre-paired-estimator advantage

**Old claim** (2026-09-02, "ESCALATE rate added to the cost sweep" entry):
at the configured cost, crediting a human reviewer one additional
`representment_cost_inr` per escalated-and-contested row overstates the
reported advantage by 22,475/1,000 against a reported advantage of
12,923/1,000 (174%), flipping the sign to roughly **−9,553**.

**New claim:** the reported advantage in that table was
`median(policy) − median(baseline_a)`, the unpaired estimator Phase 1
corrected. Recomputed with the paired advantage (`data/eval/cost_sensitivity_median_iqr.csv`,
20 seeds × 15,000 rows, this session):

| cost | escalate rate (median) | overstatement (INR/1,000) | paired advantage (INR/1,000) | overstatement / advantage | advantage after adjustment |
|---:|---:|---:|---:|---:|---:|
| **400 (configured)** | 0.056189 | 22,475 | 11,210 | **200.5%** | **−11,265** |
| 600 | 0.056189 | 33,713 | 42,923 | 78.5% | +9,209 |
| 1,000 | 0.056189 | 56,189 | 162,600 | 34.6% | +106,412 |
| 2,000 | 0.056189 | 112,377 | 685,587 | 16.4% | +573,209 |
| 4,000 | 0.056189 | 224,754 | 2,180,044 | 10.3% | +1,955,289 |
| 10,000 | 0.056189 | 561,886 | 7,555,900 | 7.4% | +6,994,014 |

**The finding gets stronger, not weaker, under the corrected estimator.** The
overstatement now exceeds the *entire* paired advantage by 200.5% (was
174%), and the after-adjustment figure is more negative (−11,265 vs −9,553).
The qualitative conclusion - this specific alternative pricing of escalation
flips the sign of the headline comparison at the configured cost, and the
effect fades fast as cost rises - is unchanged and, if anything, understated
by the original table.

**Superseded in spirit, not replaced:** this sensitivity (fixed extra cost
per escalated row) and the Phase 1
`eval.cost_sensitivity.break_even_human_review_cost_inr` (solve for the review
cost that exactly cancels the advantage, ≈₹200 at the configured cost) ask
related but different questions and both stand. Both point the same
direction: the reported advantage is thin against any real human-review cost.

**Reproduce:** the escalate-rate and advantage columns are in
`data/eval/cost_sensitivity_median_iqr.csv` after
`python -m eval.run_cost_sensitivity --n-seeds 20 --n-rows 15000`.
**Status:** DECIDED.

### Everything else checked reproduces exactly

Full re-runs at 20 seeds × 15,000 rows, compared to 4 decimal places against
their recorded values - all matched with no correction needed:

- Phase 2 model quality (`python -m eval.run_harness --n-seeds 20 --n-rows 15000`):
  model PR-AUC 0.3522, prevalence baseline 0.2377, oracle ceiling 0.4572,
  calibration error 0.0270, precision/recall at threshold 0.3537/0.6954.
- Phase 3 business metrics (`python -m eval.run_business_harness --n-seeds 20 --n-rows 15000`):
  `zero`/`oracle`/`naive_contest` recovered totals (1,559,504 / 1,727,152 /
  1,714,015), baseline A/B, FP/FN counts and costs, escalated share (0.0433).
- Oracle closed-form and replicate-mean at seed 42: 0.4556 and 0.4569,
  std 0.0173, SE 0.0008.
- Every precision/recall/escalate-rate column in the 2026-09-01/09-02
  cost-sweep entries (only the advantage column, corrected above, was wrong).

### The mechanism: a generator fingerprint gate, always run

`eval/generator_fingerprint.py` hashes `generate_dataset`'s full output - every
value of every column of both frames, not a summary statistic - at one fixed
small `(n_rows=2000, seed=0)`, and `tests/test_generator_fingerprint.py`
asserts it against a committed constant on every `pytest` run (CI included, no
flag to skip it). This is the mechanism item A asked for: a generator change -
including one with no visible numeric effect, like the `VISA_83`→`VISA_10_4`
rename, which the test suite pins as a case that must still move the hash -
now fails a fast, always-on test by name, rather than surviving until someone
happens to re-check a specific downstream figure. Existing exact-value golden
fixtures (`tests/test_eval_oracle_replicate_check.py`,
`tests/test_eval_cost_sweep_regression.py`,
`tests/test_eval_extraction_comparison_regression.py`,
`tests/test_eval_loss_tail.py`, `tests/test_eval_ablation.py`) are now tagged
`GOLDEN FIXTURE` in a comment so a failure of the fingerprint test names
exactly what else needs re-running and re-committing.

Verified live: mutating `logit_intercept` from `-1.8` to `-1.80001` (a
5th-decimal change to one config constant) fails
`test_the_generator_fingerprint_matches_the_committed_value` immediately.

**Reproduce:** `pytest tests/test_generator_fingerprint.py`. **Status:**
DECIDED.

---

## 2026-09-03 — Single-feature ablation: how much of the advantage is one geo feature (Phase 2 addendum, item B)

**Why this was run.** `eval/leakage.py`'s discrimination-ceiling guard, built
in Phase 2, measures each feature's univariate AUC as a byproduct of checking
none of them is a leak. `ip_geo_billing_distance_km` reaches 69.9% of the
Bayes ceiling on its own - not a leak (98% is the flag threshold, and the
shuffled-label control proves this isn't one), but strong enough on its own
that, against a full-model advantage over "contest everything" of only ≈0.66%
at the configured cost, the question "is the model mostly one feature"
deserved a measured answer rather than a guess.

**Method.** The exact same business harness, seeds, paired estimator, and
cost sweep as Phase 1 (`eval.cost_sensitivity.summarize_sweep`), run three
times with the model restricted to: the single strongest feature
(`ip_geo_billing_distance_km`), the top three by the guard's own ranking
(`+ prior_order_count, avs_match`), and the full twelve-feature set (unchanged
from every other headline number). `eval/ablation.py`'s
`predictions_for_feature_subset` fits its own `lgb.LGBMClassifier` directly
rather than calling `disputedesk.model.train.train`, because that function
hardcodes the categorical-feature list against the full declared set and
would be asked for a column not present in a restricted `X_train`; production
training code (`disputedesk/model/train.py`) is untouched.

**Result**, seeds 0–19, n_rows=15,000 per seed, paired advantage over
baseline A (mean of per-seed differences, 95% bootstrap CI, seeds positive):

| cost (₹) | top-1 feature | top-3 features | full (12 features) | top-1 as % of full |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | — |
| 50 | −85 (−174 to −9), 9/20 | −96 (−221 to 12), 12/20 | **−131** (−284 to −6), 12/20 | not meaningful* |
| 100 | −602 (−959 to −255), 6/20 | −629 (−991 to −278), 7/20 | **−257** (−563 to 53), 10/20 | not meaningful* |
| 200 | −839 (−1,656 to −21), 8/20 | −1,500 (−2,610 to −431), 8/20 | **+1,040** (+289 to +1,787), 14/20 | not meaningful* |
| 300 | +994 (−1,169 to +3,228), 11/20 | +1,116 (−416 to +2,568), 13/20 | **+4,184** (+2,673 to +5,717), 18/20 | 23.7% |
| **400 (configured)** | **+7,377** (+4,643 to +9,736), 18/20 | **+7,717** (+4,153 to +10,912), 17/20 | **+11,210** (+8,508 to +13,633), 19/20 | **65.8%** |
| 600 | +30,434 (+25,263 to +35,616), 20/20 | +34,957 (+29,210 to +40,047), 20/20 | +42,923 (+38,006 to +47,564), 20/20 | 70.9% |
| 800 | +78,730 (+72,104 to +85,732), 20/20 | +84,820 (+76,592 to +92,403), 20/20 | +95,731 (+88,717 to +102,444), 20/20 | 82.2% |
| 1,000 | +147,968 (+141,127 to +154,480), 20/20 | +157,738, 20/20 | +162,600 (+154,130 to +170,686), 20/20 | 91.0% |
| 2,000 | +672,273, 20/20 | +679,652, 20/20 | +685,587, 20/20 | 98.1% |
| 4,000 | +2,211,923, 20/20 | +2,203,442, 20/20 | +2,180,044, 20/20 | **101.5%** |
| 6,000 | +3,970,096, 20/20 | +3,942,748, 20/20 | +3,904,450, 20/20 | 101.7% |
| 10,000 | +7,698,799, 20/20 | +7,613,152, 20/20 | +7,555,900, 20/20 | 101.9% |

\* below ≈₹300 the full-model advantage itself is not measurably different
from zero (Phase 1's own finding), so a ratio against it is a ratio against a
number close to zero and is not a meaningful percentage - reported as raw
figures only in that region.

**The finding, stated plainly: at the configured cost, one feature -
`ip_geo_billing_distance_km` - alone captures roughly two-thirds (65.8%) of
the model's entire measured advantage over "contest everything." Three
features capture 68.8%.** The other nine features, combined, add nine more
percentage points of advantage. This is not a criticism of the model - LightGBM
finding and using the single strongest signal is exactly what it is supposed
to do - but it is a materially different story than "a twelve-feature model
learned a rich pattern," and a panel asking "what does the model actually
learn" should be answered with this table, not a guess.

**A second finding, more surprising and worth stating with equal plainness:
above ≈₹2,000, the restricted models slightly *exceed* the full model's
advantage** (101–102% at ₹4,000–10,000). The full twelve-feature model is not
dominant at every cost point; at high cost, fewer features does marginally
better on this metric. Not investigated further here - a plausible mechanism
is that at a high breakeven `p_win` threshold, the extra features contribute
noise as often as signal to the ranking of the few highest-confidence
disputes that still clear the bar, but this is offered as a hypothesis, not a
finding, and no code was changed in response to it (this session's own
standing rule: report what comes out, do not chase a better number).

**Golden fixture:** CI-scale exact values (8 seeds, 5,000 rows, cost=400 only)
committed in `tests/test_eval_ablation.py`'s
`COMMITTED_ABLATION_ADVANTAGE_AT_400`. At that reduced scale the three
variants do **not** preserve the top1 ≤ top3 ≤ full ordering the headline
table shows (top3's paired mean is actually negative there) - recorded as a
finding about how noisy a 3–8-seed read of this comparison is, not
suppressed. No ordering is asserted in the committed test for exactly this
reason.

**Reproduce:** `python -m eval.run_ablation --n-seeds 20 --n-rows 15000`.
Writes `data/eval/ablation_{top1,top3,full}_median_iqr.csv` and matching
per-seed CSVs. Unit/structural tests (feature-restriction is real, the "full"
variant matches the unrestricted harness bit-for-bit, unknown feature names
raise): `pytest tests/test_eval_ablation.py`.

**Status:** CONFIRMED-RAN (2026-09-03).

---

## 2026-09-03 — A limit on the LLM extraction-comparison's eval design, not just its sample size (Phase 2 addendum, item C)

**The gap.** `customer_communication_log` is excluded from the model's own
feature set - deliberately, per `disputedesk/features/build.py`'s own
docstring and CLAUDE.md's boundary rule - specifically *because* it is
designed to carry `true_fraud` signal (GENERATOR.md §3), and that signal is
instead measured openly by the LLM-vs-TF-IDF extraction comparison
(`eval/extraction_comparison.py`, corrected 2026-09-02). What has not been
stated plainly until now is what that comparison is actually extracting
signal *from*.

**The generator's comms-log signal is a 4×4×4 slot template, not open text.**
`disputedesk/generator/comms.py`: every `true_fraud`-carrying message is
assembled from exactly three content slots - `opening`, `claim`, `detail` -
each drawn from a **fixed pool of four fixed strings**
(`_OPENINGS`/`_CLAIMS`/`_DETAILS`), at `true_fraud`-conditioned sampling rates
documented elsewhere as a "max ratio 1.5:1" (deliberately mild, per the
session-2 fix for the exact-string-leak defect this design replaced). Tone
(`_SIGNOFFS_POLITE`/`_SIGNOFFS_TERSE`), inclusion of the detail slot, near-empty
messages, an irrelevant aside, and character-level typo/lowercase noise are
all drawn from `relationship_genuineness` or pure noise, not `true_fraud` - so
the entire `true_fraud` signal in every message lives in which 3 of 64
possible (opening, claim, detail) combinations were drawn, tilted by a small
weight difference.

**Why this matters, stated as a limit and not a defect.** This was the
correct design choice for what it was solving - the *previous* comms-log
design used disjoint, class-exclusive template sets, which let a bag-of-words
model recover `true_fraud` as a near-perfect string match, an unambiguous
leak. The fix traded that leak for a narrower, harder, weight-tilted signal
on purpose. But it means the extraction-comparison's paired result (difference
+0.1624, 95% CI −0.0648 to +0.3858, includes zero) is not a general claim
about LLM-vs-TF-IDF extraction from free-form customer messages - it is a
measurement of extracting a specific, deliberately subtle, closed-vocabulary
signal from a 64-combination template, at n=60. **A real customer-service
inbox has an unbounded vocabulary, genuine typos, code-switching, and
context a 4×4×4 template cannot represent**, and nothing in this repository
measures whether either extraction method's relative performance would
transfer to that setting.

**Two distinct limits, both real, neither substituting for the other:**

1. **Sample size** (recorded 2026-09-02): n=60 is too small for either arm to
   be distinguished from chance, let alone from each other.
2. **Eval design** (this entry): even at a larger `n`, the comparison would
   still be measuring extraction from a closed, synthetic, 64-combination
   template - not from open text. Raising `n` (the "what would settle it"
   note in `README.md`) addresses limit 1. It does not address limit 2,
   because the template's ceiling on what signal exists to extract is fixed
   regardless of how many draws from it are scored.

**What this does not change:** the LLM's narrow architectural role (drafting
text, never deciding, never doing arithmetic on money) does not rest on this
comparison and is unaffected by either limit. Both limits belong in Phase 3's
Limits section alongside the operating-point objection (0.75%→0.66% advantage,
precision 0.2543 vs. 0.2377 no-skill floor) and the sample-size caveat, per
this session's brief - carried here first so Phase 3 assembles rather than
re-derives them.

**Status:** DECIDED.

---

## 2026-09-03 — Two unseeded large-n sanity-check AUCs, re-run with a recorded seed

**Old claim** (GENERATOR.md §1 L5 and §3, both dated "session 2, out-of-band
sanity check", never corrected until now): `AUC(days_between_purchase_and_dispute,
true_fraud) = 0.3507` at n=300,000, and `AUC(amount, true_fraud) = 0.5998` at
n=200,000. Neither cited a seed.

**Found by:** the 2026-09-03 stale-number audit. Both claims' seed-42
companion measurements at n=15,000 reproduce exactly (0.3504 and 0.6082); the
two large-n figures do not hit exactly at any of five seeds tried (0, 1, 7,
11, 42), landing within 0.001–0.002 of each — consistent with genuine
run-to-run sampling variance at those scales, not a code change, but not
reproducible as recorded either way.

**New claim:** re-run at the same `n` with the seed pinned to 42, for
consistency with every other seed-42 measurement in this document:

| Claim | Old (unseeded) | New (n unchanged, seed=42) |
|---|---:|---:|
| `AUC(days_between_purchase_and_dispute, true_fraud)`, n=300,000 | 0.3507 | **0.3496** |
| `AUC(amount, true_fraud)`, n=200,000 | 0.5998 | **0.5978** |

Both deltas are within ordinary sampling noise for an AUC estimated at this
`n` from a Beta-derived population target (the θ-ratio derivations these
sections document target a population value; any single large-`n` draw lands
near it, not on it) - this is not a claim the original figures were wrong on
their own terms, only that they cannot be reproduced as recorded and are
therefore replaced with a version that can be. `GENERATOR.md` updated in
place at both citations, with this entry linked from each.

**Reproduce:**

    python -c "
    from sklearn.metrics import roc_auc_score
    from disputedesk.generator.config import GeneratorConfig
    from disputedesk.generator.pipeline import generate_dataset
    f, d = generate_dataset(300000, 42, GeneratorConfig())
    print(roc_auc_score(d['true_fraud'], f['days_between_purchase_and_dispute']))
    f2, d2 = generate_dataset(200000, 42, GeneratorConfig())
    print(roc_auc_score(d2['true_fraud'], f2['amount']))
    "

**Status:** DECIDED.

---

## 2026-09-03 — Phase 3 key run: blocked by the account's daily token budget, not by rate limiting

**What was attempted.** Two live runs of `eval.run_grounding_draft
--n-letters 250 --seed 0`, both against `openai/gpt-oss-20b` via Groq. Both
failed on `429 Too Many Requests`. The first failure, at letter 167 of 250,
turned out to be a real bug independent of the rate limit: the script held
every row in memory and wrote its output CSV once, at the end of the loop, so
the crash lost all 167 successful, budget-consuming calls with nothing
persisted. Fixed (this entry's companion commits: `81f0e2a` incremental
checkpointing and a related crash-on-long-corpus-text fix, `483999c` making
the resumable loop testable with a stub client). The second run, relaunched
with the fix and 2-second inter-letter pacing, failed again — immediately, on
the very first letter's draft call.

**The real cause, read from the 429 response body rather than assumed from
the header:**

    Rate limit reached for model `openai/gpt-oss-20b` in organization ...
    on tokens per day (TPD): Limit 200000, Used 199763, Requested 491.
    Please try again in 1m49.728s.

This is **not** the per-minute or per-day *request*-count limit — that one
(`x-ratelimit-limit-requests: 1000`) had 649 of 1,000 untouched throughout.
It is the account's **daily token budget for this model, 200,000 TPD**,
already at 99.9% by the time of the second crash. Pacing, retry counts, and
checkpointing all address *rate*; none of them touch a *daily total* that is
already spent. "Blocked until tomorrow" would still be the wrong takeaway,
worked out below.

**The arithmetic, so the actual constraint is legible rather than just
"blocked":**

- 167 letters completed successfully in the first run before the crash =
  334 individual API calls (`normalize_communication_log` +
  `draft_explanation_letter`, one each per letter).
- Treating the day's ~200,000-token budget as consumed predominantly by that
  run: **≈599 tokens per individual call** (200,000 ÷ 334), or **≈1,198
  tokens per letter** counting both calls together — this session's own
  incidental testing (a handful of probes and a 3-letter sanity check earlier
  in the session) accounts for a small, single-digit-percent share of the
  total and is folded into the same estimate rather than subtracted out, so
  the true per-call figure is very slightly lower than stated.
- A full n=250 measurement needs drafting (250 × 2 = 500 calls) plus grading.
  Grading just the clean class (the false-flag rate specifically) is another
  250 calls; the full three-class corpus is up to ~750. So the job needs
  **750–1,250 calls total**.
- At ≈599 tokens/call, that is **≈449,000–749,000 tokens** — against a
  200,000 TPD ceiling, **2.25 to 3.75 days of budget, not one.**

**The reservation, checked separately because it is the other lever a reader
would reach for.** `GroqHttpLLMClient.MAX_COMPLETION_TOKENS = 1512`
(`disputedesk/evidence/llm.py`) is sent as `max_completion_tokens` on every
call, normalize and draft alike. Its derivation comment dates to the
4,000-character letter ceiling Phase 0 replaced with the current 1,000-character
one (`disputedesk/evidence/letter.py`'s `NETWORK_SUMMARY_MAX_CHARS`) — Phase 0
left the constant unchanged, flagging it then as "over-provisions rather than
under-provisions, which is the safe direction... a token-cost optimisation,
not a correctness fix, and is not in this remediation's scope." Applying the
same derivation method the constant already documents, to the ceiling as it
actually stands today:

    1000 chars / 3.6 measured chars-per-token ≈ 278 visible tokens
    + ~100 tokens JSON scaffolding (unchanged - same schema shape)
    + ~300 tokens reasoning margin (kept, for the same reason as the
      original: a caller can raise reasoning_effort above this class's
      "low" default without touching the constant)
    278 + 100 + 300 = 678 tokens

**678 versus 1512 reserved — about 2.2× over-provisioned** for what a
1,000-character-capped letter can actually need.

**Whether shrinking it would raise the daily call ceiling is genuinely
uncertain, and stated as such rather than assumed.** The 429 bodies captured
today show a `Requested` figure (491, then 612, on two calls that both sent
`max_completion_tokens=1512`) that does not match 1512 at all — evidence that
Groq's TPD accounting is not simply "reserve the full configured cap per
call." If TPD is charged against *actual* generated tokens — consistent with
this project's own previously-recorded live measurements of 349–581
completion tokens per call at `reasoning_effort="low"`, both comfortably
under even a 678 cap — then lowering `MAX_COMPLETION_TOKENS` mainly buys a
tighter ceiling against a rare pathological long completion, not a higher
sustainable calls-per-day rate, because real usage is already running well
under a much smaller number than 1512. The true throughput driver would then
be call count and prompt size, which the completion cap does not bound. No
change was made to the constant — this is a report for a deliberate decision
later, per this session's explicit instruction, not a fix.

**Not done, and named rather than silently skipped:** 0.1 (grounding-gate
false-flag rate) and 0.2 (raised-n LLM-vs-TF-IDF) did not run. See the
"grounding gate: still not measured" and "extraction comparison: raised n not
run" entries alongside this one for exactly what stands in the README in
their place.

**Multi-day resume, verified without spending any more of today's budget:**
`draft_corpus` (`eval/run_grounding_draft.py`, refactored in `483999c`) is
tested end-to-end against a `FakeLLMClient` calling it twice at the same
checkpoint path — the second call resumes from exactly where the first left
off, and a third call against a fully-done corpus makes zero client calls at
all (an exploding stub proves it — see
`tests/test_eval_run_grounding_draft.py`). The command sequence for a real
multi-day run, once budget allows:

    # Day 1 (and any following day): identical command, safe to interrupt or
    # re-run any number of times - already-checkpointed positions are skipped.
    python -m eval.run_grounding_draft --n-letters 250 --seed 0

    # Once every position is drafted (check with):
    python -c "from pathlib import Path; from eval.run_grounding_draft import already_drafted_positions; print(len(already_drafted_positions(Path('data/reference/grounding_letters_seed0.csv'))))"

    # Then, in a session with budget remaining:
    python -m eval.run_grounding_eval

Also recorded in `NUMBERS.md` alongside the single-session command.

**Reproduce the arithmetic above:** it is post-mortem reasoning over the
crashed run's own printed progress and the 429 response bodies captured
during today's session, not a re-runnable command — there is nothing to
reproduce until budget resets, at which point the real measurement (not this
estimate) supersedes it.

**Status:** BLOCKED (external, account-level, resets on Groq's own daily
cycle — not a defect in this codebase). Arithmetic: DECIDED.

---

## 2026-09-03 — Phase 5 freeze reopened for Phase 3 documentation and audit work, and re-frozen

**Decision:** the Phase 5 freeze (declared 2026-09-01, reopened and
re-frozen once already on 2026-09-02 for the grounding gate) is reopened for
today's work and **re-frozen on 2026-09-03**.

**What reopened it, and why none of it is new capability:**

- The stale-number audit, single-feature ablation, and the eval-design limit
  on the extraction comparison (Phase 2 addendum, earlier commits today) —
  measurement and documentation work: re-running existing harnesses, adding
  a generator fingerprint test, correcting recorded figures. No production
  code path changed behaviour.
- Phase 3's write-up: `README.md`'s top-line description corrected to state
  the document-upload gap in the opening paragraph rather than as a
  footnote; a `Limits` section consolidating figures that already existed
  elsewhere in the repo; `NUMBERS.md` and `Makefile` (`make verify`,
  `make verify-key`) mapping every numeric claim to its reproduce command;
  `docs/QA-PREP.md`. All documentation - nothing here is a code path.
- Two bug fixes, found while attempting the key run and made necessary by
  it, not chosen freely: `eval/run_grounding_draft.py` now checkpoints
  incrementally instead of losing all progress to any interruption, and
  `eval/grounding_eval.py`'s `_as_letter` no longer routes corpus text
  through `DraftedLetter`'s submission-length validation, which crashed
  grading on any mutated letter over 1,000 characters. Both are `eval/`-only
  - measurement tooling, not `disputedesk/` production code, and both were
  required to make an *already-declared* capability (the grounding-gate
  measurement Phase 5's 2026-09-02 reopening already approved) actually
  runnable, not to add a new one.
- The key run itself did not complete (see the 2026-09-03 "key run: blocked"
  entry) - no measurement was added, and none of today's changes depend on
  one having happened.

**What did not change:** no new AI surface, no new production code path, no
change to `disputedesk/policy/`, `disputedesk/evidence/grounding.py`, or any
decision-making code. `docs/AI-SURFACE.md`'s ranking and the four killed
candidates are untouched.

**Status:** DECIDED.

---

## 2026-09-04 — Phase 5 freeze reopened, scoped to one defect: zero document ids on submit, fixed

**Decision:** the Phase 5 freeze (declared 2026-09-01, reopened and
re-frozen on 2026-09-02 for the grounding gate, reopened and re-frozen again
on 2026-09-03 for documentation and audit work) is reopened for exactly one
defect and **re-frozen on 2026-09-04**.

**The defect.** Razorpay's contest endpoint documents that `action="submit"`
requires at least one document id attached across the evidence-type fields
(`https://razorpay.com/docs/api/disputes/contest/`). `RazorpayHttpClient.contest()`
never populated any such field — every `action="submit"` this system ever
sent went out with zero document ids, a shape the real API's own documented
contract rejects. `tests/test_client_document_contract.py`, committed at
`48d03bf` before any fix, proved it: `RazorpayHttpClient().contest(...)`
called exactly as every other test in the repository already called it, and
the request body asserted on the wire —

    action='submit' carries zero document ids across every evidence-type
    field - Razorpay's documented contract requires at least one.
    Body: {'amount': 500000, 'summary': '...', 'action': 'submit'}

— `assert 0 >= 1` failed, against unmodified code, as the dated record of the
defect.

**What was built, scoped to exactly this:** evidence bundle assembly →
render → upload → capture document ids → include them in the submit
payload. Nothing else changed.

- `disputedesk/evidence/documents.py` (new): pure, no I/O. Renders each
  required evidence type's known facts (never fabricated — no invented
  signatures, tracking numbers, or receipts; see `docs/AI-SURFACE.md` §0.2
  and `draft_letter.py`'s deterministic-fallback discipline) into a
  hand-rolled minimal PDF — chosen because Razorpay's document endpoint
  accepts only `image/jpg`, `image/jpeg`, `image/png`, `application/pdf`,
  not plain text, and adding a PDF library was not asked for
  (`CLAUDE.md`: "Do not introduce a new library without asking"). PDF
  string-literal escaping is tested against literal injection-attempt
  strings, since the customer-communication document embeds
  attacker-influenced free text verbatim. The writer's byte-level structure
  (xref offsets, `startxref`) is independently checked against Poppler's
  `pdfinfo`/`pdftotext`, not just against itself.
- `disputedesk/client/razorpay.py`: `contest()` now takes the rendered
  bundle, uploads each document via the new `upload_document()` (real
  multipart `POST /v1/documents`, `purpose="dispute_evidence"`), and attaches
  the returned ids under their evidence-type keys before submitting. An
  empty bundle is rejected before any network call — `DocumentUploadError`,
  the same fail-closed shape `LetterNotSubmittableError` already gives the
  letter-provenance gate, deliberately duplicated in `FakeRazorpayClient` so
  the demo script and most tests can't re-open the invariant by a path only
  they exercise.
- `disputedesk/evidence/assembler.py` / `disputedesk/api/pipeline.py`:
  `EvidencePacket.evidence_bundle` and `_EvidenceOutcome.evidence_bundle`
  thread the rendered bundle through. `_file_if_needed` withholds for review
  — not filed, no network call — when the bundle is `None` or empty, exactly
  as it already withheld a non-`model` letter. An upload failure that
  happens *inside* the client call (after the bundle looked fileable)
  surfaces as `_file`'s existing "failed" outcome, the same as a contest-PATCH
  timeout always has — the system degrades, it does not crash.

**Invariants proven to still hold, each with its own test:**

- The policy engine still cannot reach the LLM or the new documents module —
  `tests/test_grounding_gate_pipeline.py::TestPolicyEngineCannotReachTheGate::test_the_policy_package_imports_nothing_from_evidence`
  source-scans all of `disputedesk/policy/*.py`; unchanged, and it already
  generically covers the new module since nothing in `policy/` imports it.
- The grounding gate still gates before any upload, not just before the
  contest PATCH —
  `tests/test_grounding_gate_pipeline.py::TestWithheldLetterIsNotFiled::test_nothing_is_filed_in_either_direction`
  now also asserts `razorpay.upload_calls == []` for a withheld letter.
- Letter provenance still gates submission via the new signature —
  `tests/test_evidence_letter_provenance.py`'s three direct-`contest()` tests
  now pass a populated `evidence_bundle` specifically, proving the provenance
  check still wins even when a bundle is present.
- Idempotency still holds for re-submitted uploads —
  `tests/test_evidence_document_pipeline_invariants.py::test_a_replayed_event_does_not_upload_documents_a_second_time`:
  a replayed webhook event adds zero further `upload_document` calls, the
  same DB-level `dispute_id` UNIQUE check every other idempotency guarantee
  in this system already rests on.
- The audit log stays append-only and its hash chain still verifies after a
  real contest-with-documents flow —
  `test_evidence_document_pipeline_invariants.py::test_the_audit_chain_still_verifies_after_a_contest_with_documents`.
- Every new failure mode fails closed and is tested: an unrenderable bundle
  and an empty-but-successfully-rendered bundle both withhold for review
  before any network call
  (`test_a_render_failure_withholds_for_review_and_never_reaches_the_client`,
  `test_an_empty_rendered_bundle_withholds_for_review`); an upload that
  raises and an upload that returns no id both degrade to a recorded
  "failed" outcome rather than crashing the request
  (`test_an_upload_failure_degrades_to_a_failed_outcome_not_a_crash`,
  `test_an_upload_returning_no_id_degrades_to_a_failed_outcome_not_a_crash`).

**`tests/test_client_document_contract.py` was deleted, not kept.** Its own
docstring, written when it was committed, claimed it would be "kept,
unmodified... as the dated record of the defect" — that claim turned out to
be wrong. Once `contest()`'s signature grew a required 4th argument, the
file's 3-argument call could no longer even be collected: it raised
`TypeError: missing required argument`, not the original `AssertionError`,
so keeping it would have broken suite collection entirely rather than
preserving a record. The record is `48d03bf` itself, inspectable with
`git show 48d03bf` — this entry corrects the file's own prior claim.

**What this does not establish.** Not claimed: that this has been run
against production Razorpay. It has not. The accurate statement is that the
contest path — evidence rendering, document upload, and the submit payload
shape — is conformant with Razorpay's documented API contract, and has never
been executed against a live merchant account. Every test runs against
`httpx.MockTransport` and recorded fixtures, never a real socket
(`CLAUDE.md`: "No test may make a network call"). The rupee figures recorded
elsewhere in `README.md` and `NUMBERS.md` are unchanged by this reopening and
remain contingent on that same unexercised submission path — closing the
document-id gap makes rejection no longer *certain*; it does not make
acceptance *verified*. `README.md`'s opening paragraph and Limits section are
updated to say exactly this, not more.

**What did not change:** the model, the feature set, policy thresholds, the
cost sweep, the grounding gate, the leakage guards, or any recorded number.
`eval.cost_sensitivity.SWEEP_ASSUMES_EVERY_SUBMISSION_IS_ACCEPTED` is
untouched and stays `False` — a documented-contract-conformant upload is not
the same fact as a production-verified acceptance.

**Status:** DECIDED.

---

## 2026-09-03 — Generator parameters sourced against published statistics (Phase 1, calibration provenance)

**Decision:** Sourced generator parameters against published statistics
without regenerating data (rule 5). Values recorded as inside/outside
published ranges; nothing changed to fit.
**Why:** `CALIBRATION.md` was added as a citation index — the value each
generator parameter with a real-world analogue uses, the published range, the
named source, and whether the configured value falls inside or outside that
range. Two rows have no generator analogue at all (chargeback base rate,
VAMP/ECM threshold) and are recorded as such rather than force-fit to a
generator field that doesn't exist. One row — friendly-fraud share — falls
outside its published range (generator: 50–60%; published ceiling: 43.8%,
Chargebacks911 2026) and is recorded as outside, not adjusted. The generator
itself was not touched; no number recorded elsewhere in this repo moved as a
result of this entry.
**Reproduce:** `cat CALIBRATION.md` (external-citation index; not reproduced
by a command, since the published-range figures are not computed by this
repo — see `NUMBERS.md`'s "Calibration provenance" row).
**Status:** DECIDED.

---

## 2026-09-03 — Phase 2: EV threshold verified, calibration and escalate band checked

**Decision:** Phase 2 was an investigation, not a refactor. It set out to
derive a cost-sensitive contest threshold (Elkan, "The Foundations of
Cost-Sensitive Learning", IJCAI 2001: `contest iff p_win > cost/amount`) and
found `disputedesk/policy/engine.py`'s `decide()` already implements exactly
that rule — `expected_value = p_win * amount - representment_cost; contest
iff expected_value > 0` has been SPEC.md §4's specification since Phase 3,
and is algebraically identical to the Elkan ratio. No refactor was made. Two
follow-up checks (calibration of `p_win`; cost of the escalate band) were
run as eval-only what-ifs on top of this finding, per instruction. No number
in `NUMBERS.md`'s existing headline rows changed as a result of any of this.

**1. Threshold — already derived, not refactored.**
`policy/config.py` has no fixed probability constant serving as a contest
threshold; the only scalar there is `low_confidence_band = (0.45, 0.55)`,
which gates ESCALATE and never participates in the CONTEST-vs-ACCEPT
boundary. Cost-model check: the ₹400 (`REPRESENTMENT_COST_INR`, including
its bundled ₹150 analyst-time component) is charged only on CONTEST, and
regardless of win/lose (`eval/business_metrics.py:recovered_rupees`);
ACCEPT is the zero-cost reference by construction; the recoverable amount
is the gross `amount` field with no separate non-refundable fee anywhere in
the codebase (grepped, none exists). All four match Elkan's assumptions
exactly, so `C/A` is the correct expression with no algebra correction
needed. `tests/test_policy_ev_threshold_is_derived.py` pins the equivalence
against real holdout predictions (seed 0, n=5,000) rather than asserting it
from reading code alone — both tests pass immediately (characterization,
not a bug fix).

**2. Calibration of `p_win` — the unchecked Elkan precondition.**
No calibration step exists anywhere in `disputedesk/model/` — grepped for
`CalibratedClassifierCV`, `isotonic`, `sigmoid`, `Platt`; none found.
`LGBMClassifier.predict_proba`'s raw output is used as `p_win` directly.
Measured on the standard 20 seeds × 15,000 rows (`eval/run_calibration_report.py`,
new eval-only module — `eval/calibration.py` gained `brier_score` and
`near_threshold_reliability`, both unit-tested):

- **Brier score:** median 0.1697 (IQR 0.1670–0.1715) across seeds, versus
  0.1812 for a baseline that always predicts the training-split prevalence
  (0.2377 × 0.7623) — the model is modestly better than that floor, not
  dramatically so, consistent with the modest PR-AUC lift over prevalence
  reported elsewhere in this file.
- **Reliability table** (10 equal-width bins, pooled across all 20 seeds'
  holdouts, n=72,130):

  | bin | count | mean predicted p | observed win rate |
  |---|---|---|---|
  | 0.0–0.1 | 15,259 | 0.0716 | 0.0892 |
  | 0.1–0.2 | 16,951 | 0.1423 | 0.1418 |
  | 0.2–0.3 | 12,623 | 0.2529 | 0.2804 |
  | 0.3–0.4 | 16,670 | 0.3500 | 0.3507 |
  | 0.4–0.5 | 8,613 | 0.4389 | 0.3768 |
  | 0.5–0.6 | 1,731 | 0.5355 | 0.3807 |
  | 0.6–0.7 | 261 | 0.6353 | 0.4330 |
  | 0.7–0.8 | 22 | 0.7208 | 0.4091 |

  Calibration is reasonably tight through 0.0–0.4 (gaps ≤0.028). Above 0.4
  the model is **overconfident and increasingly so** — gap widens from
  +0.062 (0.4–0.5 bin) to +0.312 (0.7–0.8 bin) — but those bins also carry
  collapsing sample size (8,613 down to 22), so the tail figure is as much a
  small-sample warning as a calibration verdict. `p_max = 0.75` in
  `GeneratorConfig` means the generator itself rarely produces disputes with
  a true win probability above 0.75, so this tail region is inherently
  sparse by construction, not just by model behavior.
- **Near-threshold reliability** — the region that actually matters for the
  Elkan decision, since a calibration gap elsewhere can't flip a
  contest/accept call: rows where `p_win` sits within ±0.05 of *that row's
  own* `cost/amount` (the threshold is per-dispute, so "near the threshold"
  is evaluated per-row, not against one global cutoff). Pooled across all
  20 seeds: n=15,028 of 72,130 (~20.8% of the holdout is a "close call" by
  this definition). Mean predicted p 0.1061 vs. observed win rate 0.1223 —
  gap **−0.0162**, i.e. the model is mildly *underconfident* here, the
  opposite direction from the high-p tail. Overall median derived threshold
  across the holdout: 0.0628.

  **Finding, as-is per instruction (rule 1):** calibration is not uniform.
  It is reasonably good specifically in the low-to-mid range where most
  decisions actually get made (median threshold ≈0.063, gap ≈−0.016 near
  it), and poor in a sparse, rarely-decision-relevant high-probability tail.
  No calibrator was added. This is recorded as a limit, not corrected.

**3. Cost of the escalate band.**
`low_confidence_band = (0.45, 0.55)` runs before the EV check and overrides
it regardless of sign — a deliberate, documented departure from
EV-optimality (SPEC.md §4's "I don't know" path), not an accident. Measured
eval-only (`eval/escalate_band_counterfactual.py`, new module — reimplements
the plain EV rule read-only; `disputedesk/policy/engine.py` and `config.py`
were never touched or called with a modified band), standard 20 seeds ×
15,000 rows, configured ₹400 cost:

- **Band fraction of holdout:** median 0.0562 (IQR 0.0485–0.0624) — matches
  the already-published, separately-tested escalate rate exactly
  (`tests/test_eval_cost_sensitivity.py::test_escalate_rate_is_invariant_to_cost_per_seed`),
  cross-validating the new module against the existing one.
- **Counterfactual (band-free EV rule) advantage vs. baseline A:** paired
  mean +11,478.0 (95% CI +8,746.4 to +13,936.5, 19/20 seeds positive).
- **Actual (banded) advantage:** paired mean +11,210.3 (95% CI +8,507.9 to
  +13,633.3, 19/20 seeds positive) — matches the already-published headline
  exactly, same cross-validation.
- **Delta:** +267.7 INR/1,000 (≈2.4% of the headline advantage) — what the
  escalate band costs, in exchange for routing ~5.6% of disputes to human
  review of the genuinely uncertain region instead of an automated call.
  Reported as-is; the band was not removed or narrowed.

**Reproduce:**
```
pytest tests/test_policy_ev_threshold_is_derived.py -v
python -m eval.run_calibration_report
python -m eval.run_escalate_band_counterfactual
```
**Status:** DECIDED.

---

## 2026-09-03 — Escalate-band cost corrected to a directly-paired estimate

**Decision:** The prior entry's escalate-band cost (+267.7 INR/1,000) was
reported as a bare point estimate — `counterfactual_advantage.mean_difference
- actual_advantage.mean_difference`, a difference of two *separately*
bootstrapped paired-vs-baseline-A estimates, each with its own marginal CI.
That is not a valid comparison: subtracting two point estimates and quoting
neither CI (or worse, quoting one arm's marginal CI next to the other's
point estimate) says nothing about whether the two arms actually differ.

**Why this is fixable exactly, not approximately:** for a given seed,
`advantage_EV_rule(s) = counterfactual_recovered(s) - baseline_a(s)` and
`advantage_actual(s) = actual_recovered(s) - baseline_a(s)`. Their difference
is `counterfactual_recovered(s) - actual_recovered(s)` — `baseline_a(s)`
cancels algebraically, so the correctly-paired delta is a *direct* pairing
of `counterfactual_recovered` against `actual_recovered`, bootstrapped once
over those 20 seed-pairs, not derived from the other two CIs at all.

`eval/escalate_band_counterfactual.py:summarize_band_counterfactual` now
computes this directly (`band_cost = paired_difference(counterfactual_recovered,
actual_recovered)`), replacing the old `delta_mean` scalar. Re-run at the
standard 20 seeds × 15,000 rows, configured ₹400 cost:

- **Mean point estimate unchanged:** +267.7 INR/1,000 (confirms the earlier
  arithmetic was correct as far as it went — the bug was the missing CI, not
  the mean itself; `tests/test_eval_escalate_band_counterfactual.py` now
  pins this algebraic equivalence).
- **Paired 95% CI: +146.4 to +397.3.** Excludes zero.
- **16 of 20 seeds positive.**
- **Conclusion: the band's cost is distinguishable from zero at this sample
  size.** Per instruction, had the CI included zero the correct published
  statement would have been "not distinguishable from zero," not the point
  estimate alone — that caveat does not apply here, but the interval is
  reported either way, not just the mean.

Also added, per instruction: a README Limits paragraph on band misalignment
(the band is anchored to p_win=0.5, a symmetric-classification-problem
boundary, while this system's actual decision boundary — median `cost/amount`
across the holdout — is 0.0628; the band therefore diverts clear-contest
cases into human review rather than sitting near the genuinely uncertain
region), and one sentence in the calibration section noting the near-threshold
underconfidence (predicted 0.1061 vs. observed 0.1223) biases the reported
+11,210.3 advantage toward a conservative floor, not an optimistic estimate.
No policy, model, or generator code changed; this is a statistics fix and two
README additions, eval-only.

**Reproduce:** `python -m eval.run_escalate_band_counterfactual` (prints the
band cost with its own directly-paired CI, and states explicitly whether
that CI excludes zero).
**Status:** DECIDED.
