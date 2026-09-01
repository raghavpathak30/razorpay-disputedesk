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
