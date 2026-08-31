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
