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
