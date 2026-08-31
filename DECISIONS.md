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
