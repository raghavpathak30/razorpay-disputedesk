# GENERATOR.md — the generative story for synthetic fraud-reason-code disputes

This document is written before any generator code exists, per `PHASES.md` Phase 1
and `SPEC.md` §5. It is prose first, parameters second. Every numeric parameter
below is marked **Guess** — none are final. Nothing here is implemented yet; this
file describes what `disputedesk/generator/` will do once this story is approved.

A note on location: `PHASES.md` refers to this file as `docs/GENERATOR.md`. It is
being written at the repository root instead, and no `docs/` directory has been
created, per this session's explicit instruction. Flagging the discrepancy rather
than silently resolving it.

---

## 1. The causal story

A fraud-reason-code chargeback happens when a cardholder tells their issuer "I did
not authorize this transaction." The reason code is uniform; the underlying reality
behind it is not. Two very different situations produce the identical reason code:

- **Genuine unauthorized use.** A stolen card, a taken-over account, or a synthetic
  identity made the purchase. The named cardholder truly never authorized it.
- **First-party misuse ("friendly fraud").** The named cardholder made the purchase
  themselves — or a household member with access to the card did — and is now
  disputing it as fraud anyway, whether through genuine confusion, a shared-card
  dynamic, or a knowing attempt to get goods for free.

Contesting is winnable when the merchant can produce evidence that the *charge was
properly authorized and consistent with this customer's real behavior* — that is
overwhelmingly a friendly-fraud story, not a genuine-fraud one. This is the single
governing claim of the whole generator, and it is a claim a payments person could
disagree with (e.g., "issuers side with cardholders on fraud codes regardless of
merchant evidence" is a real, more pessimistic counter-view — see §9).

Named latent causes, in the order they act:

**L1 — `true_fraud` (binary).** Was the charge genuinely unauthorized by the real
cardholder? This is the dominant cause. `true_fraud = 1` drives win probability
down hard: no amount of delivery or authentication evidence un-does the fact that
the actual cardholder didn't make the charge, and issuers know this. `true_fraud =
0` (friendly fraud) drives win probability up, because the merchant genuinely has a
paper trail proving the charge was legitimate. **Direction:** negative. **Strength:
large** (Guess: this alone should account for roughly half of the total spread in
p across the population).

**L2 — `authentication_strength` (continuous, 0–1).** How well the transaction's
authenticity can be documented at the moment of sale: AVS match quality, CVV match,
3DS/OTP success, device trust. Higher strength is persuasive evidence on its own,
independent of whether the charge was actually fraud — a fraud team reviewing a
representment cares what was *shown* to be true at checkout, not only the ultimate
fact. **Direction:** positive. **Strength: medium-large.** Correlated with
`true_fraud` (fraud tends to authenticate worse) but not perfectly — see
confounders (§6).

**L3 — `relationship_genuineness` (continuous, 0–1).** How real and established
the customer relationship looks on paper: order history, communication trail,
account tenure. High genuineness gives the merchant more supporting evidence to
cite (prior good orders, consistent communication). **Direction:** positive.
**Strength: medium.** Weakly correlated with `true_fraud` — account-takeover fraud
can inherit a genuinely long history (§6).

**L4 — `delivery_provability` (continuous, 0–1).** Whether delivery to the correct
party can be documented: signed physical delivery > unsigned physical > digital
good / service with no proof-of-receipt at all. **Direction:** positive.
**Strength: medium.**

**L5 — `filing_delay_days` (continuous, causally produced).** How long after
purchase/delivery the dispute was filed. This is not just a proxy for
`true_fraud` — timing carries independent evidential weight. Genuine-fraud
victims tend to notice and report an unrecognized charge quickly after it hits
their statement. Friendly-fraud disputes tend to arrive later, often after the
good has shipped or been used, which is itself part of why they're winnable
(SPEC.md's own worked example — "filed more than N days after delivery" — is
this cause). **Direction:** positive, with a plausible U-shape a skeptical
reviewer might raise (a dispute filed so late it blows the response deadline is
arguably *unwinnable for a different reason* — out of scope here, since we're
modeling winnability if a timely response is filed). **Strength: medium.**

Revision note: the first draft of this document treated the corresponding
feature as a near-exact reading of this latent, on the reasoning that real
systems log timestamps precisely. On reflection that's a real consideration but
not the only one, and it cuts both ways — argued out here rather than silently
changed, since the earlier draft stated it as settled:
- *For* near-exact: system timestamps genuinely are recorded precisely; making
  the feature noisy for its own sake would be less realistic, not more.
- *Against*: a near-noiseless path from a medium-strength latent straight into an
  observable feature is a second low-noise channel into `p` once L1 (binary,
  fully structural via the mixture draw) is counted alongside it. That
  compresses exactly the gap §5 exists to report — the whole reason the oracle
  ceiling is interesting is that a real model only sees noisy readings of the
  latents, never the latents themselves.
- Resolution: add modest measurement noise (Guess: ±1–3 days) to the observed
  `days_between_purchase_and_dispute` feature relative to the true
  `filing_delay_days` latent. This is independently defensible on realism
  grounds too, not just as a patch: "days between purchase and dispute" is
  genuinely ambiguous by a day or two depending on which anchor a system uses
  (order-placed vs. delivered vs. charge-posted), so a small jitter term is a
  plausible model of that ambiguity, not an arbitrary noise injection.

**Revision note (session 2, out-of-band sanity check):** the ±1–3 day
measurement noise above turned out not to be the binding constraint. The
sanity check measured `AUC(days_between_purchase_and_dispute, true_fraud) =
0.0623` — a near-deterministic read on `true_fraud`, the most serious of six
defects found, and exactly the failure mode this section's first revision note
worried about ("a second low-noise channel into `p`"). The actual cause was
the *component-conditioned means* in `latents.py` (3 days for `true_fraud=1`,
20 for `true_fraud=0`), which barely overlap once drawn from a
`Gamma(shape=2)` — the measurement noise added on top was too small to matter.
Solved analytically rather than re-tuned by trial, using the identity that for
`X ~ Gamma(k, θx)`, `Y ~ Gamma(k, θy)` independent with the same shape `k`,
`W = Y/(X+Y) ~ Beta(k, k)`, so `P(Y > X) = P(W > ρ/(1+ρ))` where `ρ = θx/θy =
mean_x/mean_y`. Solving for `P(Y > X) = 0.35` at `k=2` (shape unchanged) gives
`ρ ≈ 1.51`. Verified by full-pipeline simulation including the measurement
noise: `mean_true_fraud_days = 8.5`, `mean_genuine_days = 13.0` (ratio 1.529)
→ empirical `AUC(days_between_purchase_and_dispute, true_fraud) = 0.3507` at
n=300,000, and 0.3504 on the actual n=15,000, seed=42 dataset. The causal
story is unchanged — fraud victims still notice and report sooner than
friendly-fraud disputers file — only the two means moved closer together, so
`filing_delay_days` remains a genuine, medium-strength, *not* near-deterministic
cause, matching the "independent evidential weight" claim above instead of
contradicting it.

**L6 — `reason_subtype` (categorical, minor).** Fraud-reason-code chargebacks are
not one homogeneous bucket on real networks — different networks' fraud-category
codes carry slightly different evidentiary and procedural expectations (see the
real code identifiers in §8). **Direction:** small, code-dependent offset.
**Strength: small.** Included mainly for schema realism and as a genuinely
weak/noisy signal, not a primary driver — a case where the true effect size
might reasonably be argued to be zero.

**Revision note (session 2, out-of-band sanity check):** the sanity check measured
`AUC(reason_code, true_fraud) = 0.5000` for all four codes, to four decimal places
— too exact to be a real weak association, which prompted a check of whether this
is by construction or a measurement artifact. It is **by construction, and exactly
zero, not merely small**: `reason_subtype` is drawn in pipeline step 3 uniformly at
random, independent of the mixture component (§2 step 2) and therefore independent
of `true_fraud`. The "small, code-dependent offset" language above was accurate
about L6's effect on `p` (each code nudges the logit by a small fixed amount, per
§4's calibration) but imprecise about what that implies for `true_fraud`: since
which code a row gets does not depend on `true_fraud` at all, L6 has **zero**
statistical association with `true_fraud` specifically, even though it is not
irrelevant to `p` or to the label. This is the intended design (L6 was "included
mainly for schema realism," never conditioned on the mixture component the way
L2–L5 are), not a bug — restated here per the instruction not to keep describing
L6 as merely "small" when the true value, for this specific relationship, is zero.
`reason_code` (the feature) also now carries misclassification noise relative to
`reason_subtype` (the latent) — see §3's revision note on `reason_code` — which
does not change this: the misclassification is drawn independently of everything,
so it cannot introduce an association with `true_fraud` either.

**L7 — `dispute_propensity` (continuous, 0–1, added in revision 2).** A
customer-level behavioral trait: this customer's general tendency to file
disputes, independent of whether *this particular* transaction is fraud. Drawn
as its own latent, independent of `true_fraud` and `relationship_genuineness` —
a customer can be genuinely long-tenured and low-risk (high L3) while still
having an independent high or low tendency to dispute charges. **Direction:**
small positive effect on `p`, modeling the "repeat disputer / pattern of
behavior" evidence merchants actually cite in representment letters. **Strength:
small.** This is contestable in the same way the old `prior_dispute_count` note
was: a payments person could reasonably argue arbitration for fraud-reason
codes is scoped to *this* transaction's authorization evidence and shouldn't be
swayed by unrelated dispute history at all — if so, this latent's effect on `p`
should be zero, not small-positive. Drawing it independently of `true_fraud`/L3
rather than correlating it with either is itself a simplification, flagged for
reconsideration once real generation is running.

Why this latent exists: the original draft gave the *observed* feature
`prior_dispute_count` (a noisy reading of L3) an additional direct arrow into
`p`, on top of its arrow into the feature. That meant part of `p` was reachable
by reading a feature with no sensor noise in between — a real leakage-shaped
risk, structurally the same failure mode the leakage guard exists to catch,
just introduced from the generator side instead of the feature-builder side.
Splitting it into its own latent restores the pattern every other cause
follows: latent → noisy feature, and (separately) latent → contribution to `p`,
with noise breaking each arrow independently.

---

## 2. Generation order (the pipeline)

Strict rule, restated because it's the whole point of this document: **no feature
is derived from `p`, and no feature is derived from the label.** Both the feature
vector and `p` are computed from the latent causes in parallel; the label is drawn
last, from `p`, and nothing downstream of the label feeds back into anything.

1. **Assign a timestamp and period index.** Draw a purchase timestamp inside the
   simulation window; derive a month index used only to apply the temporal drift
   described in §7.
2. **Draw a confounder-mixture component**, conditioned only on the period index
   (drift lives here). The component determines which sub-population this record
   belongs to (§6) and, deterministically, `true_fraud` for that component.
3. **Draw the remaining continuous latent causes** (`authentication_strength`,
   `relationship_genuineness`, `delivery_provability`, `filing_delay_days`,
   `dispute_propensity`) from distributions whose parameters depend on the
   component drawn in step 2 — this is how confounding is built in, not bolted
   on. `dispute_propensity` (L7) is drawn independently of the component per its
   own note in §1. Draw `reason_subtype`.
4. **Draw an irreducible residual `ε`**, independent of every named latent, from a
   fixed noise distribution. This is the formal source of irreducible error (§ "no
   feature captures everything" — see below): even an oracle that knew every named
   latent exactly still could not predict `p` without also knowing `ε`.
5. **Compute `p`** as a logistic function of a weighted combination of L1–L7 and
   `ε`, then map into the allowed band (§4). `p` is stored for evaluation and
   generator debugging only — it is never written into the feature table the model
   sees.
6. **Derive every observable feature** from the latents drawn in steps 2–3 only
   (never from `p`, never from `ε`, never from the label), each with its own
   independent sensor noise (§3). This includes `amount`, which is a noisy,
   weakly-causal reading of `true_fraud` (L1) as of revision 2 — see §3 and §8 —
   not the parentless field it was in the first draft. The pure-noise features
   (§3) are drawn here too, independently of every latent by construction.
7. **Assign schema-plumbing fields** (`id`, `payment_id`, `phase`, `status`,
   `respond_by`, `currency`) from the timestamp and record index — not from any
   causal latent.
8. **Draw the label**, `won_if_contested ~ Bernoulli(p)`, last, using the `p` from
   step 5.

The irreducible-error mechanism deserves its own sentence since SPEC.md asks for
it explicitly: some disputes are unwinnable (or winnable) for reasons no feature
set could ever capture — an idiosyncratic reviewer, a network's mood, a detail that
never made it into any structured field. That is `ε` in step 4. It is what keeps
even the *oracle* ceiling in §5 below 1.0, as distinct from the separate noise that
keeps a real trained model below the oracle ceiling (it only sees noisy features,
never the latents or `p` directly).

---

## 3. Observable features and their causal parents

| Feature (maps to SPEC.md §3 order context) | Noisy observation of | Notes |
|---|---|---|
| `avs_match` | `authentication_strength` | threshold(latent) + flip noise (session 2, see below) — confounding flows through the component-conditioned latent distributions (§6), not a feature-level override |
| `cvv_match` | `authentication_strength` | separate flip noise from `avs_match`, so the two aren't collinear |
| `device_fingerprint_known` | `relationship_genuineness` | threshold(latent) + flip noise; confounding flows through the latent, per §6's closing note |
| `delivery_confirmed` | `delivery_provability` | threshold(latent) + flip noise |
| `prior_order_count` | `relationship_genuineness` | count-noise (Guess: Poisson-ish) |
| `prior_dispute_count` | `dispute_propensity` (L7, revision 2) | customer-level trait, no longer a proxy for `relationship_genuineness` — see L7 note in §1 for why this changed |
| `ip_geo_billing_distance_km` | `authentication_strength` (inverse) | continuous reading + Gaussian sensor noise; confounding flows through the latent (§6) |
| `days_between_purchase_and_dispute` | `filing_delay_days` | small measurement noise (Guess: ±1–3 days) — see the session-2 revision note under L5 in §1 for the real fix (component means, not this noise term) |
| `customer_communication_log` (free text) | `true_fraud` (opening/claim/detail phrasing rates) + `relationship_genuineness` (tone, length, messiness) | slot-filled from shared phrase pools at different rates per class (session 2 — see below), generated deterministically from the seed — **no LLM call**. Normalising this text is `evidence/`'s job in Phase 3, not the generator's. |
| `reason_code` | `reason_subtype`, with misclassification noise (session 2, see below) | drawn from a small set of confirmed real network fraud-category codes (§8, revision 2 — previously placeholders) |
| `amount` | `true_fraud` (weak, revision 2; rebalanced session 2) | see below |

**`amount` is a weak causal feature, not pure noise (revision 2).** The first
draft listed `amount` as pure noise. That was wrong for a reason specific to
this project: `amount` is the multiplier in the policy engine's expected-value
calculation (SPEC.md §4, `expected_value = P(win) * amount - representment_cost`).
A purely noisy `amount` would reduce that formula to "contest whatever is
expensive," and would make the false-positive-cost analysis (SPEC.md §6)
measure an artifact of the generator rather than anything real. Real chargeback
amounts do correlate with fraud type: higher-value transactions skew toward
genuine third-party/account-takeover fraud (a stolen card gets used for the
biggest purchase it can clear before the theft is noticed), while lower-value
transactions skew toward friendly fraud — forgotten recurring subscription
charges, small one-off "I don't recognize this" disputes. So: `amount` is drawn
from a `true_fraud`-conditioned distribution (Guess: lognormal, higher `μ` for
`true_fraud=1`, lower `μ` for `true_fraud=0`, with enough shared spread that the
two overlap heavily and `amount` alone is nowhere near diagnostic on its own).
It causes nothing back into `p` — the arrow runs `true_fraud → amount`, not
`amount → p` — so this doesn't add a new leakage-shaped path, and it's derived
in step 6 like every other feature, from a latent, never from `p` or the label.

**Revision note (session 2, out-of-band sanity check):** "nowhere near
diagnostic on its own," above, turned out to be wrong in the original
implementation — the sanity check measured `AUC(amount, true_fraud) = 0.7631`,
a strongly diagnostic single feature, against a `μ` gap of 0.9 between the two
lognormals. Solved analytically for the intended weak link rather than
re-tuned by trial: for two lognormals sharing `σ`, `AUC = Φ(Δμ / (σ√2))`.
Target `AUC = 0.60`: `z = Φ⁻¹(0.60) = 0.253347`, `Δμ = σ·√2·z = 0.9 × 1.414214
× 0.253347 = 0.322458`. `μ_genuine` held at its original value; `μ_true_fraud
= μ_genuine + Δμ`. Verified by simulation (n=200,000): empirical AUC = 0.5998;
on the actual n=15,000, seed=42 dataset: AUC = 0.6082. `σ` is unchanged.

**Revision note (session 2): `avs_match`/`cvv_match`/`device_fingerprint_known`/
`delivery_confirmed` mechanism fixed.** These were originally drawn as
`Bernoulli(latent)` directly (e.g. `authentication_strength = 0.3` gave only a
30% chance `avs_match = True`), which for latents spread across `[0, 1]`
compounds roughly 30% inherent disagreement with the latent's own threshold
*before* the configured flip-probability noise is even applied. The sanity
check's LightGBM gain ranking showed `avs_match`/`cvv_match` absent from the
top 8 features while `ip_geo_billing_distance_km` — a continuous reading of
the same `authentication_strength` latent — ranked first, consistent with this
mechanism destroying most of the latent's signal. Fixed by switching to
`threshold(latent) + flip_prob`, with `flip_prob` lowered so it is the single,
literal noise source (see `config.py`'s `boolean_reading_threshold` and the
`*_flip_prob` fields for the numbers). Verified by simulation:
`AUC(avs_match, true_fraud)` moved from 0.35 (0.15 from chance) under the old
mechanism to 0.23 (0.27 from chance) under the new one.

**Revision note (session 2): `customer_communication_log` redesigned.** The
original implementation drew from exactly three fixed template strings per
`true_fraud` branch (six total). The sanity check found all six templates
mapped to exactly one `true_fraud` value each — a string match recovered
`true_fraud` exactly, a low-noise path into the dominant latent arriving
through a column the leakage guard's numeric-correlation check does not
inspect, and one that also defeated the point of Phase 3's planned LLM
normalisation step (nothing to extract if the category is readable by string
match). Replaced with text slot-filled from four independent components —
opening, claim, detail, signoff — each drawn from a small shared phrase pool
per slot, weighted by `true_fraud` but never exclusively (weights range
roughly 0.20–0.30, never 0/1), so phrasings overlap across classes at
different rates rather than partitioning them. `relationship_genuineness` now
has the causal effect the table above already claimed for it but the
implementation didn't have: it raises the probability of a polite signoff and
of including the detail sentence (tone and length), and lowers the probability
of a typo (coherence). Realistic messiness was added: occasional near-empty
logs (`"n/a"`, `"?"`), an occasional irrelevant aside unrelated to either
class, inconsistent capitalization, and single-character typos. The
`"I never received what I ordered"` phrase was removed — it is a
non-delivery claim, not a fraud-family reason code, and does not belong in
either branch. Verified: across a dataset of 15,000 rows, no communication-log
string that recurs 20+ times maps to only one `true_fraud` value (checked at
seed 42 and two other seeds); the output space has 3,900 distinct strings.

**Revision note (session 2): `reason_code` misclassification noise added.**
`reason_code` was `latents.reason_subtype.copy()` — an exact latent copy, the
same low-noise-channel shape as the original `prior_dispute_count` and
`customer_communication_log` defects. §1's rule is that every latent reaches
its observable feature through independent sensor noise, with L5 named as the
sole near-exact exception; L6 was never that exception. Fixed by adding a
configured probability (`reason_code_misclassification_prob`, Guess: 0.10)
that the recorded code differs from the true `reason_subtype`, modeling
realistic issuer coding error — when triggered, the code is reassigned
uniformly among the *other* three confirmed codes, never re-drawing the same
one by chance. Verified: measured misclassification rate 0.0971 against the
0.10 target on the n=15,000, seed=42 dataset. Because `reason_subtype` has
zero association with `true_fraud` by construction (see the L6 revision note
in §1), this fix does not and cannot change `reason_code`'s (lack of)
association with `true_fraud` — it only stops `reason_code` from being a
noiseless read of the latent for the purpose of predicting `p`/the label,
which is what §1's noisy-observation rule is actually protecting.

**Pure noise, no causal parent at all** (needed as leakage-guard negative
controls, and as the "you need some" features the phase explicitly asks for):
`card_network` (Visa/Mastercard/RuPay/Amex, cosmetic only), `checkout_hour_of_day`
(Guess: uniform over 0–23, added in revision 2 as a genuine-variance replacement
— see below), and the `id`/`payment_id` identifiers themselves. `currency` was
removed from this list in revision 2 (see §8 note) since a constant column has
no variance and is useless as a negative control; `checkout_hour_of_day` was
added in its place specifically so the leakage guard has at least one
real-variance, genuinely-uncaused numeric feature to test against, not just a
categorical one. None of these are touched by step 3 or step 5 of the pipeline.
In Phase 2's leakage-guard test, a deliberately leaky control column (one that
literally copies `p` or the label) will be added to a test fixture to prove the
guard actually fires on a positive case, not just passes vacuously.

---

## 4. The p band

**Revision note:** the first draft of this section was internally inconsistent —
it stated modes of ~0.175 and ~0.70 against a ~45% `true_fraud` rate, which
arithmetically implies E[p] ≈ 0.46, not the ~0.35–0.40 the same section claimed.
Fixed below by lowering the modes, not by restating the mean, and by writing
the arithmetic out explicitly so the two numbers can't drift apart silently
again.

**Guess: `p ∈ [0.02, 0.75]`.**

Neither bound is 0 or 1. Rationale: even the strongest true-fraud, weak-evidence
case should retain some residual chance of a contest succeeding — issuer review
error, an incomplete counter-file, a sympathetic read — never zero. Even the
strongest friendly-fraud, maximal-evidence case retains real risk of losing:
arbitration outcomes are not mechanical, deadlines get missed on the issuer's side
too, and network policy varies. Collapsing either tail to certainty would make the
label a deterministic function of the latents in every way that matters, which
defeats the entire exercise (SPEC.md's warning about PR-AUC of 0.99 meaning
nothing). The band is tighter than the first draft's `[0.04, 0.93]` because the
modes below are themselves much closer together — a wide band with close modes
would mean the tails do almost no work, which is its own kind of dishonesty
about the shape of the distribution.

**Guess:** the population distribution of `p` is roughly bimodal, because
`true_fraud` (L1) dominates: a low mode around **0.08** for the `true_fraud=1`,
non-confounder majority, and a high mode around **0.39** for the `true_fraud=0`,
non-confounder majority, with the two confounder components (§6) supplying the
overlap in between.

### E[p], stated as one number, tied to the arithmetic that produces it

**E[p] ≈ 0.25.**

The arithmetic, so this number and the modes above can never silently disagree
again — recompute this whenever either mode or the `true_fraud` rate changes,
rather than restating a separate guessed mean:

```
E[p] ≈ π_fraud · low_mode + (1 − π_fraud) · high_mode
     = 0.45 · 0.08     + 0.55 · 0.39
     = 0.036           + 0.2145
     = 0.2505
```

`π_fraud = 0.45` is the midpoint of §7's drift range (40% at month 0 rising to
50% by month 23). This is a first-order approximation — it treats each class as
a point mass at its mode and ignores the small pull the two confounder
components (§6) and the `ε` residual exert on the true simulated mean. Phase 1's
gate includes checking the actual empirical mean of generated `p` against this
target range once the generator runs; if it drifts outside **[0.22, 0.28]**,
that's a bug or a parameter that needs revisiting, not a number to quietly
re-justify after the fact.

**Justification against external evidence:** industry sources commonly cite
representment win rates for fraud-reason-code chargebacks around **17%** —
these are industry estimates, not audited figures, and are noted here as
context, not as ground truth this generator is required to match. Our label,
`won_if_contested`, means something narrower and more favorable than "the
observed win rate": it is "winnable *if contested with correct, complete
evidence*." The commonly-cited ~17% figure blends together merchants who
contest well, merchants who contest with weak or incomplete evidence packets,
and merchants who contest reflexively without regard to whether the specific
case is winnable at all — SPEC.md's own framing is that today's merchants
either contest everything or accept everything, neither of which is "contest
optimally." An optimal-evidence label should sit *above* the blended observed
rate, because it strips out the execution failures baked into the observed
number. E[p] ≈ 0.25 sits about 8 points above the ~17% anchor — a modest,
defensible lift for "assume correct evidence, still governed by the same
underlying mix of genuine vs. friendly fraud," not the much larger lift a
figure like the old ~0.35–0.40 mean would have implied.

---

## 5. The implied ceiling (Bayes ceiling)

Because the generator retains true `p` per record (as a debug column, never a
model input), we can compute the best any model could ever do on this dataset —
the ceiling imposed by `ε` and the label-sampling step, not by feature quality.

**Method** (to implement in Phase 2, described here only): score every record by
its own true `p_i`. Because each label is an independent `Bernoulli(p_i)` draw,
the *expected* precision and recall at any threshold `t` have closed forms directly
from the distribution of `p` in the dataset — no repeated resampling needed:

- `precision*(t) = mean(p_i for all i with p_i ≥ t)`
- `recall*(t) = sum(p_i for all i with p_i ≥ t) / sum(p_i for all i)`
  (since `sum(p_i)` is the expected number of true positives in the whole set)

Sweeping `t` over `[p_min, p_max]` traces the oracle precision-recall curve; its
area is the oracle PR-AUC — the number no downstream feature-based model can beat,
no matter how good. (This can be cross-checked empirically by drawing many
replicate label-samples per `p`-bucket and confirming the closed-form curve
matches; that check belongs in Phase 2's tests, not here.)

**Re-derived guess (revision 2), now that §4's modes are 0.08 and 0.39 instead of
~0.175 and ~0.70:** the modes are much closer together than the first draft
assumed, so the ceiling must come down with them — this is not a case where the
oracle number can be left alone while only the mean/mode text above it changes.

**Prevalence baseline first, since the ceiling is uninterpretable without it:**
a random (no-skill) classifier's PR-AUC equals prevalence, and prevalence here
is exactly E[p] ≈ **0.25**. Any oracle number below or barely above 0.25 means
the causal structure gives almost no separability; the honest question for this
dataset is how much lift the oracle gets *over* 0.25, not its absolute value.

**Worked illustration** using the two-point idealization from §4 (all
`true_fraud=1` non-confounder records at exactly `p=0.08`, all `true_fraud=0`
non-confounder records at exactly `p=0.39`, ignoring confounder overlap and `ε`
for this pass only — this is not the real distribution, just a tractable stand-in
to get a number):

- For any threshold `t ≤ 0.08`, everyone qualifies: `recall*=1.0`,
  `precision*=E[p]=0.25` (exactly the prevalence, as it must be when the
  threshold excludes nobody).
- For `0.08 < t ≤ 0.39`, only the high-mode group qualifies: `precision*=0.39`,
  `recall* = (0.55 × 0.39) / 0.2505 ≈ 0.856`.
- Average precision over the resulting two-segment step curve:
  `AP ≈ 0.39 × 0.856 + 0.25 × (1 − 0.856) ≈ 0.334 + 0.036 ≈ 0.37`.

So the idealized, no-overlap ceiling is **≈0.37** — a lift of about **+0.12**
over the 0.25 prevalence baseline. That idealization has zero within-mode
spread and zero confounder overlap, both of which the real generator has by
design (§6); introducing them mixes some `true_fraud=1` records up toward the
high mode and some `true_fraud=0` records down toward the low mode, which can
only reduce separability relative to this clean two-point case, never improve
it. **Guess at the realistic ceiling: PR-AUC roughly 0.30–0.36**, i.e. a real
but modest lift of **+0.05 to +0.11** over baseline — a large downward
correction from the first draft's 0.75–0.88 guess, and the expected consequence
of deliberately choosing modes close enough together that E[p] resolves to
~0.25 rather than ~0.46. This is not a worse generator; a genuinely hard,
honestly-modest-ceiling problem is more defensible than an accidentally
easy one. All of these numbers are placeholders pending the actual generated
data — the gate for Phase 2 is computing them for real, not hitting this guess.
The old naive `t=0.5` operating-point illustration from the first draft is
dropped here: with `p_max=0.75` and both modes below 0.4, a fixed threshold of
0.5 would select almost nothing and isn't a meaningful reference point anymore —
the policy engine's own expected-value threshold (SPEC.md §4) is the
operating point that actually matters, and it's a Phase 3 concern, not this
document's.

A real trained model (LightGBM on noisy features, never seeing `p` or the
latents) should be expected to land measurably below this ceiling. That gap is
itself worth reporting: it's the honest answer to "how much of the true signal did
the model actually recover."

---

## 6. Confounders

At least two required by the phase; two are specified, each as a named
mixture component with a guessed population share.

**6A — The traveling legitimate customer (`true_fraud = 0`, looks risky).**
A genuine, repeat customer who is traveling, using an unfamiliar network/device,
or shipping to a temporary address (hotel, forwarding address, a relative's
house). This produces a poor `avs_match`, elevated `ip_geo_billing_distance_km`,
and a possibly-unrecognized `device_fingerprint_known` — all despite `true_fraud =
0` and genuinely high `relationship_genuineness` otherwise. **Guess: ~8% of the
`true_fraud=0` population** (roughly 4–5% of all records, given the overall mix in
§4).

**6B — Account-takeover fraud on a good account (`true_fraud = 1`, looks
clean).** A fraudster compromises a long-tenured, genuinely good customer's
account or full card details and transacts as if they were that customer.
`relationship_genuineness`-derived features (prior order count, known device — if
the session/cookie was hijacked rather than the device changed) can look
excellent, and `authentication_strength` can pass cleanly if the fraudster has
correct CVV/OTP access. **Guess: ~12% of the `true_fraud=1` population** (roughly
5–6% of all records).

Both are drawn as their own mixture components in pipeline step 2 (§2), with their
own latent-parameter distributions, rather than as post-hoc feature overrides — so
the confounding is structural, not a patch applied after the fact.

---

## 7. Temporal structure

**Guess: a 24-month simulation window**, referenced by month index 0–23 rather
than anchored to specific real calendar dates (the anchor date is an
implementation detail, decided when the code is written, not a causal choice).

- Purchase timestamps are drawn across the window (Guess: mildly seasonal/weekday
  volume pattern, not causally load-bearing — included for realism only).
- Dispute-filed timestamp = purchase timestamp + `filing_delay_days` (L5).
- `respond_by` = filed timestamp + a fixed response window. **Guess: 7 days** —
  flagged explicitly as unverified against real Razorpay/network deadlines; needs
  confirmation before Phase 4's API client work relies on it.

**What drifts, and why it must:** if nothing about the causal structure changes
over the window, a temporal split is statistically indistinguishable from a random
split, and the whole justification for using one (SPEC.md §5, CLAUDE.md invariant
2) is decorative. Two drift terms, both guesses:

- The `true_fraud` base rate in the mixture draw (step 2) rises over the window —
  **Guess: from ~40% in month 0 to ~50% by month 23** — a fraud-ring-scaling
  story.
- `authentication_strength`'s distribution improves mildly over the window
  (issuers rolling out stronger 3DS-style checks) — **Guess: a small upward shift**,
  smaller in magnitude than the fraud-rate drift.

Net effect: the joint distribution of (features, `p`) is non-stationary across the
window, so a model trained on the first N months and evaluated on the last M
months is doing genuine generalization, not just re-reading a shuffled version of
its training data.

---

## 8. Mapping to Razorpay's schema (SPEC.md §3)

**Dispute object:**

| Field | Source |
|---|---|
| `id` | generated identifier, schema plumbing, non-causal |
| `payment_id` | generated identifier, 1:1 with the dispute for this dataset (Guess/simplification: no multi-dispute-per-payment modeled) |
| `amount` | weak causal draw conditioned on `true_fraud` (L1), not pure noise as of revision 2 — see §3 (Guess: lognormal, `μ` shifted higher for `true_fraud=1`, needs real parameter calibration) |
| `currency` | constant `INR` for all records (Guess/simplification) — kept constant deliberately (revision 2 dropped it from the pure-noise/leakage-guard list *because* it's constant, see §3; a genuine noise feature was added elsewhere instead of making currency artificially variable) |
| `reason_code` | drawn from a small set of **confirmed real** fraud-category codes, conditioned weakly on `reason_subtype` (L6) — see the sourced list below, replacing the first draft's invented placeholders |
| `phase` | constant `chargeback` for every generated record (Guess/simplification — the generator produces the initial dispute event only, not a full phase-escalation lifecycle) |
| `status` | constant `open` for every generated record — the live workflow status is not fabricated by the generator; it's what the running system would update later |
| `respond_by` | filed timestamp + response window (§7) |

**`reason_code` values (revision 2 — confirmed, sourced, not invented):**
Razorpay publishes its own chargeback reason-code reference at
`https://cdn.razorpay.com/files/chargeback_codes.pdf` (fetched and read
directly for this revision). It lists each card network's codes with a
`Chargeback Category` column, and the rows marked category `Fraud` are the
ones in scope for this project. Of those, the following are card-not-present
style fraud codes — matching Razorpay's predominantly CNP (online) transaction
flow, which is what this generator's feature set (device fingerprint, IP-geo
distance, no physical POS signals) assumes:

| Network | Code | Reason text (as published) |
|---|---|---|
| Mastercard | 4837 | No Cardholder Authorization |
| Mastercard | 4840 | Fraudulent Processing of Transactions |
| Visa | 83 | Fraud-Card Absent Environment |
| Amex | FR2 | Fraudulent Transaction |

These four are used as the `reason_code` value set. Notably, every fraud-category
row in the source document is marked `Preventable: NO` and `Reversible: YES` —
i.e. Razorpay's own reference agrees these are exactly the "can't be prevented
on the front end, but can be contested after the fact" bucket this whole
project targets, which is a small but genuine external validation of the scope
decision in DECISIONS.md's track-selection entry.

**Explicitly left unconfirmed, not invented:** the source document's five pages
cover Mastercard, Visa, and Amex only — no RuPay or Diners table was present.
Since Razorpay is a predominantly India-facing gateway, a RuPay fraud code is a
real gap, not a cosmetic one. Rather than inventing a plausible-looking RuPay
code, this is left explicitly open pending either a separate RuPay-specific
source or written confirmation from Razorpay's dispute documentation team. Two
Visa fraud-category codes from the same source (`62` Counterfeit Transaction,
`81` Fraud-Card Present Environment) were also excluded, since both describe
card-present/physical-card scenarios that don't fit this generator's CNP
feature set — noted here so the exclusion reads as a scoping decision, not an
oversight.

**Known simplification:** `reason_code` is drawn from `reason_subtype`
independently of the cosmetic `card_network` pure-noise feature (§3), even
though in reality a Visa-family code could only appear on a Visa transaction.
Coupling those two fields correctly is deferred rather than folded into the
causal story now — flagged in "Open parameters" below.

**Order context (own schema):** every bullet in SPEC.md §3's order-context list
has a row in the §3 table above — `avs_match`, `cvv_match`,
`device_fingerprint_known` (device fingerprint), `delivery_confirmed`,
`prior_order_count`, `prior_dispute_count`, `ip_geo_billing_distance_km`,
`days_between_purchase_and_dispute`, `customer_communication_log`.

**Ground truth label:** `won_if_contested` ← `Bernoulli(p)`, per §2 step 8.

**Evidence objects** (`shipping_proof`, `billing_proof`, etc.) are explicitly
**not** produced by the generator. The generator only produces the order-context
fields that later determine which evidence *can* be assembled (e.g.
`delivery_confirmed = true` is what makes a `shipping_proof`/`proof_of_service`
object assemblable at all) — constructing the evidence objects themselves is
`evidence/`'s job, starting in Phase 3.

---

## 9. What this dataset cannot tell you

This section is meant to go into the submission verbatim.

- **The causal story is authored, not measured.** No real dispute-outcome data
  informed L1–L6, their directions, their strengths, or the mixture-component
  shares in §6. Every number in this document is a guess by the author, not an
  estimate from any real portfolio.
- **A model that recovers this generator's structure has recovered exactly that
  — this generator's structure.** Strong precision/recall/PR-AUC on this dataset
  is evidence the model can learn a documented synthetic pattern. It is not
  evidence the same model would perform anywhere near as well against real
  Razorpay merchant disputes, where the true causal drivers, their strengths, and
  even the right feature set may differ from what's guessed here.
- **It cannot validate the feature list itself.** Whether AVS mismatch, device
  fingerprint reuse, etc. are actually as predictive in reality as modeled here is
  an assumption, not a finding.
- **It collapses issuer- and network-specific variation into one `p` per
  record.** Real arbitration outcomes vary by issuer and by network policy in ways
  this generator does not model separately.
- **It is not a simulation of adversarial adaptation.** The temporal drift in §7
  is a scripted, non-reactive change in base rates — deliberately not a
  fraud-vs-defense game loop. Building that would cross the "defense only, no
  attack simulator" line in SPEC.md §8, which is disqualifying for this track.
- **The Bayes ceiling in §5 is a ceiling over this synthetic world only.** It says
  what's achievable against `ε` and the label-sampling step *as defined here*. It
  says nothing about what ceiling would exist against real outcomes.

Per DECISIONS.md's 2026-08-31 "Synthetic data over real data" entry, this caveat
is required to appear in the README and the pitch video, not just here.

---

## Open parameters requiring review

Every guessed number in this document, gathered in one place for review before any
of it is implemented:

1. `p` band: `[0.02, 0.75]` (§4) — **revised down from `[0.04, 0.93]`**
2. Population `p` modes: low ~0.08, high ~0.39, E[p] ≈ 0.25 (§4) — **revised down
   from low ~0.15–0.20 / high ~0.65–0.75 / mean ~0.35–0.40, which was internally
   inconsistent**
3. Traveler confounder share: ~8% of `true_fraud=0` (§6A) — unchanged
4. Account-takeover confounder share: ~12% of `true_fraud=1` (§6B) — unchanged
5. Simulation window: 24 months (§7)
6. Response window (`respond_by`): 7 days — **unverified against real deadlines** (§7)
7. `true_fraud` base-rate drift: ~40% → ~50% over the window (§7) — this is also
   the `π_fraud` term in §4's E[p] arithmetic; changing it requires re-checking
   that arithmetic, not just this line
8. Authentication-strength improvement drift: small, unspecified magnitude (§7)
9. Oracle ceiling guess: PR-AUC roughly **0.30–0.36** against a **0.25 prevalence
   baseline** (§5) — **revised down from 0.75–0.88**, a direct consequence of
   item 2 above
10. `amount` distribution shape and parameters, now conditioned on `true_fraud`
    (§3, §8) — lognormal assumed; **session 2: `μ` gap solved analytically for
    `AUC(amount, true_fraud) = 0.60`** (`μ_true_fraud = 8.922458`,
    `μ_genuine = 8.6`, `σ = 0.9` unchanged) — see the amount revision note in §3
11. `dispute_propensity` (L7) distribution parameters and its exact weight in
    `p` (§1, §2) — new latent, added in revision 2, no numbers chosen yet
12. `days_between_purchase_and_dispute` measurement-noise magnitude: ±1–3 days
    guess (§1, §3) — new in revision 2
13. `checkout_hour_of_day` distribution: assumed uniform over 0–23 (§3) — new
    pure-noise feature added in revision 2, not yet load-bearing enough to need
    more than this
14. RuPay (and Diners) fraud-category reason codes: **unconfirmed** — not present
    in the source document used for item 15, and a real gap given Razorpay's
    India-facing focus (§8)
15. `reason_code` ↔ `card_network` independence is a known simplification (§8) —
    real codes are network-specific; this dataset draws them independently
16. **(session 2)** `filing_delay_days` component means: `mean_true_fraud_days =
    8.5`, `mean_genuine_days = 13.0` (§1 L5) — solved via the `Gamma(k,k)` ratio
    identity for `AUC(days_between_purchase_and_dispute, true_fraud) ≈ 0.35`,
    replacing the original 3/20 day guess that produced AUC 0.0623 (see the L5
    revision note in §1)
17. **(session 2)** `reason_code_misclassification_prob = 0.10` (§3, §8) — Guess,
    not derived; models issuer coding error, replacing the exact-copy defect
18. **(session 2)** `customer_communication_log` slot weights
    (`comms_opening_weights_*`, `comms_claim_weights_*`,
    `comms_detail_weights_*`), tone/length slopes on `relationship_genuineness`
    (`comms_signoff_polite_*`, `comms_detail_inclusion_*`), and messiness rates
    (`comms_near_empty_prob`, `comms_irrelevant_detail_prob`,
    `comms_typo_base_prob`, `comms_typo_relationship_scale`,
    `comms_lowercase_prob`) — all Guesses, tuned only so that no recurring exact
    string is single-class (see the comms-log revision note in §3), not against
    any real support-ticket corpus
19. **(session 2)** `boolean_reading_threshold = 0.5` and the lowered
    `avs_match_flip_prob = 0.05` / `cvv_match_flip_prob = 0.08` (§3) — replacing
    the `Bernoulli(latent)` mechanism that destroyed most of
    `authentication_strength`'s signal in `avs_match`/`cvv_match`

Resolved in revision 2, kept for the record: the four `reason_code` values are
now Razorpay's own confirmed published codes (§8), not invented placeholders.

**Resolved in session 2, kept for the record:** an out-of-band sanity check on
seed 42, n=15,000 found six defects, all fixed and documented at their
respective sections above — `customer_communication_log` was perfectly
separable by string match (§1 L1, §3); `reason_code` was an exact latent copy
(§1 L6, §3); `amount`'s AUC against `true_fraud` was 0.7631 against an intended
weak link (§3, item 10 above); `days_between_purchase_and_dispute`'s AUC was
0.0623, the most serious defect (§1 L5, item 16 above); `reason_code`'s
AUC of exactly 0.5000 was confirmed as zero-by-construction, not an artifact
(§1 L6); and `avs_match`/`cvv_match` were confirmed to be losing most of
`authentication_strength`'s signal to the `Bernoulli(latent)` draw mechanism,
now fixed (§3, item 19 above).
