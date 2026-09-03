# CALIBRATION.md

Provenance for every generator parameter that has a real-world analogue: what
value the generator uses, the published range for that quantity, the named
source, and whether the chosen value falls **inside** or **outside** that
range.

This file is a citation index, not a measurement. Nothing here is reproduced
by a command — the numbers on the "published range" side come from external
sources, not from this repo — so none of these rows belong in `NUMBERS.md`
(`NUMBERS.md` is provenance for numbers *this repo computes*; adding external
citations there would dilute that invariant). `NUMBERS.md` points here
instead.

Nothing in this file changes the generator. Where a configured value sits
outside a published range, that is recorded as a limit and left alone —
CLAUDE.md rule 5 ("do not touch the generator to make a number move") and the
session's standing rules apply here exactly as they do everywhere else.

**Summary.** Of the 7 parameters listed, 5 have a checkable published
real-world analogue (rows 2–5, 7). Of those 5: 3 fall inside the cited
numeric range (representment win rate, per-dispute contest cost, analyst time
cost), 1 falls outside (friendly-fraud share), and 1 (reason-code mix) has a
qualitative published analogue but no quantitative range to be scored
inside/outside of, so it is recorded as a known simplification instead. The
remaining 2 (chargeback base rate, VAMP/ECM threshold) are not modeled in
this system at all and are listed for completeness, not scored.

---

## 1. Chargeback base rate

| | |
|---|---|
| **Generator value** | Not modeled. |
| **Published range** | Blended 0.26% (Sift benchmarking, Q3 2025); industry spread 0.12%–1.02% (Datos/Mastercard); CNP eCommerce 0.6%–1% (PayKings). |
| **Source** | Sift Q3 2025 benchmarking report [vendor]; Datos/Mastercard 2025 [regulator/network]; PayKings [vendor]. |
| **Date** | Q3 2025 / 2025. |
| **Inside or outside** | **N/A — no generator analogue exists to compare.** The generator does not model an underlying transaction population or a per-transaction chargeback incidence rate; it generates already-filed dispute records directly (`GeneratorConfig` has no field for transaction volume or dispute-per-transaction odds — checked, none exists). There is nothing in the generator this published rate could be checked against without inventing a transaction-volume parameter that doesn't currently exist anywhere in the pipeline. |

## 2. Friendly-fraud share

| | |
|---|---|
| **Generator value** | `true_fraud_rate_month0 = 0.40`, `true_fraud_rate_month_last = 0.50` (`disputedesk/generator/config.py`). **Direction checked, not inverted:** `disputedesk/generator/latents.py:55-58` computes `fraud_rate` from these two fields and draws `true_fraud = rng.random(n) < fraud_rate` — so `true_fraud_rate_month0/last` parametrizes `P(true_fraud=1)`, the share of **genuine third-party fraud**, directly (`GENERATOR.md` L1: `true_fraud=1` means "the actual cardholder didn't make the charge"). Friendly fraud is the complement, `true_fraud=0`, so friendly-fraud share = 1 − fraud_rate = **50%–60%** of the dispute population (60% at month 0, drifting to 50% at the window's end, as the genuine-fraud rate rises from 40% to 50%). |
| **Published range** | 19–20% (Javelin/Visa, denominator: **fraud-coded disputes only**) vs. 43.8% (Chargebacks911 2026 merchant survey, denominator: **all chargebacks**, not just fraud-coded ones). Both ends reported, not collapsed to one number — the gap is a denominator difference. **Which denominator applies here:** this generator's population is fraud-coded disputes only — all four `reason_codes` (`MC_4837`, `MC_4840`, `VISA_10_4`, `AMEX_FR2`) are fraud reason codes, and CLAUDE.md scopes the whole project to "fraud-reason-code chargebacks." So the Javelin/Visa 19–20% figure (same denominator: fraud-coded disputes) is the apples-to-apples comparison; the Chargebacks911 43.8% (all chargebacks, a broader population that includes non-fraud codes like 13.1/13.2 this generator never produces) is reported for completeness but is not the matched comparison. |
| **Source** | Javelin/Visa [regulator/network] (fraud-coded disputes); Chargebacks911 2026 merchant survey [vendor] (all chargebacks). |
| **Date** | Javelin/Visa: not dated in the source material available; Chargebacks911: 2026. |
| **Inside or outside** | **OUTSIDE against both denominators**, but by very different margins. Against the matched denominator (Javelin/Visa, 19–20%, fraud-coded disputes), the generator's 50–60% is dramatically outside — roughly 2.5–3x the published ceiling. Against the broader, less-matched denominator (Chargebacks911, 43.8%, all chargebacks), it is only narrowly outside (50% vs. 43.8% at the window's low end). Recorded as outside either way; the generator was not adjusted to close the gap. `GENERATOR.md`'s own §5 justification (E[p]≈0.25 sitting ~8 points above a commonly-cited ~17% blended win-rate anchor, see row 3 below) is a related but separate claim about win probability, not about friendly-fraud share — it does not resolve this gap. |

## 3. Representment / contest win rate

| | |
|---|---|
| **Generator value** | `E[p]` target 0.25 (`GeneratorConfig.e_p_target`); realized prevalence baseline (median across 20 seeds, n=15,000, `won_if_contested=1` rate on the temporal holdout) **0.2377 (IQR 0.2331–0.2422)** — `NUMBERS.md`, "Prevalence baseline". |
| **Published range** | ~20% network-derived (Datos/Mastercard 2025) vs. ~44% merchant-survey (Chargebacks911). Both ends reported — the gap is a denominator difference, not disagreement. Mastercard 2025 additionally reports FIs win 45.8% of what they represent, and a merchant *net* win rate of ~8.1% (net of disputes never contested at all). |
| **Source** | Datos/Mastercard 2025 [regulator/network]; Chargebacks911 [vendor]. |
| **Date** | 2025 (Datos/Mastercard); Chargebacks911 date not specified in source material available. |
| **Inside or outside** | **INSIDE.** 0.2377 falls between the network-derived ~20% floor and the ~44% merchant-survey ceiling, sitting closer to the network-derived end. Note the label semantics: `won_if_contested` means "winnable *if contested with correct, complete evidence*" (`GENERATOR.md` §5), which is narrower and more favorable than a blended observed win rate that includes merchants who contest with weak evidence or contest reflexively — so a value on the higher side of "inside" is expected, not a red flag. |

## 4. Per-dispute contest cost

| | |
|---|---|
| **Generator value** | `REPRESENTMENT_COST_INR = 400.0` (`disputedesk/policy/config.py`), a policy config constant, not a generator field — decomposed as ₹200 network resubmission fee + ₹150 analyst time + ₹50 excessive-representment exposure. |
| **Published range** | ₹250–₹2,000 per dispute (~₹500 typical) for Indian payment gateways. Razorpay does not publish a flat fee for this. |
| **Source** | Indian gateway fee schedules, aggregated [vendor]; no regulator/network source publishes a flat per-dispute representment fee for the Indian market. |
| **Date** | Not independently dated in source material available. |
| **Inside or outside** | **INSIDE**, and conservative relative to the ₹500 typical point (₹400 < ₹500). Not re-selected to match the ~₹500 typical figure — the existing cost-advantage sweep already spans ₹50–₹10,000, per session instruction, and this configured value is left where it already stood. |

## 5. Analyst time cost

| | |
|---|---|
| **Generator value** | ₹150 of the ₹400 `REPRESENTMENT_COST_INR`, attributed to "analyst time to assemble and submit the packet" (`disputedesk/policy/config.py` comment). |
| **Published range** | India fraud-analyst pay ~₹201–₹284/hr (Glassdoor, 2025–26). ₹150 implies ~30–45 minutes of analyst time per reviewed dispute (₹150 ÷ ₹284/hr ≈ 31.7 min; ₹150 ÷ ₹201/hr ≈ 44.8 min). |
| **Source** | Glassdoor 2025–26 [vendor]. For scale only, not a direct comparison (different market/currency): Mastercard/Datos 2025 puts US FI cost per dispute at $9.08–$10.32 [regulator/network]. |
| **Date** | 2025–2026 (Glassdoor); 2025 (Mastercard/Datos, US context only). |
| **Inside or outside** | **Plausible / slightly conservative.** ₹150 is not itself a published rate to be inside or outside of — it is a cost component whose *implied review duration* (~30–45 min) is a reasonable read for assembling and submitting an evidence packet, on the conservative (lower) side. The US FI figure ($9.08–$10.32/dispute) is cited as context only; it is a different market and currency and is not compared directly. |

## 6. VAMP / excessive-dispute-ratio threshold

| | |
|---|---|
| **Generator value** | Not modeled as a rate. Present only qualitatively, as the stated justification for the ₹50 "excessive-representment exposure" component of `REPRESENTMENT_COST_INR` (`disputedesk/policy/config.py`: "marginal risk of tripping a card network's dispute-ratio program on a representment that is later lost"). |
| **Published range** | Visa VAMP excessive threshold: 1.5% (effective 1 Apr 2026; US/Canada/EU/APAC only). Mastercard ECM: 100 chargebacks **and** 1.5% ratio (both conditions). |
| **Source** | Visa [regulator/network]; Mastercard [regulator/network]. |
| **Date** | Visa: 1 Apr 2026. Mastercard: not independently dated in source material available. |
| **Inside or outside** | **N/A — no dispute-ratio metric exists in this system to compare against a threshold.** The generator and policy engine operate on individual disputes; neither computes a merchant-level dispute-to-transaction ratio, so there is no generated number this threshold could be checked against. It is recorded here only because it is the real-world program the ₹50 cost component is a stated hedge against. |

## 7. Reason-code mix

| | |
|---|---|
| **Generator value** | Uniform draw over four codes — `MC_4837`, `MC_4840`, `VISA_10_4`, `AMEX_FR2` (`disputedesk/generator/config.py: reason_codes`, drawn via `rng.choice` with no weighting in `disputedesk/generator/latents.py:144` — confirmed by reading the call site, no `p=` argument is passed). Each code ≈25% of records by construction. |
| **Published range** | None quantitative. Qualitative evidence (PaymentBrief 2026) indicates reason-code prevalence is *not* uniform in practice: 10.4 (card-absent fraud, closest real-world analogue to this generator's `VISA_10_4`) dominates paired with 13.1 (merchandise/services not received) in physical eCommerce portfolios, and paired with 13.2 (services not as described / canceled recurring) in SaaS/digital-goods portfolios. No source quantifies the exact split. |
| **Source** | PaymentBrief 2026 [vendor] (qualitative category pairing only — no vendor or academic source publishes a quantitative mix). |
| **Date** | 2026. |
| **Inside or outside** | **Not scored — recorded as a known simplification, not as unknowable.** There is no quantitative target to be inside or outside of, so "inside/outside" does not apply. But the qualitative evidence is directly relevant: it indicates real-world reason-code prevalence skews toward whichever fraud code most commonly co-occurs with the dominant non-fraud code in a given vertical (10.4-led combinations dominate both cited verticals), which this generator's uniform 25%/25%/25%/25% draw across `MC_4837`/`MC_4840`/`VISA_10_4`/`AMEX_FR2` does not reflect. Uniform is therefore a **deliberate simplification known to be unrepresentative** of the real mix, not a defensible best estimate in the absence of data — recorded as such. |

---

## Summary

Of the parameters above with an actual generator analogue to check (rows 2–5,
7 — rows 1 and 6 have no analogue in this system by construction):

- **Inside published range:** representment/contest win rate (row 3), per-dispute
  contest cost (row 4).
- **Plausible, not a strict range comparison:** analyst time cost (row 5).
- **Outside published range:** friendly-fraud share (row 2) — direction verified
  against `disputedesk/generator/latents.py:55-58` (not inverted); the generator's
  50–60% is dramatically outside the matched fraud-coded-disputes denominator
  (19–20%, Javelin/Visa) and narrowly outside the broader all-chargebacks
  denominator (43.8%, Chargebacks911).
- **Known simplification, not scored:** reason-code mix (row 7) — no quantitative
  benchmark exists, but qualitative evidence (PaymentBrief 2026) indicates the
  real-world mix is not uniform, so the generator's uniform 25%/25%/25%/25% draw
  is recorded as a deliberate, known-unrepresentative simplification.
- **No generator analogue exists:** chargeback base rate (row 1), VAMP/ECM
  threshold (row 6).

Nothing above changes the generator. The friendly-fraud share is recorded as
outside its published range and left as-is, per CLAUDE.md rule 5 and this
session's standing rules — a limit worth stating in the README, not a defect
to quietly correct by retuning `true_fraud_rate_month0`/`true_fraud_rate_month_last`.
