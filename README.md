# Dispute Desk

Razorpay AI Buildathon, Track 02 (AI Risk Manager). One loss class:
**fraud-reason-code chargebacks** (the four confirmed card-network codes in
`GENERATOR.md` §8 — `MC_4837`, `MC_4840`, `VISA_10_4`, `AMEX_FR2`).

A merchant today either contests every fraud-reason-code chargeback it
receives (burning analyst time and accruing excessive-representment
exposure on network dispute-ratio programs) or accepts every one of them
(leaving recoverable revenue on the table). Dispute Desk takes an incoming
`open` dispute, scores the probability that contesting it would win, applies
a deterministic expected-value policy to contest, accept, or escalate the
case to a human, and — for the cases worth contesting — assembles the
evidence packet the reason code requires (drafting the explanation letter
and normalising the customer's free-text message), files it against
Razorpay's Disputes API in test mode, and writes an append-only audit row
explaining the decision end to end. Every number below is measured on a
fully synthetic, documented dataset (see "What this dataset cannot tell
you" below) — there is no real Razorpay merchant data anywhere in this
project.

Phases 0 through 4 (`PHASES.md`) are complete and frozen as of this
document. Nothing described here is a stub unless explicitly labelled one.

## What exists

| Component | Status | Where |
|---|---|---|
| Synthetic dispute generator | Built | `disputedesk/generator/` |
| Feature builder | Built | `disputedesk/features/` |
| Win-probability model (LightGBM) | Built | `disputedesk/model/` |
| Policy engine | Built | `disputedesk/policy/` |
| Evidence assembler (reason-code map + LLM drafting/normalisation) | Built | `disputedesk/evidence/` |
| Razorpay Disputes API client (test mode) | Built | `disputedesk/client/` |
| Append-only audit log (DB triggers + hash chain) | Built | `disputedesk/audit/` |
| FastAPI webhook | Built | `disputedesk/api/` |
| Demo CLI | Built | `disputedesk/cli/demo.py` |
| Eval harness (model, business, cost-sensitivity, oracle) | Built | `eval/` |

**Not built, stated plainly rather than left for a reviewer to discover:**
a document-upload pipeline (the real Razorpay `contest()` call submits the
drafted letter as `summary` text; it does not attach the per-evidence-type
document ids the live API also accepts, because this project never built a
file-storage/upload path — see `disputedesk/client/razorpay.py`'s module
docstring); an order-context lookup service (the webhook assumes
`avs_match` through `checkout_hour_of_day` arrive already joined onto the
dispute payload — a real deployment would fetch these from the merchant's
own order/customer systems by `payment_id`, which this project does not
build); a persisted, versioned model artifact (the model trains in memory
once per process from a fixed seed — see `disputedesk/model/registry.py`);
and webhook payload signature verification (Razorpay signs real webhooks;
this endpoint validates payload *shape* via Pydantic but does not verify an
`X-Razorpay-Signature` header, so it should not be exposed to the public
internet without adding that check first).

## The LLM authority boundary

The LLM is allowed exactly two jobs (`SPEC.md` §2, `DECISIONS.md`'s
"LLM authority boundary" entry):

- Drafting the `explanation_letter` evidence object from the dispute's
  order-context facts (`disputedesk/evidence/draft_letter.py`).
- Normalising the customer's free-text `customer_communication_log` into
  typed fields (`disputedesk/evidence/normalize_comms.py`).

It is explicitly forbidden from:

- Deciding contest vs. accept vs. escalate. That is `disputedesk/policy/`,
  a pure function of `P(win)` and `amount` (`SPEC.md` §4). The policy engine
  never calls the LLM and the LLM never sees the policy decision before it's
  made.
- Mapping reason codes to required evidence types. That is a deterministic
  lookup table (`disputedesk/evidence/reason_code_map.py`) sourced from
  Razorpay's published chargeback reason-code reference. Card networks
  publish this; an LLM here would be strictly worse and is disqualifying
  under the AI Judgment criterion.
- Any arithmetic on money. `amount` is only ever quoted back into a prompt,
  never computed by one.

Every LLM completion is validated against a Pydantic schema
(`disputedesk/evidence/schemas.py`) before anything downstream can use it.
On validation failure there is exactly one repair attempt (the error fed
back to the model); if that also fails, the system degrades to a
deterministic template letter / conservative-defaults normalisation and
sets a `human_review_required` flag — it never crashes and never lets
unvalidated text reach the Razorpay client (`SPEC.md` §7 failure path 2,
demoed live — see "Running the demo" below).

**We measured whether the LLM's typed extraction adds predictive value
here. At the sample size available, it cannot be shown either way — and the
comparison the earlier version of this README reported was not sound.**
Corrected 2026-09-02; the superseded claim and what was wrong with it are in
`DECISIONS.md`'s entry of that date. Reproduce everything below with
`python -m eval.run_extraction_comparison` (no API key needed — the LLM arm
is a committed recording).

`disputedesk/evidence/normalize_comms.py`'s typed extraction and a TF-IDF +
logistic-regression baseline (`eval/tfidf_baseline.py`) were scored on the
**same 60 disputes**, with the **same cross-validation folds**, on the same
`customer_communication_log` → `true_fraud` task:

| Arm | AUC (n=60, paired) | vs. chance (95% CI) |
|---|---|---|
| TF-IDF + logistic regression | 0.5392 | +0.0392 (−0.1349 to +0.2081) — **not distinguishable** |
| LLM typed fields | 0.3768 | −0.1232 (−0.2735 to +0.0304) — **not distinguishable** |

**Paired difference: +0.1624 in TF-IDF's favour, 95% paired bootstrap CI
−0.0648 to +0.3858 — the interval includes zero.** The direction survives.
The previous claim that this was "wide enough … to be a real result" does
not: at n=60 neither arm can be shown to carry any signal at all, so neither
can be shown to beat the other.

What was wrong before: the LLM arm (0.4211, n=60) was compared against a
TF-IDF figure of **0.6371 that had no code, no recorded n, no seed and no
command anywhere in this repository**. Re-implementing the baseline shows
what that number almost certainly was — a *large-sample* measurement. The
same implementation scores **0.6479 at n=3,000** and **0.5104 at n=60**
(per-fold means). So the original comparison put a large-sample baseline
against a 60-item LLM run and attributed the gap to the extraction method.
Roughly half of the apparent 0.216 gap was sample size, not method.

The LLM extraction is still *reliable* on this sample (all 60 completions
validated against schema on the first call, no repair or fallback), and the
structural argument for why its typed fields would struggle is unchanged and
worth keeping: the generator's comms-log design (`GENERATOR.md` §3) makes the
differentiating signal a subtle frequency tilt across near-synonymous
phrasings, which a bag-of-words vectorizer preserves as continuous per-token
weight and a coarse yes/no extraction collapses. But that is now an
explanation offered for a difference **this evidence cannot establish**, not a
finding.

**What would settle it:** re-running the LLM arm at n ≈ 1,000, which the
TF-IDF arm's own n=60-vs-n=3,000 spread suggests is roughly where this
comparison becomes readable. That needs a live API key and roughly 1,000
Groq calls; it was not run here (no `LLM_API_KEY` is configured in this
environment). The command is
`python -m eval.run_llm_normalization_quality --n-rows 1000 --seed 0`, and
`eval/run_extraction_comparison.py` will pair against it once the recording
is committed.

This measurement is *not* what justifies the LLM's narrow role in this
system. That justification is architectural and stands on its own: the policy
engine is a pure function of `P(win)` and `amount`, the reason-code mapping is
a published lookup table, and no LLM output does arithmetic on money
(`SPEC.md` §2). Nothing in the table above is load-bearing for that.

## Headline numbers

Every number below is `median (IQR: 25th–75th percentile)` across **20
seeds** (0–19), each seed generating **15,000 disputes** and evaluated on
its own temporal holdout only (never the training split — `CLAUDE.md`
invariant 2). Reproduce with `python -m eval.run_harness --n-seeds 20
--n-rows 15000` and `python -m eval.run_business_harness --n-seeds 20
--n-rows 15000`. Full per-seed and summary artifacts land in `data/eval/`
(gitignored; regenerate them, don't expect them checked in).

**Model quality**, temporal holdout:

| Metric | Median | IQR |
|---|---|---|
| Model PR-AUC (average precision) | 0.3522 | 0.3448–0.3696 |
| Prevalence baseline (no-skill PR-AUC) | 0.2377 | 0.2331–0.2422 |
| Oracle Bayes ceiling (closed-form, `eval/oracle.py`) | 0.4572 | 0.4527–0.4585 |
| Calibration error (ECE, 10 equal-width bins) | 0.0270 | 0.0208–0.0314 |

**Policy precision/recall**, across the same `representment_cost_inr` sweep
as "Cost sensitivity" below (0 → 10,000, 20 seeds, `low_confidence_band`
held fixed) — this, not any single-threshold figure, is the operating
result. CONTEST counts as the positive prediction; ESCALATE is folded in as
a positive prediction too, matching `ESCALATE_MODE = "naive_contest"` (the
same convention the rupee numbers below use — an escalated dispute is
credited exactly as if it had been contested, so precision/recall and the
rupee numbers describe the same set of "effectively contested" rows). ₹400
is `PolicyConfig`'s **configured default**, not "the" operating point — the
row it lands on is bolded below for reference, not singled out as special:

| Cost (₹) | Precision | Recall | Paired advantage vs. baseline A (INR/1,000) | 95% CI | Seeds + | CI excludes 0 |
|---:|---|---|---:|---|---:|---|
| 0 | 0.2377 (0.2331–0.2422) | 1.0000 (1.0000–1.0000) | +0 | +0 to +0 | 0/20 | **no** |
| 50 | 0.2378 (0.2333–0.2423) | 1.0000 (0.9985–1.0000) | -131 | -284 to -6 | 12/20 | yes |
| 100 | 0.2387 (0.2340–0.2429) | 0.9965 (0.9917–0.9977) | -257 | -563 to +53 | 10/20 | **no** |
| 150 | 0.2406 (0.2353–0.2444) | 0.9873 (0.9841–0.9931) | +141 | -416 to +675 | 11/20 | **no** |
| 200 | 0.2432 (0.2378–0.2467) | 0.9768 (0.9753–0.9812) | +1,040 | +289 to +1,787 | 14/20 | yes |
| 250 | 0.2456 (0.2400–0.2495) | 0.9657 (0.9605–0.9698) | +1,690 | +291 to +3,018 | 14/20 | yes |
| 300 | 0.2479 (0.2422–0.2524) | 0.9506 (0.9459–0.9578) | +4,184 | +2,673 to +5,717 | 18/20 | yes |
| 350 | 0.2503 (0.2441–0.2542) | 0.9324 (0.9270–0.9388) | +5,970 | +3,501 to +8,077 | 18/20 | yes |
| **400 (configured default)** | **0.2543 (0.2476–0.2569)** | **0.9155 (0.9111–0.9232)** | **+11,210** | **+8,508 to +13,633** | **19/20** | **yes** |
| 600 | 0.2641 (0.2571–0.2675) | 0.8446 (0.8301–0.8513) | +42,923 | +38,006 to +47,564 | 20/20 | yes |
| 800 | 0.2724 (0.2672–0.2775) | 0.7726 (0.7655–0.7812) | +95,731 | +88,717 to +102,444 | 20/20 | yes |
| 1,000 | 0.2791 (0.2700–0.2849) | 0.7036 (0.6957–0.7126) | +162,600 | +154,130 to +170,686 | 20/20 | yes |
| 1,500 | 0.2915 (0.2848–0.2979) | 0.5613 (0.5513–0.5715) | +396,168 | +384,822 to +407,618 | 20/20 | yes |
| 2,000 | 0.2991 (0.2949–0.3080) | 0.4521 (0.4380–0.4744) | +685,586 | +671,494 to +700,821 | 20/20 | yes |
| 3,000 | 0.3142 (0.3074–0.3278) | 0.3123 (0.2999–0.3443) | +1,390,451 | +1,369,312 to +1,413,406 | 20/20 | yes |
| 4,000 | 0.3233 (0.3137–0.3375) | 0.2339 (0.2207–0.2558) | +2,180,044 | +2,151,701 to +2,210,984 | 20/20 | yes |
| 6,000 | 0.3369 (0.3264–0.3644) | 0.1540 (0.1408–0.1759) | +3,904,450 | +3,869,015 to +3,941,954 | 20/20 | yes |
| 8,000 | 0.3467 (0.3346–0.3780) | 0.1252 (0.1100–0.1369) | +5,712,932 | +5,665,269 to +5,759,948 | 20/20 | yes |
| 10,000 | 0.3599 (0.3435–0.3868) | 0.1069 (0.0981–0.1192) | +7,555,900 | +7,501,981 to +7,607,412 | 20/20 | yes |

**The advantage column is the mean of per-seed differences**, not a
difference of medians. Every seed scores both arms on the identical
holdout, so the comparison is paired and must be estimated as one; the CI
is a 95% percentile bootstrap over the 20 seeds and "Seeds +" is the
sign-test count. Both are reported because they answer different questions
and here they sometimes disagree — see ₹50 below. The previous version of
this table reported `median(policy) − median(baseline A)`; that estimator,
and the conclusion drawn from it, are corrected in `DECISIONS.md`
2026-09-02.

Precision rises and recall falls monotonically as cost rises — a higher
cost raises the per-row breakeven `p_win` required to contest
(`expected_value = p_win * amount − cost > 0`), so the policy contests a
narrower, higher-precision, lower-recall slice as cost grows. Full detail:
`DECISIONS.md`'s 2026-09-01 "Policy precision/recall added to the cost
sweep" entry.

<details>
<summary>Superseded Phase 2 placeholder (pre-policy-engine; kept for
history, not current)</summary>

Precision/recall, **at the placeholder threshold = train-split label
prevalence (median 0.2543, IQR 0.2522–0.2559)** — from before the policy
engine existed, when there was no expected-value rule to report
precision/recall against:

| Metric | Median | IQR |
|---|---|---|
| Precision | 0.3537 | 0.3490–0.3606 |
| Recall | 0.6954 | 0.6771–0.7037 |

This threshold's median (0.2543) visually matches the configured-cost
(₹400) policy precision median above, also 0.2543 — checked per-seed across
the same 20 seeds and confirmed **coincidental**: the two quantities differ
at every individual seed (they're computed from disjoint splits by
different code paths); only their medians happen to sit near this
generator's overall label prevalence. See `DECISIONS.md`'s 2026-09-01 entry
for the full per-seed A−B spread.

</details>

**Business metrics**, at `PolicyConfig`'s configured default
(`representment_cost_inr=400.0`,
`low_confidence_band=(0.45, 0.55)`), rupees recovered per 1,000 disputes,
median test-split size n=3,609.5:

| | Median (INR / 1,000 disputes) | IQR |
|---|---|---|
| Policy (escalated disputes scored `naive_contest` — see below) | 1,714,015 | 1,632,554–1,746,598 |
| Baseline A: contest everything | 1,701,092 | 1,620,248–1,737,029 |
| Baseline B: accept everything | 0 | 0–0 |

`naive_contest` scores an escalated (low-confidence) dispute exactly the
way baseline A already scores every dispute — contested — which is the
fair apples-to-apples comparison, since baseline A doesn't have an
abstention path to be penalised for using. Under this scoring **the policy
beats both baselines**, though narrowly at this cost: **paired mean
+11,210 INR/1,000, 95% CI +8,508 to +13,633, 19 of 20 seeds positive —
≈0.66% over baseline A**. (Corrected 2026-09-02: previously reported as
+12,923 / ≈0.75%, which was a difference of medians on a paired design. The
corrected figure is smaller — see `DECISIONS.md`.) See "Cost sensitivity"
below for why that margin is cost-dependent, not fixed, and for two costs the
sweep does not charge for. Two other ways of scoring an escalated
dispute are reported for context, not as headlines: crediting it 0 (the
most conservative reading — the policy then trails baseline A, 1,559,504
vs. 1,701,092, because that scoring structurally penalises having an
abstention path at all) and crediting it the oracle outcome (the true
label as foreknowledge no real reviewer has — 1,727,152, an optimistic
upper bound). Escalated disputes are 4.33% (median) of total holdout
rupees at stake, close to their 5.5% share of dispute count, not wildly
disproportionate at this confidence band's width.

**False-positive / false-negative cost** (checklist item 6), same run,
same configured default:

| | Count / 1,000 | Cost (INR / 1,000 disputes) |
|---|---|---|
| False positive (contested, should have accepted) | 2,204.5\* | 244,384 (IQR 241,137–247,783) |
| False negative (accepted, was actually winnable) | 71.5\* | 42,357 (IQR 38,127–51,175) |

\*Counts are medians over a ~3,609-row holdout, reported per-1,000 for
comparability with the rupee figures; they are not literal per-1,000
observation counts.

FP cost = the fixed `representment_cost_inr` (INR 400 — modeled as three
named components in `disputedesk/policy/config.py`: a ~200 network
resubmission fee, ~150 analyst time, ~50 excessive-representment exposure;
this breakdown is a stated assumption, not a cited fee schedule — replace
with a real figure before this touches real money). FN cost = the full
lost dispute `amount`.

## Cost sensitivity — a curve, not a point estimate

The +11,210/1,000 paired advantage at the configured cost is one point on a
swept curve, not the whole story (`python -m eval.run_cost_sensitivity
--n-seeds 20 --n-rows 15000`, same 20 seeds, `representment_cost_inr` swept
0 → 10,000, `low_confidence_band` held fixed).

**Where the advantage is measurable, and where it is not.** Under the paired
estimator:

- **No measurable advantage at or below ₹150.** At ₹0 the two arms are
  identical by construction (`expected_value = p_win × amount ≥ 0` for
  essentially every dispute, so the policy contests what baseline A contests).
  At ₹100 and ₹150 the 95% CI includes zero.
- **At ₹50 the policy is measurably *worse*:** paired mean −131/1,000, CI
  −284 to −6, excluding zero. And yet 12 of 20 seeds are positive. Both facts
  are real, and the shape behind them is measured rather than described
  (`eval.cost_sensitivity.loss_tail`):

  | Cost (₹) | Mean | Seeds +/− | Worst seed | Best seed | Spread | Mean loss ÷ mean gain |
  |---:|---:|---:|---:|---:|---:|---:|
  | 50 | −131 | 12 / 7 | −1,167 | +120 | 1,286 | **7.65×** |
  | 100 | −257 | 10 / 10 | −1,539 | +817 | 2,355 | 2.52× |
  | 400 | +11,210 | 19 / 1 | −5,394 | +20,597 | 25,992 | 0.45× |

  **At ₹50 the policy wins slightly more often than it loses, and loses 7.65
  times harder when it does.** A majority of seeds improving is not the same
  as the expected value improving, which is why the mean and the sign count
  are both reported and neither is allowed to stand alone. The asymmetry is a
  low-cost phenomenon — the ratio falls monotonically as cost rises and is
  below 1 by the configured ₹400, where the policy wins more often *and*
  bigger.
- **The advantage becomes measurable at ₹200** (+1,040, CI +289 to +1,787,
  14/20 seeds positive) and the CI excludes zero at every swept cost above it.
- **Above ₹200 it grows monotonically.** By ₹2,000 baseline A's own recovered
  total has fallen to near zero while the policy still recovers 770,639/1,000;
  by ₹3,000 baseline A is net *negative* — contesting everything actively
  destroys value — while the policy is still positive. That divergence is the
  robust part of this curve.

So: **no measurable advantage over "contest everything" below ₹200 per
representment; the advantage appears at ₹200 and grows above it.** The
configured ₹400 sits above that threshold but not far above it.

The sentence this section previously carried — that below ≈290 the sign flips
were "noise from individual near-threshold disputes … not a real effect" — was
an assertion the unpaired estimator had no standing to make, and the paired
data contradicts it in both directions: ₹200–₹250 *are* measurable positives,
and ₹50 is a measurable negative. It is deleted rather than softened.

### Two things this sweep does not charge for

Both were implicit until 2026-09-02. Both make the numbers above **more**
favourable to the policy than a full accounting would, never less.

**1. Human review time.** Every CONTEST and ESCALATE decision above is
credited as a filed representment. In the running system a dispute whose
evidence packet is not fit to file is withheld and queued for a person
(Phase 0 remediation), and escalated disputes were always a person's problem.
The sweep charges nothing for that time. Turning that into a number: the
₹400 advantage of +11,210/1,000 is cancelled if each human-touched dispute
costs about **₹200** to review — computed at the measured ESCALATE rate of
5.62% alone, so it is an *upper* bound; any non-zero withheld rate lowers it.
For scale, `representment_cost_inr`'s own breakdown already budgets ₹150 of
analyst time per contested dispute. An advantage that survives only while a
review costs less than ₹200 is a thin one, and it is stated here rather than
left for a reader to derive.

The withheld rate itself is only half measurable. Its reason-code component
is exactly **0%** on this dataset (the generator emits only codes the system
has an evidence strategy for — asserted in
`tests/test_eval_sweep_assumptions.py`). Its letter-drafting component is
**currently unmeasured**: the only measurement of it was invalidated by the
Phase 0 schema and prompt changes and needs a live API key to redo. Modelling
it would have meant inventing a rate; it is excluded and named instead.

**2. Whether a filed contest would be accepted at all.** Razorpay's contest
endpoint requires at least one document id when `action="submit"`, and this
project has no document-upload pipeline, so it sends none. **A contest filed
by this system today would very likely be rejected by the live API.** Every
absolute "rupees recovered" figure above therefore describes what the policy
*would* recover if its filings were accepted. The *comparison* is unaffected —
baseline A files through the same client and inherits the same gap — but the
absolute totals are contingent on a component that does not exist. Recorded
in code as `eval.cost_sensitivity.SWEEP_ASSUMES_EVERY_SUBMISSION_IS_ACCEPTED
= False`, with a test that keeps it false until the upload path is built.

Full table: `DECISIONS.md`'s 2026-09-02 paired-estimator entry; raw data in
`data/eval/cost_sensitivity_median_iqr.csv` after running the command above.

## What this dataset cannot tell you

Verbatim from `GENERATOR.md` §9 — required to appear here and in the pitch
video:

> - **The causal story is authored, not measured.** No real dispute-outcome data
>   informed L1–L6, their directions, their strengths, or the mixture-component
>   shares in §6. Every number in this document is a guess by the author, not an
>   estimate from any real portfolio.
> - **A model that recovers this generator's structure has recovered exactly that
>   — this generator's structure.** Strong precision/recall/PR-AUC on this dataset
>   is evidence the model can learn a documented synthetic pattern. It is not
>   evidence the same model would perform anywhere near as well against real
>   Razorpay merchant disputes, where the true causal drivers, their strengths, and
>   even the right feature set may differ from what's guessed here.
> - **It cannot validate the feature list itself.** Whether AVS mismatch, device
>   fingerprint reuse, etc. are actually as predictive in reality as modeled here is
>   an assumption, not a finding.
> - **It collapses issuer- and network-specific variation into one `p` per
>   record.** Real arbitration outcomes vary by issuer and by network policy in ways
>   this generator does not model separately.
> - **It is not a simulation of adversarial adaptation.** The temporal drift in §7
>   is a scripted, non-reactive change in base rates — deliberately not a
>   fraud-vs-defense game loop. Building that would cross the "defense only, no
>   attack simulator" line in SPEC.md §8, which is disqualifying for this track.
> - **The Bayes ceiling in §5 is a ceiling over this synthetic world only.** It says
>   what's achievable against `ε` and the label-sampling step *as defined here*. It
>   says nothing about what ceiling would exist against real outcomes.

See `GENERATOR.md` in full for the generative story, every named causal
factor, and every parameter guess with its own revision history.

## Running the demo from a clean clone

The demo has two segments, printed under their own headers, with different
reproducibility guarantees:

- **Segment A — deterministic.** Requires Python 3.11+ and no network
  access, secrets, or `.env` file — it trains its own model in memory from
  the synthetic generator and fakes both the LLM and Razorpay API clients.
  **This segment's stdout is byte-identical across cold clones**: same seed,
  same fixtures, no LLM call, no wall-clock timestamp in anything printed.
  This is the segment any reproducibility check should run — pass
  `--deterministic-only` to run only it.
- **Segment B — LLM output, not reproducible.** Drafts the explanation
  letter for two disputes chosen to differ on both the fraud reason code and
  evidence availability (one with the full required evidence set, one with
  a documented gap), using a real `GroqHttpLLMClient` call, and prints the
  letters verbatim. Needs a populated `.env` (see `.env.example`) and
  network access to the configured `LLM_API_URL`; the same underlying
  live-Groq-model wording variance the "LLM authority boundary" section
  above measured applies here too, so a second run can print different text
  for the same dispute. Degrades to a clear skip message, not a crash, if
  `.env` isn't configured.

```bash
git clone <this repo> && cd disputedesk  # or your clone's directory name
pip install -e ".[dev]"
python -m disputedesk.cli.demo                     # both segments
python -m disputedesk.cli.demo --deterministic-only # Segment A only, offline
```

Segment A replays two dispute events with visibly different evidence
profiles through the real FastAPI webhook route, the real policy engine,
and the real evidence assembler end to end — one scores high enough to
contest, the other scores low enough to accept, so `P(win)` and the
decision can be seen responding to input, not repeating a fixed value. It
also demonstrates idempotent replay, webhook payload rejection, and all
three `SPEC.md` §7 / demo failure paths live: an LLM that returns invalid
output twice in a row (repair fails, degrades to a deterministic template,
flags for human review), a Razorpay API call that times out once and then
recovers, and a Razorpay API call that receives an HTTP 429 with
`Retry-After` and recovers — both retry paths via the same shared
exponential-backoff retry helper. Nothing crashes at any step.

Pass `--db-path <file>` instead of the default in-memory database to
replay the same webhook event across two separate process invocations and
see the decision-before-API-call idempotency guarantee hold across a
restart, not just within one process.

## Setup and testing

```bash
pip install -e ".[dev]"
cp .env.example .env   # only needed for the real webhook/LLM/Razorpay paths - not the demo or tests
pytest
ruff check .
```

No test makes a network call — the LLM and Razorpay clients are faked
throughout (`FakeLLMClient`, `FakeRazorpayClient`). Reproduce the headline
numbers above with:

```bash
python -m eval.run_harness --n-seeds 20 --n-rows 15000
python -m eval.run_business_harness --n-seeds 20 --n-rows 15000
python -m eval.run_cost_sensitivity --n-seeds 20 --n-rows 15000
```

## Further reading

- `SPEC.md` — what this system is, its components, and where the LLM is and
  is not allowed.
- `PHASES.md` — the build order and each phase's acceptance criteria.
- `GENERATOR.md` — the synthetic data generative story, written before any
  generator code existed, with every parameter guess and its revision
  history.
- `DECISIONS.md` — append-only decision and measurement log; the source for
  every number in this document and most of `ARCHITECTURE.md`.
- `ARCHITECTURE.md` — the system architecture, including the LLM authority
  boundary and why the decision layer is deterministic.
