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
| Append-only audit log | Built | `disputedesk/audit/` |
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

**We measured whether the LLM actually adds predictive value here, and it
does not.** `disputedesk/evidence/normalize_comms.py`'s typed extraction
was benchmarked against a TF-IDF + logistic-regression baseline on the same
`customer_communication_log` → `true_fraud` prediction task, 5-fold
stratified CV both times:

- TF-IDF + logistic regression: **AUC 0.6371**
- LLM normalisation (`openai/gpt-oss-20b` via Groq, real API, n=60,
  seed=0): **mean AUC 0.4211** (std 0.1378 across folds) — below a coin
  flip on 3 of 5 folds.

The LLM extraction is *reliable* (all 60 completions validated against
schema on the first call, no repair or fallback needed) but not
*predictive*: the two fields that would carry the most signal
(`claims_unauthorized_transaction`, `is_substantive`) sit near 1.0 for
both classes, because the generator's comms-log design (`GENERATOR.md` §3)
deliberately makes the differentiating signal a subtle frequency tilt
across near-synonymous phrasings — exactly what a bag-of-words vectorizer
preserves as continuous per-token weight and a coarse yes/no LLM extraction
collapses away. This is a single n=60 diagnostic run, not a multi-seed
headline (see `DECISIONS.md`'s 2026-09-01 entry for the full breakdown and
caveats on sample size) — but the gap is wide enough, and structurally
explained enough, to be a real result: it is the reason this project's LLM
role stayed narrow (draft text, don't extract structured signal from it)
rather than being tuned away.

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

| Cost (₹) | Precision | Recall | Policy advantage vs. baseline A (INR/1,000, median) |
|---:|---|---|---:|
| 0 | 0.2377 (0.2331–0.2422) | 1.0000 (1.0000–1.0000) | 0 |
| 50 | 0.2378 (0.2333–0.2423) | 1.0000 (0.9985–1.0000) | −219 |
| 100 | 0.2387 (0.2340–0.2429) | 0.9965 (0.9917–0.9977) | −163 |
| 200 | 0.2432 (0.2378–0.2467) | 0.9768 (0.9753–0.9812) | +1,772 |
| 300 | 0.2479 (0.2422–0.2524) | 0.9506 (0.9459–0.9578) | +4,209 |
| **400 (configured default)** | **0.2543 (0.2476–0.2569)** | **0.9155 (0.9111–0.9232)** | **+12,923** |
| 600 | 0.2641 (0.2571–0.2675) | 0.8446 (0.8301–0.8513) | +44,403 |
| 800 | 0.2724 (0.2672–0.2775) | 0.7726 (0.7655–0.7812) | +86,000 |
| 1,000 | 0.2791 (0.2700–0.2849) | 0.7036 (0.6957–0.7126) | +152,071 |
| 1,500 | 0.2915 (0.2848–0.2979) | 0.5613 (0.5513–0.5715) | +390,402 |
| 2,000 | 0.2991 (0.2949–0.3080) | 0.4521 (0.4380–0.4744) | +669,547 |
| 3,000 | 0.3142 (0.3074–0.3278) | 0.3123 (0.2999–0.3443) | +1,384,613 |
| 4,000 | 0.3233 (0.3137–0.3375) | 0.2339 (0.2207–0.2558) | +2,168,360 |
| 6,000 | 0.3369 (0.3264–0.3644) | 0.1540 (0.1408–0.1759) | +3,893,929 |
| 8,000 | 0.3467 (0.3346–0.3780) | 0.1252 (0.1100–0.1369) | +5,722,304 |
| 10,000 | 0.3599 (0.3435–0.3868) | 0.1069 (0.0981–0.1192) | +7,573,243 |

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
beats both baselines**, though narrowly at this cost (+12,923 INR/1,000,
≈0.75% over baseline A) — see "Cost sensitivity" below for why that margin
is cost-dependent, not fixed. Two other ways of scoring an escalated
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

The +12,923/1,000 advantage above is one point on a swept curve, not the
whole story (`python -m eval.run_cost_sensitivity`, same 20 seeds,
`representment_cost_inr` swept 0 → 10,000, `low_confidence_band` held
fixed):

- **Below ≈290**, the policy is **statistically indistinguishable from
  baseline A** — the median advantage is under 0.1% of the ~2,000,000
  base and its sign flips between adjacent swept values (noise from
  individual near-threshold disputes crossing `decide()`'s cutoff, not a
  real effect). Near cost=0 this is expected by construction:
  `expected_value ≈ p_win * amount`, positive for nearly every dispute, so
  the policy contests almost everything baseline A already contests
  everything.
- **The configured value, 400,** sits just above that noisy band — a real
  but modest edge (+0.75%).
- **Above ≈300 the advantage grows monotonically and robustly**, crossing
  10% of baseline A's own total between cost=800 (6.6%) and cost=900
  (10.4%) — roughly 2.25× the configured cost. By cost≈2,000, baseline A's
  recovered total has fallen to near zero while the policy still recovers
  770,639/1,000; by cost≈2,500 and above, baseline A goes net *negative*
  (contesting everything actively destroys value) while the policy stays
  solidly positive — the clearest divergence in the sweep, driven by the
  policy declining to contest disputes whose `p_win * amount` no longer
  clears the larger cost, versus baseline A paying the fee regardless of
  merit.

Reading a single number off this curve (the configured-cost row) without
the shape around it understates what the policy is worth in a
higher-cost regime and overstates how confidently it beats baseline A in a
near-zero-cost one. The exact crossover points are themselves noisy and
shouldn't be over-read; the qualitative shape — near-parity low, growing
and monotonic high — is the robust finding. Full table:
`DECISIONS.md`'s 2026-08-31 "representment_cost_inr sensitivity sweep"
entry; raw data in `data/eval/cost_sensitivity_median_iqr.csv` after
running the command above.

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
