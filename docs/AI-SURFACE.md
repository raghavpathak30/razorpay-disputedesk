# AI surface spike — candidate integration points

**Status: design only. No implementation code has been written. Nothing here is built.**

Written 2026-09-02, against the repo at commit `03dd265`. This document proposes where a
model could take on real judgment in this system, argues against the candidates that do not
survive scrutiny, and ranks the ones that do. Step 2 of the brief is "stop" — so this ends at
a recommendation, not a change.

---

## 0. Three premises in the brief that the repo contradicts

I am raising these first because two of them change the ranking, and one changes what
"lands before Phase 3" means.

### 0.1 The 71 reason codes do not carry different evidence requirements — not here

`data/reference/razorpay_chargeback_codes.csv` holds 71 published codes. It has seven
columns: `network, code, wire_code, reason_text, category, preventable, reversible`. There is
no per-code evidence checklist in it, and none anywhere else in the repo.
`disputedesk/evidence/reason_code_map.py` says so in its own docstring:

> That source lists each code's network and reason text but not a *per-code evidence
> checklist* — card networks publish evidence-matrix documents separately, and none was
> fetched for this project. The mapping below is therefore an ASSUMPTION, not a citation.

And the mapping it holds is four codes (`MC_4837`, `MC_4840`, `VISA_10_4`, `AMEX_FR2`) all
pointing at the *same* five evidence types, because all four are card-not-present
"I didn't authorize this" claims. That is deliberate: `SPEC.md` §1 is "exactly one class of
loss. Nothing else," and the four codes are that one class.

So an evidence-sufficiency assessor that varies its judgment by reason code has nothing to
vary against. To give it 71 distinct requirement sets I would have to either fetch real
network evidence matrices (not in this environment) or invent them — and inventing them is
`SPEC.md` §2's named disqualifier:

> Reason code to required-evidence mapping. That is a lookup table. Card networks publish it.
> An LLM here is strictly worse and will be marked down.

This does not kill the candidate. It moves it from "reads 71 requirement sets" to "reads one
requirement set of five types" and takes most of the judgment depth with it. See §2.3.

### 0.2 There is only one unstructured artifact per dispute, and it is a slot template

The brief describes reading "delivery receipts, merchant notes, chat logs, emails."
The generator produces exactly one free-text field, `customer_communication_log`, and
`disputedesk/generator/comms.py` builds it from four openings × four claims × four details ×
six sign-offs, plus a typo/lowercase pass. Nothing else unstructured exists.

An evidence-sufficiency candidate therefore needs a new artifact generator — which is
`SPEC.md` §5 work ("this is where the project dies if it dies"), the repo's own
highest-risk component, on top of the integration itself. And the measurement that follows
grades the model on recovering a template I wrote, against a ceiling I set. That is not
disqualifying, but it caps how strong the resulting claim can be, and the cap has to be
stated in the README rather than discovered by a panel.

### 0.3 This is not landing before Phase 3 — Phase 5 is already frozen

The brief says this "runs alongside Phase 2 and lands before Phase 3." The repo is past
that. `README.md` line 22: "Phases 0 through 4 (`PHASES.md`) are complete and frozen."
`DECISIONS.md` has a 2026-09-01 "Phase 5 freeze: refactor, security review, docs" entry, and
`PHASES.md` Phase 5 opens with "No new code."

Any candidate here reopens a frozen build. The real cost of that is not the feature — it is
the tail: the README's component table, the "frozen" claim itself, the security review pass
(a new LLM output reaching a submission path is exactly what that pass exists to check), and
Phase 5's "three cold runs from a fresh clone." I have priced that tail into each candidate
below rather than quoting the feature cost alone.

### 0.4 No API key is configured in this environment

`LLM_API_KEY` is unset and there is no `.env`. Every candidate below needs live calls to
produce its headline number. Under the brief's own rule, each is buildable and testable here
against a recorded-fixture stub, with the measurement left as a recorded manual command —
the same shape as the open `--n-rows 1000` normalisation run the README already carries.
This is a constraint on **all** candidates equally, so it does not separate them; it does
mean no candidate delivers a measured interval in this session.

---

## 1. The test each candidate had to pass

A candidate survives only if all six hold:

1. **Judgment, not lookup.** A well-written deterministic function cannot do it. Not "would
   be tedious to write" — cannot.
2. **Policy isolation intact.** `disputedesk/policy/engine.py` stays a pure function of
   `p_win` and `amount`. It gains no import, no argument, no field. It keeps veto.
3. **Schema-validated structured output**, through the existing
   `evidence/validated_call.py` path: one repair attempt, then fail.
4. **Fails closed** into `_withhold_for_review` in `disputedesk/api/pipeline.py`, carrying
   provenance.
5. **A measurable outcome** with an n and an interval, and a named baseline — or an explicit
   statement that no baseline exists and why that is honest rather than convenient.
6. **Ground truth that is not the model's own opinion.** This is the one that did most of the
   killing. Several attractive candidates can only be scored by asking a model whether the
   model was right.

---

## 2. Candidates that survived

### 2.1 Letter grounding gate — every factual assertion traced to a record field

*(This is the brief's "adversarial self-critique," sharpened. See the note at the end of this
section for why the sharpening matters.)*

**The judgment.** Given a drafted `explanation_letter` and the seven-field `DisputeContext`
it was drafted from, decide for each factual assertion in the letter whether the record
supports it, contradicts it, or is silent on it.

**Why deterministic code cannot do it.** A function can check the assertions it knows to look
for. `delivery_confirmed=False` plus the substring "delivery was confirmed" is a three-line
check, and I would write it. Two things defeat that function, and the second is decisive:

- *Paraphrase.* The model's ways of asserting delivery are an open set — "the parcel was
  received," "the order was fulfilled," "shipment completed to the billing address." A
  keyword list is a guess at that set, and the model is not obliged to stay inside it.
- *Fabrication.* The failure that matters is an assertion with **no corresponding field at
  all**: a tracking number, a signature name, a delivery date, a customer service call that
  never happened. A deterministic checker validates the fields it enumerates. It cannot
  enumerate what the model invented, because the set of inventable facts is not finite and
  is not known in advance. Detecting "this sentence asserts something, and nothing in the
  record is about it" requires reading the sentence.

This is the only candidate where I could not write the function, having tried to specify it.

**Position relative to the policy boundary.** Furthest from it of anything proposed. It reads
our own model's output and our own record. It never reads the customer's narrative as a
signal, never sees `p_win`, and never produces anything `policy/` could consume — its output
is consumed only by `_file_if_needed`, after the policy decision is already made and
persisted. Constraint 1 is preserved by construction, not by discipline: there is no value on
this path that the policy engine has an input slot for.

It is also the only candidate that *reduces* what the model can do. Today a hallucinated
letter with `provenance=MODEL` is submittable. This gate can only ever move a letter from
submittable to withheld.

**What it fails to, and its provenance.** A new `LetterProvenance` member —
`FAILED_GROUNDING` — joining `FALLBACK` and `LOW_CONFIDENCE` as non-`MODEL`. That is the
whole integration on the failure side: `DraftedLetter.submittable` is already
`provenance is MODEL`, `require_submittable` already raises inside the client, and
`_file_if_needed` already routes a non-submittable letter to `_withhold_for_review` with the
provenance value in the reason string. A gate failure, a gate timeout, and gate output that
fails schema validation all land on the same non-`MODEL` value — the gate cannot fail open,
because the only way to stay submittable is for the gate to affirmatively pass.

**Measurement.** This is where it wins, because the ground truth is mechanical rather than
editorial.

Build a frozen corpus of letters with known defects:

- **(a) Contradiction, n≈40.** Draft a letter against a context, then flip one boolean field
  in the record. The letter now asserts something the record denies. Ground truth is the
  flip. Zero authorial judgment.
- **(b) Fabrication, n≈40.** Insert one sentence carrying a fact with no corresponding field.
  Ground truth is the insertion. I author these, so I set their difficulty — disclosed below.
- **(c) Clean controls, n≈40.** Unmodified letters. Ground truth: nothing to find.

Reported:

- **False-flag rate on (c)**, Wilson 95% interval. This is the headline, not the detection
  rate. A gate that flags everything detects everything and is worthless; its cost is the
  `withheld_for_review` queue, which is analyst time — the same cost `SPEC.md` §6 already
  prices for a false positive. At n=40 a rate near 0.05 gives roughly [0.01, 0.17], which is
  wide; n=120 controls tightens it to about [0.02, 0.11]. I would run 120 per cell.
- **Detection rate on (a) and (b) separately**, Wilson 95% each, never pooled — they are
  different tasks and pooling would let the easy one carry the hard one.
- **Paired against a deterministic keyword baseline** on the identical items, McNemar on
  per-item correctness. McNemar rather than the bootstrap-AUC machinery in
  `eval/extraction_comparison.py` because here each item has a binary correct/incorrect
  outcome and both arms make a hard decision — there is no threshold to invent, which is
  exactly the objection that module's own docstring records against McNemar in *its* setting.
  The lesson carried over is the pairing, not the estimator.

**Baseline honesty.** A deterministic baseline exists and I will implement it, which is the
opposite of the TF-IDF situation — there the baseline was a number with no code. The risk
here is the mirror image: I author corpus (b), so I could author it easy and win. Two
guards, both stated in the README rather than left implicit: the corpus is written and
committed **before** either arm is run, and its composition (how many fabrications are
near-paraphrases of real fields versus wholly invented entities) is published as a table so a
reader can see the difficulty I chose. The strongest claim this design can support is "on
this frozen corpus, with this composition, at this n" — never "the gate catches
hallucinations." That ceiling is a property of the design and cannot be raised by running it
harder.

**Token cost.** One extra call, on the contest path only. Prompt ≈ 600 tokens (letter capped
at `NETWORK_SUMMARY_MAX_CHARS`=1000 chars ≈ 280, seven context fields ≈ 60, instructions
≈ 250); output ≈ 350 (one entry per assertion: quoted span, supporting field or `null`,
verdict). **≈ 950 tokens per contested dispute.** Measurement run: ~120 drafting calls
(committed as a fixture, drafted once) + ~240 gate calls.

**New README sentence.** Today the README can say every completion is schema-validated. It
cannot say anything about whether the letter is *true*. It would gain:

> A drafted letter cannot be filed until every factual assertion in it has been traced back
> to a field in the dispute record. A letter that asserts what the record denies, or asserts
> what the record has no field for, is withheld for human review rather than submitted. On a
> frozen 360-item corpus this gate flagged X% of clean letters incorrectly (95% CI …) while
> catching Y% of injected contradictions and Z% of injected fabrications.

**Why the sharpening from "self-critique."** The brief's version scores the letter "against
network acceptance criteria." I could not find a defensible source for those criteria — the
repo has no network evidence matrix (§0.1), so the criteria would be mine, the scores would
be against my criteria, and the "rejection rate" would measure my rubric, not the letter. It
would also have no ground truth: nothing can say whether a rejected letter deserved
rejection. Grounding replaces an editorial rubric with a factual one, and buys mechanical
ground truth for two thirds of the corpus. Same plumbing, same cost, same fail-closed path —
a claim that can actually be defended.

---

### 2.2 Narrative/transaction consistency check

**The judgment.** Decide whether what the customer says in
`customer_communication_log` is compatible with what the transaction record shows.

**Why deterministic code cannot do it.** Mostly it cannot, and this is real: "I was travelling
in Europe when this happened" against `ip_geo_billing_distance_km=3.2` is a contradiction no
field comparison reaches, because the claim has to be extracted from prose before anything
can be compared to it. The current `NormalizedCommunicationLog` has a `mentions_travel`
boolean, but "mentions travel" and "asserts a location incompatible with the recorded IP
geolocation" are different facts, and only the first is a lookup.

**Position relative to the policy boundary — and why this is the dangerous one.** Closest of
anything proposed, in two ways that need separating.

The first is the one the brief anticipates, and it is handleable: the verdict must never
reach `policy/`. Enforceable the same way the existing boundary is enforced — `policy/engine.py`
imports nothing from `evidence/`, `decide()` takes two floats, and a test asserts the import
graph.

The second is not in the brief and is the reason this ranks below §2.1.
`disputedesk/evidence/schemas.py` currently says of the normalisation output:

> These describe what the customer said, for use in assembling the evidence packet — they are
> not, and must never be treated as, a fraud signal. `policy/` never sees this model.

A consistency verdict **is** a fraud signal. That sentence would have to be rewritten, and
rewriting a constraint to admit the thing you just built is exactly the move `CLAUDE.md`
warns against ("Do not 'improve' a metric by changing how it is computed" — the same
instinct, applied to an invariant). It can be done honestly, but it must be done as a dated
correction in `DECISIONS.md` that says the boundary moved and why, not as a quiet edit.

There is a third, subtler point that applies here and to §2.1 both, and I want it on the
record rather than discovered later. Routing to `withheld_for_review` on an LLM verdict lets
the model change what gets filed. The existing provenance gate has the same shape, but it
triggers on *the model's own failure* — a letter it could not produce validly. A consistency
gate triggers on *the model's judgment about the dispute*. That is a genuine escalation of
authority, even though it only ever moves toward human review. It stays inside constraint 1
only if it is strictly one-directional: it can withhold, it can never cause a contest, it can
never convert an ESCALATE into anything, and it can never reach the accept path (accepting is
irreversible — `_withhold_for_review`'s docstring already makes this argument for the
existing gates). §2.1 is one-directional over our own output. This one is one-directional
over a judgment about the customer, which is a larger thing to be one-directional about.

**What it fails to, and its provenance.** A new `_EvidenceOutcome` validation result —
`narrative_inconsistent` — alongside the existing `reason_code_unrecognised`, routed by the
same branch in `_file_if_needed`. Provenance: prompt version, model, the quoted span, and the
record field it conflicts with, on the decision row.

**Measurement.** Detection rate against `true_fraud`, the debug-only generator column.
Precedent for using it exists (`eval/oracle.py` uses debug `p`; `eval/tfidf_baseline.py` uses
`true_fraud` directly), and both label it eval-only, never a model or policy input.

The problem is what the resulting number means. `true_fraud` is my generator's latent, and
`comms.py` encodes it as a frequency tilt across near-synonymous phrasings — the README
already explains that this is why coarse yes/no extraction struggles with it. So a
consistency detector scored against `true_fraud` measures the model's ability to recover a
tilt I chose, in a corpus with no actual narrative/record contradictions in it, because
**the generator does not currently produce any**. `comms.py` draws phrases independently of
`ip_geo_billing_distance_km`, `delivery_confirmed`, and every other record field. To measure
this I would first have to inject contradictions into the generator — which is §0.2's problem
again, plus a modification to the component `SPEC.md` §5 says to read twice.

A cheaper, more honest variant: score it as a **paired arm against the existing n=60
recording**, on the same items, using the machinery in `eval/extraction_comparison.py`
unchanged. That answers "does a consistency-framed prompt beat the current field-extraction
prompt at recovering `true_fraud`" with a paired bootstrap CI, at a cost of ~60 live calls
and no generator change at all. It is a smaller claim than the brief wants, but it is
measurable this week and it reuses a module built for exactly this comparison. If the interval
excludes zero, that is a real finding about prompt framing. If it does not, that is the same
result the README already reports honestly once.

**Token cost.** Prompt ≈ 450 (comms log ≈ 120, record fields ≈ 80, instructions ≈ 250),
output ≈ 250. **≈ 700 tokens per dispute**, on every dispute rather than only contested ones —
which is the highest per-dispute-population cost of the three.

**New README sentence.**

> The system reads the customer's own account of the dispute against the transaction record
> and flags disputes where the two conflict. It cannot contest or accept on that basis — the
> policy engine never sees the verdict — but a conflicting narrative routes the dispute to a
> person instead of being filed unsupervised.

---

### 2.3 Evidence sufficiency assessment

**The judgment.** Given the evidence artifacts actually held for a dispute and the required
evidence types for its reason code, decide whether each requirement is genuinely satisfied by
what is on hand.

**Why deterministic code cannot do it — with §0.1 and §0.2 applied.** In the general case it
genuinely cannot: "does this delivery receipt actually establish delivery to the billing
address" is reading comprehension over an artifact, not a field check. That is a real
judgment and it is the best-motivated candidate in the brief.

In *this repo*, the general case is not what would get built. The required-evidence lookup
returns the same five types for all four supported codes. Three of those five
(`billing_proof`, `access_activity_log`, `proof_of_service`) map to boolean record fields —
`avs_match`, `cvv_match`, `device_fingerprint_known`, `delivery_confirmed` — where
sufficiency genuinely is a field check, and a function is the right answer. A fourth,
`explanation_letter`, is the thing being drafted. That leaves `customer_communication` and a
template-generated log, where "is this sufficient" collapses toward the `is_substantive`
boolean the schema already has.

So to make this candidate carry real judgment I would first build the artifact generator from
§0.2 — receipts, merchant notes, emails — and only then would there be anything to read. The
integration is downstream of a generator project, and the generator project is the highest-risk
component in the repo by `SPEC.md`'s own assessment.

**Position relative to the policy boundary.** Safe, comparable to §2.1: it reads artifacts
and a lookup-table requirement list, and produces a per-requirement verdict that only
`_file_if_needed` consumes. It must not be allowed to *change* the requirement list — that is
`SPEC.md` §2's disqualifier — only to judge satisfaction of a list the lookup table produced.
That distinction is enforceable with a test: the verdict schema keys are constrained to the
tuple `required_evidence_types()` returned, and any other key fails validation.

**What it fails to, and its provenance.** `_EvidenceOutcome.validation_result` =
`evidence_insufficient`, plus the per-requirement verdict list persisted on the decision row.
Same `_withhold_for_review` route.

**Measurement.** The brief argues that having no TF-IDF baseline is a feature, "because the
claim is not hostage to a margin that does not clear noise." I do not think that survives.
A claim with no baseline is not safer than one with a baseline — it is unfalsifiable, which
is a worse position to defend from, and it is the position the TF-IDF number was in before it
was implemented (§`DECISIONS.md` 2026-09-02: the baseline "had no implementation anywhere in
the repository," and that is precisely what let the error persist).

What *would* be measurable: hold-one-out. Generate artifact sets, remove one artifact, and
measure whether the assessor now reports the corresponding requirement unsatisfied. Ground
truth is the removal — mechanical, like §2.1's contradiction cell, and the same Wilson-interval
treatment applies. But the ceiling is circular in a way §2.1's is not: I generate the
artifacts, so I decide how legible "sufficient" is, and the assessor's score is a measurement
of my generator's clarity as much as the model's reading. §2.1's contradiction cell avoids
this because the flip is applied to a *record field*, which the generator did not author for
the purpose of being read.

**Token cost.** Artifacts at 2–4k characters plus the requirement list: prompt ≈ 1,500,
output ≈ 500. **≈ 2,000 tokens per contested dispute** — the most expensive of the three, and
that is before the generator work.

**New README sentence.**

> The system reads the unstructured evidence actually held for a dispute and judges, per
> requirement, whether it is satisfied — rather than assuming a filed artifact is a
> sufficient one. Requirements themselves remain a published lookup table; the model judges
> satisfaction only.

---

## 3. Candidates killed

### 3.1 Contest strategy selection — killed on measurability

The brief lists it; I am arguing against it, because it fails test 5 in a way no design fixes.

There is no ground truth for "was this the right argument." The only outcome label in the
system is `won_if_contested`, and per `GENERATOR.md` and `CLAUDE.md` invariant 1 it is
sampled from a Bernoulli over causal latents — authentication strength, relationship
genuineness, delivery provability. **Argument choice is not among them.** The label is
constructed to be independent of anything a strategy selector could do. So the selector's
output cannot be scored against the only label available, now or after any amount of extra
work, and adding argument quality to the generator's `p` would mean inventing the effect size
I then "measure."

It also fails test 1. All four supported codes share one evidence set (§0.1), so "which
argument" reduces to a framing choice over the same five evidence types.

And it fails test 6: the remaining way to score it is to ask a model whether the argument was
good.

Advisory output that nothing consumes and nothing can score is decoration with a token cost.
Cut.

### 3.2 Reason-code classification for unrecognised codes — killed on SPEC

Tempting: 71 published codes, only 4 supported, and `_UNRECOGNISED_REASON_CODE` currently
queues the rest for a person. A model could map an unrecognised code onto the nearest
supported condition.

This is `SPEC.md` §2's named disqualifier wearing a different hat, and
`canonical_reason_code`'s docstring already forbids it in one line: "this is a rename table,
not a normaliser, and it must never guess." The `VISA_83 → VISA_10_4` entry is not a guess —
it is documented supersession under Visa Claims Resolution. A model asked to do the same job
for `MC_4855` would be guessing, and it would be guessing its way into filing on a dispute
class the system has no evidence strategy for. Cut, and the existing queue-for-a-human
behaviour is the correct answer.

### 3.3 Prompt-injection screening of customer free text — killed on approach

The concern is real and is the one live security surface: `customer_communication_log` is
attacker-controlled in production and flows into two prompts. But an LLM classifier over it
is the weak answer to that concern. The strong answers are architectural and mostly already
present — the model's output is schema-constrained, it has no tool access, it cannot reach
`policy/`, and the letter it produces cannot be filed without passing `require_submittable`.
The remaining hardening is prompt-level delimiting and an explicit instruction-precedence
line in the prompt files, which is Phase 5 security-review work, not an AI surface. Cut as a
candidate; **flagged as a real finding for the security pass**, because a model gate here
would have obscured a structural fix.

### 3.4 Triage notes for the withheld-for-review queue — killed on measurability

Drafting "here is why this dispute is in your queue and what to check" is plausible and
cheap. There is no metric with an n. Cut.

### 3.5 Non-substantive message triage — killed on test 1

A well-written function does it (length, token count, stopword ratio), and
`NormalizedCommunicationLog.is_substantive` already covers it. Cut.

---

## 4. Ranking

Scored as **judgment depth × measurability-in-this-repo ÷ risk to existing invariants**, each
1–5, risk higher = worse. The arithmetic is an ordering device, not a decision procedure; the
reasoning above is what matters.

| # | Candidate | Depth | Measurability | Risk | Score | Live calls to measure |
|---|---|---:|---:|---:|---:|---:|
| 1 | **Letter grounding gate** (§2.1) | 3 | 5 | 1 | **15.0** | ~360 |
| 2 | Narrative/transaction consistency (§2.2) | 5 | 3 | 5 | 3.0 | ~60 (paired variant) |
| 3 | Evidence sufficiency (§2.3) | 5 | 2 | 4 | 2.5 | ~200 + generator work |
| — | Contest strategy selection (§3.1) | 4 | 0 | 3 | 0.0 | killed |

This ordering disagrees with the brief's, which leads with evidence sufficiency. The reason
is §0.1 and §0.2: evidence sufficiency scores 5 on depth *in the general case*, but in this
repo it is downstream of an artifact-generator project it does not yet have, and its
measurement grades a corpus I authored for the purpose of being read. It ranks third on
cost and circularity, not on merit.

The grounding gate wins on measurability and risk, not on depth — and I want to be plain that
its depth is a 3. It is the shallowest judgment of the three. What it buys is a claim that can
be defended: mechanical ground truth on two thirds of its corpus, a real deterministic
baseline it must beat on identical paired items, a fail-closed path that already exists and is
already tested, and zero movement of the policy boundary. Given that the last thing to go
wrong in this repo was a headline number whose baseline had no code, I would rather ship a
3-depth claim that holds than a 5-depth claim resting on a corpus I wrote to be read.

---

## 5. Recommendation

**Tier 1: the letter grounding gate (§2.1), alone.**

Note that this collapses the brief's Tier 1 and Tier 2 into one item rather than adding to
them — the brief's Tier 2 self-critique gate *is* my Tier 1, sharpened per §2.1's closing
note. That is a scope reduction.

I would not propose a second integration in the same pass. The tail from §0.3 (unfreezing
Phase 5, the security review pass over a new LLM output on a submission path, three cold
runs) is paid once for one integration and roughly twice for two, and the brief's own rule
applies: a half-finished second integration is worse than one finished one.

If you want a second, take **§2.2 in its paired variant only** — ~60 live calls, no generator
change, reusing `eval/extraction_comparison.py` unchanged. It answers a smaller question than
the full consistency check, and it is the one thing here that could be measured end to end
this week. It still requires the `schemas.py` fraud-signal sentence to be corrected as a
dated `DECISIONS.md` entry, not edited quietly.

**What Tier 1 delivers in this environment**, given §0.4: the gate built, the failing test
first, a `FAILED_GROUNDING` provenance member, schema-validated output through the existing
repair path, the deterministic keyword baseline implemented, the frozen corpus committed with
its composition table, a `DECISIONS.md` entry with a `Reproduce:` line, and a recorded-fixture
stub test. **The headline number is a manual run, recorded as an exact command** — the same
shape as the outstanding `--n-rows 1000` normalisation run. No interval ships from this
session, and the README says so rather than shipping a point estimate.

---

## 6. What I need from you

1. **Which candidate**, or which two. My recommendation is §2.1 alone.
2. **Whether reopening the Phase 5 freeze is acceptable** (§0.3), and whether the README's
   "complete and frozen" line becomes "frozen except this surface, re-frozen on <date>" or
   the freeze is withdrawn and re-taken.
3. **Whether a manual-command headline is acceptable** for this surface (§0.4), or whether
   an `LLM_API_KEY` will be available before this lands. This changes what the README can
   claim on the day it ships.

No code will be written until you answer.
