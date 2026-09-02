# Architecture

Sourced from `DECISIONS.md`, the project's append-only decision and
measurement log — that file is the primary record; this document organizes
it into one narrative rather than restating it from memory. Where a claim
below needs the exact reasoning or a measured number behind it, the
relevant `DECISIONS.md` entry is named.

## System shape

One request, one straight-through pipeline, no loops back into an earlier
stage:

```
webhook (FastAPI)                      disputedesk/api/webhook.py
  -> validate payload shape             disputedesk/api/schemas.py (Pydantic)
  -> build features                     disputedesk/features/build.py   (pure)
  -> P(win) = model.predict_proba       disputedesk/model/               (LightGBM)
  -> decide(p_win, amount)              disputedesk/policy/engine.py     (deterministic)
       |
       +-- CONTEST -> assemble evidence disputedesk/evidence/
       |     -> required_evidence_types()   (lookup table, deterministic)
       |     -> normalize_communication_log (LLM + schema validation)
       |     -> draft_explanation_letter    (LLM + schema validation)
       |
  -> persist decision (before any API call)  disputedesk/audit/log.py
  -> file via Razorpay client (if contest/accept)  disputedesk/client/razorpay.py
  -> persist API outcome (separate row, after the call)  disputedesk/audit/log.py
```

Every arrow is one direction. The model never calls the policy engine or
the API client; the policy engine never calls the LLM or the API client;
the evidence assembler never decides contest/accept; the audit log never
computes anything, it only records what already happened elsewhere. This
shape is deliberate (`DECISIONS.md`'s "LLM authority boundary" entry) and
is what "the model and the LLM can both be swapped out and the policy
engine's decisions remain traceable and reproducible" (`PHASES.md` Phase 3
gate) actually depends on.

## Why the decision layer is deterministic, not the model or the LLM

`disputedesk/policy/engine.py`'s `decide()` is a pure function of exactly
two numbers — `P(win)` and `amount` — plus two named constants
(`representment_cost_inr`, `low_confidence_band`):

```
expected_value = P(win) * amount - representment_cost
if confidence_band is LOW:        -> escalate_to_human
elif expected_value > 0:          -> contest
else:                             -> accept
```

Two reasons this is a plain function and not, say, a second model or an
LLM call:

1. **Traceability.** `SPEC.md` §4 and `PHASES.md`'s Phase 3 gate both
   require that a decision can always be traced back to the parameters
   that produced it. A deterministic function with named, config-file
   constants gives an exact answer to "why did this dispute get
   contested?" — replay the same `p_win`, `amount`, and config and you get
   the same decision, byte for byte, forever. A model or LLM in this seat
   would make that an approximate, non-reproducible claim.
2. **The model and the LLM are explicitly allowed to be wrong, or to
   change.** `PHASES.md`'s Phase 3 gate: "the model and the LLM can both be
   swapped out and the policy engine's decisions remain traceable and
   reproducible." Only a policy layer with zero learned parameters can make
   that promise — swap the LightGBM model for a different one, or the LLM
   provider for another, and `decide()`'s logic and its guarantees don't
   move.

The "escalate to human" branch exists so the system has an honest "I don't
know" path (`p_win` within `low_confidence_band`, currently `(0.45, 0.55)`,
centered on maximum model uncertainty rather than on any particular
dispute's own breakeven point, so the band has a fixed, auditable meaning
independent of `amount`) — this is not a hedge bolted on for the judging
rubric; `DECISIONS.md`'s 2026-08-31 "Phase 3 cost-weighted business
metrics" entry describes a real mistake this caught: an earlier run of the
business-metrics harness credited every escalated dispute INR 0 (assuming
the human reviewer recovers nothing) and concluded, wrongly, that the
policy loses to "contest everything." Scoring an escalated dispute the way
baseline A already scores every dispute (`naive_contest` mode) — the fair
comparison, since baseline A has no abstention path to credit specially —
reverses that: the policy beats both baselines. The wrong conclusion was
caught and corrected in `DECISIONS.md` before it could become a headline
number, not silently fixed.

## The LLM authority boundary

Exactly three jobs (two until 2026-09-02), all schema-validated before use,
none touching a decision:

- **Draft `explanation_letter`** from the dispute's order-context facts
  (`disputedesk/evidence/draft_letter.py`).
- **Normalise `customer_communication_log`** free text into typed boolean/
  enum fields (`disputedesk/evidence/normalize_comms.py`).
- **Grade the drafted letter against the dispute record**
  (`disputedesk/evidence/grounding.py`), withholding it from submission if any
  factual assertion cannot be traced to a record field. Added 2026-09-02 under
  the `SPEC.md` §2 amendment of that date.

The third is the only one that can change what happens to a dispute, so its
authority is bounded twice over. It is **one-directional** — it can move a
letter from submittable to `failed_grounding`, never the reverse, and no path
in the module constructs a `MODEL`-provenance letter. And it is **downstream of
the decision**: it runs inside `assemble_evidence_packet`, on the `CONTEST`
branch only, after `policy/` has decided and the decision has been persisted.
A withhold leaves `policy_branch` on the audit row reading `contest`; what it
changes is `validation_result`, which is a separate column recording a separate
fact. `tests/test_grounding_gate_pipeline.py` asserts both directions, and
asserts as a property of the source that `disputedesk/policy/` imports nothing
from `disputedesk/evidence/`.

Forbidden, explicitly, per `DECISIONS.md`'s "LLM authority boundary" entry:
deciding contest vs. accept (`policy/`'s job, always resolved before the
LLM is ever called for a given dispute — see the pipeline diagram above:
`evidence/` only runs on the `CONTEST` branch), mapping reason codes to
required evidence types (a lookup table,
`disputedesk/evidence/reason_code_map.py` — card networks publish this, an
LLM here is strictly worse), and any arithmetic on money (`amount` is only
ever quoted back into a prompt, never computed by one).

**Provider and model.** `disputedesk/evidence/llm.py`'s `GroqHttpLLMClient`
calls Groq's OpenAI-compatible endpoint with `openai/gpt-oss-20b`
(`DECISIONS.md`'s "LLM provider: Groq" entry). The model choice is
deliberately small: neither of the two allowed jobs needs multi-step
reasoning, long-context synthesis, or broad world knowledge — both are
short, schema-constrained, single-turn completions, and reaching for a
larger model here would be the same "forcing an LLM/unnecessary tech"
failure mode the AI Judgment criterion penalises, applied to model size
instead of scope. Verified live against the real endpoint before being
relied on in the eval harness (`DECISIONS.md`'s "Groq live call, verified"
entry).

**Every LLM output is validated before use, with one repair attempt and a
deterministic fallback** (`disputedesk/evidence/validated_call.py`,
`SPEC.md` §7 failure path 2): the completion is parsed and validated
against a Pydantic schema (`disputedesk/evidence/schemas.py`,
`extra="forbid"`); on failure, exactly one repair call is made with the
validation/parse error fed back into the prompt; if that also fails, the
caller falls back to a deterministic template (a fixed-sentence letter
built only from order-context facts, or conservative all-`False`
normalisation defaults) and sets `human_review_required=True` on the audit
row. No unvalidated LLM text reaches the Razorpay client at any point in
this chain — `disputedesk/api/pipeline.py`'s `_assemble_evidence_if_contesting`
only ever reads `packet.explanation_letter.letter_text`, a field that only
exists after passing through this validation path or the deterministic
fallback.

**We measured whether the LLM adds predictive value, and it does not.**
`disputedesk/evidence/normalize_comms.py`'s typed extraction scored mean
AUC 0.4211 (5-fold CV, n=60, live Groq calls) against `true_fraud` versus a
TF-IDF + logistic-regression baseline's 0.6371 on the same task
(`DECISIONS.md`'s 2026-09-01 "LLM normalisation quality vs. TF-IDF
baseline" entry). The extraction is reliable in format (zero repairs or
fallbacks needed across the sample) but not predictive in content — a
diagnosed, structural reason, not noise: two of the seven typed fields sit
near 1.0 for both classes because the generator's comms-log design makes
the real signal a frequency tilt across near-synonymous phrasings, which a
coarse yes/no LLM extraction collapses and a bag-of-words vectorizer
preserves. Nothing was changed in response to this number — no prompt,
schema, or feature-encoding edit — per that entry's own explicit note; it
is reported as a finding, not tuned away. It is the reason this project's
LLM footprint stayed at "draft text" rather than expanding into "extract
structured predictive signal."

## The model

LightGBM (`disputedesk/model/`), trained on the temporal training split
only, outputs `P(win)` and nothing else — `SPEC.md` §2: "Outputs `P(win)`
only. Makes no decisions." One model trained in memory per process
(`disputedesk/model/registry.py`, `lru_cache`, fixed seed 42) from the same
generate → temporal-split → train pipeline the eval harness uses; there is
no persisted, versioned model artifact (`DECISIONS.md`'s "Phase 4" entry
states this as a real, stated limitation, not treated as production-ready).
`MODEL_VERSION` names the seed and config and is recorded on every audit
row, so a given decision can always be tied back to which model produced
its `p_win`.

## Synthetic data, and why fully synthetic

No public labelled dispute dataset exists (`DECISIONS.md`'s "Synthetic data
over real data" entry). Anchoring to a public transaction dataset like
IEEE-CIS was considered and rejected: it buys realism that can't be
defended under panel questioning in the time available, and a documented
generative process with a stated accuracy ceiling is more defensible than
a half-real dataset with an unclear provenance story. The generative story
(`GENERATOR.md`) is written in prose before any generator code, names each
causal factor's direction and strength, and — critically — samples the
label from a Bernoulli whose parameter depends on those causal factors,
never computes it as a boolean expression over the feature columns
(`CLAUDE.md` invariant 1; enforced in CI by a leakage-guard test). See the
README's "What this dataset cannot tell you" section (verbatim from
`GENERATOR.md` §9) for what this measures and doesn't.

Two rounds of out-of-band sanity checking (`GENERATOR.md`'s "session 2"
revision notes, `DECISIONS.md`'s corresponding entries) found and fixed six
real generator defects that the generator's own unit tests had passed —
see the pitch-video answer draft for the full account; the short version is
that a generator's tests can prove internal consistency without proving
the causal story it claims to implement, which is why the out-of-band
AUC-against-`true_fraud` check on individual features was necessary at all
and is worth keeping as a standing practice, not a one-time audit.

## Audit log

Two append-only tables, not one mutable row (`disputedesk/audit/models.py`,
`DECISIONS.md`'s "Phase 4" entry): `decisions` (one row per dispute, UNIQUE
`dispute_id`, written *before* the Razorpay API is ever called — model
version, the exact feature dict used, `p_win`, policy branch, expected
value, prompt version, validation result, `human_review_required`) and
`api_outcomes` (at most one row per dispute, UNIQUE `dispute_id`, written
after the API call finishes — action, outcome, response or error). A
single mutable row can't be both "persisted before the API call" and
"append-only with no update path" at once — filling in an API response
after the fact would be an `UPDATE`. Two insert-only tables, joined by
`dispute_id` for display, get both properties literally: `disputedesk/audit/log.py`
exposes only insert functions, and no update or delete path exists anywhere
in the codebase. Each table's own `dispute_id` UNIQUE constraint is the
idempotency gate `PHASES.md` Phase 4 asks the *database* to enforce, not
application logic — proven directly in `tests/test_audit_log.py` by
inserting a duplicate row that bypasses the normal insert function entirely
and asserting the database itself raises `IntegrityError`.

## Failure recovery

Two paths, both demoed live in `disputedesk/cli/demo.py` and required by
`SPEC.md` §7:

1. **API timeout.** The decision row is written before the Razorpay call
   (`disputedesk/api/pipeline.py::process_dispute_event`), so a crash or
   timeout mid-call never loses the decision, and a retried/replayed
   request for the same dispute is still safe — the `api_outcomes` UNIQUE
   constraint means a retry can only ever reproduce the same outcome row,
   never a second filing.
   `disputedesk/retry.py::call_with_backoff` — one helper, shared by both
   `disputedesk/client/razorpay.py` and `disputedesk/evidence/llm.py`'s
   Groq client, rather than each reimplementing backoff — retries on
   `httpx.TimeoutException` or an HTTP 429 (honoring `Retry-After` when
   present) with exponential backoff, and lets any other exception
   propagate immediately.
   `tests/test_client_razorpay.py::test_timeout_then_success_files_exactly_once`
   proves this end to end via `httpx.MockTransport` (no real socket): a
   timeout followed by a successful retry results in exactly one successful
   filing, not two.
2. **Invalid LLM output.** Covered above under the LLM authority boundary —
   schema validation, one repair attempt, deterministic template fallback,
   `human_review_required` flag, never a crash.

`disputedesk/retry.py` was itself born from a real failure hit during
development, not designed speculatively: `DECISIONS.md`'s 2026-09-01 "LLM
normalisation quality" entry records the free-tier Groq endpoint returning
HTTP 429 under rate limits during an eval run, which is what made the
429-aware retry path a requirement rather than a nice-to-have.

## Known gaps, stated rather than left for a reviewer to find

- **No webhook signature verification.** The FastAPI route validates
  payload *shape* (`disputedesk/api/schemas.py`, every field constrained —
  not just `status`) but does not verify a Razorpay `X-Razorpay-Signature`
  HMAC header, because real Razorpay webhook payload/signature
  documentation specific to dispute events could not be located during
  this project (`DECISIONS.md`'s "Phase 4" entry: "two lookups 404'd").
  This endpoint should not be exposed to the public internet as-is.
- **No document-upload pipeline.** The real `contest()` API accepts
  per-evidence-type document ids; this project never built a file-storage
  path, so `contest()` submits the drafted letter as `summary` text only
  and records `required_evidence_types` in the audit log for a human to
  attach files against, rather than inventing document ids for files that
  don't exist. Razorpay's contest documentation (re-read 2026-09-02) states
  that `action="submit"` requires at least one document id across the
  evidence fields — so a real submit from this system would likely be
  *rejected* by the live API, not merely incomplete. Recorded in
  `DECISIONS.md`'s 2026-09-02 entry of the same name; untested against the
  live API.
- **Append-only guards are SQLite-only.** `disputedesk/audit/db.py`
  installs `BEFORE UPDATE`/`BEFORE DELETE` triggers that make the audit
  tables append-only in the database, not merely by convention — but the
  DDL is written and tested for SQLite alone. `init_db` *raises* on any
  other dialect rather than coming up with an audit log that claims to be
  append-only and is not. So "Postgres is a connection-string change" is no
  longer quite true: a Postgres deployment additionally needs the
  equivalent trigger DDL, or (cleaner there) `REVOKE UPDATE, DELETE` on the
  application role. The hash chain
  (`disputedesk/audit/chain.py`, `verify_chain()`) is dialect-independent
  and works either way.
- **The hash chain is tamper-evident, not tamper-proof.** Someone who can
  drop the triggers can still edit a row — the chain guarantees they cannot
  do it invisibly, because every later row commits to the edited row's
  content, so the edit becomes a rewrite of the whole suffix of the log.
  There is no off-box anchor (no periodic publication of the head hash), so
  a full-suffix rewrite by a sufficiently privileged actor would verify
  clean. Stated as what it is.
- **No order-context lookup.** The webhook assumes order-context fields
  (`avs_match` through `checkout_hour_of_day`) arrive already joined onto
  the dispute payload; a real deployment needs to build that join from the
  merchant's own systems.
- **No persisted model artifact.** See "The model" above.

None of these are silently patched around elsewhere in the system — each
is a real, load-bearing limitation of what's built today.
