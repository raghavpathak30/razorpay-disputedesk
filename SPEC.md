# Dispute Desk — build spec

Razorpay AI Buildathon, Track 02 (AI Risk Manager).
Loss class: **fraud-reason-code chargebacks**. Exactly one. Nothing else.

---

## 1. What it does

An incoming dispute arrives. The system:

1. Pulls the linked order, payment, and customer history.
2. Scores the probability the dispute is winnable if contested.
3. Applies a deterministic policy: contest, accept, or escalate to a human.
4. If contesting: assembles the typed evidence objects that reason code requires, drafts the explanation letter, validates the packet against a schema.
5. Files it via the Razorpay Disputes API in test mode, idempotently.
6. Writes an audit row that explains the decision end to end.

Merchants currently either contest everything (wasting analyst time and accruing
excessive-representment exposure) or accept everything (leaving recoverable money
on the table). Both are the baselines this is measured against.

---

## 2. Components

| # | Component | Deterministic or LLM | Notes |
|---|---|---|---|
| 1 | Synthetic dispute generator | Deterministic | Highest-risk component. See §5. |
| 2 | Feature builder | Deterministic | Pure function, no I/O. |
| 3 | Win-probability model | LightGBM | Outputs `P(win)` only. Makes no decisions. |
| 4 | Policy engine | Deterministic | Owns every contest/accept/escalate decision. |
| 5 | Evidence assembler | Hybrid | Lookup table picks *which* evidence. LLM drafts *content*. |
| 6 | Disputes API client | Deterministic | Test mode. Idempotency key = dispute id. |
| 7 | Audit log | Deterministic | Append-only. One row per decision. |
| 8 | Eval harness | Deterministic | Runs in CI. See §6. |

### Where the LLM is allowed

Only two places:

- Drafting the `explanation_letter` from unstructured customer-communication logs.
- Normalising messy free-text order context into typed evidence fields.

**Amended 2026-09-02 — a third place, and the test it had to pass.** The two
above stand unchanged. Added:

- Grading a drafted `explanation_letter` against the dispute record, and
  withholding it from submission if any assertion in it cannot be traced to a
  record field (`disputedesk/evidence/grounding.py`).

The original two-place limit was written to keep the LLM away from the
*decision*. This addition does not approach it: the gate reads this system's
own model output against this system's own record, after the policy engine has
already decided and its decision has been persisted. It is one-directional — it
can withhold a letter, never cause one to be filed — and it produces no value
`policy/` has an input slot for. It earned the exception by being the one
candidate where a deterministic function provably cannot do the job: it can
check the fields it enumerates, but it cannot enumerate what a model invented.

Four other candidates were considered against this same bar and rejected; see
`docs/AI-SURFACE.md` and the README's "What we did not build" section. The
forbidden list below is unchanged and still binding.

### Where the LLM is forbidden

- Reason code to required-evidence mapping. That is a lookup table. Card networks
  publish it. An LLM here is strictly worse and will be marked down.
- The contest/accept decision. That is §4.
- Any arithmetic on money.

Write this boundary into the README. It is the single strongest thing you can say
on the AI Judgment criterion, and most submissions will get it backwards.

---

## 3. Data model

Match Razorpay's real schema so the test-mode API accepts your objects unchanged.

**Dispute:** `id`, `payment_id`, `amount`, `currency`, `reason_code`, `phase`
(`fraud` | `retrieval` | `chargeback` | `pre_arbitration` | `arbitration`),
`status` (`open` | `under_review` | `won` | `lost` | `closed`), `respond_by`.

**Evidence object types:** `shipping_proof`, `billing_proof`, `cancellation_proof`,
`customer_communication`, `proof_of_service`, `explanation_letter`,
`refund_confirmation`, `access_activity_log`, `refund_cancellation_policy`,
`terms_and_conditions`, `others`.

**Order context (your own):** AVS match, CVV match, device fingerprint,
delivery-confirmation flag, prior order count for this customer, prior dispute
count, IP-geo vs billing-address distance, days between purchase and dispute,
customer communication log (free text).

**Ground truth label:** `won_if_contested` — boolean. Generated per §5.

---

## 4. Policy engine

```
expected_value = P(win) * amount - representment_cost
if confidence_band is LOW:        -> escalate_to_human
elif expected_value > 0:          -> contest
else:                             -> accept
```

`representment_cost` is a named constant with a comment explaining where the
number came from. The escalate band exists so the system has an honest "I don't
know" path — panels ask about this.

The model never calls the API. The policy engine does.

---

## 5. Synthetic data — read this twice

This is where the project dies if it dies.

You are generating both the features and the label. If the label is a
deterministic function of the features, the model scores 0.99 PR-AUC and the
number means nothing. You have already been burned by exactly this shape of
mistake.

Rules:

- Write the generative story first, in prose, before any code. "A dispute is
  winnable when delivery was confirmed AND the customer had prior successful
  orders AND the dispute was filed more than N days after delivery" — that kind
  of statement.
- The label must be sampled, not computed. `won_if_contested ~ Bernoulli(p)`
  where `p` depends on the causal factors. Never `won = (delivery_confirmed and
  ...)`.
- Include irreducible error. Some disputes are unwinnable for reasons not in the
  feature set. Target a ceiling well below perfect and say what it is.
- Include confounders and near-misses: legitimate customers who look fraudulent,
  and vice versa.
- Split temporally, not randomly. Train on earlier disputes, test on later.
- Write a leakage guard as a standalone testable function that asserts no
  feature column is derivable from the label. Run it in CI.
- Document the generator's parameters in the README. A panel will ask how you
  know the numbers mean anything, and the honest answer is "here is exactly how
  the data was made, and here is the ceiling that implies."

Generate 10k–20k disputes. Do not anchor to IEEE-CIS. It buys realism you cannot
defend in the time available.

---

## 6. Evaluation

Reported on the temporal holdout only. Never on training data.

**Model metrics**
- Precision and recall of the contest decision against `won_if_contested`
- PR-AUC
- Calibration: is `P(win) = 0.7` right about 70% of the time?

**Business metrics**
- Rupees recovered per 1,000 disputes
- Against baseline A: contest everything
- Against baseline B: accept everything
- If you do not beat both, say so. That is still a result.

**False-positive cost (checklist item 6)**
- FP = contesting a dispute you should have accepted. Cost = representment fee +
  analyst time + excessive-representment exposure.
- FN = accepting a winnable dispute. Cost = the full dispute amount.
- Both stated in rupees, both traced to a named source or a stated assumption.

Report a distribution, not one number. Multiple seeds, median and IQR. A single
split is a claim; a distribution is a result.

---

## 7. Failure recovery (checklist item 12)

Two paths, both demoed on video:

1. **API timeout.** Decision is persisted before the API call. Retry with
   exponential backoff, keyed on dispute id, so a retry cannot double-file.
   Kill the network mid-demo and show it recover.
2. **LLM returns invalid output.** Schema validation fails. One repair attempt
   with the validation error fed back. If it fails again, fall back to a
   deterministic template letter and flag the record for human review. The
   system degrades, it does not crash.

---

## 8. Explicitly out of scope

Say no to all of these:

- A dashboard or web UI. A CLI and a demo script are enough.
- Authentication, multi-tenancy, user accounts.
- A second loss class.
- Real payment integration. Test mode only.
- Any offense-capable code. No fraud-pattern generator, no attack simulator, not
  even as a test harness. Track 02 disqualifies it.
- RAG. There is no corpus here. Adding one to look impressive fails AI Judgment.

---

## 9. Submission checklist

Nothing ships until all twelve are ticked.

**From Razorpay's official track text and submission steps:**

1. [ ] Exactly one class of loss
2. [ ] A working detector, verifier, or auto-responder
3. [ ] Measured precision
4. [ ] Measured recall
5. [ ] On a held-out test set
6. [ ] False-positive cost stated explicitly, in rupees
7. [ ] Defense-only — nothing offense-capable anywhere in the repo
8. [ ] Public GitHub repo
9. [ ] Five-minute pitch video
10. [ ] Architecture documented

**Reported by aggregators, not confirmed on razorpay.com — treat as likely:**

11. [ ] Full audit trail
12. [ ] One runtime failure identified and handled gracefully
