# Q&A prep

Not for the judges — for whoever is standing in front of them. The six
questions most likely to sink this in the room, each with the honest answer
and the number that backs it, written as they would actually be said out
loud. Built 2026-09-03 (Phase 3), after the key run. Every number here has a
`Reproduce:` command in `NUMBERS.md` — check it before quoting this document
from memory in six months.

---

## 1. "So does this actually file the dispute, or not?"

**Say:** "It assembles, validates, and grounds the evidence packet, and it
calls Razorpay's contest endpoint — but that call would almost certainly be
rejected. The contest endpoint requires at least one document id when you
submit, and we never built a file-upload path, so we send zero. The `accept`
path is real end to end — that endpoint just needs a dispute id and it goes
through in test mode. The honest way to describe this system is: it decides,
it drafts, it validates, it grounds, and it stops one step short of a working
filer for the contest side specifically."

**Number:** zero document ids sent, always. `disputedesk/client/razorpay.py`'s
`contest()` — read the code, there's no upload call anywhere in this
repository.

**Don't say:** "it files disputes with Razorpay." That was the README's own
first paragraph until this phase, and it's the fastest way to lose the room
if a judge who knows the API asks a follow-up you can't back up.

---

## 2. "Why should I trust a 0.66% advantage over just contesting everything?"

**Say:** "You shouldn't trust it as a settled result — you should trust that
we measured it honestly and told you where it breaks. At the configured
cost the paired advantage is +11,210 rupees per 1,000 disputes, 95%
confidence interval +8,508 to +13,633, positive on 19 of 20 seeds — that's
real, not noise. But there's no measurable advantage at all below ₹200 per
representment, and the sweep doesn't charge for human review time. Solve for
the review cost that cancels the advantage and you get about ₹200 per
dispute — against ₹150 of analyst time we already budget for filing alone,
before a review is added on top. Charge that back and the policy trails
baseline A by about 11,265 rupees per 1,000, not leads it."

**Number:** +11,210/1,000 (95% CI +8,508 to +13,633); break-even review cost
≈₹200; overstatement 200.5% at the configured point.

**If pushed further:** "The thinness of the margin is why we built the cost
sweep in the first place — it's not a single number, it's a curve, and we
show you the whole thing, including the part where it doesn't hold."

---

## 3. "What does your model actually learn?"

**Say:** "Mostly one feature. We ran a single-feature ablation after
noticing `ip_geo_billing_distance_km` — the distance between the IP
geolocation and the billing address — reaches 70% of the theoretical best
possible discrimination on its own. Retrained the model on that feature
alone, same seeds, same evaluation: it captures 65.8% of the full twelve-
feature model's entire advantage over contest-everything. The top three
features get you to 68.8%. The other nine, combined, add nine points. And
above about ₹2,000 in representment cost, the restricted models actually
*beat* the full model slightly — we don't have an explanation for that, and
we're not offering one we haven't checked."

**Number:** 65.8% (top-1), 68.8% (top-3), 101–102% at high cost
(unexplained).

**Don't say:** "the model learned a rich twelve-feature pattern." It didn't,
by our own measurement, and this table exists specifically so we don't get
shown that in the room instead of saying it ourselves.

---

## 4. "Isn't this synthetic data just telling you what you already built into it?"

**Say:** "Yes, largely, and we say so in the README under 'What this
dataset cannot tell you' — that's copied verbatim from `GENERATOR.md`, which
was written before any generator code existed. The causal story is authored,
not measured; no real dispute-outcome data went into it. What we can defend
is that the *label* is sampled, not computed — `won_if_contested` is a
Bernoulli draw over causal latents, never a boolean expression over the
feature columns — and we built a leakage guard specifically to catch the
alternative. It's not a claim that this model would perform this well on
real Razorpay disputes. It's a claim that, given the causal story we
documented, the model recovers it about as well as the label's own
irreducible noise allows — 0.3522 PR-AUC against a 0.4572 theoretical
ceiling."

**Number:** oracle ceiling 0.4572; model 0.3522; prevalence baseline 0.2377.

---

## 5. "The LLM doesn't seem to be doing much. Why is it here at all?"

**Say:** "By design, not by accident. It drafts the letter text and
normalises free-text into typed fields — nothing that decides money moves,
nothing that maps reason codes to evidence, nothing that touches the
contest/accept decision. We measured whether its typed extraction adds
predictive value over a plain TF-IDF baseline and, honestly, at n=60 we
can't show it does — the paired confidence interval spans zero. We added
one more job this week: a grounding gate that grades the model's own
drafted letter against the dispute record and withholds it from filing if it
asserts something the record can't back up. That's the one place a model
does something a deterministic function genuinely can't — enumerate what the
model invented — and it's one-directional: it can only withhold, never
submit."

**If asked "so what's the gate's own false-flag rate":** "Unmeasured, and I'll
say exactly why rather than dodge it. We attempted the measurement — drafted
letters against the real model, twice — and both runs were blocked by the
account's daily token budget, not a missing key. The arithmetic's in
`DECISIONS.md`: a proper n=250 measurement needs 750 to 1,250 calls, which is
two to four days of this account's daily allowance, not one sitting. What
does stand, and doesn't depend on that number at all, is the economics: at
the ₹150 analyst-time figure we already budget per contested dispute, the
gate's false-flag rate has to stay under 2.3% or it cancels the whole
measured advantage on its own. That's the number we'd be checking the gate
against, once we have it."

**Number:** 2.3% false-flag budget (measured, key-independent); gate's own
rate: not yet measured, 750–1,250 calls needed, ~2.25–3.75 days of daily
budget at this account's current usage.

**If asked "why not let it do more":** "We looked at four other places — a
strategy selector for how to argue the contest, a reason-code guesser for
unsupported codes, a narrative-consistency checker against the customer's
own story, an evidence-sufficiency judge. We killed all four and wrote down
why in `docs/AI-SURFACE.md`: the strategy selector has no ground truth to be
scored against, the reason-code guesser is the exact thing SPEC.md's 'never
guess' rule forbids, the consistency checker would have meant rewriting our
own invariant that customer communication is never a fraud signal, and the
sufficiency judge needs an artifact generator we didn't build. Where we
didn't put the model is as much of an answer as where we did."

---

## 6. "What happens if the LLM output is garbage, or the API is down?"

**Say:** "Two separate failure paths, both demoed live in the demo script,
not just asserted. If the LLM returns invalid JSON, there's one repair
attempt with the error fed back to the model; if that also fails, it falls
back to a deterministic template letter — and that template letter is
never filed. It gets a `provenance` tag that only allows submission for the
model's own validated output; a fallback letter is withheld and queued for
a person, even though the old code used to file it. We found and fixed that
ourselves during a verification pass — it's in `DECISIONS.md`, dated, with
the wrong behaviour named. If the Razorpay API times out, the decision is
already persisted to the database before the API call happens, so a retry
after a crash can't double-file — there's a UNIQUE constraint on dispute id
backing that up, not just application logic."

**Number:** provenance gate pinned by 11 tests
(`tests/test_evidence_letter_provenance.py`); idempotency enforced by a
database UNIQUE constraint, not a check in code.

---

## Housekeeping, if asked

- **"Is the audit log really append-only?"** — Yes, by database trigger, not
  convention; and no, not tamper-*proof* — a privileged actor rewriting the
  whole suffix of the log would verify clean, because there's no off-box
  anchor. Say both halves.
- **"Why SQLite and not Postgres?"** — SQLite for the demo; the append-only
  triggers are SQLite-specific and `init_db` raises rather than silently
  running unprotected on another dialect. Named as a gap, not hidden.
- **"Did you find your own mistakes, or did someone else?"** — Both are on
  the record. `DECISIONS.md` has multiple dated self-corrections, including
  one correcting an earlier correction (the 0.4335 root-cause diagnosis was
  wrong the first time too, and that's written down as well).
- **"Why does the README say the grounding gate is unmeasured — didn't you
  have a key?"** — We did, and we tried. Two live runs, both blocked by the
  account's daily token budget, not a rate limit or a missing credential.
  The first crash also surfaced a real bug — the drafting script held every
  row in memory and lost 167 successful, paid-for calls when it crashed —
  which we fixed the same session, checkpointed, and tested against a stub
  client before touching the real API again. Publishing an unmeasured
  capability behind a made-up number was the mistake that started this whole
  audit; we weren't going to make it twice in the same repo.
