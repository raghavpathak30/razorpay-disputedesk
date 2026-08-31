# CLAUDE.md

Context document for this repository. Read this first, every session.

Two other documents are authoritative and you must read them before acting:

- `SPEC.md` — what this system is, its components, and where the LLM is and is not allowed.
- `PHASES.md` — the build order and the acceptance criteria for each phase.

If anything in this file conflicts with `SPEC.md`, `SPEC.md` wins.

---

## Project

Dispute Desk: an agent that triages fraud-reason-code chargebacks, decides whether
contesting is worth it, assembles the required evidence, files it through the
Razorpay Disputes API in test mode, and logs every decision for audit.

This is a submission to the Razorpay AI Buildathon, Track 02. It is judged on
measured results and honest metrics, not on feature count. A smaller system with
defensible numbers beats a larger system with impressive-looking ones.

---

## Invariants

These are not preferences. Breaking one invalidates the submission.

1. **The label is sampled, never computed.** In the synthetic generator,
   `won_if_contested` is drawn from a Bernoulli distribution whose parameter
   depends on causal factors. It is never a boolean expression over feature
   columns. If you find yourself writing `won = delivery_confirmed and ...`, stop.
2. **No metric is ever reported from the training split.** Temporal holdout only.
3. **No headline number comes from a single seed.** Report median and IQR across
   seeds. A single split is a claim, not a result.
4. **The LLM has no authority.** It drafts text and normalises unstructured input.
   It does not decide contest vs accept, does not map reason codes to evidence
   types, and does not do arithmetic on money.
5. **Defense only.** No fraud-pattern generator, no attack simulator, not even as
   a test harness. This is disqualifying for the track.
6. **Nothing unbuilt is described as built.** README, docstrings, and comments
   describe what exists today. If a component is a stub, it says so.

---

## Stack

Python 3.11. FastAPI for the webhook endpoint. SQLAlchemy over SQLite (so Postgres
is a connection-string change, not a rewrite). Pydantic v2 for every schema
boundary. LightGBM for the model. pytest. ruff for lint and format.

Do not introduce a new library without asking. Popular and boring over new and
clever.

---

## Module layout

```
disputedesk/
  generator/    synthetic dispute + order generation
  features/     feature builder (pure functions, no I/O)
  model/        train, predict, calibration
  policy/       the decision engine
  evidence/     reason-code map, LLM drafter, schema validation
  client/       Razorpay test-mode API client
  audit/        append-only decision log
  api/          FastAPI webhook
  cli/          demo entry points
eval/           harness, baselines, metric reporting
tests/
```

Rules:

- One responsibility per module. If a file exceeds 300 lines, split it.
- Functions stay under 50 lines. If one grows past that, it is doing two things.
- `features/` is pure. No database calls, no network, no randomness. It must be
  testable with a dict in and a dict out.
- Nothing outside `client/` talks to the network.
- Nothing outside `evidence/` calls the LLM.

---

## What NOT to do

Learned constraints. Do not do these even if they seem reasonable in the moment.

- Do not build a web UI or dashboard. A CLI and a demo script are the deliverable.
- Do not add authentication, user accounts, or multi-tenancy.
- Do not add RAG. There is no corpus. It will be marked down as forced.
- Do not add a second loss class.
- Do not hardcode API keys, tokens, or secrets. Environment variables only, loaded
  through one config module.
- Do not append to a growing file when a new module is the right answer.
- Do not silently change a number that is already recorded in a document. If a
  measurement changes, add a dated correction, keep the old value visible, and say
  what caused the change.
- Do not "improve" a metric by changing how it is computed. Change the model or
  accept the number.
- Do not write code for a phase that has not started. See `PHASES.md`.

---

## Testing

- Every feature ships with tests in the same session. Not later.
- When a bug is found: write a failing test that reproduces it, then fix it.
- `features/` and `policy/` get unit tests with hand-built cases, including
  near-miss confounders.
- The leakage guard in `tests/` is a first-class test, not a script. It asserts no
  feature column is derivable from the label. It runs in CI.
- The eval harness runs in CI on a fixed seed set so a metric regression is caught
  at commit time.

---

## Git

- Commit after every working unit of change, not at the end of a session.
- Commit messages say what changed and why, one line, no ceremony.
- Start each phase on a clean tree.
- Revert with git, never by asking for the previous version back.

---

## Session protocol

- One task per prompt. If asked for five things, do the first and stop.
- Think before writing code on anything non-trivial. Say what you plan to do,
  briefly, then do it.
- If the same problem survives three attempts, stop and say so rather than trying
  a fourth variation. The approach is probably wrong, not the implementation.
- When a session produces a durable lesson, a convention, or a constraint, append
  it to the "What NOT to do" section above before the session ends.
- At the end of a phase, run a refactor pass: remove dead code, split oversized
  files, delete anything written for a phase that was later cut.
- Explicitly run a security review pass before the final freeze: secrets,
  injection surfaces on the webhook endpoint, and unvalidated LLM output reaching
  the API client.
