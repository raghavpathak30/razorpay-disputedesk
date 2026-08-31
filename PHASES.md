# PHASES.md

Build order. Each phase has acceptance criteria. **Do not begin a phase until every
criterion of the previous phase is ticked.** Not "mostly done". Ticked.

The reason this file exists: the last competition was lost by building past the
brief. The twelve submission requirements in `SPEC.md` §9 are the brief. Every
phase below exists to tick some of them, and nothing is built that ticks none.

---

## Phase 0 — Skeleton

Half a day. Get the shape right before any logic exists.

- [ ] Repo initialised, `.gitignore`, `pyproject.toml`, ruff configured
- [ ] Module directories from `CLAUDE.md` created, each with an empty `__init__.py`
- [ ] One config module reading every secret from environment variables
- [ ] `.env.example` committed; real `.env` gitignored
- [ ] pytest runs and passes on an empty suite
- [ ] CI workflow runs lint and tests on push

**Gate:** `pytest` and `ruff check` both pass on a clean clone.

---

## Phase 1 — Synthetic generator

The highest-risk phase. If this is wrong, every number downstream is meaningless
and nothing later can rescue it.

- [ ] The generative story written in prose in `GENERATOR.md` **before** any
      generator code exists
- [ ] Causal factors named, with the direction and rough strength of each effect
- [ ] `won_if_contested` sampled from a Bernoulli whose parameter depends on those
      factors — never computed as a boolean expression
- [ ] Irreducible error present and its implied ceiling stated as a number in
      `GENERATOR.md`
- [ ] Confounders present: legitimate customers that look fraudulent, and fraudulent
      ones that look legitimate
- [ ] Records match Razorpay's `dispute` and `evidence` schema field-for-field
- [ ] Temporal ordering: every record carries a timestamp, and the split is by time
- [ ] Leakage guard written as a test, asserting no feature column is derivable
      from the label; it passes, and a deliberately leaky control case fails it
- [ ] 10k–20k disputes generated reproducibly from a seed

**Gate:** read the generator line by line yourself. Trace every feature back to its
source. If any path runs from a feature to the label without noise in between, the
phase is not done. Do not look at a model metric until this gate is passed.

---

## Phase 2 — Model and evaluation

Ticks checklist items 3, 4, 5, 6.

- [ ] `features/` built as pure functions with unit tests
- [ ] LightGBM trained on the temporal training split only
- [ ] Precision and recall reported on the temporal holdout
- [ ] PR-AUC reported on the temporal holdout
- [ ] Calibration checked: does P(win)=0.7 come out right about 70% of the time
- [ ] Baseline A implemented: contest everything
- [ ] Baseline B implemented: accept everything
- [ ] Rupees recovered per 1,000 disputes, model vs both baselines
- [ ] False-positive cost stated in rupees, with its components named
- [ ] False-negative cost stated in rupees
- [ ] Every headline number reported as median and IQR across at least 20 seeds
- [ ] The single-seed number, if kept anywhere, is stored under a key that says it
      must not be reported as the headline

**Gate:** if the model does not beat both baselines, that is written down as the
result. It is not a reason to change the metric.

---

## Phase 3 — Policy and evidence

Ticks checklist item 2.

- [ ] Policy engine implemented exactly as `SPEC.md` §4, with unit tests covering
      contest, accept, and escalate branches
- [ ] `representment_cost` is a named constant with a sourced comment
- [ ] Reason-code to required-evidence mapping implemented as a lookup table
- [ ] LLM drafts `explanation_letter` and normalises free-text order context — and
      nothing else
- [ ] Every LLM output validated against a Pydantic schema before use
- [ ] One repair attempt on validation failure, feeding the error back
- [ ] Deterministic template fallback if repair fails, plus a human-review flag
- [ ] Prompt text versioned in the repo, not inline in a function

**Gate:** the model and the LLM can both be swapped out and the policy engine's
decisions remain traceable and reproducible.

---

## Phase 4 — Integration, audit, failure recovery

Ticks checklist items 11 and 12.

- [ ] FastAPI webhook receives an `open` dispute event
- [ ] `client/` calls the Razorpay test-mode Disputes API for contest and accept
- [ ] Idempotency keyed on dispute id; a replayed event cannot double-file
- [ ] Decision persisted **before** the API call
- [ ] Retry with exponential backoff on timeout
- [ ] Append-only audit row per decision: model version, features used, P(win),
      policy branch, prompt version, validation result, API response
- [ ] Failure path 1 demonstrated: API timeout, recovered
- [ ] Failure path 2 demonstrated: invalid LLM output, repaired or degraded to
      template, never crashed

**Gate:** kill the network mid-run. The system must recover without double-filing
and without losing the audit row.

---

## Phase 5 — Freeze

No new code. Ticks checklist items 8, 9, 10.

- [ ] Refactor pass: dead code removed, oversized files split
- [ ] Security review pass: secrets, webhook input validation, unvalidated LLM
      output reaching the API client
- [ ] README describes only what exists; stubs are labelled as stubs
- [ ] Architecture document written, including the LLM authority boundary
- [ ] `GENERATOR.md` published as part of the submission — the honest account
      of how the data was made is a strength, not a liability
- [ ] Three cold runs from a fresh clone, all passing
- [ ] Five-minute pitch video recorded: architecture and trade-offs, both failure
      paths shown live
- [ ] All twelve items in `SPEC.md` §9 ticked

**Gate:** a stranger clones the repo and reproduces the headline number.

---

## Cut list

If time runs short, cut in this order. Do not cut upward.

1. Calibration check (Phase 2)
2. The escalate-to-human band (Phase 3) — collapse to a two-way decision
3. FastAPI webhook (Phase 4) — replace with a CLI that replays a fixture event

Never cut: the leakage guard, the temporal holdout, the multi-seed distribution,
the false-positive cost, or either failure path. Those are the submission.
