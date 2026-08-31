# Dispute Desk

Razorpay AI Buildathon, Track 02. Triages fraud-reason-code chargebacks,
decides whether contesting is worth it, assembles evidence, files through the
Razorpay Disputes API in test mode, and logs every decision for audit.

See `SPEC.md` for what this system is and where the LLM is and is not allowed,
and `PHASES.md` for the build order.

## Status

**Phase 0 (skeleton) only.** Nothing below is implemented yet:

- No synthetic data generator.
- No feature builder.
- No model.
- No policy engine.
- No evidence assembler.
- No Razorpay API client.
- No audit log.
- No FastAPI webhook or CLI.

What exists: the module layout, a single config module reading required
settings from environment variables, and CI running lint and tests.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env  # fill in real values
pytest
ruff check .
```
