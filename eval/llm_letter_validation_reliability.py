"""Measures how often `disputedesk.evidence.draft_letter`'s real Groq-backed
explanation-letter drafting (SPEC.md §7 failure path 2) validates on the
first attempt, needs the one repair call, and - if repair also fails -
falls back to the deterministic template. Sampled empirically against a
fixed dispute context, not scripted.

Uses `draft_explanation_letter` directly, not the full
`assemble_evidence_packet` - and a single fixed `NormalizedCommunicationLog`
(obtained once per dispute by the caller, reused across every
letter-drafting attempt) - so the measurement isolates letter-drafting
reliability specifically, not conflated with `normalize_comms`'s own
separate repair path. Neither the drafting prompt
(`disputedesk/evidence/prompts/explanation_letter_v1.txt`) nor
`disputedesk/evidence/draft_letter.py` itself is touched or reimplemented
here - `_validation_error_text` below reuses `validated_call._parse`, the
exact function production validation calls, so a reported error is the
real error production would hit, not a re-derived approximation of it.

This module makes real network calls through whatever `LLMClient` it is
given. `_validation_error_text` and the record-building logic are unit-
tested with `FakeLLMClient`; the real measurement against the live API is a
one-off script run, recorded in DECISIONS.md, not a repeatable automated
test (CLAUDE.md: no test may make a network call).
"""

import json
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.draft_letter import draft_explanation_letter
from disputedesk.evidence.llm import LLMClient
from disputedesk.evidence.schemas import ExplanationLetterOutput, NormalizedCommunicationLog
from disputedesk.evidence.validated_call import _parse


@dataclass(frozen=True)
class DraftAttemptRecord:
    run_index: int
    first_draft_valid: bool
    first_draft_error: str | None
    repair_attempted: bool
    repair_succeeded: bool | None  # None when repair was never attempted
    repair_error: str | None
    final_path: str  # "letter" or "template_fallback"
    raw_responses: list[str] = field(default_factory=list)


class _RecordingLLMClient:
    """Wraps a real `LLMClient`, recording every raw completion in call
    order - purely observational. Every `complete()` call is delegated
    unchanged, so `draft_explanation_letter` sees identical behaviour to a
    direct call; this only lets the caller inspect what happened afterward.
    """

    def __init__(self, inner: LLMClient):
        self._inner = inner
        self.responses: list[str] = []

    def complete(self, prompt: str) -> str:
        response = self._inner.complete(prompt)
        self.responses.append(response)
        return response


def _validation_error_text(raw_response: str) -> str | None:
    """`None` if `raw_response` parses and validates against
    `ExplanationLetterOutput` via the same `_parse` helper
    `disputedesk/evidence/validated_call.py` uses internally; otherwise the
    exact error text that helper raised.
    """
    try:
        _parse(raw_response, ExplanationLetterOutput)
    except (json.JSONDecodeError, ValidationError) as error:
        return repr(error)
    return None


def run_one_draft_attempt(
    run_index: int,
    context: DisputeContext,
    evidence_types: tuple[str, ...],
    normalized_comms: NormalizedCommunicationLog,
    llm_client: LLMClient,
) -> DraftAttemptRecord:
    recorder = _RecordingLLMClient(llm_client)
    result = draft_explanation_letter(context, evidence_types, normalized_comms, recorder)

    first_error = _validation_error_text(recorder.responses[0]) if recorder.responses else None
    repair_attempted = len(recorder.responses) >= 2
    repair_error = _validation_error_text(recorder.responses[1]) if repair_attempted else None

    return DraftAttemptRecord(
        run_index=run_index,
        first_draft_valid=first_error is None,
        first_draft_error=first_error,
        repair_attempted=repair_attempted,
        repair_succeeded=(repair_error is None) if repair_attempted else None,
        repair_error=repair_error,
        final_path="template_fallback" if result.human_review_required else "letter",
        raw_responses=list(recorder.responses),
    )


def run_letter_reliability_sample(
    context: DisputeContext,
    evidence_types: tuple[str, ...],
    normalized_comms: NormalizedCommunicationLog,
    llm_client: LLMClient,
    n_runs: int,
    sleep_seconds: float = 0.0,
) -> list[DraftAttemptRecord]:
    """`n_runs` independent letter-drafting attempts against the same fixed
    context/evidence/comms - the only thing that can vary between runs is
    the live model's own non-determinism. `sleep_seconds` paces real API
    calls to respect rate limits; leave at 0 for `FakeLLMClient`.
    """
    records = []
    for i in range(n_runs):
        if i > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)
        records.append(
            run_one_draft_attempt(i, context, evidence_types, normalized_comms, llm_client)
        )
    return records


def failure_rate(records: list[DraftAttemptRecord]) -> float:
    """Fraction of runs whose *final* path was the deterministic template -
    i.e. both the first draft and the one repair attempt failed validation.
    """
    if not records:
        return 0.0
    return sum(r.final_path == "template_fallback" for r in records) / len(records)
