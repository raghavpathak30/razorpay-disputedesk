"""Pure classification logic for the letter-drafting validation-reliability
measurement (`eval/llm_letter_validation_reliability.py`): exercised only
with `FakeLLMClient` (CLAUDE.md: no test may make a network call). The real
measurement against the live API is a one-off script run, recorded in
DECISIONS.md, not covered here.
"""

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.llm import FakeLLMClient
from disputedesk.evidence.reason_code_map import required_evidence_types
from disputedesk.evidence.schemas import NormalizedCommunicationLog
from eval.llm_letter_validation_reliability import (
    failure_rate,
    run_letter_reliability_sample,
    run_one_draft_attempt,
)

_CONTEXT = DisputeContext(
    reason_code="MC_4837",
    amount=6500.0,
    avs_match=True,
    cvv_match=True,
    device_fingerprint_known=True,
    delivery_confirmed=True,
    prior_order_count=6,
)
_EVIDENCE_TYPES = required_evidence_types("MC_4837")
_NORMALIZED = NormalizedCommunicationLog(
    claims_unauthorized_transaction=True,
    mentions_prior_bank_contact=True,
    mentions_shared_card_access=False,
    mentions_travel=False,
    tone="polite",
    is_substantive=True,
    summary="Customer disputes the charge.",
)

_VALID_LETTER = (
    '{"letter_text": "'
    + ("We are contesting this chargeback. " * 3)
    + '", "cites_evidence_types": ["billing_proof"]}'
)


def test_first_draft_valid_records_no_repair():
    client = FakeLLMClient(responses=[_VALID_LETTER])

    record = run_one_draft_attempt(0, _CONTEXT, _EVIDENCE_TYPES, _NORMALIZED, client)

    assert record.first_draft_valid is True
    assert record.first_draft_error is None
    assert record.repair_attempted is False
    assert record.repair_succeeded is None
    assert record.final_path == "letter"
    assert record.raw_responses == [_VALID_LETTER]


def test_repair_succeeds_after_a_bad_first_draft():
    client = FakeLLMClient(responses=["not json", _VALID_LETTER])

    record = run_one_draft_attempt(0, _CONTEXT, _EVIDENCE_TYPES, _NORMALIZED, client)

    assert record.first_draft_valid is False
    assert record.first_draft_error is not None
    assert record.repair_attempted is True
    assert record.repair_succeeded is True
    assert record.repair_error is None
    assert record.final_path == "letter"
    assert record.raw_responses == ["not json", _VALID_LETTER]


def test_repair_also_fails_falls_back_to_template():
    client = FakeLLMClient(responses=["not json", "still not json"])

    record = run_one_draft_attempt(0, _CONTEXT, _EVIDENCE_TYPES, _NORMALIZED, client)

    assert record.first_draft_valid is False
    assert record.repair_attempted is True
    assert record.repair_succeeded is False
    assert record.repair_error is not None
    assert record.final_path == "template_fallback"


def test_too_short_letter_text_is_a_real_validation_error_not_a_parse_error():
    # Valid JSON, wrong per-field constraint (letter_text min_length=50) -
    # exercises the schema-validation branch, not just JSON parsing.
    too_short = '{"letter_text": "short", "cites_evidence_types": []}'
    client = FakeLLMClient(responses=[too_short, too_short])

    record = run_one_draft_attempt(0, _CONTEXT, _EVIDENCE_TYPES, _NORMALIZED, client)

    assert record.first_draft_valid is False
    assert "letter_text" in record.first_draft_error
    assert record.final_path == "template_fallback"


def test_run_letter_reliability_sample_returns_one_record_per_run():
    client = FakeLLMClient(responses=[_VALID_LETTER])

    records = run_letter_reliability_sample(
        _CONTEXT, _EVIDENCE_TYPES, _NORMALIZED, client, n_runs=5, sleep_seconds=0.0
    )

    assert [r.run_index for r in records] == [0, 1, 2, 3, 4]
    assert all(r.final_path == "letter" for r in records)


def test_failure_rate_counts_only_template_fallback_final_paths():
    client = FakeLLMClient(responses=["not json", "still not json"])
    all_fallback = run_letter_reliability_sample(
        _CONTEXT, _EVIDENCE_TYPES, _NORMALIZED, client, n_runs=4, sleep_seconds=0.0
    )

    assert failure_rate(all_fallback) == 1.0
    assert failure_rate([]) == 0.0
