"""CLAUDE.md Day-1 Phase 4 verification: drafts the same letter twice back to
back against the real Groq API and checks three things end to end:

1. Both calls' prompts share an identical static prefix up to the dispute-
   record boundary (Phase 1's caching precondition) - a hard assertion,
   true regardless of what Groq's cache actually does.
2. Both calls' `prompt_tokens`/`completion_tokens`/`cached_tokens` are
   printed and persisted onto real audit rows, and the hash chain still
   verifies across both rows - also a hard assertion.
3. Whether call 2 got a cache hit (`cached_tokens > 0`) is *reported*, not
   asserted. `_STATIC_PREFIX` measures 470 tokens by Groq's own
   `usage.prompt_tokens` (a `tiktoken` cl100k_base/o200k_base estimate said
   399 - the real gpt-oss tokenizer is not publicly available via tiktoken,
   so that number was an approximation; 470 is the corrected, authoritative
   figure - see DECISIONS.md's 2026-09-08 correction). Console.groq.com's
   docs (checked live, 2026-09-08) give only a 128-1024 token range for the
   minimum cacheable prefix, with no `openai/gpt-oss-20b`-specific number,
   and a same-day empirical test (identical 470-token prefix, no dynamic
   tail, sent twice back to back) got `cached_tokens: 0` on the second call -
   near-conclusive evidence 470 tokens sits under this model's real
   threshold. A miss here is a real, now well-supported finding, not a bug
   in this script.

Makes real network calls - run manually, not part of the test suite (this
project's convention: no test may make a network call). Requires a
populated `.env`. Groq's prompt cache is volatile and expires after roughly
2 hours of non-use, so both calls happen in this one run, back to back.
"""

import sys

from disputedesk.audit.chain import verify_chain
from disputedesk.audit.db import get_engine, init_db, make_session_factory
from disputedesk.audit.log import record_decision
from disputedesk.audit.models import DecisionRecord
from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.draft_letter import _STATIC_PREFIX, draft_explanation_letter
from disputedesk.evidence.llm import GroqHttpLLMClient, LLMClient
from disputedesk.evidence.schemas import NormalizedCommunicationLog

_CONTEXT = DisputeContext(
    reason_code="VISA_10_4",
    amount=5000.0,
    avs_match=True,
    cvv_match=True,
    device_fingerprint_known=True,
    delivery_confirmed=True,
    prior_order_count=12,
)
_EVIDENCE_TYPES = (
    "billing_proof",
    "access_activity_log",
    "proof_of_service",
    "customer_communication",
    "explanation_letter",
)
_NORMALIZED_COMMS = NormalizedCommunicationLog(
    claims_unauthorized_transaction=True,
    mentions_prior_bank_contact=False,
    mentions_shared_card_access=False,
    mentions_travel=False,
    tone="polite",
    is_substantive=True,
    summary="Customer says they did not authorize this charge.",
)


class _PromptRecordingClient:
    """Wraps a real `LLMClient`, recording every prompt in call order -
    purely observational, same pattern as
    `eval/llm_letter_validation_reliability.py`'s `_RecordingLLMClient`
    (which records responses; this records prompts, the thing this script
    needs to check). `usage_log` delegates straight through so callers that
    read it (this script) see the wrapped client's real Groq usage data.
    """

    def __init__(self, inner: LLMClient):
        self._inner = inner
        self.prompts: list[str] = []

    @property
    def usage_log(self) -> list[dict]:
        return self._inner.usage_log

    def complete(self, prompt: str, *, response_format: dict | None = None) -> str:
        self.prompts.append(prompt)
        return self._inner.complete(prompt, response_format=response_format)


def _decision_kwargs(dispute_id: str, usage: dict) -> dict:
    """Fields `record_decision` needs that are unrelated to what this script
    checks - plausible, fixed values consistent with `_CONTEXT`, not derived
    from a real policy/model run. This script's job is the audit-row/hash-
    chain/token-column path, not re-proving the policy engine.
    """
    return dict(
        dispute_id=dispute_id,
        reason_code=_CONTEXT.reason_code,
        amount_inr=_CONTEXT.amount,
        model_version="phase4-verification-script",
        features={"amount": _CONTEXT.amount, "avs_match": _CONTEXT.avs_match},
        p_win=0.8,
        policy_branch="contest",
        expected_value_inr=3600.0,
        representment_cost_inr=400.0,
        low_confidence=False,
        prompt_version="explanation_letter_v4",
        validation_result="validated",
        human_review_required=False,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        cached_tokens=usage["cached_tokens"],
    )


def main() -> int:
    client = _PromptRecordingClient(GroqHttpLLMClient())

    print("Drafting the same letter twice, back to back...")
    letter_1 = draft_explanation_letter(_CONTEXT, _EVIDENCE_TYPES, _NORMALIZED_COMMS, client)
    letter_2 = draft_explanation_letter(_CONTEXT, _EVIDENCE_TYPES, _NORMALIZED_COMMS, client)

    assert len(client.usage_log) == 2, (
        f"expected exactly 2 calls total (repair=False means one call per draft), "
        f"got {len(client.usage_log)} - a repair fired, which should not happen "
        f"unless the live model returned a malformed/non-schema response"
    )
    call_1, call_2 = client.usage_log

    print()
    print(f"Call 1 usage: {call_1}")
    print(f"Call 2 usage: {call_2}")
    print()

    # --- 1. Hard assertion: byte-identical static prefix ------------------
    prefix_len = len(_STATIC_PREFIX)
    assert client.prompts[0][:prefix_len] == _STATIC_PREFIX, "call 1's prompt prefix drifted"
    assert client.prompts[1][:prefix_len] == _STATIC_PREFIX, "call 2's prompt prefix drifted"
    print(f"PASS: both prompts share an identical {prefix_len}-char static prefix.")

    # --- 2. Report, don't assert, on the actual caching outcome ------------
    print()
    print(
        f"_STATIC_PREFIX measured length: {prefix_len} chars "
        "(470 tokens per Groq's own usage.prompt_tokens, DECISIONS.md 2026-09-08)"
    )
    print(f"Call 2 cached_tokens: {call_2['cached_tokens']}")
    if call_2["cached_tokens"] > 0:
        print("RESULT: call 2 got a cache hit.")
    else:
        print(
            "RESULT: call 2 got NO cache hit. Consistent with a same-day empirical "
            "test (DECISIONS.md 2026-09-08) showing this same 470-token prefix, "
            "sent identically twice with no dynamic tail, also gets no cache hit - "
            "470 tokens is under this model's real minimum cacheable-prefix length."
        )

    assert letter_1.submittable, "letter 1 did not validate - not a meaningful caching test"
    assert letter_2.submittable, "letter 2 did not validate - not a meaningful caching test"

    # --- 3. Hard assertions: audit rows carry token fields, chain verifies -
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    session = make_session_factory(engine)()

    record_decision(session, **_decision_kwargs("phase4_verify_call_1", call_1))
    record_decision(session, **_decision_kwargs("phase4_verify_call_2", call_2))

    row_1, row_2 = session.query(DecisionRecord).order_by(DecisionRecord.id).all()
    assert row_1.prompt_tokens == call_1["prompt_tokens"]
    assert row_1.completion_tokens == call_1["completion_tokens"]
    assert row_1.cached_tokens == call_1["cached_tokens"]
    assert row_2.prompt_tokens == call_2["prompt_tokens"]
    assert row_2.completion_tokens == call_2["completion_tokens"]
    assert row_2.cached_tokens == call_2["cached_tokens"]
    print()
    print("PASS: both audit rows carry the token fields from their respective calls.")

    result = verify_chain(session)
    assert result.ok, f"chain verification failed: {result.problems}"
    assert result.rows_checked == 2
    print("PASS: hash chain verifies end to end across both rows.")

    session.close()
    print()
    print("ALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
