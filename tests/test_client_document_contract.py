"""Proves the defect this reopening exists to fix, before any fix lands.

Razorpay's contest endpoint documents that `action="submit"` requires at
least one document id attached across the evidence fields
(https://razorpay.com/docs/api/disputes/contest/, verified 2026-09-02 per
`DECISIONS.md`'s entry of that name). `RazorpayHttpClient.contest()` has
never populated any such field - `disputedesk/client/razorpay.py`'s own
module docstring said so outright: "`contest()` therefore never populates
the per-evidence-type document-id list fields... doing so would mean
inventing document ids that do not correspond to any real uploaded file."
So every `action="submit"` this system has ever sent goes out with zero
document ids attached - not "unproven," a request the real API's own
documented contract would reject.

This test calls `contest()` exactly as it exists today (the same
`(dispute_id, amount_inr, letter)` signature every other test in this
repository already calls it with) and inspects the literal request body a
real HTTP call would send. It fails against unmodified code - that failure
*is* the defect, demonstrated without touching a single line of production
code first.

**This test is expected to become obsolete, not just pass, once the fix
lands.** Attaching document ids means `contest()` needs to receive
documents to upload, which changes its signature - a 3-argument call
proving "zero document ids" cannot exist once a 4th, required argument
carries the evidence bundle. The commit that builds the pipeline replaces
this file with `tests/test_client_razorpay.py`'s updated contest tests,
which assert the same contract (non-empty document ids under
`action="submit"`) against the new signature. This file is kept, unmodified
from the commit that introduced it, as the dated record of the defect.
"""

import json

import httpx
import pytest

from disputedesk.client.razorpay import RazorpayHttpClient
from disputedesk.evidence.letter import DraftedLetter, LetterProvenance

# Every SPEC.md §3 evidence-type field the real API accepts a document-id
# list under. If none of these are non-empty when action="submit", the real
# API's documented contract - "at least one document id" - is violated.
_EVIDENCE_TYPE_FIELDS = (
    "shipping_proof",
    "billing_proof",
    "cancellation_proof",
    "customer_communication",
    "proof_of_service",
    "explanation_letter",
    "refund_confirmation",
    "access_activity_log",
    "refund_cancellation_policy",
    "terms_and_conditions",
    "others",
)


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_id")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "rzp_test_secret")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_API_URL", "https://example.test/llm")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    from disputedesk.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_a_real_submit_carries_at_least_one_document_id(monkeypatch):
    """THE DEFECT, DEMONSTRATED. Fails against unmodified
    `disputedesk/client/razorpay.py`: the request body's `action` is
    "submit" and every evidence-type field is absent or empty - exactly the
    shape Razorpay's real API rejects.
    """
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "disp_1", "status": "under_review"})

    def fake_request(method, url, **kwargs):
        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            return http_client.request(method, url, **kwargs)

    monkeypatch.setattr(httpx, "request", fake_request)

    letter = DraftedLetter(
        letter_text="a fully drafted, model-authored explanation letter body.",
        cites_evidence_types=("billing_proof",),
        provenance=LetterProvenance.MODEL,
    )
    client = RazorpayHttpClient()

    client.contest("disp_1", amount_inr=5000.0, letter=letter)

    body = seen["json"]
    assert body["action"] == "submit"

    total_document_ids = sum(len(body.get(field) or []) for field in _EVIDENCE_TYPE_FIELDS)
    assert total_document_ids >= 1, (
        "action='submit' carries zero document ids across every evidence-type field "
        f"- Razorpay's documented contract requires at least one. Body: {body}"
    )
