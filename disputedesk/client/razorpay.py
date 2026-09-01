"""Test-mode Razorpay Disputes API client (SPEC.md §1 step 5, §2's "Disputes
API client" row). Along with `evidence/llm.py`, the only module allowed to
talk to the network (CLAUDE.md: "Nothing outside client/ talks to the
network").

Endpoint shapes below were verified against Razorpay's own documentation on
2026-09-01 (today), not recalled from training data:
- Auth: HTTP Basic Auth, `base64(key_id:key_secret)` in the `Authorization`
  header - https://razorpay.com/docs/api/authentication/
- Accept: `POST /v1/disputes/{id}/accept`, empty request body. Response
  includes `status: "lost"` - accepting is irreversible.
  https://razorpay.com/docs/api/disputes/accept/
- Contest: `PATCH /v1/disputes/{id}/contest`, body `{amount, summary,
  <evidence_type>: [doc_id, ...], ..., others: [{type, document_ids}], action:
  "draft"|"submit"}`. https://razorpay.com/docs/api/disputes/contest/
- Dispute entity fields (id, entity, payment_id, amount, currency,
  amount_deducted, reason_code, respond_by, status, phase, created_at,
  evidence) - https://razorpay.com/docs/api/disputes/fetch-all/

"Test mode" is not a separate base URL - Razorpay determines it from which
key pair is configured (test keys are prefixed `rzp_test_`), so
`razorpay_api_base_url` (`disputedesk/config.py`) is the same host either
way; using a real vs. test key pair is entirely a `.env` concern.

Unit convention: every other module in this codebase (`policy/`,
`evidence/`, the generator) treats `amount` as rupees, matching
`DisputeRecord.amount`'s own convention. Razorpay's wire format is paise
(the smallest currency unit) - the rupees -> paise conversion happens at
this module's boundary only, nowhere else in the codebase.

This project does not build a document-upload pipeline (not listed in
SPEC.md or PHASES.md - out of scope). `contest()` therefore never populates
the per-evidence-type document-id list fields (`shipping_proof`,
`billing_proof`, ...): doing so would mean inventing document ids that do
not correspond to any real uploaded file, which is worse than omitting them.
It submits the drafted `explanation_letter` text as `summary` (truncated to
the API's documented 1000-character limit) with `action="submit"`; the
`required_evidence_types` the evidence assembler computed are recorded in
the audit log for a human reviewer, not attached as files here.
"""

import time
from collections.abc import Callable
from typing import Protocol

import httpx

from disputedesk.config import get_settings
from disputedesk.retry import call_with_backoff

_CONTEST_SUMMARY_MAX_CHARS = 1000  # documented limit, contest()'s `summary` field


class RazorpayClient(Protocol):
    def accept(self, dispute_id: str) -> dict:
        """POST the accept action for `dispute_id`. Returns the parsed
        dispute entity (irreversible - real status moves to "lost")."""
        ...

    def contest(self, dispute_id: str, amount_inr: float, summary: str) -> dict:
        """PATCH the contest action for `dispute_id` with the full disputed
        `amount_inr` and the drafted explanation letter as `summary`.
        Returns the parsed dispute entity."""
        ...


class FakeRazorpayClient:
    """Deterministic stand-in for tests and the demo script. Never touches
    the network. `accept_responses`/`contest_responses` are consumed one per
    call (the last entry repeats once exhausted, same convention as
    `evidence.llm.FakeLLMClient`); an `Exception` instance in the list is
    raised instead of returned, so a test can queue a timeout followed by a
    success to exercise the retry path.
    """

    def __init__(
        self,
        accept_responses: list[dict | Exception] | None = None,
        contest_responses: list[dict | Exception] | None = None,
    ):
        self._accept_responses = accept_responses or [{"id": "disp_fake", "status": "lost"}]
        self._contest_responses = contest_responses or [
            {"id": "disp_fake", "status": "under_review"}
        ]
        self.accept_calls: list[str] = []
        self.contest_calls: list[tuple[str, float, str]] = []

    @staticmethod
    def _next(queue: list[dict | Exception], index: int) -> dict:
        item = queue[min(index, len(queue) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    def accept(self, dispute_id: str) -> dict:
        index = len(self.accept_calls)
        self.accept_calls.append(dispute_id)
        return self._next(self._accept_responses, index)

    def contest(self, dispute_id: str, amount_inr: float, summary: str) -> dict:
        index = len(self.contest_calls)
        self.contest_calls.append((dispute_id, amount_inr, summary))
        return self._next(self._contest_responses, index)


class RazorpayHttpClient:
    """Real implementation: plain `httpx` calls against Razorpay's test-mode
    Disputes API, retried on timeout/429 via the shared
    `disputedesk.retry.call_with_backoff` (PHASES.md Phase 4). Every
    provider detail is read from `get_settings()`, never hardcoded, per the
    Phase 0 config-module rule.
    """

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        settings = get_settings()
        self._base_url = settings.razorpay_api_base_url
        self._auth = (settings.razorpay_key_id, settings.razorpay_key_secret)
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        # Seam for tests/the demo script to observe or compress the wait
        # between retries without touching call_with_backoff itself - real
        # callers never pass this, so production behaviour is unchanged.
        self._sleep_fn = sleep_fn

    def _call(self, method: str, path: str, json_body: dict | None) -> dict:
        def _do_request() -> httpx.Response:
            response = httpx.request(
                method,
                f"{self._base_url}{path}",
                auth=self._auth,
                json=json_body,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            return response

        response = call_with_backoff(
            _do_request, max_retries=self._max_retries, sleep_fn=self._sleep_fn
        )
        return response.json()

    def accept(self, dispute_id: str) -> dict:
        return self._call("POST", f"/disputes/{dispute_id}/accept", None)

    def contest(self, dispute_id: str, amount_inr: float, summary: str) -> dict:
        body = {
            "amount": round(amount_inr * 100),  # rupees -> paise, at this boundary only
            "summary": summary[:_CONTEST_SUMMARY_MAX_CHARS],
            "action": "submit",
        }
        return self._call("PATCH", f"/disputes/{dispute_id}/contest", body)
