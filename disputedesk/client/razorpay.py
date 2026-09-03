"""Test-mode Razorpay Disputes API client (SPEC.md §1 step 5, §2's "Disputes
API client" row). Along with `evidence/llm.py`, the only module allowed to
talk to the network (CLAUDE.md: "Nothing outside client/ talks to the
network").

Endpoint shapes below were verified against Razorpay's own documentation:
- Auth: HTTP Basic Auth, `base64(key_id:key_secret)` in the `Authorization`
  header - https://razorpay.com/docs/api/authentication/ (2026-09-01).
- Accept: `POST /v1/disputes/{id}/accept`, empty request body. Response
  includes `status: "lost"` - accepting is irreversible.
  https://razorpay.com/docs/api/disputes/accept/ (2026-09-01).
- Contest: `PATCH /v1/disputes/{id}/contest`, body `{amount, summary,
  <evidence_type>: [doc_id, ...], ..., others: [{type, document_ids}], action:
  "draft"|"submit"}`. `action="submit"` requires at least one document id
  across the evidence fields. https://razorpay.com/docs/api/disputes/contest/
  (2026-09-01, re-verified 2026-09-02).
- Documents: `POST /v1/documents`, `multipart/form-data`, fields `file` and
  `purpose` (`"dispute_evidence"` is a documented valid value), both
  required. Accepts `image/jpg`, `image/jpeg`, `image/png`,
  `application/pdf` only - not plain text. Response includes the created
  document's `id`. https://razorpay.com/docs/api/documents/create/
  (verified 2026-09-04).
- Dispute entity fields (id, entity, payment_id, amount, currency,
  amount_deducted, reason_code, respond_by, status, phase, created_at,
  evidence) - https://razorpay.com/docs/api/disputes/fetch-all/ (2026-09-01).

"Test mode" is not a separate base URL - Razorpay determines it from which
key pair is configured (test keys are prefixed `rzp_test_`), so
`razorpay_api_base_url` (`disputedesk/config.py`) is the same host either
way; using a real vs. test key pair is entirely a `.env` concern.

Unit convention: every other module in this codebase (`policy/`,
`evidence/`, the generator) treats `amount` as rupees, matching
`DisputeRecord.amount`'s own convention. Razorpay's wire format is paise
(the smallest currency unit) - the rupees -> paise conversion happens at
this module's boundary only, nowhere else in the codebase.

**2026-09-04 scoped reopening: the document-upload gap is closed.** Until
this change, `contest()` never populated the per-evidence-type document-id
list fields - `action="submit"` always went out with zero document ids,
which Razorpay's own documented contract rejects
(`tests/test_client_document_contract.py` proves it against the code exactly
as it stood before this change). `contest()` now takes the rendered evidence
bundle (`disputedesk.evidence.documents.EvidenceDocument`, one per required
evidence type - pure, no network, built in `evidence/`), uploads each
document via the new `upload_document()`, and attaches the returned ids
under their evidence-type keys before submitting. An empty bundle is
rejected here, before any network call, the same way an unreviewed letter
already was.

**Not claimed: this has been run against production Razorpay.** It has not.
The accurate statement is that the contest path is conformant with the
documented API contract (SPEC.md's evidence-type keys, the multipart
document-upload shape, the "at least one document id" requirement) and has
never been executed against a live merchant account - tested throughout
against `httpx.MockTransport` and recorded fixtures, never a real socket
(CLAUDE.md: "No test may make a network call").

`contest()` takes a `DraftedLetter`, not a `str` (2026-09-02 remediation,
defect 0.1). Two things follow from that type, both of which this module used
to get wrong:

- Only a letter with `provenance == "model"` may be filed. The check runs
  here, inside the client and before the request is built, so it holds even
  for a caller that bypasses `disputedesk/api/pipeline.py`. Until this change
  the deterministic fallback letter - whose own body says it has not been
  reviewed by a person - was filed with `action="submit"`.
- The letter's text is already bounded by the network's documented 1,000
  character `summary` limit at construction, so this module no longer
  truncates. Until this change it sent `summary[:1000]` of a letter drafted
  against a 4,000-character ceiling.
"""

import time
from collections.abc import Callable
from typing import Protocol

import httpx

from disputedesk.config import get_settings
from disputedesk.evidence.documents import EvidenceDocument
from disputedesk.evidence.letter import DraftedLetter, require_submittable
from disputedesk.retry import call_with_backoff

DOCUMENT_UPLOAD_PURPOSE = "dispute_evidence"
"""The one documented `purpose` value this project's uploads use - Razorpay's
create-document endpoint requires it and this is the value it names for
dispute evidence specifically (verified 2026-09-04)."""


class DocumentUploadError(RuntimeError):
    """Raised when there is nothing to upload, an upload's response carries
    no usable id, or (for `RazorpayHttpClient`) the upload request itself
    fails after retries. Deliberately an error, not a silently-degraded
    submit: Razorpay's contest endpoint requires at least one document id
    under `action="submit"`, so a caller that cannot produce one must not
    reach that endpoint at all - the same fail-closed shape
    `LetterNotSubmittableError` already gives the letter-provenance gate.
    """


def _require_nonempty_bundle(evidence_bundle: tuple[EvidenceDocument, ...]) -> None:
    """The contract check `tests/test_client_document_contract.py` proves is
    missing without it. Called before any network work in both
    implementations, the same way `require_submittable` already is - a
    laxer fake would let this invariant be re-opened by a path only tests or
    the demo script exercise.
    """
    if not evidence_bundle:
        raise DocumentUploadError(
            "evidence_bundle is empty - Razorpay's contest endpoint requires at least "
            "one document id under action='submit'; refusing to submit with none"
        )


class RazorpayClient(Protocol):
    def accept(self, dispute_id: str) -> dict:
        """POST the accept action for `dispute_id`. Returns the parsed
        dispute entity (irreversible - real status moves to "lost")."""
        ...

    def upload_document(self, document: EvidenceDocument) -> str:
        """Upload one rendered evidence document and return its Razorpay
        document id. Implementations raise `DocumentUploadError` if the
        response carries no usable id."""
        ...

    def contest(
        self,
        dispute_id: str,
        amount_inr: float,
        letter: DraftedLetter,
        evidence_bundle: tuple[EvidenceDocument, ...],
    ) -> dict:
        """PATCH the contest action for `dispute_id` with the full disputed
        `amount_inr`, `letter` as `summary`, and every document in
        `evidence_bundle` uploaded and attached under its evidence type.
        Implementations must call `require_submittable(letter)` and reject
        an empty `evidence_bundle` before any network work, so neither an
        unreviewed letter nor a submit with zero document ids can be filed.
        Returns the parsed dispute entity."""
        ...


class FakeRazorpayClient:
    """Deterministic stand-in for tests and the demo script. Never touches
    the network. `accept_responses`/`contest_responses`/`upload_responses`
    are each consumed one per call (the last entry repeats once exhausted,
    same convention as `evidence.llm.FakeLLMClient`); an `Exception`
    instance in a list is raised instead of returned, so a test can queue a
    timeout followed by a success to exercise the retry path.
    """

    def __init__(
        self,
        accept_responses: list[dict | Exception] | None = None,
        contest_responses: list[dict | Exception] | None = None,
        upload_responses: list[str | Exception] | None = None,
    ):
        self._accept_responses = accept_responses or [{"id": "disp_fake", "status": "lost"}]
        self._contest_responses = contest_responses or [
            {"id": "disp_fake", "status": "under_review"}
        ]
        self._upload_responses = upload_responses or ["doc_fake"]
        self.accept_calls: list[str] = []
        self.contest_calls: list[
            tuple[str, float, DraftedLetter, tuple[EvidenceDocument, ...]]
        ] = []
        self.upload_calls: list[EvidenceDocument] = []

    @staticmethod
    def _next(queue: list, index: int):
        item = queue[min(index, len(queue) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    def accept(self, dispute_id: str) -> dict:
        index = len(self.accept_calls)
        self.accept_calls.append(dispute_id)
        return self._next(self._accept_responses, index)

    def upload_document(self, document: EvidenceDocument) -> str:
        index = len(self.upload_calls)
        self.upload_calls.append(document)
        doc_id = self._next(self._upload_responses, index)
        if not doc_id:
            raise DocumentUploadError(f"upload of {document.filename!r} returned no document id")
        return doc_id

    def contest(
        self,
        dispute_id: str,
        amount_inr: float,
        letter: DraftedLetter,
        evidence_bundle: tuple[EvidenceDocument, ...],
    ) -> dict:
        # Same gates as RazorpayHttpClient, deliberately duplicated rather
        # than left to the real client alone: the demo script and most tests
        # run against this fake, so a laxer fake would let either invariant
        # be re-opened by a path only they exercise.
        require_submittable(letter)
        _require_nonempty_bundle(evidence_bundle)
        for document in evidence_bundle:
            self.upload_document(document)  # ids not otherwise used by the fake
        index = len(self.contest_calls)
        self.contest_calls.append((dispute_id, amount_inr, letter, evidence_bundle))
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

    def upload_document(self, document: EvidenceDocument) -> str:
        def _do_upload() -> httpx.Response:
            response = httpx.request(
                "POST",
                f"{self._base_url}/documents",
                auth=self._auth,
                data={"purpose": DOCUMENT_UPLOAD_PURPOSE},
                files={"file": (document.filename, document.content, document.mime_type)},
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            return response

        response = call_with_backoff(
            _do_upload, max_retries=self._max_retries, sleep_fn=self._sleep_fn
        )
        doc_id = response.json().get("id")
        if not doc_id:
            raise DocumentUploadError(
                f"upload of {document.filename!r} returned no document id: {response.json()!r}"
            )
        return doc_id

    def contest(
        self,
        dispute_id: str,
        amount_inr: float,
        letter: DraftedLetter,
        evidence_bundle: tuple[EvidenceDocument, ...],
    ) -> dict:
        # Raises before the request body exists, let alone a socket - see
        # this module's docstring and disputedesk/evidence/letter.py.
        require_submittable(letter)
        _require_nonempty_bundle(evidence_bundle)

        document_ids_by_type: dict[str, list[str]] = {}
        for document in evidence_bundle:
            doc_id = self.upload_document(document)
            document_ids_by_type.setdefault(document.evidence_type, []).append(doc_id)

        body = {
            "amount": round(amount_inr * 100),  # rupees -> paise, at this boundary only
            "summary": letter.letter_text,  # never truncated - bounded at construction
            "action": "submit",
            **document_ids_by_type,
        }
        return self._call("PATCH", f"/disputes/{dispute_id}/contest", body)
