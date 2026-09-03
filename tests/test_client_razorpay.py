"""Tests for the Razorpay Disputes API client (PHASES.md Phase 4 items 1-2,
6-7; 2026-09-04 scoped reopening for document upload). `RazorpayHttpClient`
is exercised against `httpx.MockTransport`, never a real socket (CLAUDE.md:
"No test may make a network call.") - this is what proves "a timeout
followed by a retry files exactly once" (item 6) against the real client
code, not just the generic retry helper, and now also proves the same for
document uploads specifically.
"""

import base64
import json

import httpx
import pytest

from disputedesk.client.razorpay import (
    DOCUMENT_UPLOAD_PURPOSE,
    DocumentUploadError,
    FakeRazorpayClient,
    RazorpayHttpClient,
)
from disputedesk.evidence.documents import EvidenceDocument
from disputedesk.evidence.letter import DraftedLetter, LetterProvenance


def _model_letter(text: str = "a fully drafted, model-authored explanation letter body."):
    """A submittable letter. `contest()` takes a `DraftedLetter`, not a
    string, so that only the model's own validated output can be filed -
    see `disputedesk/evidence/letter.py`.
    """
    return DraftedLetter(
        letter_text=text,
        cites_evidence_types=("billing_proof",),
        provenance=LetterProvenance.MODEL,
    )


def _document(evidence_type: str = "billing_proof", filename: str = "billing_proof.pdf"):
    return EvidenceDocument(evidence_type=evidence_type, filename=filename, content=b"%PDF-fake")


def _bundle(*evidence_types: str) -> tuple[EvidenceDocument, ...]:
    types = evidence_types or ("billing_proof",)
    return tuple(_document(t, f"{t}.pdf") for t in types)


RAZORPAY_KEY_ID = "rzp_test_id"
RAZORPAY_KEY_SECRET = "rzp_test_secret"


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_ID", RAZORPAY_KEY_ID)
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", RAZORPAY_KEY_SECRET)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_API_URL", "https://example.test/llm")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    from disputedesk.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _patch_transport(monkeypatch, transport: httpx.MockTransport, call_counter: dict | None = None):
    """Redirect every `httpx.request(...)` call `disputedesk/client/razorpay.py`
    makes through `transport` instead of a real socket.
    """

    def fake_request(method, url, **kwargs):
        if call_counter is not None:
            call_counter["n"] = call_counter.get("n", 0) + 1
        with httpx.Client(transport=transport) as http_client:
            return http_client.request(method, url, **kwargs)

    monkeypatch.setattr(httpx, "request", fake_request)


def _upload_ok_handler(doc_id: str = "doc_1"):
    """A handler that answers `/documents` uploads successfully and leaves
    everything else to be composed by the caller - most tests below only
    care about the *contest* call, and don't want upload retries/failures
    muddying what they're checking.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/documents"
        return httpx.Response(200, json={"id": doc_id, "entity": "document"})

    return handler


def test_accept_sends_basic_auth_and_no_body(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth_header"] = request.headers.get("authorization")
        seen["body"] = request.content
        return httpx.Response(200, json={"id": "disp_1", "status": "lost"})

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    client = RazorpayHttpClient()

    result = client.accept("disp_1")

    assert result == {"id": "disp_1", "status": "lost"}
    assert seen["method"] == "POST"
    assert seen["url"] == "https://api.razorpay.com/v1/disputes/disp_1/accept"
    assert seen["body"] in (b"", b"null")
    expected_auth = (
        "Basic " + base64.b64encode(f"{RAZORPAY_KEY_ID}:{RAZORPAY_KEY_SECRET}".encode()).decode()
    )
    assert seen["auth_header"] == expected_auth


# --------------------------------------------------------------------------
# Document upload
# --------------------------------------------------------------------------


def test_upload_document_sends_multipart_with_file_and_purpose(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth_header"] = request.headers.get("authorization")
        seen["content_type"] = request.headers.get("content-type", "")
        seen["body"] = request.content
        return httpx.Response(200, json={"id": "doc_abc123", "entity": "document"})

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    client = RazorpayHttpClient()

    doc_id = client.upload_document(_document())

    assert doc_id == "doc_abc123"
    assert seen["method"] == "POST"
    assert seen["url"] == "https://api.razorpay.com/v1/documents"
    assert seen["content_type"].startswith("multipart/form-data")
    expected_auth = (
        "Basic " + base64.b64encode(f"{RAZORPAY_KEY_ID}:{RAZORPAY_KEY_SECRET}".encode()).decode()
    )
    assert seen["auth_header"] == expected_auth
    # multipart bodies are boundary-delimited raw bytes, not JSON - check the
    # two required fields and the file content landed somewhere in the body.
    assert b'name="purpose"' in seen["body"]
    assert DOCUMENT_UPLOAD_PURPOSE.encode() in seen["body"]
    assert b"billing_proof.pdf" in seen["body"]
    assert b"%PDF-fake" in seen["body"]


def test_upload_document_raises_when_the_response_has_no_id(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"entity": "document"})  # no "id"

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    client = RazorpayHttpClient()

    with pytest.raises(DocumentUploadError, match="no document id"):
        client.upload_document(_document())


def test_upload_document_retries_a_timeout_then_succeeds(monkeypatch):
    responses_sent = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        responses_sent["n"] += 1
        if responses_sent["n"] == 1:
            raise httpx.ConnectTimeout("simulated timeout", request=request)
        return httpx.Response(200, json={"id": "doc_1"})

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    monkeypatch.setattr("time.sleep", lambda _s: None)
    client = RazorpayHttpClient(max_retries=3)

    doc_id = client.upload_document(_document())

    assert responses_sent["n"] == 2
    assert doc_id == "doc_1"


# --------------------------------------------------------------------------
# Contest - the document-id contract itself
# --------------------------------------------------------------------------


def test_contest_rejects_an_empty_evidence_bundle_before_any_network_call(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("no network call should happen for an empty bundle")

    monkeypatch.setattr(httpx, "request", explode)
    client = RazorpayHttpClient()

    with pytest.raises(DocumentUploadError, match="empty"):
        client.contest("disp_1", 100.0, _model_letter(), evidence_bundle=())


def test_contest_attaches_uploaded_document_ids_under_their_evidence_types(monkeypatch):
    uploaded = {"billing_proof.pdf": "doc_bill", "proof_of_service.pdf": "doc_pos"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/documents":
            # crude but sufficient: the filename appears in the multipart body
            for filename, doc_id in uploaded.items():
                if filename.encode() in request.content:
                    return httpx.Response(200, json={"id": doc_id})
            raise AssertionError(f"unexpected upload body: {request.content!r}")
        assert request.url.path == "/v1/disputes/disp_1/contest"
        return httpx.Response(200, json={"id": "disp_1", "status": "under_review"}, request=request)

    seen_contest_body = {}
    orig_handler = handler

    def wrapped(request: httpx.Request) -> httpx.Response:
        response = orig_handler(request)
        if request.url.path == "/v1/disputes/disp_1/contest":
            seen_contest_body["json"] = json.loads(request.content)
        return response

    _patch_transport(monkeypatch, httpx.MockTransport(wrapped))
    client = RazorpayHttpClient()
    bundle = _bundle("billing_proof", "proof_of_service")

    result = client.contest(
        "disp_1", amount_inr=650.50, letter=_model_letter(), evidence_bundle=bundle
    )

    assert result == {"id": "disp_1", "status": "under_review"}
    body = seen_contest_body["json"]
    assert body["billing_proof"] == ["doc_bill"]
    assert body["proof_of_service"] == ["doc_pos"]
    assert body["action"] == "submit"
    assert body["amount"] == 65050


def test_contest_sends_amount_in_paise_and_the_letter_body_untouched(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/documents":
            return httpx.Response(200, json={"id": "doc_1"})
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "disp_2", "status": "under_review"})

    seen = {}
    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    client = RazorpayHttpClient()

    # At the network's documented ceiling exactly: this is the longest letter
    # that can exist, since `DraftedLetter` rejects anything longer at
    # construction. Nothing here truncates it - until 2026-09-02 this client
    # sent `summary[:1000]` of a letter drafted against a 4,000-character
    # schema, silently discarding most of the evidence (defect 0.2).
    body = "y" * 1000
    result = client.contest(
        "disp_2", amount_inr=650.50, letter=_model_letter(body), evidence_bundle=_bundle()
    )

    assert result == {"id": "disp_2", "status": "under_review"}
    assert seen["method"] == "PATCH"
    assert seen["url"] == "https://api.razorpay.com/v1/disputes/disp_2/contest"
    assert seen["json"]["amount"] == 65050  # rupees -> paise
    assert seen["json"]["action"] == "submit"
    assert seen["json"]["summary"] == body


def test_contest_propagates_an_upload_failure_without_reaching_the_patch(monkeypatch):
    """If a document can't be uploaded, the contest PATCH must never be
    attempted with a partial/missing document set - SPEC.md §7's "the
    system degrades, it does not crash" applies at the pipeline layer
    (`disputedesk/api/pipeline.py`'s `_file` catches this and records a
    failed outcome); at the client layer the contract is simpler: no PATCH
    without every document uploaded first.
    """
    calls = {"contest_patch": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/documents":
            raise httpx.ConnectTimeout("simulated", request=request)
        calls["contest_patch"] += 1
        return httpx.Response(200, json={"id": "disp_1", "status": "under_review"})

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    monkeypatch.setattr("time.sleep", lambda _s: None)
    client = RazorpayHttpClient(max_retries=1)

    with pytest.raises(httpx.ConnectTimeout):
        client.contest("disp_1", 100.0, _model_letter(), evidence_bundle=_bundle())

    assert calls["contest_patch"] == 0


def test_timeout_then_success_on_the_contest_patch_files_exactly_once(monkeypatch):
    contest_attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/documents":
            return httpx.Response(200, json={"id": "doc_1"})
        contest_attempts["n"] += 1
        if contest_attempts["n"] == 1:
            raise httpx.ConnectTimeout("simulated timeout", request=request)
        return httpx.Response(200, json={"id": "disp_3", "status": "under_review"})

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    monkeypatch.setattr("time.sleep", lambda _s: None)
    client = RazorpayHttpClient(max_retries=3)

    result = client.contest(
        "disp_3", amount_inr=1000.0, letter=_model_letter(), evidence_bundle=_bundle()
    )

    assert contest_attempts["n"] == 2  # one timeout, one success - never a third
    assert result == {"id": "disp_3", "status": "under_review"}


def test_persistent_timeout_raises_after_exhausting_retries(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("always times out", request=request)

    call_counter: dict = {}
    _patch_transport(monkeypatch, httpx.MockTransport(handler), call_counter)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    client = RazorpayHttpClient(max_retries=1)

    with pytest.raises(httpx.ConnectTimeout):
        client.accept("disp_4")

    assert call_counter["n"] == 2  # 1 initial + 1 retry, then gives up


def test_429_is_retried_and_then_succeeds(monkeypatch):
    responses_sent = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        responses_sent["n"] += 1
        if responses_sent["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"id": "disp_5", "status": "lost"})

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    monkeypatch.setattr("time.sleep", lambda _s: None)
    client = RazorpayHttpClient()

    result = client.accept("disp_5")

    assert responses_sent["n"] == 2
    assert result == {"id": "disp_5", "status": "lost"}


def test_429_retry_after_reaches_injected_sleep_fn(monkeypatch):
    """`RazorpayHttpClient`'s `sleep_fn` seam (used by the demo script to
    compress and observe the wait) must receive the honoured Retry-After
    value from `call_with_backoff`, not the default exponential schedule -
    complements `tests/test_retry.py::test_429_honors_retry_after_header_over_computed_delay`,
    which already proves the generic helper honours the header; this proves
    the client actually wires that behaviour through, at the client level.
    """
    responses_sent = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        responses_sent["n"] += 1
        if responses_sent["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "5"})
        return httpx.Response(200, json={"id": "disp_6", "status": "under_review"})

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    sleeps: list[float] = []
    client = RazorpayHttpClient(sleep_fn=sleeps.append)

    result = client.accept("disp_6")

    assert result == {"id": "disp_6", "status": "under_review"}
    assert sleeps == [5.0]  # the honoured header value, not base_delay_seconds * 2**0


# --------------------------------------------------------------------------
# FakeRazorpayClient
# --------------------------------------------------------------------------


def test_fake_client_queues_an_exception_then_a_response():
    fake = FakeRazorpayClient(
        contest_responses=[
            httpx.TimeoutException("simulated"),
            {"id": "disp_fake", "status": "under_review"},
        ]
    )

    with pytest.raises(httpx.TimeoutException):
        fake.contest("disp_fake", 100.0, _model_letter(), _bundle())

    result = fake.contest("disp_fake", 100.0, _model_letter(), _bundle())
    assert result == {"id": "disp_fake", "status": "under_review"}
    assert len(fake.contest_calls) == 2


def test_fake_client_records_calls_made():
    fake = FakeRazorpayClient()
    fake.accept("disp_a")
    letter = _model_letter()
    bundle = _bundle()
    fake.contest("disp_b", 500.0, letter, bundle)

    assert fake.accept_calls == ["disp_a"]
    assert fake.contest_calls == [("disp_b", 500.0, letter, bundle)]


def test_fake_client_rejects_an_empty_bundle_the_same_as_the_real_client():
    fake = FakeRazorpayClient()

    with pytest.raises(DocumentUploadError, match="empty"):
        fake.contest("disp_1", 100.0, _model_letter(), evidence_bundle=())

    assert fake.contest_calls == []
    assert fake.upload_calls == []


def test_fake_client_uploads_every_document_in_the_bundle():
    fake = FakeRazorpayClient(upload_responses=["doc_1", "doc_2"])
    bundle = _bundle("billing_proof", "proof_of_service")

    fake.contest("disp_1", 100.0, _model_letter(), bundle)

    assert fake.upload_calls == list(bundle)


def test_fake_client_upload_document_raises_on_an_empty_id_response():
    fake = FakeRazorpayClient(upload_responses=[""])

    with pytest.raises(DocumentUploadError, match="no document id"):
        fake.upload_document(_document())
