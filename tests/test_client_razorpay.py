"""Tests for the Razorpay Disputes API client (PHASES.md Phase 4 items 1-2,
6-7). `RazorpayHttpClient` is exercised against `httpx.MockTransport`, never
a real socket (CLAUDE.md: "No test may make a network call.") - this is
what proves "a timeout followed by a retry files exactly once" (item 6)
against the real client code, not just the generic retry helper.
"""

import base64
import json

import httpx
import pytest

from disputedesk.client.razorpay import FakeRazorpayClient, RazorpayHttpClient
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


def test_contest_sends_amount_in_paise_and_the_letter_body_untouched(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "disp_2", "status": "under_review"})

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    client = RazorpayHttpClient()

    # At the network's documented ceiling exactly: this is the longest letter
    # that can exist, since `DraftedLetter` rejects anything longer at
    # construction. Nothing here truncates it - until 2026-09-02 this client
    # sent `summary[:1000]` of a letter drafted against a 4,000-character
    # schema, silently discarding most of the evidence (defect 0.2).
    body = "y" * 1000
    result = client.contest("disp_2", amount_inr=650.50, letter=_model_letter(body))

    assert result == {"id": "disp_2", "status": "under_review"}
    assert seen["method"] == "PATCH"
    assert seen["url"] == "https://api.razorpay.com/v1/disputes/disp_2/contest"
    assert seen["json"]["amount"] == 65050  # rupees -> paise
    assert seen["json"]["action"] == "submit"
    assert seen["json"]["summary"] == body


def test_timeout_then_success_files_exactly_once(monkeypatch):
    responses_sent = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        responses_sent["n"] += 1
        if responses_sent["n"] == 1:
            raise httpx.ConnectTimeout("simulated timeout", request=request)
        return httpx.Response(200, json={"id": "disp_3", "status": "under_review"})

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    monkeypatch.setattr("time.sleep", lambda _s: None)
    client = RazorpayHttpClient(max_retries=3)

    result = client.contest("disp_3", amount_inr=1000.0, letter=_model_letter())

    assert responses_sent["n"] == 2  # one timeout, one success - never a third
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


def test_fake_client_queues_an_exception_then_a_response():
    fake = FakeRazorpayClient(
        contest_responses=[
            httpx.TimeoutException("simulated"),
            {"id": "disp_fake", "status": "under_review"},
        ]
    )

    with pytest.raises(httpx.TimeoutException):
        fake.contest("disp_fake", 100.0, _model_letter())

    result = fake.contest("disp_fake", 100.0, _model_letter())
    assert result == {"id": "disp_fake", "status": "under_review"}
    assert len(fake.contest_calls) == 2


def test_fake_client_records_calls_made():
    fake = FakeRazorpayClient()
    fake.accept("disp_a")
    letter = _model_letter()
    fake.contest("disp_b", 500.0, letter)

    assert fake.accept_calls == ["disp_a"]
    assert fake.contest_calls == [("disp_b", 500.0, letter)]
