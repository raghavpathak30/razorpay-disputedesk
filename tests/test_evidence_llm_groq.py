"""Tests for `GroqHttpLLMClient` (`disputedesk/evidence/llm.py`), added
alongside the 2026-09-01 fix (DECISIONS.md: "Letter-drafting reliability,
re-measured") for the deprecated `max_tokens` parameter, the missing
`reasoning_effort` control, and the usage-logging instrument that measured
the fix. Exercised against `httpx.MockTransport`, never a real socket
(CLAUDE.md: "No test may make a network call.") - the same pattern
`tests/test_client_razorpay.py` uses for the real Razorpay client.
"""

import json

import httpx
import pytest

from disputedesk.evidence.llm import GroqHttpLLMClient

LLM_API_URL = "https://example.test/llm"


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_id")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "rzp_test_secret")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_API_URL", LLM_API_URL)
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    from disputedesk.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _patch_post(monkeypatch, transport: httpx.MockTransport):
    def fake_post(url, **kwargs):
        with httpx.Client(transport=transport) as http_client:
            return http_client.post(url, **kwargs)

    monkeypatch.setattr(httpx, "post", fake_post)


def _response_with_usage(
    content: str = "hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    reasoning_tokens: int | None = 3,
) -> dict:
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {"choices": [{"message": {"content": content}}], "usage": usage}


def test_sends_max_completion_tokens_not_the_deprecated_max_tokens_alias(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_response_with_usage())

    _patch_post(monkeypatch, httpx.MockTransport(handler))
    client = GroqHttpLLMClient()

    client.complete("draft a letter")

    assert "max_tokens" not in seen["body"]
    assert seen["body"]["max_completion_tokens"] == GroqHttpLLMClient.MAX_COMPLETION_TOKENS


def test_reasoning_effort_defaults_to_low(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_response_with_usage())

    _patch_post(monkeypatch, httpx.MockTransport(handler))
    client = GroqHttpLLMClient()

    client.complete("draft a letter")

    assert seen["body"]["reasoning_effort"] == "low"


def test_reasoning_effort_is_configurable(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_response_with_usage())

    _patch_post(monkeypatch, httpx.MockTransport(handler))
    client = GroqHttpLLMClient(reasoning_effort="medium")

    client.complete("draft a letter")

    assert seen["body"]["reasoning_effort"] == "medium"


def test_records_usage_including_reasoning_tokens(monkeypatch):
    handler = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json=_response_with_usage(
                prompt_tokens=377, completion_tokens=581, reasoning_tokens=5
            ),
        )
    )
    _patch_post(monkeypatch, handler)
    client = GroqHttpLLMClient()

    client.complete("draft a letter")

    assert client.usage_log == [
        {"prompt_tokens": 377, "completion_tokens": 581, "reasoning_tokens": 5}
    ]


def test_records_usage_with_missing_reasoning_tokens_field_as_none(monkeypatch):
    handler = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json=_response_with_usage(reasoning_tokens=None)
        )
    )
    _patch_post(monkeypatch, handler)
    client = GroqHttpLLMClient()

    client.complete("draft a letter")

    assert client.usage_log[0]["reasoning_tokens"] is None


def test_usage_log_accumulates_across_multiple_calls(monkeypatch):
    handler = httpx.MockTransport(lambda request: httpx.Response(200, json=_response_with_usage()))
    _patch_post(monkeypatch, handler)
    client = GroqHttpLLMClient()

    client.complete("first")
    client.complete("second")

    assert len(client.usage_log) == 2


def test_missing_usage_field_entirely_does_not_crash(monkeypatch):
    handler = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})
    )
    _patch_post(monkeypatch, handler)
    client = GroqHttpLLMClient()

    result = client.complete("draft a letter")

    assert result == "x"
    assert client.usage_log == [
        {"prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None}
    ]
