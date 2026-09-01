"""The LLM interface (SPEC.md §2, PHASES.md Phase 3 gate: "the model and the
LLM can both be swapped out"). Everything in `evidence/` that needs a
completion goes through `LLMClient`, never a concrete SDK/HTTP call directly,
so tests can substitute `FakeLLMClient` and make no network call at all.

Nothing outside `disputedesk/evidence/` may import this module (CLAUDE.md).
"""

from typing import Protocol

import httpx

from disputedesk.config import get_settings
from disputedesk.retry import call_with_backoff


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        """Return the model's raw text completion for `prompt`. Callers are
        responsible for parsing/validating the result - this interface makes
        no promise about output shape.
        """
        ...


class FakeLLMClient:
    """Deterministic stand-in for tests and local runs with no `LLM_API_KEY`
    configured. Never touches the network. Returns each entry of `responses`
    in order, one per call; repeats the last entry once exhausted, so a test
    can queue a broken response followed by a good one to exercise the one
    repair attempt (SPEC.md §7 failure path 2).
    """

    def __init__(self, responses: list[str]):
        if not responses:
            raise ValueError("FakeLLMClient needs at least one response")
        self._responses = responses
        self._call_count = 0

    def complete(self, prompt: str) -> str:
        index = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[index]

    @property
    def call_count(self) -> int:
        return self._call_count


class GroqHttpLLMClient:
    """Real implementation: a plain `httpx` call to Groq's OpenAI-compatible
    chat completions endpoint, using `httpx` (already a project dependency)
    rather than adding an SDK dependency without asking (CLAUDE.md). Every
    provider detail - endpoint, model, key - is read from `get_settings()`
    (`LLM_API_URL`, `LLM_MODEL`, `LLM_API_KEY`), never hardcoded here: the
    Phase 0 rule is that `disputedesk/config.py` is the only place that reads
    `os.environ`, and provider configuration is exactly the kind of thing
    that rule exists to keep out of a class constant.

    See the 2026-08-31 "LLM provider: Groq" DECISIONS.md entry for which
    model is configured in `.env.example` and why.

    Groq's free tier returns HTTP 429 under rate limits (see the 2026-09-01
    "LLM normalisation quality" DECISIONS.md entry, which first hit this
    against the real API) - `complete()` retries a 429 or a timeout with
    `disputedesk.retry.call_with_backoff`, the same helper
    `disputedesk/client/razorpay.py` uses, so this backoff lives in the
    client itself (PHASES.md Phase 4) rather than being re-implemented by
    every caller/eval script that happens to hit a rate limit.
    """

    def __init__(self, timeout_seconds: float = 30.0, max_retries: int = 3):
        settings = get_settings()
        self._api_url = settings.llm_api_url
        self._model = settings.llm_model
        self._api_key = settings.llm_api_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    def complete(self, prompt: str) -> str:
        def _call() -> httpx.Response:
            response = httpx.post(
                self._api_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            return response

        response = call_with_backoff(_call, max_retries=self._max_retries)
        body = response.json()
        return body["choices"][0]["message"]["content"]
