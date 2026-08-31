"""The LLM interface (SPEC.md §2, PHASES.md Phase 3 gate: "the model and the
LLM can both be swapped out"). Everything in `evidence/` that needs a
completion goes through `LLMClient`, never a concrete SDK/HTTP call directly,
so tests can substitute `FakeLLMClient` and make no network call at all.

Nothing outside `disputedesk/evidence/` may import this module (CLAUDE.md).
"""

from typing import Protocol

import httpx

from disputedesk.config import get_settings


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


class AnthropicHttpLLMClient:
    """Real implementation: a plain `httpx` call to Anthropic's Messages API,
    using `httpx` (already a project dependency) rather than adding an SDK
    dependency without asking (CLAUDE.md). ASSUMPTION: the model id default
    below is a plausible current choice, not confirmed against Anthropic's
    live model list at deploy time - callers needing a specific model should
    pass `model` explicitly.
    """

    _API_URL = "https://api.anthropic.com/v1/messages"
    _ANTHROPIC_VERSION = "2023-06-01"
    _DEFAULT_MODEL = "claude-3-5-haiku-20241022"

    def __init__(self, model: str | None = None, timeout_seconds: float = 30.0):
        self._api_key = get_settings().llm_api_key
        self._model = model or self._DEFAULT_MODEL
        self._timeout_seconds = timeout_seconds

    def complete(self, prompt: str) -> str:
        response = httpx.post(
            self._API_URL,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": self._ANTHROPIC_VERSION,
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
        body = response.json()
        return "".join(block["text"] for block in body["content"] if block["type"] == "text")
