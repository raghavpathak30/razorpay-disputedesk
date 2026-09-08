"""The LLM interface (SPEC.md §2, PHASES.md Phase 3 gate: "the model and the
LLM can both be swapped out"). Everything in `evidence/` that needs a
completion goes through `LLMClient`, never a concrete SDK/HTTP call directly,
so tests can substitute `FakeLLMClient` and make no network call at all.

Nothing outside `disputedesk/evidence/` may import this module (CLAUDE.md).
"""

import logging
from typing import Protocol

import httpx

from disputedesk.config import get_settings
from disputedesk.retry import call_with_backoff

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    usage_log: list[dict]
    """Every `complete()` call this instance has made, in order, as
    `{"prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens"}`
    dicts (CLAUDE.md Day-1 Phase 3). `FakeLLMClient` never touches the
    network, so its entries are all zero; `GroqHttpLLMClient`'s are the
    provider's real reported usage. Callers that persist per-dispute token
    totals (`disputedesk/api/pipeline.py`) sum this list rather than reaching
    into a concrete client type, so the aggregation works against either."""

    def complete(self, prompt: str, *, response_format: dict | None = None) -> str:
        """Return the model's raw text completion for `prompt`. Callers are
        responsible for parsing/validating the result - this interface makes
        no promise about output shape.

        `response_format` (default `None`) is passed to the provider
        verbatim when given - e.g. Groq/OpenAI-style structured-outputs
        `{"type": "json_schema", "json_schema": {...}}` (CLAUDE.md Day-1
        Phase 2). `None` means "no constraint beyond the prompt", which is
        every call site's behaviour before Phase 2 and still is for every
        call site except the drafting call.
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
        # Zero-valued, not omitted: keeps this attribute's shape identical to
        # `GroqHttpLLMClient.usage_log` (CLAUDE.md Day-1 Phase 3) so callers
        # that sum it (`disputedesk/api/pipeline.py`) work against either
        # client without a type check.
        self.usage_log: list[dict] = []

    def complete(self, prompt: str, *, response_format: dict | None = None) -> str:
        index = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        self.usage_log.append(
            {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
        )
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

    Uses `max_completion_tokens`, Groq's documented parameter
    (console.groq.com/docs/reasoning, verified 2026-09-01) - not the
    deprecated `max_tokens` alias this client sent until this same date.
    Confirmed live (2026-09-01) that Groq *was* honouring `max_tokens` as an
    alias, not silently ignoring it (`max_tokens=50` produced exactly
    `usage.completion_tokens=50`, `finish_reason="length"`) - the prior
    parameter name was not the bug. The bug was the *value*: for
    `openai/gpt-oss-20b`, `usage.completion_tokens_details.reasoning_tokens`
    counts against the same budget as visible output (confirmed live, same
    date), so `max_tokens=1024` could be, and per DECISIONS.md's 2026-09-01
    "Letter-drafting validation reliability" entry regularly was, spent
    entirely on hidden reasoning before any letter text - or any letter text
    at all - was emitted, truncating the JSON response.
    """

    # `ExplanationLetterOutput.letter_text` permits up to 4000 characters
    # (disputedesk/evidence/schemas.py) - the budget must be able to reach
    # that ceiling. Derivation, from real measurements on this model against
    # both explanation-letter demo fixtures (2026-09-01, DECISIONS.md):
    #   4000 chars / ~3.6 measured chars-per-token (live: 2804 chars/581
    #   completion tokens and 1678 chars/349 completion tokens, both at
    #   reasoning_effort="low") ~= 1112 visible tokens
    #   + ~100 tokens of JSON scaffolding (the object's keys, braces, and up
    #     to 5 `cites_evidence_types` strings)
    #   + ~300 tokens reserved for reasoning - measured only 4-5 reasoning
    #     tokens per call at reasoning_effort="low" (this class's default),
    #     but kept wide because a caller can override `reasoning_effort` to
    #     "medium"/"high" (measured 300-700 reasoning tokens at the
    #     provider's un-set default) without touching this constant.
    # 1112 + 100 + 300 = 1512
    MAX_COMPLETION_TOKENS = 1512

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        reasoning_effort: str = "low",
    ):
        """`reasoning_effort` ("low" | "medium" | "high", per Groq's gpt-oss
        parameter) defaults to "low": both of this client's SPEC.md §2 jobs -
        drafting a letter from already-decided, already-supplied order-context
        facts, and normalising free text into typed fields - are prose
        generation and extraction, not reasoning tasks; the policy decision
        that would need reasoning is made upstream, deterministically, before
        the LLM is ever called. Live measurement (2026-09-01, same prompt,
        same fixtures) found "low" cut `reasoning_tokens` from 300-700 to 4-5
        per call with no loss of a complete, schema-valid letter - see
        DECISIONS.md's 2026-09-01 "Letter-drafting reliability, re-measured"
        entry for the before/after pass rates this produced.
        """
        settings = get_settings()
        self._api_url = settings.llm_api_url
        self._model = settings.llm_model
        self._api_key = settings.llm_api_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._reasoning_effort = reasoning_effort
        # One entry per `complete()` call - the instrument DECISIONS.md's
        # 2026-09-01 letter-drafting entries asked for, read directly by
        # callers that need it (e.g. eval/run_llm_letter_validation_reliability.py)
        # in addition to the `logger.info` line below.
        self.usage_log: list[dict] = []

    def complete(self, prompt: str, *, response_format: dict | None = None) -> str:
        body = {
            "model": self._model,
            "max_completion_tokens": self.MAX_COMPLETION_TOKENS,
            "reasoning_effort": self._reasoning_effort,
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_format is not None:
            # Omitted entirely rather than sent as `null` when unset, so the
            # request body every existing call site sends is byte-identical
            # to before this parameter existed (`tests/test_evidence_llm_groq.py`).
            body["response_format"] = response_format

        def _call() -> httpx.Response:
            response = httpx.post(
                self._api_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "content-type": "application/json",
                },
                json=body,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            return response

        response = call_with_backoff(_call, max_retries=self._max_retries)
        body = response.json()
        self._record_usage(body.get("usage") or {})
        return body["choices"][0]["message"]["content"]

    def _record_usage(self, usage: dict) -> None:
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        # `cached_tokens` (CLAUDE.md Day-1 Phase 3) is stored as 0, never
        # `None`, whether the field is absent entirely or explicitly `null` -
        # both mean "this call had no cache hit", and the audit row's
        # `cached_tokens` side column (`disputedesk/audit/models.py`) is meant
        # to be summed across a dispute's calls, which a stray `None` would
        # break. Contrast with `reasoning_tokens` above, which stays `None`
        # on absence - that field is diagnostic-only, never summed or
        # persisted to the audit row, so there is no such requirement on it.
        cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        record = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
        }
        self.usage_log.append(record)
        logger.info(
            "groq usage model=%s reasoning_effort=%s prompt_tokens=%s "
            "completion_tokens=%s reasoning_tokens=%s cached_tokens=%s (ceiling=%s)",
            self._model,
            self._reasoning_effort,
            record["prompt_tokens"],
            record["completion_tokens"],
            record["reasoning_tokens"],
            record["cached_tokens"],
            self.MAX_COMPLETION_TOKENS,
        )
