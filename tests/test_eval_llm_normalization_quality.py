"""Pure post-processing for the LLM-normalization-quality check
(`eval/llm_normalization_quality.py`): feature extraction and the AUC
computation, tested with synthetic data - never a real LLM call.
`run_llm_normalization_sample` is exercised only with `FakeLLMClient`
(CLAUDE.md: no test may make a network call); the real measurement against
the live API is a one-off, recorded in DECISIONS.md, not covered here.
"""

import json

import httpx
import numpy as np
import pytest

from disputedesk.evidence.llm import FakeLLMClient
from disputedesk.evidence.schemas import NormalizedCommunicationLog
from disputedesk.generator.config import GeneratorConfig
from eval.llm_normalization_quality import (
    FEATURE_COLUMNS,
    auc_of_normalized_fields,
    normalized_to_feature_vector,
    run_llm_normalization_sample,
)


def _rate_limit_error(retry_after: str = "0") -> httpx.HTTPStatusError:
    # Built entirely in-memory - no network call - to simulate what
    # `GroqHttpLLMClient.complete` raises on a real HTTP 429.
    request = httpx.Request("POST", "https://example.invalid/chat/completions")
    response = httpx.Response(429, request=request, headers={"retry-after": retry_after})
    return httpx.HTTPStatusError("429 Too Many Requests", request=request, response=response)


class _FlakyThenValidLLMClient:
    """Raises a 429 for the first `fail_times` calls, then behaves like
    `FakeLLMClient` for the rest - a test double for the retry path only;
    never touches the network.
    """

    def __init__(self, fail_times: int, responses: list[str]):
        self._fail_times = fail_times
        self._responses = responses
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise _rate_limit_error()
        index = min(self.call_count - self._fail_times - 1, len(self._responses) - 1)
        return self._responses[index]


def _normalized(**overrides) -> NormalizedCommunicationLog:
    defaults = dict(
        claims_unauthorized_transaction=False,
        mentions_prior_bank_contact=False,
        mentions_shared_card_access=False,
        mentions_travel=False,
        tone="neutral",
        is_substantive=True,
        summary="x",
    )
    return NormalizedCommunicationLog(**{**defaults, **overrides})


def test_normalized_to_feature_vector_encodes_every_typed_field():
    normalized = _normalized(
        claims_unauthorized_transaction=True,
        mentions_prior_bank_contact=True,
        mentions_shared_card_access=False,
        mentions_travel=False,
        tone="polite",
        is_substantive=True,
    )
    vector = normalized_to_feature_vector(normalized)

    assert set(vector) == set(FEATURE_COLUMNS)
    assert vector["claims_unauthorized_transaction"] == 1
    assert vector["mentions_prior_bank_contact"] == 1
    assert vector["mentions_shared_card_access"] == 0
    assert vector["tone_polite"] == 1
    assert vector["tone_terse"] == 0


def test_neutral_tone_sets_neither_one_hot_column():
    vector = normalized_to_feature_vector(_normalized(tone="neutral"))
    assert vector["tone_polite"] == 0
    assert vector["tone_terse"] == 0


def test_terse_tone_sets_only_the_terse_column():
    vector = normalized_to_feature_vector(_normalized(tone="terse"))
    assert vector["tone_polite"] == 0
    assert vector["tone_terse"] == 1


def test_auc_recovers_a_perfectly_separable_synthetic_signal():
    # One feature exactly equals the label - a sanity check that the
    # methodology itself (not the LLM) can detect a real signal when one
    # exists, before trusting it on ambiguous real output.
    n = 40
    true_fraud = [i % 2 == 0 for i in range(n)]
    feature_rows = [
        {
            col: (1 if col == "claims_unauthorized_transaction" and label else 0)
            for col in FEATURE_COLUMNS
        }
        for label in true_fraud
    ]
    result = auc_of_normalized_fields(feature_rows, true_fraud, n_splits=5, random_state=0)

    assert result["n"] == n
    assert result["mean_auc"] > 0.95


def test_auc_is_near_chance_for_a_feature_independent_of_the_label():
    rng = np.random.RandomState(0)
    n = 60
    true_fraud = list(rng.random(n) < 0.5)
    feature_rows = [{col: int(rng.random() < 0.5) for col in FEATURE_COLUMNS} for _ in range(n)]
    result = auc_of_normalized_fields(feature_rows, true_fraud, n_splits=5, random_state=0)

    assert 0.25 < result["mean_auc"] < 0.75


def test_run_llm_normalization_sample_returns_expected_shape_using_fake_client():
    valid_response = json.dumps(
        {
            "claims_unauthorized_transaction": True,
            "mentions_prior_bank_contact": False,
            "mentions_shared_card_access": False,
            "mentions_travel": False,
            "tone": "polite",
            "is_substantive": True,
            "summary": "Customer disputes the charge.",
        }
    )
    client = FakeLLMClient(responses=[valid_response])

    sample = run_llm_normalization_sample(
        n_rows=6, seed=0, generator_config=GeneratorConfig(), llm_client=client, sleep_seconds=0.0
    )

    assert len(sample) == 6
    assert set(FEATURE_COLUMNS) <= set(sample.columns)
    assert "true_fraud" in sample.columns
    assert "human_review_required" in sample.columns
    assert sample["human_review_required"].eq(False).all()


def test_run_llm_normalization_sample_flags_human_review_when_the_llm_fails():
    client = FakeLLMClient(responses=["broken", "still broken"])

    sample = run_llm_normalization_sample(
        n_rows=3, seed=0, generator_config=GeneratorConfig(), llm_client=client, sleep_seconds=0.0
    )

    assert sample["human_review_required"].eq(True).all()


def test_run_llm_normalization_sample_recovers_from_a_transient_429():
    valid_response = json.dumps(
        {
            "claims_unauthorized_transaction": True,
            "mentions_prior_bank_contact": False,
            "mentions_shared_card_access": False,
            "mentions_travel": False,
            "tone": "neutral",
            "is_substantive": True,
            "summary": "Customer disputes the charge.",
        }
    )
    # Fails twice (rate limited) then succeeds - within the default 3 retries.
    client = _FlakyThenValidLLMClient(fail_times=2, responses=[valid_response])

    sample = run_llm_normalization_sample(
        n_rows=1,
        seed=0,
        generator_config=GeneratorConfig(),
        llm_client=client,
        sleep_seconds=0.0,
        retry_sleep_seconds=0.0,
    )

    assert len(sample) == 1
    assert bool(sample["human_review_required"].iloc[0]) is False
    assert client.call_count == 3


def test_run_llm_normalization_sample_gives_up_after_max_429_retries():
    client = _FlakyThenValidLLMClient(fail_times=99, responses=["unused"])

    with pytest.raises(httpx.HTTPStatusError):
        run_llm_normalization_sample(
            n_rows=1,
            seed=0,
            generator_config=GeneratorConfig(),
            llm_client=client,
            sleep_seconds=0.0,
            max_429_retries=2,
            retry_sleep_seconds=0.0,
        )
