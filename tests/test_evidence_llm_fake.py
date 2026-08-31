"""`FakeLLMClient` is what makes every other evidence/ test network-free
(CLAUDE.md: no test may make a network call). These tests are about the fake
itself: it must return responses in order and repeat the last one, so tests
can queue a broken-then-good pair to exercise the one-repair-attempt path.
"""

import pytest

from disputedesk.evidence.llm import FakeLLMClient


def test_returns_responses_in_order():
    client = FakeLLMClient(responses=["first", "second"])
    assert client.complete("prompt") == "first"
    assert client.complete("prompt") == "second"


def test_repeats_the_last_response_once_exhausted():
    client = FakeLLMClient(responses=["only"])
    assert client.complete("prompt") == "only"
    assert client.complete("prompt") == "only"
    assert client.complete("prompt") == "only"


def test_call_count_tracks_number_of_calls():
    client = FakeLLMClient(responses=["a", "b"])
    assert client.call_count == 0
    client.complete("prompt")
    client.complete("prompt")
    assert client.call_count == 2


def test_requires_at_least_one_response():
    with pytest.raises(ValueError):
        FakeLLMClient(responses=[])
