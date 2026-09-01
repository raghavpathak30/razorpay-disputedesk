"""The shared validate-then-repair-once mechanics (SPEC.md §7 failure path
2), tested directly against a toy schema so the behaviour isn't entangled
with either caller's prompt content. No network call - `FakeLLMClient` only.
"""

import json

from pydantic import BaseModel

from disputedesk.evidence.llm import FakeLLMClient
from disputedesk.evidence.validated_call import call_llm_and_validate, validate_or_repair


class _ToySchema(BaseModel):
    value: int


def test_valid_first_response_is_returned_without_a_repair_call():
    client = FakeLLMClient(responses=[json.dumps({"value": 1})])
    result = call_llm_and_validate(client, "prompt", _ToySchema)
    assert result == _ToySchema(value=1)
    assert client.call_count == 1


def test_validate_or_repair_makes_no_call_at_all_when_the_given_response_is_valid():
    # The caller already has a completion (e.g. one it fetched itself, or
    # for logging) - validating it must not cost a second API call for the
    # same logical request.
    client = FakeLLMClient(responses=["should never be used"])
    result = validate_or_repair(client, "prompt", json.dumps({"value": 7}), _ToySchema)
    assert result == _ToySchema(value=7)
    assert client.call_count == 0


def test_validate_or_repair_makes_exactly_one_call_to_repair_an_invalid_given_response():
    client = FakeLLMClient(responses=[json.dumps({"value": 8})])
    result = validate_or_repair(client, "prompt", "not json", _ToySchema)
    assert result == _ToySchema(value=8)
    assert client.call_count == 1


def test_invalid_then_valid_response_succeeds_via_one_repair_attempt():
    client = FakeLLMClient(responses=["not json", json.dumps({"value": 2})])
    result = call_llm_and_validate(client, "prompt", _ToySchema)
    assert result == _ToySchema(value=2)
    assert client.call_count == 2


def test_invalid_twice_returns_none_after_exactly_one_repair_attempt():
    client = FakeLLMClient(responses=["not json", "still not json"])
    result = call_llm_and_validate(client, "prompt", _ToySchema)
    assert result is None
    assert client.call_count == 2


def test_schema_mismatch_triggers_repair_same_as_bad_json():
    client = FakeLLMClient(responses=[json.dumps({"wrong_field": 1}), json.dumps({"value": 3})])
    result = call_llm_and_validate(client, "prompt", _ToySchema)
    assert result == _ToySchema(value=3)


def test_markdown_code_fence_is_stripped_before_parsing():
    fenced = "```json\n" + json.dumps({"value": 5}) + "\n```"
    client = FakeLLMClient(responses=[fenced])
    result = call_llm_and_validate(client, "prompt", _ToySchema)
    assert result == _ToySchema(value=5)
    assert client.call_count == 1
