"""Shared "call the LLM, validate, repair once" mechanics (SPEC.md §7 failure
path 2, PHASES.md Phase 3 gate). Used by both `normalize_comms.py` and
`draft_letter.py` so the repair behaviour is identical and tested once.

Fetching and validating are two separate functions (`call_llm_and_validate`
delegates to `validate_or_repair`) precisely so a caller with a completion
already in hand never has to make a second, redundant call just to run it
through validation - one logical normalisation costs exactly one API call
when the first response validates, two only when it needs repair.
"""

import json

from pydantic import BaseModel, ValidationError

from disputedesk.evidence.llm import LLMClient
from disputedesk.evidence.prompts import load_prompt


def _strip_code_fence(text: str) -> str:
    """LLMs frequently wrap JSON in a ```json ... ``` fence despite being
    told not to; tolerate it rather than fail validation on formatting.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _parse(text: str, schema: type[BaseModel]) -> BaseModel:
    payload = json.loads(_strip_code_fence(text))
    return schema.model_validate(payload)


def validate_or_repair(
    llm_client: LLMClient,
    prompt: str,
    raw_response: str,
    schema: type[BaseModel],
    *,
    repair: bool = True,
) -> BaseModel | None:
    """Validate `raw_response` - a completion already obtained for `prompt` -
    against `schema`. Makes **no** call to `llm_client` if `raw_response`
    validates; on failure, makes exactly one repair call, feeding the
    validation/parse error back to the model. Returns `None` if the repair
    attempt also fails - the caller is responsible for the deterministic
    template fallback and setting a human-review flag (SPEC.md §7: "the
    system degrades, it does not crash").

    `repair` (default `True`, today's only behaviour) governs only what
    happens on a *validation failure*: when `False`, a malformed/invalid
    first response returns `None` immediately - same return value and same
    caller-side fallback as a failed repair attempt, just without spending a
    second call to get there. Used by `draft_letter.py` (CLAUDE.md Day-1
    Phase 2) once the drafting call constrains output at the sampling layer
    via `response_format`, making a "the model returned malformed JSON, ask
    it again" retry redundant for that call specifically -
    `normalize_comms.py` and `grounding.py` keep the default and are
    unaffected (`tests/test_evidence_validated_call.py` pins this).

    Split out from `call_llm_and_validate` so a caller that already has a
    completion (e.g. one it fetched for logging, or a connectivity check)
    can validate it without a second, redundant API call for the same
    logical request.
    """
    try:
        return _parse(raw_response, schema)
    except (json.JSONDecodeError, ValidationError) as first_error:
        if not repair:
            return None
        repair_prompt = (
            prompt + "\n\n" + load_prompt("repair_addendum_v1").format(error=str(first_error))
        )
        raw_repair = llm_client.complete(repair_prompt)
        try:
            return _parse(raw_repair, schema)
        except (json.JSONDecodeError, ValidationError):
            return None


def call_llm_and_validate(
    llm_client: LLMClient,
    prompt: str,
    schema: type[BaseModel],
    *,
    response_format: dict | None = None,
    repair: bool = True,
) -> BaseModel | None:
    """Call `llm_client` with `prompt` exactly once, then validate the result
    via `validate_or_repair` (which makes a second call only on failure, as
    the one repair attempt, and only when `repair` is `True`).

    `response_format` (default `None`, today's only behaviour) is passed to
    `llm_client.complete` verbatim - see `LLMClient.complete`'s docstring.
    """
    raw = llm_client.complete(prompt, response_format=response_format)
    return validate_or_repair(llm_client, prompt, raw, schema, repair=repair)
