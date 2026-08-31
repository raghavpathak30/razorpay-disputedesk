"""Shared "call the LLM, validate, repair once" mechanics (SPEC.md §7 failure
path 2, PHASES.md Phase 3 gate). Used by both `normalize_comms.py` and
`draft_letter.py` so the repair behaviour is identical and tested once.
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


def call_llm_and_validate(
    llm_client: LLMClient, prompt: str, schema: type[BaseModel]
) -> BaseModel | None:
    """Call `llm_client` with `prompt`, parse the result as JSON, and
    validate it against `schema`. On failure, make exactly one repair
    attempt, feeding the validation/parse error back to the model. Returns
    `None` if both attempts fail - the caller is responsible for the
    deterministic template fallback and setting a human-review flag
    (SPEC.md §7: "the system degrades, it does not crash").
    """
    raw = llm_client.complete(prompt)
    try:
        return _parse(raw, schema)
    except (json.JSONDecodeError, ValidationError) as first_error:
        repair_prompt = (
            prompt + "\n\n" + load_prompt("repair_addendum_v1").format(error=str(first_error))
        )
        raw_repair = llm_client.complete(repair_prompt)
        try:
            return _parse(raw_repair, schema)
        except (json.JSONDecodeError, ValidationError):
            return None
