"""Loads versioned prompt text from `disputedesk/evidence/prompts/*.txt`
(PHASES.md Phase 3 gate: "prompt text lives in versioned files in the repo,
not inline in a function"). Each file name carries its own version suffix
(`_v1`); a prompt change that alters model behaviour should add a new
versioned file rather than editing one in place, so old runs stay
reproducible against the prompt they were actually run with.
"""

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache
def load_prompt(name: str) -> str:
    """`name` is the file stem, e.g. `"explanation_letter_v1"`."""
    return (_PROMPTS_DIR / f"{name}.txt").read_text()
