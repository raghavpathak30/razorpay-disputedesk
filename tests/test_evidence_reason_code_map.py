"""The reason-code -> evidence-types lookup is a deterministic table (SPEC.md
§2): every real generator reason code must resolve, and an unknown code must
fail loudly rather than silently return an empty or guessed evidence set.
"""

import pytest

from disputedesk.evidence.reason_code_map import (
    REQUIRED_EVIDENCE_BY_REASON_CODE,
    required_evidence_types,
)
from disputedesk.features.build import REASON_CODES


@pytest.mark.parametrize("reason_code", REASON_CODES)
def test_every_generator_reason_code_has_a_mapping(reason_code):
    evidence_types = required_evidence_types(reason_code)
    assert isinstance(evidence_types, tuple)
    assert len(evidence_types) > 0


def test_explanation_letter_is_always_required():
    # The LLM's drafted output has to actually be used by every packet, or
    # Part C's LLM boundary is dead code.
    for evidence_types in REQUIRED_EVIDENCE_BY_REASON_CODE.values():
        assert "explanation_letter" in evidence_types


def test_unknown_reason_code_raises_instead_of_guessing():
    with pytest.raises(KeyError):
        required_evidence_types("NOT_A_REAL_CODE")


def test_mapping_covers_exactly_the_generator_vocabulary():
    assert set(REQUIRED_EVIDENCE_BY_REASON_CODE) == set(REASON_CODES)
