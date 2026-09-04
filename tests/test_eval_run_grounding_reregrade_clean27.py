"""Pure selection logic for the second pre-registration's n=27 re-grade
(DECISIONS.md, 2026-09-03 "Second pre-registration"). The live grading loop
makes real network calls and is untested here, per this project's convention
for scripts that call a real LLM (see the script's own module docstring) -
this test only proves the fixed item-id list actually selects 27 items from
the real committed corpus, and that a changed scope is refused rather than
silently run.
"""

import pytest

from eval.grounding_corpus import build_corpus
from eval.run_grounding_draft import CONTEXT_FIELDS
from eval.run_grounding_eval import load_drafts
from eval.run_grounding_reregrade_clean27 import (
    LETTERS_PATH,
    PRE_REGISTERED_DRAFT_INDICES,
    select_pre_registered_items,
)


def test_pre_registered_indices_are_27_unique_values():
    assert len(PRE_REGISTERED_DRAFT_INDICES) == 27
    assert len(set(PRE_REGISTERED_DRAFT_INDICES)) == 27


def test_select_pre_registered_items_finds_exactly_27_in_the_real_corpus():
    """Uses the actual committed letters file and seed - the same corpus the
    live run builds from - so this catches a mismatch between the hardcoded
    id list and what `data/reference/grounding_letters_seed0_n45.csv` and
    `build_corpus(seed=0)` actually produce, with no network call."""
    drafts = load_drafts(LETTERS_PATH)
    all_items = build_corpus(drafts, seed=0)

    selected = select_pre_registered_items(all_items)

    assert len(selected) == 27
    assert all(item.item_class == "clean" for item in selected)
    expected_ids = {f"d{i:04d}_clean" for i in PRE_REGISTERED_DRAFT_INDICES}
    assert {item.item_id for item in selected} == expected_ids


def test_select_pre_registered_items_stops_on_a_missing_id():
    drafts = load_drafts(LETTERS_PATH)
    all_items = build_corpus(drafts, seed=0)
    truncated = [item for item in all_items if item.item_id != "d0028_clean"]

    with pytest.raises(SystemExit, match="d0028_clean"):
        select_pre_registered_items(truncated)


def test_context_fields_matches_load_drafts_columns():
    """Sanity check that `load_drafts` (imported from run_grounding_eval, which
    this script reuses rather than re-implementing) still reads the same
    context fields the corpus builder expects."""
    assert set(CONTEXT_FIELDS) == {
        "reason_code",
        "amount",
        "avs_match",
        "cvv_match",
        "device_fingerprint_known",
        "delivery_confirmed",
        "prior_order_count",
    }
