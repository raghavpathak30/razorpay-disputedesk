"""Checkpointing and resume logic for `eval.run_grounding_draft` (found while
running the real n=250 key run, 2026-09-03).

The script originally held every drafted row in memory and wrote the output
CSV once, after the full loop. A live run crashed on an exhausted-retries 429
at letter 167 of 250 - all 167 successful, budget-consuming API calls were
lost, because nothing had been persisted. This module tests the pure
resume/merge logic; the live drafting loop itself is untested here, per this
project's convention for scripts that make real network calls (see this
script's own module docstring).
"""

import pandas as pd

from eval.run_grounding_draft import already_drafted_positions, merge_and_write


def test_already_drafted_positions_is_empty_when_no_file_exists(tmp_path):
    assert already_drafted_positions(tmp_path / "does_not_exist.csv") == set()


def test_already_drafted_positions_reads_existing_checkpoint(tmp_path):
    path = tmp_path / "partial.csv"
    pd.DataFrame({"draft_index": [0, 1, 2], "letter_text": ["a", "b", "c"]}).to_csv(
        path, index=False
    )

    assert already_drafted_positions(path) == {0, 1, 2}


def test_merge_and_write_appends_new_rows_to_an_existing_checkpoint(tmp_path):
    path = tmp_path / "partial.csv"
    pd.DataFrame({"draft_index": [0, 1], "letter_text": ["a", "b"]}).to_csv(path, index=False)

    merge_and_write(path, [{"draft_index": 2, "letter_text": "c"}])

    result = pd.read_csv(path)
    assert list(result["draft_index"]) == [0, 1, 2]
    assert list(result["letter_text"]) == ["a", "b", "c"]


def test_merge_and_write_creates_a_fresh_file_when_none_exists(tmp_path):
    path = tmp_path / "fresh.csv"

    merge_and_write(path, [{"draft_index": 0, "letter_text": "a"}])

    result = pd.read_csv(path)
    assert list(result["draft_index"]) == [0]


def test_merge_and_write_does_not_duplicate_a_row_written_twice(tmp_path):
    """A crash immediately after a row is flushed but before some other bookkeeping
    step must not double-write that row on the next resume."""
    path = tmp_path / "partial.csv"
    pd.DataFrame({"draft_index": [0], "letter_text": ["a"]}).to_csv(path, index=False)

    merge_and_write(path, [{"draft_index": 0, "letter_text": "a"}])

    result = pd.read_csv(path)
    assert len(result) == 1
