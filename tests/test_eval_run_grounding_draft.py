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

from disputedesk.evidence.llm import FakeLLMClient
from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset
from eval.run_grounding_draft import already_drafted_positions, draft_corpus, merge_and_write


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


# --------------------------------------------------------------------------
# Full stop-and-restart resume, with a stub client - no live calls
# --------------------------------------------------------------------------

VALID_NORMALIZED = (
    '{"claims_unauthorized_transaction": true, "mentions_prior_bank_contact": false, '
    '"mentions_shared_card_access": false, "mentions_travel": false, "tone": "polite", '
    '"is_substantive": true, "summary": "Customer disputes the charge."}'
)
VALID_LETTER = '{"letter_text": "%s", "cites_evidence_types": ["billing_proof"]}' % ("x" * 80)


def _stub_client() -> FakeLLMClient:
    # Every call gets a valid response regardless of which prompt it was for -
    # normalize and draft expect different schemas, but FakeLLMClient just
    # replays responses in order, so both are provided per letter and the
    # loop's own call order determines which is consumed when.
    return FakeLLMClient([VALID_NORMALIZED, VALID_LETTER] * 20)


def test_draft_corpus_resumes_after_a_simulated_stop_and_restart(tmp_path):
    """The exact scenario 0.c asks for: draft some letters, stop (a fresh
    process boundary is simulated by calling draft_corpus again with a fresh
    stub client and the same output path), and confirm the second call only
    drafts what the first call left undone - no live network call anywhere.
    """
    out = tmp_path / "letters.csv"
    features_df, _debug = generate_dataset(6, seed=0, config=GeneratorConfig())

    # "Run 1": only get through the first 3 positions, as if a crash (or a
    # deliberate stop) happened after them - simulated by handing
    # draft_corpus a features_df sliced to those positions only.
    draft_corpus(features_df.iloc[:3], out, _stub_client(), sleep_seconds=0.0)
    assert already_drafted_positions(out) == {0, 1, 2}

    # "Run 2": a fresh process, fresh client, same output path, full
    # features_df. Must skip 0-2 and draft only 3-5.
    draft_corpus(features_df, out, _stub_client(), sleep_seconds=0.0)

    result = pd.read_csv(out)
    assert set(result["draft_index"]) == {0, 1, 2, 3, 4, 5}
    assert len(result) == 6  # no duplicates from the re-covered 0-2 range


def test_draft_corpus_makes_zero_calls_when_everything_is_already_done(tmp_path):
    """A third restart, with nothing left to do, must not call the client at
    all - the strongest form of "resume doesn't redo work."
    """
    out = tmp_path / "letters.csv"
    features_df, _debug = generate_dataset(3, seed=0, config=GeneratorConfig())
    draft_corpus(features_df, out, _stub_client(), sleep_seconds=0.0)

    class ExplodingClient:
        def complete(self, prompt: str) -> str:
            raise AssertionError("draft_corpus called the client with nothing left to draft")

    draft_corpus(features_df, out, ExplodingClient(), sleep_seconds=0.0)

    assert len(pd.read_csv(out)) == 3
