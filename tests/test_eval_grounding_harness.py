"""The scoring harness that drives both arms over the corpus.

Driven entirely by `FakeLLMClient` - no network. What is pinned here is the
scoring contract, not the model's judgment: that a gate failure counts as a
flag (because that is what production does), that "correct" is oriented so
higher is better on every class, and that the report always carries n.
"""

import numpy as np
import pytest

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.llm import FakeLLMClient
from eval.grounding_corpus import CorpusItem, build_corpus
from eval.grounding_eval import compare_class, false_flag_rate, report, score_corpus, score_item

CONTEXT = DisputeContext(
    reason_code="MC_4837",
    amount=4999.0,
    avs_match=True,
    cvv_match=True,
    device_fingerprint_known=True,
    delivery_confirmed=True,
    prior_order_count=6,
)
LETTER = (
    "This letter responds to a chargeback under reason code MC_4837 for a transaction "
    "of INR 4999.00. Address Verification Service matched the billing address. "
    "Delivery of the order was confirmed. This customer has 6 prior orders."
)

GROUNDED = (
    '{"assertions": [{"quote": "q", "supporting_field": "avs_match", "verdict": "supported"}]}'
)
UNGROUNDED = '{"assertions": [{"quote": "q", "supporting_field": null, "verdict": "unsupported"}]}'


def _item(item_class="clean", text=LETTER) -> CorpusItem:
    return CorpusItem("i0", text, CONTEXT, item_class, None)


class TestScoring:
    def test_a_grounded_verdict_is_not_a_flag(self):
        score = score_item(_item(), FakeLLMClient([GROUNDED]))
        assert score.gate_flagged is False
        assert score.gate_failed is False

    def test_an_ungrounded_verdict_is_a_flag(self):
        score = score_item(_item(), FakeLLMClient([UNGROUNDED]))
        assert score.gate_flagged is True
        assert score.gate_failed is False

    def test_a_gate_failure_counts_as_a_flag(self):
        """Production withholds a letter the gate could not grade. Scoring it
        any other way would report a gate that is not the one being shipped."""
        score = score_item(_item(), FakeLLMClient(["not json"]))
        assert score.gate_flagged is True
        assert score.gate_failed is True

    def test_the_baseline_arm_is_scored_on_the_same_item(self):
        dirty = _item("unrecorded", LETTER + " Tracking reference 1Z9X4A2210 was scanned.")
        score = score_item(dirty, FakeLLMClient([GROUNDED]))
        assert score.baseline_flagged is True
        assert score.gate_flagged is False  # the fake said grounded; the arms disagree

    def test_score_corpus_returns_one_row_per_item(self):
        items = build_corpus([(LETTER, CONTEXT)] * 3, seed=0)
        scores = score_corpus(items, FakeLLMClient([GROUNDED]))
        assert len(scores) == len(items)
        assert set(scores["item_id"]) == {i.item_id for i in items}


class TestOrientation:
    def _scores(self, gate_flags, baseline_flags, item_class):
        import pandas as pd

        return pd.DataFrame(
            {
                "item_id": [f"i{i}" for i in range(len(gate_flags))],
                "item_class": [item_class] * len(gate_flags),
                "mutation": [None] * len(gate_flags),
                "gate_flagged": gate_flags,
                "baseline_flagged": baseline_flags,
                "gate_failed": [False] * len(gate_flags),
                "n_assertions": [1] * len(gate_flags),
            }
        )

    def test_on_a_positive_class_flagging_is_correct(self):
        scores = self._scores([True] * 8 + [False] * 2, [False] * 10, "unrecorded")
        result = compare_class(scores, "unrecorded")
        assert result.gate.value == pytest.approx(0.8)
        assert result.baseline.value == pytest.approx(0.0)
        assert result.difference > 0

    def test_on_the_clean_class_not_flagging_is_correct(self):
        """Orientation flip: the same raw flags now mean the opposite, so a
        gate that flags 8 of 10 clean letters must score 0.2, not 0.8."""
        scores = self._scores([True] * 8 + [False] * 2, [False] * 10, "clean")
        result = compare_class(scores, "clean")
        assert result.gate.value == pytest.approx(0.2)
        assert result.baseline.value == pytest.approx(1.0)
        assert result.difference < 0

    def test_false_flag_rate_reads_the_clean_class_directly(self):
        scores = self._scores([True] * 3 + [False] * 17, [False] * 20, "clean")
        rate = false_flag_rate(scores)
        assert rate.value == pytest.approx(0.15)
        assert rate.denominator == 20

    def test_an_absent_class_raises_rather_than_reporting_an_empty_rate(self):
        scores = self._scores([True], [False], "clean")
        with pytest.raises(ValueError, match="no items of class"):
            compare_class(scores, "unrecorded")


class TestReport:
    def test_the_report_carries_n_and_every_class(self):
        items = build_corpus([(LETTER, CONTEXT)] * 4, seed=0)
        scores = score_corpus(items, FakeLLMClient([UNGROUNDED]))
        text = report(items, scores)
        assert "n=" in text
        assert "false-flag on clean letters" in text
        assert "Class A" in text and "Class B" in text
        assert "could not reach a verdict" in text

    def test_the_report_places_the_false_flag_rate_against_the_budget(self):
        """DECISIONS.md 2026-09-03: the false-flag rate must be placed against
        the review-cost budget explicitly, in the same table, not left for a
        reader to compute from a separate module."""
        items = build_corpus([(LETTER, CONTEXT)] * 4, seed=0)
        scores = score_corpus(items, FakeLLMClient([GROUNDED]))
        text = report(items, scores)
        assert "budget at INR" in text
        assert "CLEARS" in text or "MISSES" in text or "STRADDLES" in text

    def test_token_usage_is_reported_per_letter_when_available(self):
        items = build_corpus([(LETTER, CONTEXT)] * 2, seed=0)
        scores = score_corpus(items, FakeLLMClient([GROUNDED]))
        usage = [{"prompt_tokens": 600, "completion_tokens": 300}] * len(scores)
        text = report(items, scores, usage=usage)
        assert "900 per letter" in text

    def test_no_network_client_is_constructed_anywhere_in_this_file(self):
        """CLAUDE.md: no test makes a network call. Pins that the harness
        needs only an `LLMClient`, so a fake fully drives it."""
        items = build_corpus([(LETTER, CONTEXT)], seed=0)
        client = FakeLLMClient([GROUNDED])
        score_corpus(items, client)
        assert client.call_count == len(items)


class TestCorpusScoringIsPaired:
    def test_both_arms_see_the_identical_item_list(self):
        items = build_corpus([(LETTER, CONTEXT)] * 5, seed=3)
        scores = score_corpus(items, FakeLLMClient([GROUNDED]))
        assert list(scores["item_id"]) == [i.item_id for i in items]
        assert list(scores["item_class"]) == [i.item_class for i in items]
        assert np.all(scores["gate_flagged"].to_numpy() == False)  # noqa: E712
