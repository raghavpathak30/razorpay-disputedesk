"""The grounding-gate evaluation corpus, its deterministic baseline, and the
interval estimators the comparison is reported with.

No network and no LLM anywhere in this file - every arm tested here is either
deterministic or driven by a hand-built verdict.
"""

import numpy as np
import pytest

from disputedesk.evidence.context import DisputeContext
from eval.grounding_baseline import (
    baseline_findings,
    baseline_flags,
    field_contradictions,
    numeric_contradictions,
    unrecorded_entities,
)
from eval.grounding_corpus import (
    FLIPPABLE_FIELDS,
    UNRECORDED_TEMPLATES,
    build_corpus,
    composition,
    make_contradiction,
    make_unrecorded,
    mentioned_fields,
)
from eval.grounding_stats import clopper_pearson, paired_comparison, wilson

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
    "This letter responds to a chargeback filed under reason code MC_4837 for a "
    "transaction of INR 4999.00. Address Verification Service matched the billing "
    "address and the Card Verification Value matched. The device used at checkout "
    "was recognised from prior activity. Delivery of the order was confirmed. "
    "This customer has 6 prior orders on the account."
)


class TestCorpusLabelsAreMechanical:
    def test_a_flip_only_targets_a_field_the_letter_mentions(self):
        """A contradiction the letter does not contain is a mislabelled
        positive and would punish both arms equally for nothing."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            mutated, field = make_contradiction(LETTER, CONTEXT, rng)
            assert field in mentioned_fields(LETTER)
            assert getattr(mutated, field) is not getattr(CONTEXT, field)

    def test_a_flip_changes_exactly_one_field(self):
        rng = np.random.default_rng(1)
        mutated, field = make_contradiction(LETTER, CONTEXT, rng)
        changed = [f for f in FLIPPABLE_FIELDS if getattr(mutated, f) != getattr(CONTEXT, f)]
        assert changed == [field]
        assert mutated.amount == CONTEXT.amount
        assert mutated.prior_order_count == CONTEXT.prior_order_count

    def test_a_letter_mentioning_no_flippable_field_yields_no_class_a_item(self):
        bare = "We ask the issuer to reconsider this chargeback. Thank you for your time."
        assert mentioned_fields(bare) == ()
        assert make_contradiction(bare, CONTEXT, np.random.default_rng(0)) is None
        items = build_corpus([(bare, CONTEXT)], seed=0)
        assert [i.item_class for i in items] == ["clean", "unrecorded"]

    def test_an_insertion_adds_exactly_one_committed_template(self):
        rng = np.random.default_rng(2)
        text, template_id = make_unrecorded(LETTER, rng)
        sentence = dict(UNRECORDED_TEMPLATES)[template_id]
        assert sentence in text
        assert len(text) == len(LETTER) + len(sentence) + 1

    def test_the_corpus_is_reproducible_from_its_seed(self):
        a = build_corpus([(LETTER, CONTEXT)] * 5, seed=7)
        b = build_corpus([(LETTER, CONTEXT)] * 5, seed=7)
        assert [(i.item_id, i.letter_text, i.mutation) for i in a] == [
            (i.item_id, i.letter_text, i.mutation) for i in b
        ]

    def test_only_clean_items_are_unflagged_ground_truth(self):
        items = build_corpus([(LETTER, CONTEXT)] * 3, seed=0)
        for item in items:
            assert item.should_be_flagged == (item.item_class != "clean")

    def test_composition_reports_every_item(self):
        items = build_corpus([(LETTER, CONTEXT)] * 6, seed=0)
        table = composition(items)
        assert table["n_items"] == len(items)
        assert sum(table["by_class"].values()) == len(items)


class TestTheBaselineIsAnHonestAttempt:
    def test_it_does_not_flag_a_clean_letter(self):
        assert baseline_flags(LETTER, CONTEXT) is False

    def test_it_catches_a_flipped_boolean(self):
        rng = np.random.default_rng(0)
        mutated, field = make_contradiction(LETTER, CONTEXT, rng)
        findings = field_contradictions(LETTER, mutated)
        assert any(field in f.detail for f in findings)

    def test_it_catches_a_wrong_amount(self):
        wrong = LETTER.replace("INR 4999.00", "INR 8888.00")
        assert any(f.kind == "contradiction" for f in numeric_contradictions(wrong, CONTEXT))

    def test_it_catches_a_wrong_prior_order_count(self):
        wrong = LETTER.replace("6 prior orders", "19 prior orders")
        assert any(f.kind == "contradiction" for f in numeric_contradictions(wrong, CONTEXT))

    def test_it_catches_at_least_some_unrecorded_shapes(self):
        """It is not a strawman: the shape list is a real attempt at Class B
        and it must land on the obvious cases."""
        caught = [
            template_id
            for template_id, sentence in UNRECORDED_TEMPLATES
            if unrecorded_entities(sentence)
        ]
        assert len(caught) >= 4, f"baseline is too weak to be a fair comparison: {caught}"

    def test_its_scope_limit_is_structural(self):
        """The case the gate exists for: a confident, unremarkable sentence
        asserting a fact no field covers and no shape pattern matches."""
        subtle = "The customer accepted our published terms and conditions at checkout."
        assert unrecorded_entities(subtle) == []
        assert baseline_findings(subtle, CONTEXT) == []


class TestIntervals:
    def test_wilson_stays_inside_the_unit_interval_at_the_boundary(self):
        for numerator in (0, 40):
            rate = wilson(numerator, 40, "edge")
            assert 0.0 <= rate.ci_low <= rate.ci_high <= 1.0

    def test_wilson_matches_a_known_value(self):
        rate = wilson(6, 120, "false-flag")
        assert rate.value == pytest.approx(0.05)
        assert rate.ci_low == pytest.approx(0.023, abs=0.002)
        assert rate.ci_high == pytest.approx(0.105, abs=0.002)

    def test_a_rate_always_carries_its_n(self):
        assert "120" in str(wilson(6, 120, "false-flag"))

    def test_identical_arms_produce_an_interval_that_includes_zero(self):
        arm = np.array([True, False] * 20)
        result = paired_comparison("identical", arm, arm)
        assert result.difference == 0.0
        assert not result.excludes_zero
        assert result.p_value == 1.0

    def test_pairing_is_preserved_across_bootstrap_draws(self):
        """A small consistent edge on paired data must be detectable. Drawing
        the two arms independently would wash it out - the failure
        `eval/extraction_comparison.py` documents for the AUC case."""
        gate = np.array([True] * 30 + [False] * 10)
        baseline = np.array([True] * 22 + [False] * 18)
        result = paired_comparison("small edge", gate, baseline)
        assert result.difference == pytest.approx(0.2)
        assert result.excludes_zero

    def test_mismatched_arm_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            paired_comparison("bad", np.array([True]), np.array([True, False]))


class TestClopperPearson:
    def test_stays_inside_the_unit_interval_at_the_boundary(self):
        for numerator in (0, 45):
            rate = clopper_pearson(numerator, 45, "edge")
            assert 0.0 <= rate.ci_low <= rate.ci_high <= 1.0

    def test_zero_events_gives_a_lower_bound_of_exactly_zero(self):
        rate = clopper_pearson(0, 45, "zero")
        assert rate.ci_low == 0.0
        assert rate.ci_high > 0.0

    def test_all_events_gives_an_upper_bound_of_exactly_one(self):
        rate = clopper_pearson(45, 45, "all")
        assert rate.ci_high == 1.0
        assert rate.ci_low < 1.0

    def test_matches_scipy_binomtest_exact_ci(self):
        """Cross-checked against a second, independent scipy code path
        (`binomtest(...).proportion_ci(method="exact")`), not just this
        module's own Beta-quantile construction re-run on itself."""
        from scipy.stats import binomtest

        expected = binomtest(6, 120).proportion_ci(confidence_level=0.95, method="exact")
        rate = clopper_pearson(6, 120, "known")
        assert rate.ci_low == pytest.approx(expected.low)
        assert rate.ci_high == pytest.approx(expected.high)

    def test_is_wider_than_wilson_at_small_n(self):
        """The documented trade at small n: exact coverage costs width."""
        cp = clopper_pearson(2, 45, "cp")
        w = wilson(2, 45, "wilson")
        assert cp.ci_low <= w.ci_low
        assert cp.ci_high >= w.ci_high

    def test_str_labels_itself_clopper_pearson_not_wilson(self):
        """Rate.__str__ must not mislabel a Clopper-Pearson interval as
        Wilson - both functions build the same Rate class."""
        assert "Clopper-Pearson CI" in str(clopper_pearson(6, 120, "cp"))
        assert "Wilson CI" in str(wilson(6, 120, "w"))
