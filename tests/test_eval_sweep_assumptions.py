"""What the cost sweep assumes, made checkable (remediation item 1.0).

Phase 0 added a `withheld_for_review` outcome: a dispute the policy engine
decided to contest, whose evidence packet was not fit to file, goes to a
person and nothing is sent to the card network. The cost sweep was built when
every CONTEST decision was auto-filed, so it credits recovered value to every
contested row.

Two assumptions therefore sit under every rupee figure in the sweep, and both
are now written down as named constants with tests, rather than left implicit:

1. **Every CONTEST/ESCALATE decision results in a filed representment.** The
   withheld path is excluded from the sweep. One component of the withheld
   rate is measurable and pinned here (reason codes); the other is not
   currently measurable at all.
2. **Every filed representment is accepted for review by Razorpay.** This one
   is currently false for this implementation, and the test below says so
   rather than letting it pass unstated.
"""

import pytest

from disputedesk.evidence.published_reason_codes import is_supported_reason_code
from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset
from eval.cost_sensitivity import (
    SWEEP_ASSUMES_EVERY_SUBMISSION_IS_ACCEPTED,
    break_even_human_review_cost_inr,
)


def test_the_generator_never_produces_a_reason_code_that_would_be_withheld():
    """The reason-code component of the withheld rate is exactly zero on
    every dataset the sweep scores - so excluding it costs the sweep nothing
    on this axis. Asserted, not assumed: a future generator change that added
    an out-of-scope code would silently make the sweep's arm and the running
    system's arm different systems.
    """
    features_df, _debug = generate_dataset(2000, seed=0, config=GeneratorConfig())

    unsupported = {
        code for code in features_df["reason_code"].unique() if not is_supported_reason_code(code)
    }

    assert unsupported == set()


def test_the_document_id_gap_is_recorded_as_a_failing_assumption():
    """Razorpay's contest endpoint requires at least one document id when
    `action="submit"`, and this project has no document-upload pipeline. So
    the sweep's second assumption - that a filed contest is accepted for
    review - does not currently hold, and the constant that records it must
    say `False`, not `True`.

    This test exists to make the flag impossible to flip to `True` casually:
    flipping it means claiming the upload path was built.
    """
    assert SWEEP_ASSUMES_EVERY_SUBMISSION_IS_ACCEPTED is False


def test_break_even_human_review_cost_on_a_hand_computed_case():
    """Advantage 12,000 INR per 1,000 disputes, 6% of rows human-touched
    (60 rows per 1,000): the advantage is wiped out if each of those reviews
    costs 12,000 / 60 = 200 INR.
    """
    assert break_even_human_review_cost_inr(12_000.0, 0.06) == pytest.approx(200.0)


def test_break_even_human_review_cost_falls_as_more_rows_are_human_touched():
    """The direction that matters: any withheld rate above zero adds to the
    human-touched fraction, which lowers the human-review cost the advantage
    can survive. Excluding the withheld path therefore *overstates* the
    advantage - it never understates it.
    """
    escalation_only = break_even_human_review_cost_inr(12_000.0, 0.06)
    plus_withheld = break_even_human_review_cost_inr(12_000.0, 0.09)

    assert plus_withheld < escalation_only


def test_a_zero_human_touched_rate_has_no_finite_break_even():
    assert break_even_human_review_cost_inr(12_000.0, 0.0) == float("inf")


def test_a_non_positive_advantage_has_no_break_even_to_report():
    """If the policy is already behind at this cost point, "how expensive can
    a review be before the advantage disappears" is not a question with an
    answer - it must not come back as a positive-looking number.
    """
    assert break_even_human_review_cost_inr(-5_000.0, 0.06) == 0.0


# --------------------------------------------------------------------------
# The sweep summary reports the paired estimator, and only the paired one
# --------------------------------------------------------------------------


def test_the_summary_no_longer_exposes_a_difference_of_medians():
    """`policy_advantage_median` was `median(policy) - median(baseline_a)` -
    an unpaired statistic over a paired design. It is removed rather than
    left alongside the paired columns, so no caller can quote it by accident.
    """
    from eval.cost_sensitivity import summarize_sweep, sweep_representment_cost
    from eval.harness import fixed_seed_set

    per_seed = sweep_representment_cost(fixed_seed_set(4), 2000, [0.0, 400.0])
    summary = summarize_sweep(per_seed)

    assert "policy_advantage_median" not in summary.columns
    for column in (
        "advantage_paired_mean",
        "advantage_ci_low",
        "advantage_ci_high",
        "advantage_n_positive",
        "n_seeds",
    ):
        assert column in summary.columns


def test_the_paired_mean_matches_a_hand_rolled_computation_per_cost():
    """The summary must be the same arithmetic as doing it by hand on the
    per-seed frame - not a re-derivation that could drift.
    """
    from eval.cost_sensitivity import summarize_sweep, sweep_representment_cost
    from eval.harness import fixed_seed_set

    per_seed = sweep_representment_cost(fixed_seed_set(4), 2000, [400.0])
    summary = summarize_sweep(per_seed)

    at_cost = per_seed[per_seed["representment_cost_inr"] == 400.0].sort_values("seed")
    expected = (
        at_cost["policy_recovered_per_1000_inr"] - at_cost["baseline_a_recovered_per_1000_inr"]
    ).mean()

    assert summary.iloc[0]["advantage_paired_mean"] == pytest.approx(expected)
    assert summary.iloc[0]["n_seeds"] == 4
