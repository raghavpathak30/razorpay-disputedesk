"""The loss-tail analysis at a swept cost (remediation item 2.5).

At ₹50 the paired mean is negative (−131) while 12 of 20 seeds are positive.
Both facts are true, and a report that gave only one of them would mislead in
opposite directions depending on which. The reason they disagree is always the
shape of the per-seed distribution, so that shape gets measured rather than
described.
"""

import pandas as pd
import pytest

from eval.cost_sensitivity import loss_tail


def _frame(differences, cost=50.0):
    """A per-seed frame with a chosen advantage per seed. Baseline A is fixed
    at zero so the difference is exactly what is passed in.
    """
    return pd.DataFrame(
        {
            "seed": range(len(differences)),
            "representment_cost_inr": [cost] * len(differences),
            "policy_recovered_per_1000_inr": list(differences),
            "baseline_a_recovered_per_1000_inr": [0.0] * len(differences),
        }
    )


def test_loss_tail_on_a_hand_computed_case():
    """Four seeds: +10, +20, −60, −40.
    mean = (10+20-60-40)/4 = -17.5; mean gain = 15; mean loss = -50;
    ratio = 50/15 = 3.333...; spread = 20 - (-60) = 80.
    """
    tail = loss_tail(_frame([10.0, 20.0, -60.0, -40.0]), 50.0)

    assert tail.n_seeds == 4
    assert tail.n_positive == 2
    assert tail.n_negative == 2
    assert tail.mean_difference == pytest.approx(-17.5)
    assert tail.best_seed_difference == 20.0
    assert tail.worst_seed_difference == -60.0
    assert tail.spread == 80.0
    assert tail.mean_gain_on_winning_seeds == pytest.approx(15.0)
    assert tail.mean_loss_on_losing_seeds == pytest.approx(-50.0)
    assert tail.loss_to_gain_ratio == pytest.approx(10.0 / 3.0)


def test_a_majority_of_winning_seeds_can_still_have_a_negative_mean():
    """The ₹50 shape in miniature, and the whole reason this exists: three
    seeds win small, two lose big.
    """
    tail = loss_tail(_frame([5.0, 5.0, 5.0, -50.0, -50.0]), 50.0)

    assert tail.n_positive > tail.n_negative
    assert tail.mean_difference < 0
    assert tail.loss_to_gain_ratio == pytest.approx(10.0)


def test_exact_zero_differences_count_as_neither_win_nor_loss():
    """A seed where the policy made identical decisions to baseline A is not
    a win. `n_positive + n_negative` need not equal `n_seeds`.
    """
    tail = loss_tail(_frame([0.0, 0.0, 4.0, -4.0]), 50.0)

    assert tail.n_seeds == 4
    assert tail.n_positive == 1
    assert tail.n_negative == 1


def test_no_losing_seeds_gives_a_zero_ratio():
    tail = loss_tail(_frame([1.0, 2.0, 3.0]), 50.0)

    assert tail.mean_loss_on_losing_seeds == 0.0
    assert tail.loss_to_gain_ratio == 0.0


def test_no_winning_seeds_gives_an_infinite_ratio():
    tail = loss_tail(_frame([-1.0, -2.0]), 50.0)

    assert tail.loss_to_gain_ratio == float("inf")


def test_an_unswept_cost_raises_rather_than_returning_an_empty_summary():
    with pytest.raises(ValueError):
        loss_tail(_frame([1.0, 2.0]), 999.0)


# --------------------------------------------------------------------------
# The measured ₹50 tail, pinned
# --------------------------------------------------------------------------


def test_the_measured_fifty_rupee_tail(tmp_path):
    """GOLDEN FIXTURE - pinned against disputedesk.generator output, seeds
    0-7, n_rows=5000. If eval/generator_fingerprint.py's committed fingerprint
    ever changes, re-run and re-commit these values too (see that module's
    docstring).

    Regression on the real sweep at CI scale. The headline 20-seed x
    15,000-row figures are in DECISIONS.md; these are the same computation at
    a smaller, faster, frozen scale, committed so the shape cannot drift
    unnoticed.
    """
    from disputedesk.generator.config import GeneratorConfig
    from disputedesk.model.config import ModelConfig
    from eval.cost_sensitivity import sweep_representment_cost
    from eval.harness import fixed_seed_set

    per_seed = sweep_representment_cost(
        fixed_seed_set(8), 5000, [50.0], GeneratorConfig(), ModelConfig()
    )
    tail = loss_tail(per_seed, 50.0)

    assert tail.n_seeds == 8
    assert tail.mean_difference == pytest.approx(-112.5258, abs=1e-4)
    assert tail.worst_seed_difference == pytest.approx(-1266.4701, abs=1e-4)
    assert tail.best_seed_difference == pytest.approx(544.5808, abs=1e-4)


def test_the_loss_to_gain_asymmetry_shrinks_as_cost_rises():
    """The interpretation, asserted rather than asserted-in-prose: the
    asymmetry is a low-cost phenomenon. If it ever stopped shrinking, the
    README's account of the curve would be wrong.
    """
    from disputedesk.generator.config import GeneratorConfig
    from disputedesk.model.config import ModelConfig
    from eval.cost_sensitivity import sweep_representment_cost
    from eval.harness import fixed_seed_set

    per_seed = sweep_representment_cost(
        fixed_seed_set(8), 5000, [50.0, 1000.0], GeneratorConfig(), ModelConfig()
    )

    assert (
        loss_tail(per_seed, 50.0).loss_to_gain_ratio
        > loss_tail(per_seed, 1000.0).loss_to_gain_ratio
    )
