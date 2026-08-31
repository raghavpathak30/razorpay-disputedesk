"""Unit tests for the policy engine (SPEC.md §4, PHASES.md Phase 3 gate):
every branch, plus the boundary cases the gate explicitly asks for -
expected value near zero, and confidence at the band edges.
"""

from disputedesk.policy.config import PolicyConfig
from disputedesk.policy.engine import Decision, decide

CONFIG = PolicyConfig(representment_cost_inr=400.0, low_confidence_band=(0.45, 0.55))


def test_contests_when_expected_value_is_clearly_positive():
    result = decide(p_win=0.8, amount=5000.0, config=CONFIG)
    assert result.decision == Decision.CONTEST
    assert result.expected_value_inr == 0.8 * 5000.0 - 400.0
    assert result.low_confidence is False


def test_accepts_when_expected_value_is_clearly_negative():
    result = decide(p_win=0.1, amount=1000.0, config=CONFIG)
    assert result.decision == Decision.ACCEPT
    assert result.expected_value_inr == 0.1 * 1000.0 - 400.0
    assert result.low_confidence is False


def test_escalates_when_confidence_is_low_even_with_positive_expected_value():
    # p_win=0.5 is inside the band, but 0.5 * 10000 - 400 = 4600 > 0: the
    # band check must win over a positive expected_value.
    result = decide(p_win=0.5, amount=10000.0, config=CONFIG)
    assert result.decision == Decision.ESCALATE
    assert result.low_confidence is True


def test_escalates_when_confidence_is_low_even_with_negative_expected_value():
    result = decide(p_win=0.5, amount=100.0, config=CONFIG)
    assert result.decision == Decision.ESCALATE
    assert result.low_confidence is True


def test_low_confidence_band_lower_edge_is_inclusive():
    result = decide(p_win=0.45, amount=5000.0, config=CONFIG)
    assert result.decision == Decision.ESCALATE


def test_low_confidence_band_upper_edge_is_inclusive():
    result = decide(p_win=0.55, amount=5000.0, config=CONFIG)
    assert result.decision == Decision.ESCALATE


def test_just_below_the_band_is_not_escalated():
    result = decide(p_win=0.4499, amount=5000.0, config=CONFIG)
    assert result.decision != Decision.ESCALATE
    assert result.low_confidence is False


def test_just_above_the_band_is_not_escalated():
    result = decide(p_win=0.5501, amount=5000.0, config=CONFIG)
    assert result.decision != Decision.ESCALATE
    assert result.low_confidence is False


def test_expected_value_exactly_zero_accepts_not_contests():
    # p_win * amount == representment_cost exactly, and p_win is outside the
    # confidence band - expected_value == 0 is not "> 0", so this accepts.
    amount = 1000.0
    p_win = CONFIG.representment_cost_inr / amount  # 0.4, outside (0.45, 0.55)
    result = decide(p_win=p_win, amount=amount, config=CONFIG)
    assert result.expected_value_inr == 0.0
    assert result.decision == Decision.ACCEPT


def test_expected_value_just_above_zero_contests():
    amount = 1000.0
    p_win = (CONFIG.representment_cost_inr / amount) + 0.001
    result = decide(p_win=p_win, amount=amount, config=CONFIG)
    assert result.decision == Decision.CONTEST


def test_expected_value_just_below_zero_accepts():
    amount = 1000.0
    p_win = (CONFIG.representment_cost_inr / amount) - 0.001
    result = decide(p_win=p_win, amount=amount, config=CONFIG)
    assert result.decision == Decision.ACCEPT


def test_default_config_is_used_when_none_is_passed():
    result = decide(p_win=0.9, amount=5000.0)
    assert result.representment_cost_inr == PolicyConfig().representment_cost_inr


def test_decision_result_carries_the_inputs_for_audit():
    result = decide(p_win=0.8, amount=5000.0, config=CONFIG)
    assert result.p_win == 0.8
    assert result.amount == 5000.0
    assert result.representment_cost_inr == 400.0
