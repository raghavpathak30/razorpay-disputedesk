"""The policy engine: the only thing in this system that decides contest,
accept, or escalate (SPEC.md §4, CLAUDE.md invariant 4). Pure and
deterministic - `decide` takes only the model's `P(win)` and the dispute
amount. It has no knowledge of reason codes, evidence, or free text, and
never calls or is called by anything in `evidence/`.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict

from disputedesk.policy.config import PolicyConfig


class Decision(Enum):
    """Plain `Enum`, deliberately not `(str, Enum)`: numpy silently garbles a
    str-subclass enum member during a vectorized `==` comparison or array
    construction (it coerces the member through `Enum.__str__`, e.g.
    `"Decision.ESCALATE"`, truncated, instead of its value) - a real bug hit
    while writing `eval.business_metrics`'s vectorized decision counting.
    Pydantic v2 still serializes a plain `Enum` field by its `.value`.
    """

    CONTEST = "contest"
    ACCEPT = "accept"
    ESCALATE = "escalate"


class PolicyDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: Decision
    p_win: float
    amount: float
    representment_cost_inr: float
    expected_value_inr: float
    low_confidence: bool


def decide(p_win: float, amount: float, config: PolicyConfig | None = None) -> PolicyDecision:
    """SPEC.md §4's decision rule, exactly:

        expected_value = P(win) * amount - representment_cost
        if confidence_band is LOW:        -> escalate_to_human
        elif expected_value > 0:          -> contest
        else:                             -> accept

    The band check runs first: a low-confidence `p_win` escalates regardless
    of what `expected_value` says. `expected_value == 0` is not a positive
    value, so it accepts, not contests.
    """
    config = config or PolicyConfig()
    expected_value = p_win * amount - config.representment_cost_inr
    low, high = config.low_confidence_band
    low_confidence = low <= p_win <= high

    if low_confidence:
        decision = Decision.ESCALATE
    elif expected_value > 0:
        decision = Decision.CONTEST
    else:
        decision = Decision.ACCEPT

    return PolicyDecision(
        decision=decision,
        p_win=p_win,
        amount=amount,
        representment_cost_inr=config.representment_cost_inr,
        expected_value_inr=expected_value,
        low_confidence=low_confidence,
    )
