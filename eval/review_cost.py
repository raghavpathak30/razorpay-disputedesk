"""What the grounding gate's false-flag rate does to the cost sweep.

Why this is a separate module rather than an edit to `eval/cost_sensitivity.py`
(CLAUDE.md: "Do not append to a growing file when a new module is the right
answer"): the sweep answers "what does the policy recover at a given
representment cost", and this answers "at what gate false-flag rate does the
review queue eat that". Different question, different inputs, and this one has
to be readable as a curve over an input the sweep does not have.

**The gap this closes.** DECISIONS.md's 2026-09-02 "Cost sweep assumptions"
entry excluded human-review cost from the sweep because the withheld rate had
two components and only one was measurable:

- the reason-code component, measured at exactly 0% on every dataset the sweep
  scores;
- the letter-drafting component, unmeasured, because the run that would have
  measured it was invalidated when the schema and prompt changed.

The grounding gate adds a third component - letters the gate withholds that
would otherwise have been filed - and unlike the second it is measurable, in
the same units, by `eval/run_grounding_eval.py`. This module takes that rate
and reports what it does to the break-even.

**What is *not* claimed here.** The gate's false-flag rate is an *addition* to
the human-touched rate, not a replacement for the drafting component, which
remains unmeasured. So every break-even below is still an upper bound, for the
same reason the sweep's own is: a component excluded from the denominator can
only make the true break-even lower.

No LLM. No network. `false_flag_rate` is an input, so this module is fully
evaluable without an API key - what needs the key is the measurement that
supplies the argument.
"""

from dataclasses import dataclass

import numpy as np

from eval.cost_sensitivity import break_even_human_review_cost_inr


@dataclass(frozen=True)
class ReviewCostPoint:
    """The break-even review cost at one assumed gate false-flag rate."""

    false_flag_rate: float
    escalate_rate: float
    contest_rate: float
    human_touched_rate: float
    advantage_per_1000_inr: float
    break_even_review_cost_inr: float

    @property
    def survives_at(self) -> str:
        return f"a review costing less than about INR {self.break_even_review_cost_inr:,.0f}"


def human_touched_rate(escalate_rate: float, contest_rate: float, false_flag_rate: float) -> float:
    """The fraction of holdout disputes a person must handle with no automated
    filing produced.

    The gate can only withhold letters that were going to be *filed*, so its
    false-flag rate applies to the CONTEST share, not to the whole holdout -
    an escalated dispute never reaches the gate (no evidence is assembled on
    that branch) and an accepted one has no letter. Multiplying the rate
    across the full population instead would overstate the gate's cost by
    roughly the reciprocal of the contest rate.

    The two terms cannot double-count: a dispute is on exactly one policy
    branch, so a row counted in `escalate_rate` is not in `contest_rate`.
    """
    for name, value in (
        ("escalate_rate", escalate_rate),
        ("contest_rate", contest_rate),
        ("false_flag_rate", false_flag_rate),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a rate in [0, 1], got {value}")
    if escalate_rate + contest_rate > 1.0 + 1e-9:
        raise ValueError(
            f"escalate_rate + contest_rate must not exceed 1, got {escalate_rate} + {contest_rate}"
        )
    return escalate_rate + contest_rate * false_flag_rate


def review_cost_point(
    false_flag_rate: float,
    escalate_rate: float,
    contest_rate: float,
    advantage_per_1000_inr: float,
) -> ReviewCostPoint:
    touched = human_touched_rate(escalate_rate, contest_rate, false_flag_rate)
    return ReviewCostPoint(
        false_flag_rate=false_flag_rate,
        escalate_rate=escalate_rate,
        contest_rate=contest_rate,
        human_touched_rate=touched,
        advantage_per_1000_inr=advantage_per_1000_inr,
        break_even_review_cost_inr=break_even_human_review_cost_inr(
            advantage_per_1000_inr, touched
        ),
    )


def review_cost_curve(
    escalate_rate: float,
    contest_rate: float,
    advantage_per_1000_inr: float,
    false_flag_rates=(0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50),
) -> list[ReviewCostPoint]:
    """The break-even as a curve over the gate's false-flag rate.

    Reported as a curve rather than a single number because the rate itself
    needs a live API key to measure. A reader can locate the measured rate on
    this curve when it exists; until then the curve says what the answer would
    be for any value it could take, which is strictly more informative than
    waiting.
    """
    return [
        review_cost_point(rate, escalate_rate, contest_rate, advantage_per_1000_inr)
        for rate in false_flag_rates
    ]


def false_flag_rate_at_review_cost(
    review_cost_inr: float,
    escalate_rate: float,
    contest_rate: float,
    advantage_per_1000_inr: float,
) -> float:
    """The gate false-flag rate at which the policy's advantage is exactly
    cancelled, given what one review actually costs.

    This is the number to quote when the review cost is known and the gate's
    rate is not - the inverse of `review_cost_point`, and the more useful
    direction while the rate is unmeasured. Returns `nan` when the advantage
    is already non-positive (nothing to cancel), and `inf` when no false-flag
    rate is high enough to cancel it.
    """
    if advantage_per_1000_inr <= 0.0:
        return float("nan")
    if review_cost_inr <= 0.0 or contest_rate <= 0.0:
        return float("inf")
    # advantage = 1000 * review_cost * (escalate + contest * f)  ->  solve for f
    budget_rate = advantage_per_1000_inr / (1000.0 * review_cost_inr)
    remaining = budget_rate - escalate_rate
    if remaining <= 0.0:
        return 0.0  # the escalate rate alone already exhausts the budget
    return (
        float(np.minimum(remaining / contest_rate, 1.0))
        if remaining / contest_rate <= 1.0
        else float("inf")
    )


def format_curve(points: list[ReviewCostPoint]) -> str:
    lines = [
        "gate false-flag rate | human-touched rate | break-even review cost (INR)",
        "---------------------|--------------------|-----------------------------",
    ]
    for p in points:
        cost = (
            "no advantage to erode"
            if p.break_even_review_cost_inr == 0.0
            else f"{p.break_even_review_cost_inr:,.0f}"
        )
        lines.append(f"{p.false_flag_rate:>20.2f} | {p.human_touched_rate:>18.4f} | {cost:>29}")
    return "\n".join(lines)
