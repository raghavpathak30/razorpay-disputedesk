"""Rupee-denominated business metrics (SPEC.md §6, PHASES.md Phase 2 gate item
deferred to Phase 3): rupees recovered per 1,000 disputes under the policy
engine versus the two baselines, and the false-positive/false-negative costs
in rupees with components named.

All three decision paths are priced relative to a single reference: doing
nothing recovers INR 0 per dispute (the merchant simply eats the chargeback).
That is exactly what "accept everything" is, so it is the zero baseline every
other number is measured against - a dispute you never contest costs its
`amount` in real terms, but "recovered rupees" reports the *delta* from that
default outcome, not the raw loss.

- accept: recovered = 0 (the reference outcome itself).
- contest, won (`won_if_contested` True): recovered = amount - representment_cost.
- contest, lost (`won_if_contested` False): recovered = -representment_cost -
  the false-positive cost. The dispute amount is still lost either way (that
  loss is already "priced in" to the zero baseline); contesting-and-losing
  adds the representment cost on top of it.
- escalate: SPEC.md §4 gives the escalate band an "I don't know" path
  precisely because no automated outcome should be claimed for it - but
  crediting it 0 rupees is itself a claim: it says the human who reviews an
  escalated dispute takes no action and recovers nothing. That is not what
  "escalate to a human" means, and it structurally penalizes the policy for
  having an abstention path at all (a decision-maker with no "I don't know"
  option can't be scored worse than one who has to hedge). Rather than pick
  one number, `recovered_rupees` supports three `escalate_mode`s, all
  reported side by side (never blended into one headline):
    - `"zero"`: no outcome claimed (the original, most conservative choice).
    - `"oracle"`: the human always makes the correct call, using the true
      `won_if_contested` label as foreknowledge - an upper bound on what
      escalation could be worth, not a claim about a real human's accuracy.
    - `"naive_contest"`: the human contests every escalated dispute, i.e.
      escalated rows are scored exactly like baseline A treats them. This is
      the fairest apples-to-apples comparison against "contest everything",
      since that baseline already assumes this outcome for these same rows.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from disputedesk.policy.config import PolicyConfig
from disputedesk.policy.engine import Decision, decide

EscalateMode = Literal["zero", "oracle", "naive_contest"]


@dataclass(frozen=True)
class BusinessOutcome:
    recovered_inr: np.ndarray  # per-row, policy-decided recovery
    decisions: np.ndarray  # per-row Decision values


def decide_batch(p_win: np.ndarray, amount: np.ndarray, config: PolicyConfig) -> np.ndarray:
    """`decide` applied row-wise; returns an object array of `Decision`
    values (`dtype=object` explicit, though `Decision` being a plain `Enum`
    - not `(str, Enum)`, see its docstring - already makes numpy infer that).
    """
    return np.array(
        [decide(float(p), float(a), config).decision for p, a in zip(p_win, amount, strict=True)],
        dtype=object,
    )


def recovered_rupees(
    decisions: np.ndarray,
    won_if_contested: np.ndarray,
    amount: np.ndarray,
    representment_cost_inr: float,
    escalate_mode: EscalateMode = "zero",
) -> np.ndarray:
    """Per-row recovered rupees, relative to the accept-everything reference
    outcome (see module docstring). Vectorized over the three decision paths;
    `escalate_mode` selects what an escalated row is credited (see module
    docstring for the three modes and why none of them is "the" answer).
    """
    won_if_contested = np.asarray(won_if_contested, dtype=bool)
    amount = np.asarray(amount, dtype=float)
    is_contest = decisions == Decision.CONTEST
    is_escalate = decisions == Decision.ESCALATE
    contest_recovered = np.where(
        won_if_contested,
        amount - representment_cost_inr,
        -representment_cost_inr,
    )

    if escalate_mode == "zero":
        escalate_recovered = np.zeros_like(amount)
    elif escalate_mode == "oracle":
        # Best of {contest, accept} per row, using the true label as
        # foreknowledge a real human reviewer would not actually have.
        escalate_recovered = np.where(won_if_contested, amount - representment_cost_inr, 0.0)
    elif escalate_mode == "naive_contest":
        escalate_recovered = contest_recovered
    else:
        raise ValueError(f"unknown escalate_mode: {escalate_mode!r}")

    return np.where(is_contest, contest_recovered, np.where(is_escalate, escalate_recovered, 0.0))


def contest_everything_recovered(
    won_if_contested: np.ndarray, amount: np.ndarray, representment_cost_inr: float
) -> np.ndarray:
    """Baseline A (SPEC.md §6): every dispute is contested."""
    decisions = np.full(len(amount), Decision.CONTEST, dtype=object)
    return recovered_rupees(decisions, won_if_contested, amount, representment_cost_inr)


def accept_everything_recovered(n: int) -> np.ndarray:
    """Baseline B (SPEC.md §6): every dispute is accepted - the zero reference
    by construction.
    """
    return np.zeros(n, dtype=float)


def false_positive_cost(
    decisions: np.ndarray, won_if_contested: np.ndarray, representment_cost_inr: float
) -> tuple[int, float]:
    """FP = contested a dispute that was not actually winnable (SPEC.md §6).
    Cost per FP is the fixed `representment_cost_inr` (fee + analyst time +
    excessive-representment exposure - see `disputedesk.policy.config`).
    Returns (count, total_cost_inr).
    """
    won_if_contested = np.asarray(won_if_contested, dtype=bool)
    fp_mask = (decisions == Decision.CONTEST) & ~won_if_contested
    count = int(fp_mask.sum())
    return count, count * representment_cost_inr


def false_negative_cost(
    decisions: np.ndarray, won_if_contested: np.ndarray, amount: np.ndarray
) -> tuple[int, float]:
    """FN = accepted a dispute that was actually winnable (SPEC.md §6). Cost
    is the full dispute amount, per row. Returns (count, total_cost_inr).
    """
    won_if_contested = np.asarray(won_if_contested, dtype=bool)
    amount = np.asarray(amount, dtype=float)
    fn_mask = (decisions == Decision.ACCEPT) & won_if_contested
    count = int(fn_mask.sum())
    return count, float(amount[fn_mask].sum())


def escalated_summary(decisions: np.ndarray, amount: np.ndarray) -> tuple[int, float]:
    """Count and total amount of escalated disputes - reported separately,
    never folded into recovered rupees (see module docstring).
    """
    amount = np.asarray(amount, dtype=float)
    escalate_mask = decisions == Decision.ESCALATE
    return int(escalate_mask.sum()), float(amount[escalate_mask].sum())


def escalated_amount_share(decisions: np.ndarray, amount: np.ndarray) -> float:
    """What fraction of the *total holdout rupees at stake* (sum of `amount`
    over every row, not just escalated ones) sits in escalated disputes. A
    small escalated *count* can still carry an outsized share of the money if
    escalated disputes skew toward higher amounts - this is what makes that
    visible, separately from `escalated_summary`'s raw count/total.
    """
    amount = np.asarray(amount, dtype=float)
    total = amount.sum()
    if total == 0:
        return 0.0
    _count, escalated_total = escalated_summary(decisions, amount)
    return escalated_total / total


def per_1000(total: float, n: int) -> float:
    """Normalizes a rupee total to a rate per 1,000 disputes, so headline
    numbers are comparable across seeds/test-split sizes.
    """
    return total / n * 1000.0


def build_business_row(
    p_win: np.ndarray,
    won_if_contested: np.ndarray,
    amount: np.ndarray,
    config: PolicyConfig,
) -> dict:
    """One full set of business metrics for one seed's temporal-holdout test
    split. Every input is holdout-only, consistent with `eval.harness`.
    """
    n = len(amount)
    decisions = decide_batch(p_win, amount, config)

    policy_recovered = {
        mode: recovered_rupees(
            decisions, won_if_contested, amount, config.representment_cost_inr, escalate_mode=mode
        )
        for mode in ("zero", "oracle", "naive_contest")
    }
    baseline_a_recovered = contest_everything_recovered(
        won_if_contested, amount, config.representment_cost_inr
    )
    baseline_b_recovered = accept_everything_recovered(n)

    fp_count, fp_total_inr = false_positive_cost(
        decisions, won_if_contested, config.representment_cost_inr
    )
    fn_count, fn_total_inr = false_negative_cost(decisions, won_if_contested, amount)
    escalate_count, escalate_total_amount_inr = escalated_summary(decisions, amount)

    return {
        "n": n,
        # Default headline (most conservative: no outcome claimed for
        # escalated rows). See module docstring / the two variants below for
        # why this alone can understate what the policy is worth.
        "policy_recovered_per_1000_inr": per_1000(policy_recovered["zero"].sum(), n),
        "policy_recovered_per_1000_inr_escalate_oracle": per_1000(
            policy_recovered["oracle"].sum(), n
        ),
        "policy_recovered_per_1000_inr_escalate_naive_contest": per_1000(
            policy_recovered["naive_contest"].sum(), n
        ),
        "baseline_a_contest_everything_recovered_per_1000_inr": per_1000(
            baseline_a_recovered.sum(), n
        ),
        "baseline_b_accept_everything_recovered_per_1000_inr": per_1000(
            baseline_b_recovered.sum(), n
        ),
        "false_positive_count": fp_count,
        "false_positive_cost_per_1000_inr": per_1000(fp_total_inr, n),
        "false_negative_count": fn_count,
        "false_negative_cost_per_1000_inr": per_1000(fn_total_inr, n),
        "escalated_count": escalate_count,
        "escalated_amount_per_1000_inr": per_1000(escalate_total_amount_inr, n),
        "escalated_amount_share_of_holdout": escalated_amount_share(decisions, amount),
    }


def summarize_business(results: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR (25th/75th percentile) across seeds - same convention as
    `eval.harness.summarize` (CLAUDE.md invariant 3: no headline from a single
    seed).
    """
    numeric_cols = [c for c in results.columns if c != "seed"]
    summary = pd.DataFrame(
        {
            "median": results[numeric_cols].median(),
            "q25": results[numeric_cols].quantile(0.25),
            "q75": results[numeric_cols].quantile(0.75),
        }
    )
    summary.index.name = "metric"
    return summary
