"""Sensitivity of the cost-weighted business metrics to
`representment_cost_inr`, holding the model, seeds, and `low_confidence_band`
fixed. This is a sensitivity analysis reported *alongside* the configured
value in `disputedesk.policy.config`, not a retune - nothing here writes to
that module, and the sweep never changes what the running system actually
uses.

`P(win)` does not depend on `representment_cost_inr` at all - only
`decide()` does - so each seed's generate/train/predict cycle
(`eval.harness.run_seed_pipeline`) runs exactly once per seed, and the sweep
over cost values reuses those same predictions. Escalated disputes are
scored under `escalate_mode="naive_contest"` throughout (see
`eval.business_metrics`'s module docstring and the 2026-08-31 "cost-weighted
business metrics" DECISIONS.md entry): that is the fair apples-to-apples
mode against baseline A, which already contests every escalated dispute.

Precision/recall of the policy's own decisions (CONTEST as the positive
prediction, `won_if_contested` as the label) are reported alongside the
recovered-rupee numbers at every swept cost. ESCALATE rows must be folded
into one side of that binary choice, and `_predicted_positive` below is
where that happens - kept in lockstep with `ESCALATE_MODE` on purpose, so
the same definition of "did the policy effectively contest this row" feeds
both the rupee and the precision/recall numbers. With `ESCALATE_MODE =
"naive_contest"`, escalated rows are credited as contested for recovery, so
they count as positive predictions here too.

What this sweep assumes, stated here because both assumptions were
implicit until 2026-09-02 and both are load-bearing on every rupee figure
below (remediation item 1.0):

**1. Every CONTEST/ESCALATE decision results in a filed representment.**
Phase 0 added a `withheld_for_review` outcome - a dispute the policy engine
decided to contest, whose evidence packet was not fit to file, goes to a
person and nothing reaches the card network. That path is **excluded** from
this sweep rather than modelled, and the reason is that only half of the
withheld rate is measurable:

- The reason-code half is exactly **zero** on every dataset this sweep
  scores: the generator emits only the four codes with an evidence strategy
  (pinned by `tests/test_eval_sweep_assumptions.py`).
- The letter-drafting half is **currently unmeasured**. Its only empirical
  input was the 2026-09-01 letter-drafting reliability run, which Phase 0
  invalidated by changing both the output schema and the prompt, and which
  cannot be re-measured without a live API key.

Modelling it would mean choosing a withheld rate with no measurement behind
it and letting that guess propagate into a headline. Excluding it and saying
so is the defensible option - but the exclusion is not neutral: it can only
*overstate* the automated system's advantage, never understate it, because
every withheld dispute is credited a filing that did not happen and charged
no human-review cost for the review that did.
`break_even_human_review_cost_inr` below turns that into a number a reader
can check the claim against.

**2. Every filed representment is accepted for review by Razorpay.** This
one is currently **false** - see `SWEEP_ASSUMES_EVERY_SUBMISSION_IS_ACCEPTED`.

The ESCALATE rate (fraction of holdout rows `decide()` sends to
`Decision.ESCALATE`) is also reported per swept cost, for the same reason
precision/recall is: it's a property of the decisions actually made at that
cost, not a fixed constant to take on faith. Structurally it should come out
*invariant* to `representment_cost_inr` - `decide()`'s low-confidence check
(`low <= p_win <= high`) runs before the cost-dependent
`expected_value`/CONTEST-vs-ACCEPT branch and never reads `cost` or `amount`,
so which rows escalate is fixed by `p_win` and `low_confidence_band` alone,
both held fixed across this sweep. Reported per cost anyway, not hardcoded
once, so that invariance is a measured fact, not an assumption.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from disputedesk.policy.config import PolicyConfig
from disputedesk.policy.engine import Decision
from eval.business_metrics import (
    contest_everything_recovered,
    decide_batch,
    per_1000,
    recovered_rupees,
)
from eval.harness import LABEL_COLUMN, run_seed_pipeline
from eval.paired import paired_difference

ESCALATE_MODE = "naive_contest"

SWEEP_ASSUMES_EVERY_SUBMISSION_IS_ACCEPTED = False
"""Whether a contest this system files would actually be accepted for review.

Every recovered rupee in this sweep is credited to a contest that was filed
and taken up. Razorpay's contest endpoint documents that `action="submit"`
requires at least one document id across the evidence fields
(https://razorpay.com/docs/api/disputes/contest/, verified 2026-09-02), and
`disputedesk/client/razorpay.py` sends none, because this project never built
a document-upload pipeline (ARCHITECTURE.md "Known gaps").

So this flag is `False`: the sweep's recovered-rupee figures describe what the
policy *would* recover if its filings were accepted, and today they would very
likely be rejected. That does not invalidate the *comparison* - baseline A
files through the same client and inherits the same gap, so the relative
advantage is unaffected - but it does mean the absolute
"rupees recovered per 1,000 disputes" figures are contingent on a component
that does not exist. Recorded as a constant rather than prose so a future
change that builds the upload path has an obvious place to flip, and so
flipping it is a deliberate claim rather than a silent one."""


def break_even_human_review_cost_inr(
    advantage_per_1000_inr: float, human_touched_rate: float
) -> float:
    """The per-review cost at which the policy's advantage over baseline A is
    exactly cancelled by the human time the sweep does not charge for.

    `human_touched_rate` is the fraction of holdout rows a person must handle
    without an automated filing being produced: the ESCALATE rate, plus the
    withheld-for-review rate once that is measurable. The sweep charges those
    rows nothing, so the advantage survives only while a review costs less
    than this.

    Returns `inf` when no row is human-touched (nothing is being excluded) and
    `0.0` when the advantage is already non-positive (there is no advantage
    for a review cost to erode, and returning a positive number there would
    read as a margin that does not exist).
    """
    if advantage_per_1000_inr <= 0.0:
        return 0.0
    if human_touched_rate <= 0.0:
        return float("inf")
    return advantage_per_1000_inr / (1000.0 * human_touched_rate)


def _predicted_positive(decisions: np.ndarray) -> np.ndarray:
    """CONTEST is always a positive prediction; ESCALATE's treatment must
    match `ESCALATE_MODE`'s treatment in `recovered_rupees` above so the
    precision/recall table and the rupee table describe the same decisions.
    Raises rather than guessing if `ESCALATE_MODE` is ever changed to a mode
    this function has not been updated for.
    """
    if ESCALATE_MODE != "naive_contest":
        raise NotImplementedError(
            f"no precision/recall mapping defined for ESCALATE_MODE={ESCALATE_MODE!r}"
        )
    return (decisions == Decision.CONTEST) | (decisions == Decision.ESCALATE)


def score_predictions_across_costs(
    seed: int,
    predicted_p: np.ndarray,
    amount: np.ndarray,
    won_if_contested: np.ndarray,
    costs: list[float],
    low_confidence_band: tuple[float, float],
) -> list[dict]:
    """Every cost in `costs` scored against one already-computed
    `predicted_p` - no retraining inside the loop.

    Split out from `sweep_seed` on 2026-09-03 (Phase 2 addendum item B) so
    `eval.ablation` can score a restricted-feature model's predictions through
    the exact same per-cost logic, and therefore the exact same paired
    estimator (`summarize_sweep`), as the full-feature sweep - a second,
    hand-copied implementation of this scoring is exactly the kind of drift
    this remediation exists to catch.
    """
    n = len(amount)
    rows = []
    for cost in costs:
        config = PolicyConfig(representment_cost_inr=cost, low_confidence_band=low_confidence_band)
        decisions = decide_batch(predicted_p, amount, config)
        policy_recovered = recovered_rupees(
            decisions, won_if_contested, amount, cost, escalate_mode=ESCALATE_MODE
        )
        baseline_a_recovered = contest_everything_recovered(won_if_contested, amount, cost)
        predicted_positive = _predicted_positive(decisions)
        escalate_rate = float(np.mean(decisions == Decision.ESCALATE))
        rows.append(
            {
                "seed": seed,
                "representment_cost_inr": cost,
                "n": n,
                "policy_recovered_per_1000_inr": per_1000(policy_recovered.sum(), n),
                "baseline_a_recovered_per_1000_inr": per_1000(baseline_a_recovered.sum(), n),
                "policy_precision": precision_score(
                    won_if_contested, predicted_positive, zero_division=0
                ),
                "policy_recall": recall_score(
                    won_if_contested, predicted_positive, zero_division=0
                ),
                "policy_escalate_rate": escalate_rate,
            }
        )
    return rows


def sweep_seed(
    seed: int,
    n_rows: int,
    costs: list[float],
    generator_config: GeneratorConfig,
    model_config: ModelConfig,
    low_confidence_band: tuple[float, float],
) -> list[dict]:
    """One seed's train/predict cycle, then every cost in `costs` scored
    against those same predictions - no retraining inside the loop.
    """
    run = run_seed_pipeline(seed, n_rows, generator_config, model_config)
    amount = run.test_df["amount"].to_numpy()
    won_if_contested = run.test_df[LABEL_COLUMN].to_numpy()
    return score_predictions_across_costs(
        seed, run.predicted_p, amount, won_if_contested, costs, low_confidence_band
    )


def sweep_representment_cost(
    seeds: list[int],
    n_rows: int,
    costs: list[float],
    generator_config: GeneratorConfig | None = None,
    model_config: ModelConfig | None = None,
    low_confidence_band: tuple[float, float] = (0.45, 0.55),
) -> pd.DataFrame:
    """One row per (seed, cost). `low_confidence_band` defaults to
    `PolicyConfig`'s own default, held fixed across the sweep per the "hold
    everything else fixed" instruction.
    """
    generator_config = generator_config or GeneratorConfig()
    model_config = model_config or ModelConfig()

    rows: list[dict] = []
    for seed in seeds:
        rows.extend(
            sweep_seed(seed, n_rows, costs, generator_config, model_config, low_confidence_band)
        )
    return pd.DataFrame(rows)


def _paired_advantage_rows(results: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """One paired comparison per swept cost: policy minus baseline A, seed by
    seed. Sorted by seed on both sides so element `i` of each arm is the same
    seed - the pairing is the whole point and must not depend on groupby's
    row order.
    """
    rows = []
    for cost, group in results.groupby("representment_cost_inr"):
        ordered = group.sort_values("seed")
        paired = paired_difference(
            ordered["policy_recovered_per_1000_inr"].to_numpy(),
            ordered["baseline_a_recovered_per_1000_inr"].to_numpy(),
            random_state=random_state,
        )
        rows.append(
            {
                "representment_cost_inr": cost,
                "n_seeds": paired.n_pairs,
                "advantage_paired_mean": paired.mean_difference,
                "advantage_paired_median": paired.median_difference,
                "advantage_ci_low": paired.ci_low,
                "advantage_ci_high": paired.ci_high,
                "advantage_n_positive": paired.n_positive,
                "advantage_excludes_zero": paired.excludes_zero,
            }
        )
    return pd.DataFrame(rows)


def summarize_sweep(results: pd.DataFrame, random_state: int = 0) -> pd.DataFrame:
    """Median and IQR of both recovered-rupee series, the policy's own
    precision/recall (ESCALATE folded in per `_predicted_positive`), the
    ESCALATE rate itself, and - the headline - the **paired** advantage over
    baseline A, one row per cost value.

    The advantage columns replaced `policy_advantage_median`
    (`median(policy) - median(baseline_a)`) on 2026-09-02. That statistic
    threw away the pairing this sweep is built on: every seed scores both arms
    on the identical holdout, so seed-to-seed variation is shared and a
    difference of medians leaves all of it in. It also had no interval, which
    is why a per-point sign change was read as noise rather than tested. It is
    removed rather than kept alongside, so nothing can quote it by accident.

    `random_state` fixes the bootstrap resampling so a reported interval
    reproduces exactly; it is a reporting parameter, not a tuning one.
    """
    summary = (
        results.groupby("representment_cost_inr")
        .agg(
            policy_median=("policy_recovered_per_1000_inr", "median"),
            policy_q25=("policy_recovered_per_1000_inr", lambda s: s.quantile(0.25)),
            policy_q75=("policy_recovered_per_1000_inr", lambda s: s.quantile(0.75)),
            baseline_a_median=("baseline_a_recovered_per_1000_inr", "median"),
            baseline_a_q25=("baseline_a_recovered_per_1000_inr", lambda s: s.quantile(0.25)),
            baseline_a_q75=("baseline_a_recovered_per_1000_inr", lambda s: s.quantile(0.75)),
            precision_median=("policy_precision", "median"),
            precision_q25=("policy_precision", lambda s: s.quantile(0.25)),
            precision_q75=("policy_precision", lambda s: s.quantile(0.75)),
            recall_median=("policy_recall", "median"),
            recall_q25=("policy_recall", lambda s: s.quantile(0.25)),
            recall_q75=("policy_recall", lambda s: s.quantile(0.75)),
            escalate_rate_median=("policy_escalate_rate", "median"),
            escalate_rate_q25=("policy_escalate_rate", lambda s: s.quantile(0.25)),
            escalate_rate_q75=("policy_escalate_rate", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    summary = summary.merge(
        _paired_advantage_rows(results, random_state), on="representment_cost_inr", how="left"
    ).sort_values("representment_cost_inr")

    # The human-review cost at which the paired advantage is cancelled by the
    # time this sweep does not charge for (see the module docstring's
    # assumption 1). Computed at the *measured* human-touched rate - the
    # ESCALATE rate alone - so it is an upper bound: any non-zero withheld
    # rate lowers it further.
    summary["break_even_human_review_cost_inr"] = [
        break_even_human_review_cost_inr(advantage, rate)
        for advantage, rate in zip(
            summary["advantage_paired_mean"], summary["escalate_rate_median"], strict=True
        )
    ]
    return summary


@dataclass(frozen=True)
class LossTail:
    """The shape of the per-seed difference distribution at one swept cost.

    Reported because a paired mean and a sign count can point in opposite
    directions, and when they do the reason is always in this shape. At ₹50 the
    mean is negative while a majority of seeds are positive: the policy wins
    slightly more often than it loses and loses far harder when it does. That
    asymmetry is a property of the policy worth stating in its own voice, not
    an artifact to average away.
    """

    representment_cost_inr: float
    n_seeds: int
    n_positive: int
    n_negative: int
    mean_difference: float
    worst_seed_difference: float
    best_seed_difference: float
    mean_loss_on_losing_seeds: float
    mean_gain_on_winning_seeds: float

    @property
    def spread(self) -> float:
        return self.best_seed_difference - self.worst_seed_difference

    @property
    def loss_to_gain_ratio(self) -> float:
        """How many times larger the average loss is than the average gain.
        `inf` when no seed wins; 0.0 when none loses."""
        if self.mean_gain_on_winning_seeds == 0.0:
            return float("inf") if self.mean_loss_on_losing_seeds != 0.0 else 0.0
        return abs(self.mean_loss_on_losing_seeds) / self.mean_gain_on_winning_seeds


def loss_tail(results: pd.DataFrame, cost: float) -> LossTail:
    """The per-seed advantage distribution at one swept `cost`."""
    at_cost = results[results["representment_cost_inr"] == cost].sort_values("seed")
    if at_cost.empty:
        raise ValueError(f"no swept rows at representment_cost_inr={cost}")
    differences = (
        at_cost["policy_recovered_per_1000_inr"] - at_cost["baseline_a_recovered_per_1000_inr"]
    ).to_numpy()

    losing = differences[differences < 0]
    winning = differences[differences > 0]
    return LossTail(
        representment_cost_inr=cost,
        n_seeds=int(differences.size),
        n_positive=int(winning.size),
        n_negative=int(losing.size),
        mean_difference=float(differences.mean()),
        worst_seed_difference=float(differences.min()),
        best_seed_difference=float(differences.max()),
        mean_loss_on_losing_seeds=float(losing.mean()) if losing.size else 0.0,
        mean_gain_on_winning_seeds=float(winning.mean()) if winning.size else 0.0,
    )


def fixed_seed_set(n_seeds: int, start: int = 0) -> list[int]:
    return list(np.arange(start, start + n_seeds))
