"""The eval harness runs in CI on a fixed seed set so a metric regression is
caught at commit time (PHASES.md Phase 2 gate). Deliberately small n_rows and
seed count here for CI speed - the headline >=20-seed, n_rows=15000 report is
produced by `eval/run_harness.py`, not by this test.

Bounds below are sanity/regression bounds, not the headline numbers: wide
enough to tolerate normal variation, tight enough to catch a real break (e.g.
a leaked column collapsing PR-AUC to ~1.0, or a broken pipeline collapsing it
to prevalence).
"""

import pytest

from disputedesk.generator.config import GeneratorConfig
from disputedesk.model.config import ModelConfig
from eval.harness import fixed_seed_set, run_harness, summarize
from eval.report import format_precision_recall_headline

CI_SEEDS = fixed_seed_set(8, start=0)
CI_N_ROWS = 5000


@pytest.fixture(scope="module")
def ci_summary():
    per_seed = run_harness(CI_SEEDS, CI_N_ROWS, GeneratorConfig(), ModelConfig())
    return per_seed, summarize(per_seed)


def test_harness_runs_end_to_end_on_the_fixed_ci_seed_set(ci_summary):
    per_seed, _summary = ci_summary
    assert len(per_seed) == len(CI_SEEDS)
    assert set(per_seed["seed"]) == set(CI_SEEDS)


def test_model_beats_prevalence_baseline_on_median_pr_auc(ci_summary):
    _per_seed, summary = ci_summary
    assert summary.loc["model_pr_auc", "median"] > summary.loc["prevalence_baseline", "median"]


def test_model_pr_auc_never_exceeds_the_oracle_ceiling_on_median(ci_summary):
    _per_seed, summary = ci_summary
    assert summary.loc["model_pr_auc", "median"] < summary.loc["oracle_pr_auc", "median"]


def test_oracle_pr_auc_stays_in_a_sane_range(ci_summary):
    """Regression guard against a generator change silently moving the ceiling
    without GENERATOR.md being updated to match (CLAUDE.md: never silently
    change a recorded number).
    """
    _per_seed, summary = ci_summary
    assert 0.30 < summary.loc["oracle_pr_auc", "median"] < 0.60


def test_calibration_error_stays_bounded(ci_summary):
    _per_seed, summary = ci_summary
    assert summary.loc["calibration_error", "median"] < 0.15


def test_precision_and_recall_are_non_degenerate(ci_summary):
    _per_seed, summary = ci_summary
    assert 0.0 < summary.loc["precision_at_threshold", "median"] < 1.0
    assert 0.0 < summary.loc["recall_at_threshold", "median"] < 1.0


def test_precision_recall_headline_names_the_threshold(ci_summary):
    """Structural check for the coupling itself: the one sentence meant to be
    quoted in the README/pitch video must contain the threshold value, not
    just the precision/recall numbers.
    """
    _per_seed, summary = ci_summary
    headline = format_precision_recall_headline(summary)

    assert "precision" in headline
    assert "recall" in headline
    assert "threshold" in headline
    assert f"{summary.loc['threshold', 'median']:.4f}" in headline
    assert f"{summary.loc['precision_at_threshold', 'median']:.4f}" in headline
    assert f"{summary.loc['recall_at_threshold', 'median']:.4f}" in headline
