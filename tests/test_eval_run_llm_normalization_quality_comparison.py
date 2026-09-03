"""Pins that a fresh run of `eval.run_llm_normalization_quality` cannot
reproduce the Phase 1 defect it was corrected for (stale-number audit, item A,
2026-09-03).

`eval/run_llm_normalization_quality.py` is the script the README's "What
would settle it" note tells a future reader to run when a live API key is
available, to re-measure the LLM arm at a larger n. As found, it still
imported `TFIDF_BASELINE_AUC = 0.6371` - the exact large-sample-vs-small-sample
comparison Phase 1 corrected - and printed "beats baseline: YES/NO" against
it. Following the README's own instruction would have reproduced the
defect it corrects.

This module makes real network calls and is explicitly excluded from CI's
network ban by never being imported under `tests/` for that reason - so this
test exercises the pure comparison logic it now delegates to
(`paired_comparison_against_tfidf`), which takes an already-collected LLM
sample and makes no network call itself.
"""

import numpy as np
import pandas as pd
import pytest

from disputedesk.generator.config import GeneratorConfig
from eval.run_llm_normalization_quality import paired_comparison_against_tfidf


def _fake_llm_sample(n_rows: int, seed: int) -> pd.DataFrame:
    """A `run_llm_normalization_sample`-shaped frame whose `true_fraud` column
    is real (from the generator) but whose typed fields are synthetic - stands
    in for a real LLM run without a network call, and is enough to exercise
    the comparison wiring end to end.
    """
    from eval.tfidf_baseline import comms_and_true_fraud

    _logs, labels = comms_and_true_fraud(n_rows, seed, GeneratorConfig())
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "true_fraud": labels,
            "human_review_required": False,
            "claims_unauthorized_transaction": rng.integers(0, 2, size=n_rows),
            "mentions_prior_bank_contact": rng.integers(0, 2, size=n_rows),
            "mentions_shared_card_access": rng.integers(0, 2, size=n_rows),
            "mentions_travel": rng.integers(0, 2, size=n_rows),
            "is_substantive": rng.integers(0, 2, size=n_rows),
            "tone_polite": rng.integers(0, 2, size=n_rows),
            "tone_terse": rng.integers(0, 2, size=n_rows),
        }
    )


def test_no_stale_hardcoded_baseline_survives_in_the_run_script():
    """The literal defect: a hardcoded 0.6371 measured at a different n must
    not be importable from the CLI module any more.
    """
    import eval.run_llm_normalization_quality as run_module

    assert not hasattr(run_module, "TFIDF_BASELINE_AUC")


def test_paired_comparison_scores_both_arms_on_the_identical_items():
    """The TF-IDF arm is computed on the same generator items as the LLM
    sample - same n, same seed - not a baseline recorded elsewhere at a
    different sample size.
    """
    sample = _fake_llm_sample(60, seed=0)

    result = paired_comparison_against_tfidf(sample, n_rows=60, seed=0)

    assert result.n_items == 60


def test_the_comparison_reports_an_interval_not_a_bare_point_estimate():
    """The methodological fix itself: a paired bootstrap CI, not a bare
    `mean_auc > baseline` boolean.
    """
    sample = _fake_llm_sample(60, seed=0)

    result = paired_comparison_against_tfidf(sample, n_rows=60, seed=0)

    assert hasattr(result, "ci_low")
    assert hasattr(result, "ci_high")
    assert result.ci_low <= result.difference <= result.ci_high


def test_a_sample_whose_labels_do_not_match_the_generator_is_rejected():
    """The same pairing safeguard `run_extraction_comparison.py` has: if the
    passed-in sample's `true_fraud` column does not match what the generator
    produces for this n/seed, the items are not actually paired and the
    comparison must refuse rather than silently report a meaningless number.
    """
    sample = _fake_llm_sample(60, seed=0)
    sample["true_fraud"] = ~sample["true_fraud"]  # corrupt the pairing

    with pytest.raises(ValueError, match="does not match"):
        paired_comparison_against_tfidf(sample, n_rows=60, seed=0)
