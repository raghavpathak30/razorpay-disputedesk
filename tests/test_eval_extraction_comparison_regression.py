"""Exact-value regression on the LLM-vs-TF-IDF extraction comparison
(remediation defect 1.2).

Unlike `tests/test_eval_cost_sweep_regression.py`, **these numbers are the
reported results**, not a reduced CI-scale fixture: the comparison is n=60
because that is the size of the one recorded LLM run, so the test scores the
same items the README quotes. Reproduce with:

    python -m eval.run_extraction_comparison

The LLM arm cannot be re-measured in this environment (no `LLM_API_KEY`); the
recorded run is committed at `data/reference/llm_normalization_arm_n60_seed0.csv`
so this reproduces with no network call. The first test below pins the two
arms to the *same* items, which is the property the original comparison could
not demonstrate.
"""

import numpy as np
import pandas as pd
import pytest

from disputedesk.generator.config import GeneratorConfig
from eval.extraction_comparison import (
    auc_vs_chance,
    out_of_fold_probabilities,
    paired_auc_difference,
)
from eval.llm_normalization_quality import FEATURE_COLUMNS, auc_of_normalized_fields
from eval.run_extraction_comparison import LLM_ARM_FIXTURE, LLM_ARM_N_ROWS, LLM_ARM_SEED
from eval.tfidf_baseline import comms_and_true_fraud, tfidf_auc, tfidf_out_of_fold_probabilities


@pytest.fixture(scope="module")
def arms():
    llm_arm = pd.read_csv(LLM_ARM_FIXTURE, comment="#")
    logs, labels = comms_and_true_fraud(LLM_ARM_N_ROWS, LLM_ARM_SEED, GeneratorConfig())
    llm_oof = out_of_fold_probabilities(
        llm_arm[list(FEATURE_COLUMNS)].to_numpy(dtype=float), labels, random_state=0
    )
    tfidf_oof = tfidf_out_of_fold_probabilities(logs, labels, random_state=0)
    return llm_arm, logs, labels, llm_oof, tfidf_oof


def test_the_committed_llm_arm_is_scored_on_the_generators_current_items(arms):
    """The pairing itself. The recorded LLM run carries its own `true_fraud`
    column; regenerating the dataset must reproduce it row for row, or row `i`
    of the two arms is not the same dispute and every number below is
    meaningless while still looking fine.
    """
    llm_arm, _logs, labels, _llm_oof, _tfidf_oof = arms

    assert len(llm_arm) == LLM_ARM_N_ROWS
    assert np.array_equal(llm_arm["true_fraud"].to_numpy().astype(bool), labels)


def test_the_llm_arms_recorded_per_fold_auc_still_reproduces(arms):
    """0.4211 is the number the README has always quoted for the LLM arm. It
    reproduces exactly from the committed fixture - which is what makes the
    *baseline* side of the comparison the part that was wrong, not this side.
    """
    llm_arm, _logs, labels, _llm_oof, _tfidf_oof = arms

    result = auc_of_normalized_fields(
        llm_arm[list(FEATURE_COLUMNS)].to_dict(orient="records"), labels, random_state=0
    )

    assert result["mean_auc"] == pytest.approx(0.4210648148, abs=1e-9)
    assert result["std_auc"] == pytest.approx(0.1377959505, abs=1e-9)
    assert result["n"] == 60


def test_the_tfidf_baseline_on_the_same_60_items(arms):
    """0.5104, not the 0.6371 the README compared against. The recorded
    baseline was not wrong as a measurement of TF-IDF - it was measured on a
    much larger sample (see the n=3000 test below) and then used as the
    comparator for an n=60 LLM run.
    """
    _llm_arm, logs, labels, _llm_oof, _tfidf_oof = arms

    result = tfidf_auc(logs, labels, random_state=0)

    assert result["mean_auc"] == pytest.approx(0.5104166667, abs=1e-9)
    assert result["std_auc"] == pytest.approx(0.1361359104, abs=1e-9)
    assert result["n"] == 60


def test_the_paired_difference_includes_zero(arms):
    """The correction that matters. The README called this "a wide margin,
    not a close call decided by noise". On identical items, paired, the 95%
    interval includes zero: the direction survives, the claim about it being
    beyond noise does not.
    """
    _llm_arm, _logs, labels, llm_oof, tfidf_oof = arms

    result = paired_auc_difference(labels, tfidf_oof, llm_oof, random_state=0)

    assert result.n_items == 60
    assert result.auc_a == pytest.approx(0.5391527599, abs=1e-9)
    assert result.auc_b == pytest.approx(0.3767650834, abs=1e-9)
    assert result.difference == pytest.approx(0.1623876765, abs=1e-9)
    assert result.ci_low == pytest.approx(-0.0648189314, abs=1e-9)
    assert result.ci_high == pytest.approx(0.3858363858, abs=1e-9)
    assert result.excludes_zero is False


@pytest.mark.parametrize(
    ("arm", "expected_auc", "expected_ci"),
    [
        ("tfidf", 0.5391527599, (-0.1349469315, 0.2081343257)),
        ("llm", 0.3767650834, (-0.2735147528, 0.0304258242)),
    ],
)
def test_neither_arm_is_distinguishable_from_chance_at_this_n(arm, expected_auc, expected_ci):
    """Stronger than the paired result and independent of it: at n=60 neither
    arm can be shown to carry signal at all. Any claim that one extraction
    method beats the other on this evidence is claiming more than the sample
    supports.
    """
    llm_arm = pd.read_csv(LLM_ARM_FIXTURE, comment="#")
    logs, labels = comms_and_true_fraud(LLM_ARM_N_ROWS, LLM_ARM_SEED, GeneratorConfig())
    scores = (
        tfidf_out_of_fold_probabilities(logs, labels, random_state=0)
        if arm == "tfidf"
        else out_of_fold_probabilities(
            llm_arm[list(FEATURE_COLUMNS)].to_numpy(dtype=float), labels, random_state=0
        )
    )

    result = auc_vs_chance(labels, scores, random_state=0)

    assert result.auc_a == pytest.approx(expected_auc, abs=1e-9)
    assert result.ci_low == pytest.approx(expected_ci[0], abs=1e-9)
    assert result.ci_high == pytest.approx(expected_ci[1], abs=1e-9)
    assert result.excludes_zero is False


def test_the_tfidf_baseline_at_a_large_n_recovers_the_originally_recorded_value():
    """The diagnosis of what the 0.6371 figure actually was. At n=3000 the
    same baseline implementation scores 0.6479 - close to the recorded 0.6371
    and nowhere near its own n=60 value of 0.5104. The recorded number was a
    large-sample measurement of TF-IDF used as the comparator for a 60-item
    LLM run.
    """
    logs, labels = comms_and_true_fraud(3000, LLM_ARM_SEED)

    result = tfidf_auc(logs, labels, random_state=0)

    assert result["mean_auc"] == pytest.approx(0.6479149586, abs=1e-9)
    assert result["n"] == 3000
    assert abs(result["mean_auc"] - 0.6371) < 0.02  # the recorded figure, near-reproduced
