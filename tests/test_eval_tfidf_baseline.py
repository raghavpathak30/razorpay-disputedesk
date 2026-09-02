"""The TF-IDF baseline arm (remediation defect 1.2).

The README's strongest AI-judgment claim - "we measured whether the LLM adds
predictive value here, and it does not" - rested on comparing an LLM arm
(n=60, seed 0, recorded) against a TF-IDF baseline of **AUC 0.6371 with no
code, no recorded n, no seed, and no command** anywhere in the repo. It
appeared as a bare `**Result:**` line appended to a 2026-08-31 DECISIONS.md
*decision* entry, not in the measurement format that file mandates, and not
marked CONFIRMED-RAN - which DECISIONS.md's own header forbids for anything
quoted in the README.

This module makes the baseline a real, testable thing: same task
(`customer_communication_log` -> `true_fraud`), same downstream classifier,
same 5-fold stratified CV, no network.

The leakage question matters more here than anywhere else in `eval/`: a
TF-IDF vectorizer fit on all rows before splitting would leak test-fold
vocabulary and idf weights into training, inflating the baseline the LLM is
being judged against. The vectorizer is therefore fit inside each training
fold, and `test_a_shuffled_label_scores_near_chance` is the control that
would catch it if that ever stopped being true.
"""

import numpy as np
import pytest

from eval.tfidf_baseline import comms_and_true_fraud, tfidf_auc, tfidf_out_of_fold_probabilities


def _planted_corpus(n: int = 120) -> tuple[list[str], np.ndarray]:
    """Half the documents carry a marker token that perfectly predicts the
    label; the rest carry filler. Any working text pipeline finds this.
    """
    logs, labels = [], []
    for i in range(n):
        positive = i % 2 == 0
        logs.append(
            "marker token appears here plus filler words"
            if positive
            else "only filler words appear in this one"
        )
        labels.append(positive)
    return logs, np.array(labels)


def test_a_planted_signal_is_recovered():
    logs, labels = _planted_corpus()

    result = tfidf_auc(logs, labels)

    assert result["mean_auc"] > 0.95
    assert result["n"] == len(labels)


def test_a_shuffled_label_scores_near_chance():
    """The leakage control. If the vectorizer were fit on all rows before
    splitting - or if predictions were scored in-sample - a shuffled label
    would still be partly recoverable. Out of fold, it cannot be.
    """
    logs, labels = _planted_corpus()
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(labels)

    result = tfidf_auc(logs, shuffled)

    assert 0.3 < result["mean_auc"] < 0.7


def test_out_of_fold_probabilities_cover_every_row_exactly_once():
    logs, labels = _planted_corpus(60)

    oof = tfidf_out_of_fold_probabilities(logs, labels)

    assert oof.shape == (60,)
    assert np.isfinite(oof).all()
    assert ((oof >= 0.0) & (oof <= 1.0)).all()


def test_the_result_shape_matches_the_llm_arms_result_shape():
    """Both arms are reported side by side, so they must report the same
    fields - a comparison whose two halves are shaped differently invites
    exactly the kind of loose point-estimate pairing this defect was.
    """
    from eval.llm_normalization_quality import auc_of_normalized_fields

    logs, labels = _planted_corpus(60)
    tfidf_result = tfidf_auc(logs, labels)
    llm_result = auc_of_normalized_fields(
        [dict.fromkeys(
            (
                "claims_unauthorized_transaction",
                "mentions_prior_bank_contact",
                "mentions_shared_card_access",
                "mentions_travel",
                "is_substantive",
                "tone_polite",
                "tone_terse",
            ),
            0,
        )
        | {"is_substantive": int(label)}
        for label in labels],
        labels,
    )

    assert set(tfidf_result) == set(llm_result)


def test_comms_and_true_fraud_reads_the_generator_deterministically():
    logs_a, labels_a = comms_and_true_fraud(40, seed=3)
    logs_b, labels_b = comms_and_true_fraud(40, seed=3)

    assert logs_a == logs_b
    assert np.array_equal(labels_a, labels_b)
    assert len(logs_a) == 40


def test_a_different_seed_gives_different_items():
    logs_a, _ = comms_and_true_fraud(40, seed=3)
    logs_b, _ = comms_and_true_fraud(40, seed=4)

    assert logs_a != logs_b


def test_cross_validation_needs_both_classes_present():
    logs = ["all the same text here"] * 20
    labels = np.zeros(20, dtype=bool)

    with pytest.raises(ValueError):
        tfidf_auc(logs, labels)
