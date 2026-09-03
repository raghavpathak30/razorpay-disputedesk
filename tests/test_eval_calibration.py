"""Calibration: does P(win)=0.7 come out right about 70% of the time?"""

import numpy as np

from eval.calibration import (
    brier_score,
    calibration_table,
    expected_calibration_error,
    near_threshold_reliability,
)


def test_perfectly_calibrated_predictions_have_zero_error():
    rng = np.random.default_rng(0)
    n = 20_000
    predicted_p = rng.uniform(0.05, 0.95, size=n)
    labels = rng.random(n) < predicted_p

    error = expected_calibration_error(predicted_p, labels, n_bins=10)

    assert error < 0.02


def test_systematically_overconfident_predictions_have_positive_error():
    rng = np.random.default_rng(0)
    n = 20_000
    true_p = rng.uniform(0.05, 0.55, size=n)
    labels = rng.random(n) < true_p
    overconfident_p = np.clip(true_p + 0.3, 0, 1)

    error = expected_calibration_error(overconfident_p, labels, n_bins=10)

    assert error > 0.2


def test_calibration_table_bins_sum_to_total_count():
    rng = np.random.default_rng(1)
    n = 500
    predicted_p = rng.uniform(0, 1, size=n)
    labels = rng.random(n) < 0.3

    table = calibration_table(predicted_p, labels, n_bins=10)

    assert table["count"].sum() == n
    assert (table["mean_predicted_p"].between(0, 1)).all()
    assert (table["observed_win_rate"].between(0, 1)).all()


def test_expected_calibration_error_nan_on_empty_input():
    result = expected_calibration_error(np.array([]), np.array([]))
    assert np.isnan(result)


def test_brier_score_is_zero_for_perfect_predictions():
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(labels, labels) == 0.0


def test_brier_score_is_worse_for_confidently_wrong_predictions():
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    confident_right = np.array([0.9, 0.1, 0.9, 0.1])
    confident_wrong = np.array([0.1, 0.9, 0.1, 0.9])
    assert brier_score(confident_wrong, labels) > brier_score(confident_right, labels)


def test_near_threshold_reliability_selects_only_rows_close_to_their_own_c_over_a():
    # cost=100. Row 0: amount=1000 -> threshold 0.10, predicted 0.11 -> near (within 0.05).
    # Row 1: amount=200 -> threshold 0.50, predicted 0.11 -> far (0.39 away).
    predicted_p = np.array([0.11, 0.11])
    labels = np.array([1.0, 0.0])
    amount = np.array([1000.0, 200.0])

    result = near_threshold_reliability(
        predicted_p, labels, amount, representment_cost_inr=100.0, band=0.05
    )

    assert result["count"] == 1
    assert result["mean_predicted_p"] == 0.11
    assert result["observed_win_rate"] == 1.0


def test_near_threshold_reliability_reports_zero_count_when_nothing_is_near():
    predicted_p = np.array([0.9])
    labels = np.array([1.0])
    amount = np.array([1000.0])  # threshold = 100/1000 = 0.10, far from 0.9

    result = near_threshold_reliability(
        predicted_p, labels, amount, representment_cost_inr=100.0, band=0.05
    )

    assert result["count"] == 0
    assert np.isnan(result["gap"])
    assert result["median_threshold_overall"] == 0.10
