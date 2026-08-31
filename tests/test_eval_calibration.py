"""Calibration: does P(win)=0.7 come out right about 70% of the time?"""

import numpy as np

from eval.calibration import calibration_table, expected_calibration_error


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
