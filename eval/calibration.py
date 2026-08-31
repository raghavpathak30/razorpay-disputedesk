"""Calibration: does `P(win) = 0.7` come out right about 70% of the time?
Computed on the holdout only, per CLAUDE.md invariant 2.
"""

import numpy as np
import pandas as pd


def calibration_table(
    predicted_p: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Bucket predictions into `n_bins` equal-width bins over [0, 1]; for each
    non-empty bin, report the mean predicted probability against the observed
    win rate in that bin.
    """
    predicted_p = np.asarray(predicted_p, dtype=float)
    labels = np.asarray(labels, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(predicted_p, bin_edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_index == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": bin_edges[b],
                "bin_high": bin_edges[b + 1],
                "count": int(mask.sum()),
                "mean_predicted_p": predicted_p[mask].mean(),
                "observed_win_rate": labels[mask].mean(),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    predicted_p: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Sample-weighted mean absolute gap between predicted probability and
    observed win rate across bins - one number summarizing the table above.
    """
    table = calibration_table(predicted_p, labels, n_bins=n_bins)
    if table.empty:
        return float("nan")
    gaps = (table["mean_predicted_p"] - table["observed_win_rate"]).abs()
    weights = table["count"] / table["count"].sum()
    return float((gaps * weights).sum())
