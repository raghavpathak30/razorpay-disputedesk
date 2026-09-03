"""Calibration: does `P(win) = 0.7` come out right about 70% of the time?
Computed on the holdout only, per CLAUDE.md invariant 2.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


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


def brier_score(predicted_p: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted `P(win)` and the realized {0,1}
    outcome (Brier, 1950) - a proper scoring rule, unlike ECE's binned
    summary, so it can't be gamed by bin placement. Thin wrapper around
    sklearn so every caller in this package uses the same definition.
    """
    return float(
        brier_score_loss(np.asarray(labels, dtype=float), np.asarray(predicted_p, dtype=float))
    )


def near_threshold_reliability(
    predicted_p: np.ndarray,
    labels: np.ndarray,
    amount: np.ndarray,
    representment_cost_inr: float,
    band: float = 0.05,
) -> dict:
    """Calibration restricted to the rows that actually matter for the
    Elkan/EV decision: those whose predicted `p_win` sits within `band` of
    *that row's own* derived threshold `representment_cost_inr / amount`
    (Phase 2 STEP A - the threshold is per-dispute, not a scalar, so "near
    the threshold" must be evaluated per-dispute too, not against one global
    cutoff). This is where a calibration gap would actually flip a
    contest/accept decision; calibration elsewhere on the curve does not.
    """
    predicted_p = np.asarray(predicted_p, dtype=float)
    labels = np.asarray(labels, dtype=float)
    amount = np.asarray(amount, dtype=float)
    per_row_threshold = representment_cost_inr / amount
    near_mask = np.abs(predicted_p - per_row_threshold) <= band

    if not near_mask.any():
        return {
            "band": band,
            "count": 0,
            "mean_predicted_p": float("nan"),
            "observed_win_rate": float("nan"),
            "gap": float("nan"),
            "median_threshold_overall": float(np.median(per_row_threshold)),
        }

    mean_p = float(predicted_p[near_mask].mean())
    observed = float(labels[near_mask].mean())
    return {
        "band": band,
        "count": int(near_mask.sum()),
        "mean_predicted_p": mean_p,
        "observed_win_rate": observed,
        "gap": mean_p - observed,
        "median_threshold_overall": float(np.median(per_row_threshold)),
    }
