"""Feature importance on the holdout, by permutation only.

Do not use LightGBM's built-in gain importance for any reported figure. On
this dataset gain ranks `checkout_hour_of_day` - a pure-noise control
(GENERATOR.md §3) - fourth, above four real causal features, because gain
favours high-cardinality continuous columns regardless of whether they
actually help predictions. Permutation importance measures the thing that
matters instead: how much holdout PR-AUC drops when a column is shuffled.
"""

import lightgbm as lgb
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_report(
    model: lgb.LGBMClassifier,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    random_state: int = 0,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Permutation importance of every feature, scored by holdout average
    precision (PR-AUC), sorted most-important first.
    """
    result = permutation_importance(
        model,
        X_holdout,
        y_holdout,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=random_state,
    )
    report = pd.DataFrame(
        {
            "feature": X_holdout.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)
    return report
