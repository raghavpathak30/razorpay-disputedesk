"""The TF-IDF + logistic-regression baseline for recovering `true_fraud` from
`customer_communication_log`.

Why this module exists (2026-09-02 remediation, defect 1.2). This baseline was
the bar the LLM's typed extraction was measured against, and the number quoted
for it - AUC 0.6371 - had no implementation anywhere in the repository. It
appeared once, as a bare `**Result:**` line appended to a 2026-08-31
DECISIONS.md *decision* entry, with no n, no seed, no command, and no
CONFIRMED-RAN status. Every downstream claim about the LLM adding no
predictive value rested on it.

Methodology is deliberately identical to
`eval.llm_normalization_quality.auc_of_normalized_fields` - logistic
regression, 5-fold stratified CV, ROC AUC, same `random_state` convention -
so the comparison isolates the feature-extraction step and nothing else.

**The vectorizer is fit inside each training fold**, never on the full corpus.
Fitting TF-IDF before splitting leaks test-fold vocabulary and idf weights
into training and inflates the baseline; since this baseline is the bar the
LLM is judged against, inflating it would bias the whole AI-judgment claim in
the direction the README already argued for. `tests/test_eval_tfidf_baseline.py`
holds a shuffled-label control against exactly that.

No network. Nothing here calls an LLM.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset

DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 0


def _make_pipeline(random_state: int) -> Pipeline:
    """Word-level TF-IDF with unigrams and bigrams into logistic regression.

    Frozen before the baseline was ever scored against the LLM arm: these are
    the ordinary defaults for a short-text classifier, chosen without looking
    at a result. Do not tune them after seeing a number - the point of a
    baseline is that it is not optimised to win or to lose.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )


def _folds(y: np.ndarray, n_splits: int, random_state: int) -> StratifiedKFold:
    if len(np.unique(y)) < 2:
        raise ValueError("cross-validated AUC needs both classes present in the labels")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def tfidf_out_of_fold_probabilities(
    logs: list[str],
    true_fraud,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """One predicted probability per item, produced by a model that never saw
    that item in training.

    Returned separately from `tfidf_auc` because the paired comparison in
    `eval.extraction_comparison` needs per-item scores, not a summary - a
    comparison of two mean AUCs cannot be paired, which is what went wrong.
    """
    documents = np.asarray(logs, dtype=object)
    y = np.asarray(true_fraud, dtype=int)
    oof = np.empty(len(y), dtype=float)

    for train_idx, test_idx in _folds(y, n_splits, random_state).split(documents, y):
        pipeline = _make_pipeline(random_state)
        pipeline.fit(documents[train_idx], y[train_idx])
        oof[test_idx] = pipeline.predict_proba(documents[test_idx])[:, 1]

    return oof


def tfidf_auc(
    logs: list[str],
    true_fraud,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Per-fold ROC AUC and its mean. Same return shape as the LLM arm's
    `auc_of_normalized_fields`, so the two can be printed side by side without
    either being reshaped at the call site.
    """
    documents = np.asarray(logs, dtype=object)
    y = np.asarray(true_fraud, dtype=int)

    fold_aucs = []
    for train_idx, test_idx in _folds(y, n_splits, random_state).split(documents, y):
        pipeline = _make_pipeline(random_state)
        pipeline.fit(documents[train_idx], y[train_idx])
        predicted_p = pipeline.predict_proba(documents[test_idx])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[test_idx], predicted_p)))

    return {
        "fold_aucs": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "n": len(y),
        "prevalence": float(y.mean()),
    }


def comms_and_true_fraud(
    n_rows: int, seed: int, generator_config: GeneratorConfig | None = None
) -> tuple[list[str], np.ndarray]:
    """The identical items the LLM arm is scored on: `generate_dataset`'s
    communication logs in generator index order, with the debug-only
    `true_fraud` column as the label (eval-use only, never a model or policy
    input - same convention as `eval.oracle`'s use of the debug `p` column).

    Index order is load-bearing: `eval.llm_normalization_quality.
    run_llm_normalization_sample` iterates `features_df.index` in this same
    order, which is what makes row `i` of the two arms the same dispute.
    """
    features_df, debug_df = generate_dataset(n_rows, seed, generator_config or GeneratorConfig())
    logs = [str(features_df.loc[idx, "customer_communication_log"]) for idx in features_df.index]
    labels = np.array([bool(debug_df.loc[idx, "true_fraud"]) for idx in features_df.index])
    return logs, labels


def load_llm_arm_fixture(path) -> pd.DataFrame:
    """The recorded LLM arm, committed so the paired comparison reproduces
    without an API key. See `data/reference/`'s own header for provenance.
    """
    return pd.read_csv(path)
