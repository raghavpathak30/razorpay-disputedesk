"""Formats the headline precision/recall numbers so the operating threshold
travels with them everywhere they're quoted - README, pitch video, submission
checklist items 3 and 4 (SPEC.md §9). Phase 2 has no policy engine yet, so
that threshold is a documented placeholder (train-split label prevalence),
not a policy decision, and a reader must see that in the same sentence as the
numbers, not several rows away in a table.
"""

import pandas as pd


def _fmt(summary: pd.DataFrame, metric: str) -> str:
    row = summary.loc[metric]
    return f"{row['median']:.4f} (IQR {row['q25']:.4f}-{row['q75']:.4f})"


def format_precision_recall_headline(summary: pd.DataFrame) -> str:
    """One sentence bundling precision, recall, and the threshold they were
    measured at - written so pulling precision or recall out on its own means
    visibly cutting a sentence, not just picking cells from a table.
    """
    return (
        f"precision {_fmt(summary, 'precision_at_threshold')}, "
        f"recall {_fmt(summary, 'recall_at_threshold')}, "
        f"at threshold {_fmt(summary, 'threshold')} = train-split label prevalence "
        "(placeholder pending Phase 3's policy engine, SPEC.md §4 - not a policy decision)"
    )
