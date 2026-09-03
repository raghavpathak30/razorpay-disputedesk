"""Scores the grounding gate and the deterministic baseline on identical
corpus items, paired, and reports each class separately.

Three numbers, never pooled:

- **False-flag rate on clean letters.** The headline. A gate that flags
  everything detects everything and is worthless, and its cost is the review
  queue - see `eval/review_cost.py` for what that does to the sweep.
- **Class A (contradiction) detection.** The baseline is expected to do well.
  Reported to show the gate does not lose the easy half.
- **Class B (unrecorded assertion) detection.** The load-bearing claim.

The gate arm makes real network calls through whatever `LLMClient` it is
given; `FakeLLMClient` drives it in tests. The baseline arm never does. The
pure scoring and reporting functions are unit-tested; the live measurement is
a script run recorded in DECISIONS.md, per CLAUDE.md's no-network-in-tests
rule.
"""

import time
from dataclasses import dataclass

import pandas as pd

from disputedesk.evidence.grounding import grade_letter
from disputedesk.evidence.llm import LLMClient
from eval.grounding_baseline import baseline_flags
from eval.grounding_corpus import CorpusItem, composition
from eval.grounding_stats import PairedComparison, paired_comparison, wilson
from eval.review_cost import budget_verdict

CLASS_LABELS = {
    "clean": "false-flag on clean letters",
    "contradiction": "Class A (contradiction) detection",
    "unrecorded": "Class B (unrecorded assertion) detection",
}


@dataclass(frozen=True)
class ItemScore:
    item_id: str
    item_class: str
    mutation: str | None
    gate_flagged: bool
    baseline_flagged: bool
    gate_failed: bool  # the gate could not reach a verdict (counts as flagged)
    n_assertions: int


class _GradeableText:
    """Duck-types `DraftedLetter` for exactly what `grounding.build_prompt`
    reads (`.letter_text`) - nothing else.

    Not a `DraftedLetter`, deliberately (found running the real n=250 corpus,
    2026-09-03): `make_unrecorded` (`eval/grounding_corpus.py`) inserts a
    fabricated sentence into an already-drafted letter, which can push the
    *mutated* corpus text past `DraftedLetter`'s 1,000-character submission
    ceiling even when the original drafted letter was comfortably under it.
    Corpus text is a deliberate test perturbation that is never filed
    anywhere - grading it does not require it to satisfy the production
    submission schema, only that the grader can read it. Routing it through
    `DraftedLetter` crashed the whole eval run with a `ValidationError` on any
    such item instead of scoring it.
    """

    __slots__ = ("letter_text",)

    def __init__(self, letter_text: str) -> None:
        self.letter_text = letter_text


def score_item(item: CorpusItem, llm_client: LLMClient) -> ItemScore:
    verdict, failure = grade_letter(_GradeableText(item.letter_text), item.context, llm_client)
    # A gate that cannot reach a verdict withholds the letter, so for scoring
    # it counts as a flag - that is what production does, and scoring it any
    # other way would report a gate that is not the one being shipped.
    gate_flagged = failure is not None or not verdict.grounded
    return ItemScore(
        item_id=item.item_id,
        item_class=item.item_class,
        mutation=item.mutation,
        gate_flagged=gate_flagged,
        baseline_flagged=baseline_flags(item.letter_text, item.context),
        gate_failed=failure is not None,
        n_assertions=0 if verdict is None else len(verdict.assertions),
    )


def score_corpus(
    items: list[CorpusItem], llm_client: LLMClient, sleep_seconds: float = 0.0
) -> pd.DataFrame:
    rows = []
    for i, item in enumerate(items):
        if i > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)
        rows.append(score_item(item, llm_client).__dict__)
    return pd.DataFrame(rows)


def compare_class(scores: pd.DataFrame, item_class: str) -> PairedComparison:
    """Gate vs baseline on one class, paired on identical items.

    "Correct" is oriented so higher is better in both directions: on a
    positive class it is "flagged"; on the clean class it is "did not flag".
    """
    subset = scores[scores["item_class"] == item_class]
    if subset.empty:
        raise ValueError(f"no items of class {item_class!r} in the corpus")
    if item_class == "clean":
        gate_correct = ~subset["gate_flagged"].to_numpy()
        baseline_correct = ~subset["baseline_flagged"].to_numpy()
    else:
        gate_correct = subset["gate_flagged"].to_numpy()
        baseline_correct = subset["baseline_flagged"].to_numpy()
    return paired_comparison(CLASS_LABELS[item_class], gate_correct, baseline_correct)


def false_flag_rate(scores: pd.DataFrame):
    """The headline, as a rate rather than a comparison: the fraction of clean
    letters the gate withholds. This is the number `eval/review_cost.py` takes
    as its argument."""
    clean = scores[scores["item_class"] == "clean"]
    return wilson(
        int(clean["gate_flagged"].sum()), int(len(clean)), "gate false-flag rate on clean letters"
    )


def report(items: list[CorpusItem], scores: pd.DataFrame, usage: list[dict] | None = None) -> str:
    lines = ["=== grounding gate vs deterministic baseline ===", ""]
    table = composition(items)
    lines.append(f"corpus: n={table['n_items']} items, by class {table['by_class']}")
    lines.append("")

    rate = false_flag_rate(scores)
    lines.append(str(rate))
    lines.append(budget_verdict(rate))
    lines.append("")
    for item_class in ("clean", "contradiction", "unrecorded"):
        if (scores["item_class"] == item_class).any():
            lines.append(str(compare_class(scores, item_class)))
    lines.append("")

    failed = int(scores["gate_failed"].sum())
    lines.append(
        f"gate could not reach a verdict on {failed}/{len(scores)} items (counted as flagged)"
    )

    if usage:
        prompt = sum(u.get("prompt_tokens") or 0 for u in usage)
        completion = sum(u.get("completion_tokens") or 0 for u in usage)
        n = len(usage)
        lines.append(
            f"tokens: {prompt + completion} over {n} graded letters "
            f"({(prompt + completion) / n:.0f} per letter: "
            f"{prompt / n:.0f} prompt + {completion / n:.0f} completion)"
        )
    return "\n".join(lines)
