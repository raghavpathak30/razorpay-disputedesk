"""Step 2 of the grounding-gate measurement: build the corpus from the
committed letters, run both arms on identical items, and report each class
separately with its interval.

The baseline arm needs no API key and no network. The gate arm does. Run
`--baseline-only` to reproduce everything that does not need a key, including
the corpus composition table the README publishes.

Run as `python -m eval.run_grounding_eval` (both arms, live) or
`python -m eval.run_grounding_eval --baseline-only` (no key, no network).
"""

import argparse
from pathlib import Path

import pandas as pd

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.grounding import PROMPT_VERSION
from disputedesk.evidence.llm import GroqHttpLLMClient
from eval.grounding_baseline import baseline_flags
from eval.grounding_corpus import build_corpus, composition
from eval.grounding_eval import report, score_corpus
from eval.grounding_stats import wilson
from eval.run_grounding_draft import CONTEXT_FIELDS

DEFAULT_LETTERS = Path("data/reference/grounding_letters_seed0.csv")


def load_drafts(path: Path) -> list[tuple[str, DisputeContext]]:
    frame = pd.read_csv(path)
    model_drafts = frame[frame["provenance"] == "model"]
    return [
        (str(row["letter_text"]), DisputeContext(**{f: row[f] for f in CONTEXT_FIELDS}))
        for _, row in model_drafts.iterrows()
    ]


def _baseline_only(items) -> str:
    """Everything reproducible without an API key: the composition table, and
    the baseline's own rates. Not a comparison - there is no gate arm here -
    so nothing in this output may be reported as a gate result."""
    lines = [f"corpus: {composition(items)}", ""]
    for item_class in ("clean", "contradiction", "unrecorded"):
        subset = [i for i in items if i.item_class == item_class]
        if not subset:
            continue
        flagged = sum(baseline_flags(i.letter_text, i.context) for i in subset)
        correct = len(subset) - flagged if item_class == "clean" else flagged
        lines.append(str(wilson(correct, len(subset), f"baseline on {item_class}")))
    lines.append("")
    lines.append("BASELINE ARM ONLY - no gate arm was run, so no comparison is reported.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure the grounding gate against the baseline.")
    parser.add_argument("--letters", type=Path, default=DEFAULT_LETTERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    args = parser.parse_args(argv)

    if not args.letters.exists():
        raise SystemExit(
            f"{args.letters} does not exist. Run `python -m eval.run_grounding_draft` first "
            "(needs LLM_API_KEY) - the corpus is built from real drafted letters, and "
            "substituting template text would measure the wrong thing."
        )

    items = build_corpus(load_drafts(args.letters), seed=args.seed)

    if args.baseline_only:
        print(_baseline_only(items))
        return

    llm = GroqHttpLLMClient()
    scores = score_corpus(items, llm, sleep_seconds=args.sleep_seconds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out_dir / "grounding_gate_scores.csv", index=False)

    print(report(items, scores, usage=llm.usage_log))
    print(f"\nprompt version: {PROMPT_VERSION}")
    print(
        "\nFeed the false-flag rate above into `eval/review_cost.py` to see what it "
        "does to the cost sweep's break-even."
    )


if __name__ == "__main__":
    main()
