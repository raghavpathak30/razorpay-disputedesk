"""Step 1 of the grounding-gate measurement: draft real letters and commit
them as a fixture.

Split from the measurement runner on purpose. Drafting is the expensive,
non-reproducible half - it needs a live model and its output varies run to
run - so it happens once and its result is committed, exactly as
`data/reference/llm_normalization_arm_n60_seed0.csv` was after the TF-IDF
correction. Everything downstream (corpus construction, the baseline arm, the
statistics) then reproduces from the committed file with no API key.

Makes real network calls. Never run in CI, never imported by anything under
`tests/` except for the pure checkpoint/resume helpers below.

**Why the default is 250, not 120.** The false-flag rate on clean letters is
the number that decides whether the gate is economically viable
(`eval/review_cost.py`'s ₹150-budget line, 2.3% as of DECISIONS.md
2026-09-02). At n=120 a Wilson interval around a plausible low false-flag
rate straddles that budget and resolves nothing. At n=250 clean letters, the
95% Wilson upper bound if zero letters are flagged is about 1.5% - comfortably
under 2.3%, so a genuinely low rate is distinguishable from the budget rather
than merely consistent with clearing it. See DECISIONS.md's 2026-09-03
"Grounding-gate corpus resized before the key run" entry for the full power
table.

**Checkpointed and resumable (added 2026-09-03, after a live run lost 167
successful, budget-consuming API calls to one exhausted-retries 429 - nothing
had been persisted).** Every row is flushed to `--out` immediately after it
drafts, not held in memory until the loop finishes. Re-running the identical
command after an interruption skips positions already present in the output
file and drafts only what's missing - `generate_dataset` is deterministic at
a fixed `(n_letters, seed)`, so a resumed run's positions line up with the
original run's.

Run as `python -m eval.run_grounding_draft --n-letters 250 --seed 0`.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.draft_letter import PROMPT_VERSION, draft_explanation_letter
from disputedesk.evidence.llm import GroqHttpLLMClient
from disputedesk.evidence.normalize_comms import normalize_communication_log
from disputedesk.evidence.reason_code_map import required_evidence_types
from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset

CONTEXT_FIELDS = (
    "reason_code",
    "amount",
    "avs_match",
    "cvv_match",
    "device_fingerprint_known",
    "delivery_confirmed",
    "prior_order_count",
)


def already_drafted_positions(path: Path) -> set[int]:
    """`draft_index` values already checkpointed at `path`, or the empty set
    if it does not exist yet (a fresh run, not a resume)."""
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["draft_index"].tolist())


def merge_and_write(path: Path, new_rows: list[dict]) -> None:
    """Append `new_rows` to whatever checkpoint already exists at `path` (or
    create it) and write the merged result back, keyed on `draft_index` so a
    row flushed twice - e.g. by a resume that re-drafted a position already
    present - is kept once, not duplicated.
    """
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset="draft_index", keep="last")
    combined = combined.sort_values("draft_index").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Draft letters for the grounding-gate corpus.")
    parser.add_argument("--n-letters", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        help="Delay between letters (each letter is 2 calls) - paces requests "
        "to stay under the provider's burst rate limit rather than relying "
        "on retry-after-the-fact backoff alone.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/reference/grounding_letters_seed0.csv"),
        help="Committed alongside the code - this is a reproducibility artifact, "
        "not a scratch output, so it does not go to gitignored data/eval/. "
        "Checkpointed incrementally; re-running the same command resumes.",
    )
    args = parser.parse_args(argv)

    done = already_drafted_positions(args.out)
    if done:
        print(
            f"resuming: {len(done)}/{args.n_letters} positions already checkpointed in {args.out}"
        )

    features_df, _debug = generate_dataset(args.n_letters, args.seed, GeneratorConfig())
    llm = GroqHttpLLMClient()

    for position, idx in enumerate(features_df.index):
        if position in done:
            continue
        if position > 0 and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
        row = features_df.loc[idx]
        context = DisputeContext(**{f: row[f] for f in CONTEXT_FIELDS})
        comms = normalize_communication_log(str(row["customer_communication_log"]), llm)
        letter = draft_explanation_letter(
            context, required_evidence_types(context.reason_code), comms.normalized, llm
        )
        new_row = {
            "draft_index": position,
            "letter_text": letter.letter_text,
            "provenance": letter.provenance.value,
            **{f: row[f] for f in CONTEXT_FIELDS},
        }
        merge_and_write(args.out, [new_row])
        print(f"[{position + 1}/{args.n_letters}] {letter.provenance.value}", flush=True)

    frame = pd.read_csv(args.out)
    kept = int((frame["provenance"] == "model").sum())
    print(f"\nwrote {args.out} ({len(frame)} rows, {kept} with provenance='model')")
    print(f"prompt version: {PROMPT_VERSION}")
    print(
        "Only provenance='model' rows enter the corpus - a template fallback is "
        "not the drafting model's own output and grading it would measure the "
        "wrong thing."
    )


if __name__ == "__main__":
    main()
