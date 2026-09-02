"""Step 1 of the grounding-gate measurement: draft real letters and commit
them as a fixture.

Split from the measurement runner on purpose. Drafting is the expensive,
non-reproducible half - it needs a live model and its output varies run to
run - so it happens once and its result is committed, exactly as
`data/reference/llm_normalization_arm_n60_seed0.csv` was after the TF-IDF
correction. Everything downstream (corpus construction, the baseline arm, the
statistics) then reproduces from the committed file with no API key.

Makes real network calls. Never run in CI, never imported by anything under
`tests/`.

Run as `python -m eval.run_grounding_draft --n-letters 120 --seed 0`.
"""

import argparse
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Draft letters for the grounding-gate corpus.")
    parser.add_argument("--n-letters", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/reference/grounding_letters_seed0.csv"),
        help="Committed alongside the code - this is a reproducibility artifact, "
        "not a scratch output, so it does not go to gitignored data/eval/.",
    )
    args = parser.parse_args(argv)

    features_df, _debug = generate_dataset(args.n_letters, args.seed, GeneratorConfig())
    llm = GroqHttpLLMClient()

    rows = []
    for position, idx in enumerate(features_df.index):
        row = features_df.loc[idx]
        context = DisputeContext(**{f: row[f] for f in CONTEXT_FIELDS})
        comms = normalize_communication_log(str(row["customer_communication_log"]), llm)
        letter = draft_explanation_letter(
            context, required_evidence_types(context.reason_code), comms.normalized, llm
        )
        rows.append(
            {
                "draft_index": position,
                "letter_text": letter.letter_text,
                "provenance": letter.provenance.value,
                **{f: row[f] for f in CONTEXT_FIELDS},
            }
        )
        print(f"[{position + 1}/{args.n_letters}] {letter.provenance.value}", flush=True)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

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
