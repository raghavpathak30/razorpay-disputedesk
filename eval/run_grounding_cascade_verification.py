"""CLAUDE.md Day-2 Phase 3 done-criterion: prove the grounding cascade
(`disputedesk/evidence/grounding_cascade.py`) actually withholds a known
Class-B (fabricated/ungrounded) letter and names the offending sentence, and
that a known-clean letter passes rather than being withheld wholesale.

Requires a real MiniCheck model (`MINICHECK_MODEL_PATH` in the environment/
`.env` - see `disputedesk/evidence/minicheck.py`) and llama-cpp-python
installed. Not a pytest test: this repo keeps checks that need a real local
model or network call as standalone `eval/` scripts, not in the default
`tests/` suite (mirrors `eval/run_grounding_eval.py`'s split between
`--baseline-only` and the live LLM arm).

Run as `python -m eval.run_grounding_cascade_verification`.

**The Class-B fixture is not hand-written for this script.** It is
`d0000_unrec` from the frozen, committed corpus
(`data/reference/grounding_letters_seed0_n45.csv` + `eval/grounding_corpus.py`,
seed 0) - already confirmed fabricated-and-flagged in the n45 measurement
(`data/eval/n45/grounding_gate_scores.csv`), reconstructed here exactly via
`build_corpus`, not re-typed.

**The clean fixture is hand-constructed, not pulled from the n45/clean27
corpus.** That corpus's "clean" class was later shown to be largely
false-flagged under the v1 Groq gate's stricter evidence-citation reading
(DECISIONS.md's 2026-09-03/09-04 entries) - a different gate, a different
failure mode, but not a label this script should lean on uncritically. The
letter below asserts only the seven `DisputeContext` fields, each an exact
match, so every sentence is expected to resolve at stage 1 without needing
MiniCheck at all.
"""

import sys
from pathlib import Path

from disputedesk.config import get_settings
from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.grounding_cascade import apply_grounding_cascade
from disputedesk.evidence.letter import DraftedLetter, LetterProvenance
from disputedesk.evidence.minicheck import LlamaCppMiniCheckClient, MiniCheckUnavailableError
from eval.grounding_corpus import build_corpus
from eval.run_grounding_draft import CONTEXT_FIELDS

LETTERS_PATH = Path("data/reference/grounding_letters_seed0_n45.csv")

CLEAN_CONTEXT = DisputeContext(
    reason_code="VISA_10_4",
    amount=1240.00,
    avs_match=True,
    cvv_match=True,
    device_fingerprint_known=True,
    delivery_confirmed=True,
    prior_order_count=3,
)

CLEAN_LETTER_TEXT = (
    "We are contesting the chargeback filed under reason code VISA_10_4 for a "
    "transaction of INR 1240.00. The billing address and card verification value "
    "both matched at the time of purchase. The device used for this order was "
    "already known from the customer's account history. The order was delivered "
    "and its delivery was confirmed. This customer has placed 3 prior orders "
    "with us."
)


def _load_class_b_item():
    import pandas as pd

    frame = pd.read_csv(LETTERS_PATH)
    model_drafts = frame[frame["provenance"] == "model"]
    drafts = [
        (str(row["letter_text"]), DisputeContext(**{f: row[f] for f in CONTEXT_FIELDS}))
        for _, row in model_drafts.iterrows()
    ]
    items = build_corpus(drafts, seed=0)
    return next(item for item in items if item.item_id == "d0000_unrec")


def _letter(text: str) -> DraftedLetter:
    return DraftedLetter(
        letter_text=text, cites_evidence_types=("billing_proof",), provenance=LetterProvenance.MODEL
    )


def main() -> int:
    model_path = get_settings().minicheck_model_path
    if not model_path:
        print("MINICHECK_MODEL_PATH is not set; cannot run the live cascade verification.")
        return 1

    try:
        minicheck = LlamaCppMiniCheckClient(model_path)
    except MiniCheckUnavailableError as error:
        print(f"MiniCheck unavailable: {error}")
        return 1

    ok = True

    class_b = _load_class_b_item()
    print(f"=== Class B fixture: {class_b.item_id} (mutation={class_b.mutation}) ===")
    result = apply_grounding_cascade(_letter(class_b.letter_text), class_b.context, minicheck)
    print(f"withheld: {result.withheld}")
    print(f"failure_reason: {result.failure_reason}")
    failure = result.verdict.minicheck_failure if result.verdict else None
    if failure is not None:
        print(f"offending sentence (worst MiniCheck score): {failure.sentence!r} "
              f"score={failure.minicheck_score:.3f}")
    if not result.withheld:
        print("FAIL: known Class-B letter was not withheld")
        ok = False
    elif failure is None:
        print("FAIL: withheld, but no sentence-level MiniCheck failure was named")
        ok = False
    else:
        print("PASS: withheld, offending sentence named")
    print()

    print("=== Hand-constructed clean fixture ===")
    clean_result = apply_grounding_cascade(_letter(CLEAN_LETTER_TEXT), CLEAN_CONTEXT, minicheck)
    print(f"withheld: {clean_result.withheld}")
    print(f"failure_reason: {clean_result.failure_reason}")
    if clean_result.withheld:
        print("FAIL: known-clean letter was withheld")
        ok = False
    else:
        print("PASS: clean letter passed")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
