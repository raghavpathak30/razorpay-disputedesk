"""Second pre-registration (DECISIONS.md, 2026-09-03 "Second pre-registration:
why the clean-class false-flag rate is so high"): re-grade the exact 27
clean-class items that received a grader verdict in the original n=45 run, now
that verdict persistence (`assertions_json`, `failure_reason`) is active, and
categorize every "unsupported" assertion by whether `RECORD_FIELDS` could ever
have supported it.

n=27, fixed, not adjustable - these are exactly the clean-class draft indices
whose `gate_failed` was `False` in the original run
(`data/eval/n45/grounding_gate_scores.csv`, pre-dating the persistence patch).
Hardcoded below rather than re-derived from that file, so this script cannot
silently pick up a different set on a re-run.

Same letters (`data/reference/grounding_letters_seed0_n45.csv`), same seed
(0), same `grounding_gate_v1` prompt as the original run. This is a re-grade,
not a re-draft: no new letters are drafted, only the existing 27 letters are
sent to the grader again so the *verdict content* - discarded by the original
run - is captured this time.

Makes real network calls (needs `LLM_API_KEY`). Never run in CI.

Run as `python -m eval.run_grounding_reregrade_clean27`.
"""

import json
from pathlib import Path

import pandas as pd

from disputedesk.evidence.llm import GroqHttpLLMClient
from eval.grounding_corpus import build_corpus
from eval.grounding_eval import score_item
from eval.run_grounding_eval import load_drafts

LETTERS_PATH = Path("data/reference/grounding_letters_seed0_n45.csv")
OUT_PATH = Path("data/eval/n27_reregrade/grounding_gate_scores_clean27.csv")

# The 27 clean-class draft indices that received a grader verdict
# (gate_failed=False) in the original n=45 run - fixed by the
# pre-registration, not re-derived here. d0012 and d0027 failed for reasons
# unrelated to budget exhaustion (isolated failures, not a contiguous tail);
# d0029-d0044 failed as the contiguous budget-exhaustion tail. Neither set is
# part of this re-grade.
PRE_REGISTERED_DRAFT_INDICES: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    28,
)


def select_pre_registered_items(all_items):
    """Filter a full corpus down to exactly the 27 pre-registered clean items.

    Raises `SystemExit` rather than silently proceeding on a changed scope -
    either the fixed-n list itself was edited, or `all_items` was built from
    a different letters file/seed than the pre-registration specifies.
    """
    if len(PRE_REGISTERED_DRAFT_INDICES) != 27:
        raise SystemExit(
            f"pre-registered n is 27; this list has {len(PRE_REGISTERED_DRAFT_INDICES)} - "
            "stop, do not proceed on a changed scope"
        )
    wanted_ids = {f"d{i:04d}_clean" for i in PRE_REGISTERED_DRAFT_INDICES}
    items = [item for item in all_items if item.item_id in wanted_ids]
    found_ids = {item.item_id for item in items}
    if found_ids != wanted_ids:
        missing = wanted_ids - found_ids
        raise SystemExit(f"pre-registered item ids not found in corpus: {sorted(missing)}")
    return items


def main() -> None:
    drafts = load_drafts(LETTERS_PATH)
    all_items = build_corpus(drafts, seed=0)
    items = select_pre_registered_items(all_items)

    llm = GroqHttpLLMClient()
    print("tokens used before starting this run: 0 (fresh client, usage_log starts empty)")
    print(f"grading {len(items)} clean-class letters (pre-registered, fixed n)...\n")

    rows = []
    for i, item in enumerate(items, start=1):
        score = score_item(item, llm)
        rows.append(score.__dict__)
        assertions = json.loads(score.assertions_json)
        print(
            f"[{i}/{len(items)}] {item.item_id} "
            f"gate_flagged={score.gate_flagged} gate_failed={score.gate_failed} "
            f"n_assertions={score.n_assertions}"
        )
        for a in assertions:
            print(
                f"    verdict={a['verdict']!r} supporting_field={a['supporting_field']!r} "
                f"quote={a['quote']!r}"
            )
        if score.failure_reason:
            print(f"    failure_reason={score.failure_reason!r}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}")

    total_prompt = sum(u.get("prompt_tokens") or 0 for u in llm.usage_log)
    total_completion = sum(u.get("completion_tokens") or 0 for u in llm.usage_log)
    n_calls = len(llm.usage_log)
    print(
        f"\ntokens used by this run: {total_prompt + total_completion} "
        f"over {n_calls} API calls ({total_prompt} prompt + {total_completion} completion)"
    )

    failed = sum(1 for r in rows if r["gate_failed"])
    print(f"grader failures this run: {failed}/{len(rows)}")


if __name__ == "__main__":
    main()
