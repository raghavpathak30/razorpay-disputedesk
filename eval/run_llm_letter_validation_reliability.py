"""Runs `eval.llm_letter_validation_reliability` against the real Groq API
for both of `disputedesk/cli/demo.py`'s Segment B fixtures
(`CONTEST_WORTHY_EVENT` / `WEAK_EVIDENCE_EVENT` - imported directly from
there, not duplicated), 20 letter-drafting attempts each by default, and
prints a side-by-side failure-rate report plus median completion/reasoning
token usage per fixture (from `GroqHttpLLMClient.usage_log` - a separate
client instance per fixture, so each fixture's usage is isolated). If one
dispute's failure rate is materially higher, also prints up to 3 raw failing
completions and the exact validation error each tripped. Also prints a
handful of successful weak-evidence letters in full, for a human to read.

The classification methodology itself (`eval/llm_letter_validation_reliability.py`:
`run_letter_reliability_sample`, `run_one_draft_attempt`, `failure_rate`) is
unchanged from the 2026-09-01 "Letter-drafting validation reliability"
measurement - only this entry point gained the usage/sample-letter reporting
below, so a rerun after a `disputedesk/evidence/llm.py` fix is a controlled
before/after comparison, not a new methodology.

Makes real network calls and costs real (free-tier) API usage - never run
in CI, never imported by anything under `tests/`.

Run as `python -m eval.run_llm_letter_validation_reliability`.
Run as `python -m eval.run_llm_letter_validation_reliability --reasoning-effort medium`
to compare against a higher reasoning-effort setting.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from disputedesk.cli.demo import CONTEST_WORTHY_EVENT, WEAK_EVIDENCE_EVENT
from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.llm import GroqHttpLLMClient
from disputedesk.evidence.normalize_comms import normalize_communication_log
from disputedesk.evidence.reason_code_map import required_evidence_types
from disputedesk.evidence.schemas import ExplanationLetterOutput
from disputedesk.evidence.validated_call import _parse
from eval.llm_letter_validation_reliability import (
    DraftAttemptRecord,
    failure_rate,
    run_letter_reliability_sample,
)

_LABELED_EVENTS = (
    ("full_evidence (MC_4837)", CONTEST_WORTHY_EVENT),
    ("weak_evidence (VISA_83)", WEAK_EVIDENCE_EVENT),
)


def _context_from_entity(entity: dict) -> DisputeContext:
    return DisputeContext(
        reason_code=entity["reason_code"],
        amount=entity["amount"],
        avs_match=entity["avs_match"],
        cvv_match=entity["cvv_match"],
        device_fingerprint_known=entity["device_fingerprint_known"],
        delivery_confirmed=entity["delivery_confirmed"],
        prior_order_count=entity["prior_order_count"],
    )


def _run_for_event(
    event: dict, reasoning_effort: str, n_runs: int, sleep_seconds: float
) -> tuple[list[DraftAttemptRecord], list[dict]]:
    """A fresh `GroqHttpLLMClient` per fixture, so `usage_log` covers only
    this fixture's calls. The one `normalize_communication_log` call happens
    first and is excluded from the returned usage list - the report's
    "tokens per call" is about the letter-drafting calls specifically, the
    thing this measurement is about.
    """
    llm_client = GroqHttpLLMClient(reasoning_effort=reasoning_effort)
    entity = event["payload"]["dispute"]["entity"]
    context = _context_from_entity(entity)
    evidence_types = required_evidence_types(context.reason_code)
    normalize_result = normalize_communication_log(
        entity["customer_communication_log"], llm_client
    )
    draft_usage_start = len(llm_client.usage_log)
    records = run_letter_reliability_sample(
        context, evidence_types, normalize_result.normalized, llm_client, n_runs, sleep_seconds
    )
    return records, llm_client.usage_log[draft_usage_start:]


def _write_csv(path: Path, results: dict[str, list[DraftAttemptRecord]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dispute",
                "run_index",
                "first_draft_valid",
                "first_draft_error",
                "repair_attempted",
                "repair_succeeded",
                "repair_error",
                "final_path",
            ]
        )
        for label, records in results.items():
            for r in records:
                writer.writerow(
                    [
                        label,
                        r.run_index,
                        r.first_draft_valid,
                        r.first_draft_error or "",
                        r.repair_attempted,
                        "" if r.repair_succeeded is None else r.repair_succeeded,
                        r.repair_error or "",
                        r.final_path,
                    ]
                )


def _write_raw_failures_json(path: Path, results: dict[str, list[DraftAttemptRecord]]) -> None:
    failures = []
    for label, records in results.items():
        for r in records:
            if not r.first_draft_valid:
                failures.append(
                    {
                        "dispute": label,
                        "run_index": r.run_index,
                        "attempt": "first",
                        "raw_response": r.raw_responses[0],
                        "error": r.first_draft_error,
                    }
                )
            if r.repair_attempted and not r.repair_succeeded:
                failures.append(
                    {
                        "dispute": label,
                        "run_index": r.run_index,
                        "attempt": "repair",
                        "raw_response": r.raw_responses[1],
                        "error": r.repair_error,
                    }
                )
    path.write_text(json.dumps(failures, indent=2))


def _print_usage_medians(label: str, usage: list[dict]) -> None:
    completion = [u["completion_tokens"] for u in usage if u["completion_tokens"] is not None]
    reasoning = [u["reasoning_tokens"] for u in usage if u["reasoning_tokens"] is not None]
    print(f"\n{label} token usage per letter-drafting call (n={len(usage)} calls):")
    if completion:
        print(f"  median completion_tokens : {statistics.median(completion):.0f}")
    if reasoning:
        print(f"  median reasoning_tokens  : {statistics.median(reasoning):.0f}")


def _print_report(
    results: dict[str, list[DraftAttemptRecord]], usage_by_label: dict[str, list[dict]]
) -> None:
    print()
    print("=" * 72)
    print("Letter-drafting validation reliability, live Groq, per dispute")
    print("=" * 72)
    for label, records in results.items():
        n = len(records)
        first_valid = sum(r.first_draft_valid for r in records)
        repaired = sum(r.repair_attempted for r in records)
        repair_ok = sum(bool(r.repair_succeeded) for r in records)
        print(f"\n{label} (n={n}):")
        print(f"  first draft passed validation : {first_valid}/{n} ({first_valid / n:.0%})")
        print(f"  repair attempted               : {repaired}/{n} ({repaired / n:.0%})")
        if repaired:
            print(
                f"  repair succeeded (of attempted): {repair_ok}/{repaired} "
                f"({repair_ok / repaired:.0%})"
            )
        print(f"  final path = template_fallback : {failure_rate(records):.0%}")
        _print_usage_medians(label, usage_by_label[label])

    labels = list(results)
    if len(labels) == 2:
        rate_a, rate_b = failure_rate(results[labels[0]]), failure_rate(results[labels[1]])
        print(f"\nfailure rate side by side: {labels[0]}={rate_a:.0%}  {labels[1]}={rate_b:.0%}")


def _print_sample_letters(label: str, records: list[DraftAttemptRecord], n: int) -> None:
    successes = [r for r in records if r.final_path == "letter"]
    print()
    print("=" * 72)
    print(f"{min(n, len(successes))} sample letter(s) for {label}, verbatim")
    print("=" * 72)
    for r in successes[:n]:
        raw = r.raw_responses[-1]  # the response that actually validated
        letter = _parse(raw, ExplanationLetterOutput)
        print(f"\n--- run {r.run_index} ---")
        print(letter.letter_text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=8.0,
        help="Delay between LLM calls, to stay under Groq free-tier rate limits.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
    )
    parser.add_argument(
        "--sample-letters",
        type=int,
        default=2,
        help="How many successful weak-evidence letters to print in full.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    args = parser.parse_args(argv)

    results: dict[str, list[DraftAttemptRecord]] = {}
    usage_by_label: dict[str, list[dict]] = {}
    for label, event in _LABELED_EVENTS:
        records, usage = _run_for_event(
            event, args.reasoning_effort, args.n_runs, args.sleep_seconds
        )
        results[label] = records
        usage_by_label[label] = usage

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "llm_letter_validation_reliability.csv", results)
    _write_raw_failures_json(
        args.out_dir / "llm_letter_validation_reliability_raw_failures.json", results
    )
    _print_report(results, usage_by_label)

    weak_label = _LABELED_EVENTS[1][0]
    if args.sample_letters:
        _print_sample_letters(weak_label, results[weak_label], args.sample_letters)


if __name__ == "__main__":
    main()
