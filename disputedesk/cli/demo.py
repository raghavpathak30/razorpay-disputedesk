"""Demo script (PHASES.md Phase 4 item 8): replays fixture dispute events
through the real webhook route end to end - webhook in, features, `P(win)`,
policy decision, evidence assembly, API call, audit row out - and
demonstrates both SPEC.md §7 failure paths live.

Two segments, printed under their own headers:

- **Segment A - Deterministic.** Every step above, plus the 429/Retry-After
  recovery path. Runnable from a clean clone with no `.env` and no network:
  the LLM and Razorpay clients are fakes (CLAUDE.md: "No test may make a
  network call. Fake both the LLM and the Razorpay API."), and the model is
  trained in memory from the same synthetic generator every other command in
  this project already uses. This segment's stdout is byte-identical across
  cold clones (same seed, same fixtures, no LLM, no wall-clock timestamps).
  Pass `--deterministic-only` to run only this segment - this is what any
  reproducibility check should invoke, since Segment B below never can be.
- **Segment B - LLM output (not reproducible).** Two disputes that differ on
  both reason code and evidence availability, drafted by a real
  `GroqHttpLLMClient` (SPEC.md §2's other allowed LLM job) and printed
  verbatim. Needs a populated `.env` (all of it - `get_settings()` validates
  one `Settings` object, not per-feature) and network access; skipped with a
  clear message if that's not available, never a crash.

Run: `python -m disputedesk.cli.demo` (both segments)
Run: `python -m disputedesk.cli.demo --deterministic-only` (Segment A only)

By default the audit DB is an in-memory SQLite database, so each run starts
clean. Pass `--db-path` to use a file instead - this is what lets the same
event be replayed across two separate process invocations, the scenario the
decision-before-API-call ordering (PHASES.md Phase 4 item 3) actually exists
for; the in-memory default only proves idempotency within one process.
Run: `python -m disputedesk.cli.demo --db-path data/demo/disputedesk.db`
"""

import argparse
import json
import os
import time
import warnings
from unittest import mock

import httpx

# fastapi.testclient imports starlette.testclient, which warns on every
# import that `httpx2` isn't installed (it falls back to `httpx`, which this
# project already depends on and uses correctly - see tests/test_client_razorpay.py
# and tests/test_api_webhook.py's own TestClient use). Silenced narrowly by
# warning class, not by category, so an unrelated deprecation elsewhere in
# this script still surfaces.
with warnings.catch_warnings():
    from starlette.exceptions import StarletteDeprecationWarning

    warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
    from fastapi.testclient import TestClient

from disputedesk.api.webhook import (
    app,
    get_db_session,
    get_llm_client,
    get_model,
    get_razorpay_client,
)
from disputedesk.audit.db import get_engine, init_db, make_session_factory
from disputedesk.audit.log import get_audit_trail
from disputedesk.client.razorpay import FakeRazorpayClient, RazorpayHttpClient
from disputedesk.config import get_settings
from disputedesk.evidence.assembler import assemble_evidence_packet
from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.llm import FakeLLMClient, GroqHttpLLMClient
from disputedesk.evidence.reason_code_map import required_evidence_types
from disputedesk.features.matrix import build_feature_matrix
from disputedesk.generator.config import GeneratorConfig
from disputedesk.generator.pipeline import generate_dataset, temporal_split
from disputedesk.model.config import ModelConfig
from disputedesk.model.train import train
from disputedesk.retry import call_with_backoff

DEMO_MODEL_VERSION = "demo-lgbm-v1-seed7"

VALID_NORMALIZED = json.dumps(
    {
        "claims_unauthorized_transaction": True,
        "mentions_prior_bank_contact": True,
        "mentions_shared_card_access": False,
        "mentions_travel": False,
        "tone": "polite",
        "is_substantive": True,
        "summary": "Customer says they don't recognize the charge, already contacted their bank.",
    }
)
VALID_LETTER = json.dumps(
    {
        "letter_text": (
            "We are contesting this chargeback. Our authentication and fulfillment "
            "records support that this was a genuine, authorized transaction, and "
            "the customer's own communication is consistent with our records. "
            "Full evidence is attached alongside this letter."
        ),
        "cites_evidence_types": ["billing_proof", "proof_of_service"],
    }
)

CONTEST_WORTHY_EVENT = {
    "event": "dispute.created",
    "payload": {
        "dispute": {
            "entity": {
                "id": "disp_demo_contest_001",
                "payment_id": "pay_demo_contest_001",
                "amount": 6500.0,
                "currency": "INR",
                "reason_code": "MC_4837",
                "phase": "chargeback",
                "status": "open",
                "avs_match": True,
                "cvv_match": True,
                "device_fingerprint_known": True,
                "delivery_confirmed": True,
                "prior_order_count": 6,
                "prior_dispute_count": 0,
                "ip_geo_billing_distance_km": 12.0,
                "days_between_purchase_and_dispute": 4.0,
                "customer_communication_log": (
                    "I don't recognize this charge on my card, I already called my bank."
                ),
                "card_network": "Mastercard",
                "checkout_hour_of_day": 14,
            }
        }
    },
}

# Deliberately weak-evidence, low-amount, filed-fast dispute (poor AVS/CVV,
# unrecognized device, no delivery proof, large IP-geo/billing distance, no
# order history, a prior dispute already on file) - the opposite feature
# profile from CONTEST_WORTHY_EVENT above. Shown alongside it so a viewer can
# see `P(win)` and the policy decision actually move with the input, instead
# of both demoed disputes displaying the same score.
WEAK_EVIDENCE_EVENT = {
    "event": "dispute.created",
    "payload": {
        "dispute": {
            "entity": {
                "id": "disp_demo_weak_001",
                "payment_id": "pay_demo_weak_001",
                "amount": 2200.0,
                "currency": "INR",
                "reason_code": "VISA_10_4",
                "phase": "chargeback",
                "status": "open",
                "avs_match": False,
                "cvv_match": False,
                "device_fingerprint_known": False,
                "delivery_confirmed": False,
                "prior_order_count": 0,
                "prior_dispute_count": 2,
                "ip_geo_billing_distance_km": 4800.0,
                "days_between_purchase_and_dispute": 1.0,
                "customer_communication_log": "this isnt me??? never bought this. refund now",
                "card_network": "Visa",
                "checkout_hour_of_day": 3,
            }
        }
    },
}


# Same feature profile as CONTEST_WORTHY_EVENT (real duplication would fail
# schema validation on `id` reuse) - only the id/payment_id differ, since
# this step needs a CONTEST decision so RazorpayHttpClient.contest() is the
# call that hits the stubbed 429.
RETRY_AFTER_EVENT = json.loads(json.dumps(CONTEST_WORTHY_EVENT))
RETRY_AFTER_EVENT["payload"]["dispute"]["entity"]["id"] = "disp_demo_429_recover"
RETRY_AFTER_EVENT["payload"]["dispute"]["entity"]["payment_id"] = "pay_demo_429_recover"

# Placeholder credentials for this step only - RazorpayHttpClient.__init__
# reads them via get_settings(), but the transport is stubbed below, so
# nothing with these values ever reaches a real network call.
_DEMO_SETTINGS_ENV = {
    "RAZORPAY_KEY_ID": "rzp_test_demo",
    "RAZORPAY_KEY_SECRET": "demo_secret",
    "LLM_API_KEY": "demo",
    "LLM_API_URL": "https://example.test/llm",
    "LLM_MODEL": "demo-model",
    "DATABASE_URL": "sqlite:///:memory:",
}

_DEMO_SLEEP_COMPRESSION = 0.01  # real sleep is compressed 100x; printed values are the real ones


def _train_demo_model():
    print("Training a small in-memory model for this demo (seed=7, n_rows=3000)...")
    generator_config = GeneratorConfig()
    model_config = ModelConfig()
    features_df, _debug_df = generate_dataset(3000, seed=7, config=generator_config)
    train_df, _test_df, _boundary = temporal_split(features_df, generator_config)
    X_train = build_feature_matrix(train_df)
    y_train = train_df["won_if_contested"]
    model = train(X_train, y_train, model_config)
    print(f"  trained on {len(train_df)} rows, model_version={DEMO_MODEL_VERSION}\n")
    return model


def _print_audit_trail(engine, dispute_id: str) -> None:
    session = make_session_factory(engine)()
    try:
        trail = get_audit_trail(session, dispute_id)
    finally:
        session.close()
    if trail is None:
        print(f"  no audit row found for {dispute_id}")
        return
    d = trail.decision
    print(f"  audit row (decisions table, dispute_id={d.dispute_id}):")
    print(f"    model_version        = {d.model_version}")
    print(f"    p_win                = {d.p_win:.4f}")
    print(f"    policy_branch        = {d.policy_branch}")
    print(f"    expected_value_inr   = {d.expected_value_inr:.2f}")
    print(f"    prompt_version       = {d.prompt_version}")
    print(f"    validation_result    = {d.validation_result}")
    print(f"    human_review_required= {d.human_review_required}")
    if trail.api_outcome is not None:
        o = trail.api_outcome
        print(f"  api outcome (api_outcomes table): action={o.action} outcome={o.outcome}")
        print(f"    response = {o.response_json}")
    else:
        print("  api outcome: none yet (escalated, or not filed)")


def demo_end_to_end(client: TestClient, engine) -> None:
    print("=" * 72)
    print("1. Webhook in: an 'open' MC_4837 dispute, INR 6,500")
    print("=" * 72)
    response = client.post("/webhooks/disputes", json=CONTEST_WORTHY_EVENT)
    print(f"  POST /webhooks/disputes -> {response.status_code}")
    print(f"  {response.json()}")
    _print_audit_trail(engine, "disp_demo_contest_001")

    print()
    print("=" * 72)
    print("2. Idempotency: replaying the exact same webhook event")
    print("=" * 72)
    response2 = client.post("/webhooks/disputes", json=CONTEST_WORTHY_EVENT)
    body2 = response2.json()
    print(f"  POST /webhooks/disputes (same dispute_id again) -> {response2.status_code}")
    print(f"  already_processed = {body2['already_processed']}")
    print("  (no second decision row, no second API call - see the audit row above, unchanged)")


def demo_second_dispute(client: TestClient, engine) -> None:
    print()
    print("=" * 72)
    print("3. A second dispute with a very different evidence profile")
    print("=" * 72)
    print("  (weak AVS/CVV, unrecognized device, no delivery proof, filed fast,")
    print("   no order history, a prior dispute on file - contrast with #1 above)")
    response = client.post("/webhooks/disputes", json=WEAK_EVIDENCE_EVENT)
    print(f"  POST /webhooks/disputes -> {response.status_code}")
    print(f"  {response.json()}")
    _print_audit_trail(engine, "disp_demo_weak_001")


def demo_malformed_webhook(client: TestClient) -> None:
    print()
    print("=" * 72)
    print("4. A malformed webhook (status is not 'open') is rejected, not processed")
    print("=" * 72)
    bad_event = json.loads(json.dumps(CONTEST_WORTHY_EVENT))
    bad_event["payload"]["dispute"]["entity"]["id"] = "disp_demo_malformed"
    bad_event["payload"]["dispute"]["entity"]["status"] = "won"
    response = client.post("/webhooks/disputes", json=bad_event)
    print(f"  POST /webhooks/disputes (status='won') -> {response.status_code} (expected 422)")


def demo_failure_path_llm_degrades(client: TestClient, engine) -> None:
    print()
    print("=" * 72)
    print("5. Failure path 2: the LLM returns invalid output twice in a row")
    print("=" * 72)
    print("   (repair attempt also fails -> deterministic template -> human_review flag)")
    broken_llm = FakeLLMClient(responses=["not json", "still not json"])
    app.dependency_overrides[get_llm_client] = lambda: broken_llm

    event = json.loads(json.dumps(CONTEST_WORTHY_EVENT))
    event["payload"]["dispute"]["entity"]["id"] = "disp_demo_llm_fallback"
    response = client.post("/webhooks/disputes", json=event)
    print(f"  POST /webhooks/disputes -> {response.status_code}")
    print(f"  {response.json()}")
    print("  the system did not crash; it degraded to a template letter:")
    _print_audit_trail(engine, "disp_demo_llm_fallback")

    app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient(
        responses=[VALID_NORMALIZED, VALID_LETTER]
    )


def demo_failure_path_timeout_retry() -> None:
    print()
    print("=" * 72)
    print("6. Failure path 1: the Razorpay API times out, then recovers")
    print("=" * 72)
    print("   (exercising the exact disputedesk.retry.call_with_backoff both")
    print("    disputedesk/client/razorpay.py and disputedesk/evidence/llm.py use)")

    attempts = {"n": 0}

    def flaky_call() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectTimeout("simulated network timeout")
        return "under_review"

    def on_attempt(attempt_number: int, outcome: str, error: Exception | None) -> None:
        if outcome == "timeout_retry":
            print(f"  attempt {attempt_number}: timed out ({error!r}) - backing off and retrying")
        elif outcome == "success":
            print(f"  attempt {attempt_number}: succeeded")

    result = call_with_backoff(
        flaky_call,
        max_retries=3,
        base_delay_seconds=0.01,  # demo-fast; production default is 0.5s
        sleep_fn=lambda _seconds: None,
        on_attempt=on_attempt,
    )
    print(f"  final result after recovery: status={result!r}")
    print(f"  total attempts made: {attempts['n']} (filed exactly once, not once per attempt)")


def _make_429_then_200_handler(retry_after_seconds: str, dispute_id: str):
    """A `httpx.MockTransport` handler: 429 with `Retry-After` on the first
    call, 200 on every call after - the transport-layer fixture that injects
    the failure, never touching `call_with_backoff` itself. Returns the
    handler plus the shared call counter, so a caller can report how many
    attempts were actually made.
    """
    responses_sent = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        responses_sent["n"] += 1
        if responses_sent["n"] == 1:
            print(f"  attempt 1: Razorpay responds 429, Retry-After: {retry_after_seconds}")
            return httpx.Response(429, headers={"retry-after": retry_after_seconds})
        print(f"  attempt {responses_sent['n']}: Razorpay responds 200")
        return httpx.Response(200, json={"id": dispute_id, "status": "under_review"})

    return handler, responses_sent


def _compressed_sleep_fn(seconds: float) -> None:
    """`RazorpayHttpClient`'s injected `sleep_fn` for this demo step: prints
    the real honoured interval `call_with_backoff` passed in, then sleeps a
    compressed fraction of it so the demo doesn't actually wait 3 seconds.
    """
    print(
        f"  call_with_backoff sleeps {seconds:.1f}s (the honoured Retry-After "
        f"value - matches the header above) - compressed to "
        f"{seconds * _DEMO_SLEEP_COMPRESSION:.3f}s of real wall-clock time for this demo"
    )
    time.sleep(seconds * _DEMO_SLEEP_COMPRESSION)


def demo_failure_path_429_retry_after(
    client: TestClient, engine, fake_razorpay: FakeRazorpayClient
) -> None:
    print()
    print("=" * 72)
    print("7. Failure path 3: Razorpay returns HTTP 429 with Retry-After, then recovers")
    print("=" * 72)
    print("   (a real 429 response is injected at the httpx transport layer via")
    print("    httpx.MockTransport - disputedesk.retry.call_with_backoff, the same")
    print("    helper failure path 1 above uses, honours the Retry-After header and")
    print("    retries. No new retry logic; this runs through the real")
    print("    RazorpayHttpClient over the real webhook route.)")

    handler, responses_sent = _make_429_then_200_handler("3", "disp_demo_429_recover")
    transport = httpx.MockTransport(handler)

    def _routed_request(method, url, **kwargs):
        with httpx.Client(transport=transport) as http_client:
            return http_client.request(method, url, **kwargs)

    # Scoped, not `os.environ.setdefault` (which would permanently leave these
    # placeholder values in the process environment - since pydantic-settings
    # prefers a real env var over a real `.env` file, that would silently
    # poison `get_settings()` for Segment B's real Groq client too).
    # `mock.patch.dict` restores the exact prior environment on exit.
    with mock.patch.dict(os.environ, _DEMO_SETTINGS_ENV, clear=False):
        get_settings.cache_clear()
        real_razorpay = RazorpayHttpClient(sleep_fn=_compressed_sleep_fn)
        with mock.patch("httpx.request", _routed_request):
            app.dependency_overrides[get_razorpay_client] = lambda: real_razorpay
            response = client.post("/webhooks/disputes", json=RETRY_AFTER_EVENT)
        app.dependency_overrides[get_razorpay_client] = lambda: fake_razorpay
    get_settings.cache_clear()  # drop the placeholder-backed Settings, restore real env

    print(f"  POST /webhooks/disputes -> {response.status_code}")
    print(f"  {response.json()}")
    print(f"  total attempts made against Razorpay: {responses_sent['n']} (filed exactly once)")
    _print_audit_trail(engine, "disp_demo_429_recover")


def _print_segment_header(title: str) -> None:
    print()
    print()
    print("#" * 72)
    print(f"# SEGMENT {title}")
    print("#" * 72)


# Demo-only, printing-only judgment of which of a reason code's required
# evidence types this *specific* dispute's own order-context facts actually
# support - not part of the production evidence assembler, which does not
# gate assembly on availability today (a stated gap, see ARCHITECTURE.md's
# "Known gaps": no document-upload pipeline, no per-type file existence
# check). Mirrors reason_code_map.py's own comments on what each evidence
# type stands for. `customer_communication` and `explanation_letter` are
# always produced (the raw log is always present in these fixtures; the
# letter is always drafted, by the LLM or the deterministic fallback).
_ALWAYS_AVAILABLE_EVIDENCE = ("customer_communication", "explanation_letter")


def _available_evidence_types(entity: dict) -> set[str]:
    available = set(_ALWAYS_AVAILABLE_EVIDENCE)
    if entity["avs_match"] and entity["cvv_match"]:
        available.add("billing_proof")
    if entity["device_fingerprint_known"]:
        available.add("access_activity_log")
    if entity["delivery_confirmed"]:
        available.add("proof_of_service")
    return available


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


def _print_letter_sample(label: str, event: dict, llm_client: GroqHttpLLMClient) -> None:
    entity = event["payload"]["dispute"]["entity"]
    context = _context_from_entity(entity)
    required = required_evidence_types(context.reason_code)
    available = _available_evidence_types(entity)
    missing = [t for t in required if t not in available]

    print()
    print("-" * 72)
    print(f"{label}: dispute_id={entity['id']}")
    print("-" * 72)
    print(f"  reason_code                  = {context.reason_code}")
    print(f"  required evidence types      = {list(required)}")
    print(f"  available for this dispute   = {[t for t in required if t in available]}")
    print(f"  documented gap (unavailable) = {missing or 'none - full required set available'}")

    packet = assemble_evidence_packet(context, entity["customer_communication_log"], llm_client)
    if packet.human_review_required:
        print("  NOTE: the live LLM call failed validation twice - this is the")
        print("  deterministic fallback template, not a real completion:")
    else:
        print("  drafted explanation_letter (live Groq completion, verbatim, no truncation):")
    print(f"  {packet.explanation_letter.letter_text}")


def demo_segment_b_llm_letters() -> bool:
    """Returns whether Segment B actually made live LLM calls, so `main()`
    can report accurately in the closing banner rather than assuming success.
    """
    print()
    print("8. Real drafted letters for two disputes, chosen to differ on both the")
    print("   fraud reason code and evidence availability")
    print("   (live Groq completions - a second run will very likely differ in")
    print("    wording; only Segment A above is claimed byte-identical)")
    try:
        llm_client = GroqHttpLLMClient()
    except Exception as error:  # noqa: BLE001 - missing/invalid .env must degrade, not crash
        print(f"  skipped: could not construct a real LLM client ({error!r})")
        print("  Segment B needs a populated .env (all of it - see .env.example) and")
        print("  network access to the configured LLM_API_URL. Not run.")
        return False

    _print_letter_sample(
        "Dispute A - MC_4837, full required evidence set available",
        CONTEST_WORTHY_EVENT,
        llm_client,
    )
    _print_letter_sample(
        "Dispute B - VISA_10_4, documented evidence gap (no AVS/CVV/device/delivery signal)",
        WEAK_EVIDENCE_EVENT,
        llm_client,
    )
    return True


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help=(
            "Path to a SQLite file for the audit DB. Omit for the default "
            "in-memory (fresh-per-run) database."
        ),
    )
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help=(
            "Run only Segment A (no LLM calls, no network, no .env needed). "
            "This is the segment a byte-identical-across-cold-clones check should use - "
            "Segment B (live LLM letter drafts) is never reproducible by design."
        ),
    )
    return parser.parse_args()


def _wire_dependency_overrides(engine, model) -> tuple[FakeLLMClient, FakeRazorpayClient]:
    fake_llm = FakeLLMClient(responses=[VALID_NORMALIZED, VALID_LETTER])
    fake_razorpay = FakeRazorpayClient()

    def _override_session():
        session = make_session_factory(engine)()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = _override_session
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_razorpay_client] = lambda: fake_razorpay
    app.dependency_overrides[get_model] = lambda: (model, DEMO_MODEL_VERSION)
    return fake_llm, fake_razorpay


def main() -> None:
    args = _parse_args()

    model = _train_demo_model()

    database_url = f"sqlite:///{args.db_path}" if args.db_path else "sqlite:///:memory:"
    engine = get_engine(database_url)
    init_db(engine)

    _fake_llm, fake_razorpay = _wire_dependency_overrides(engine, model)
    client = TestClient(app)

    _print_segment_header("A - Deterministic (byte-identical across cold clones, no LLM calls)")
    demo_end_to_end(client, engine)
    demo_second_dispute(client, engine)
    demo_malformed_webhook(client)
    demo_failure_path_llm_degrades(client, engine)
    demo_failure_path_timeout_retry()
    demo_failure_path_429_retry_after(client, engine, fake_razorpay)

    if args.deterministic_only:
        _print_closing_banner(ran_segment_b=False)
        return

    _print_segment_header("B - LLM output (not reproducible)")
    ran_segment_b = demo_segment_b_llm_letters()

    _print_closing_banner(ran_segment_b=ran_segment_b)


def _print_closing_banner(*, ran_segment_b: bool) -> None:
    print()
    print("=" * 72)
    print("Done. Every step above ran against the real webhook route, the real")
    print("policy engine, the real evidence assembler, and the real retry helper -")
    print("step 7 additionally ran the real RazorpayHttpClient (network calls")
    print("stubbed at the httpx transport layer). Segment A made no live network")
    print("call and read no secret from a real .env.")
    if ran_segment_b:
        print("Segment B made real, live Groq API calls - its output is not part of")
        print("the byte-identical-across-cold-clones claim.")
    print("=" * 72)


if __name__ == "__main__":
    main()
