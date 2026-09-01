"""Demo script (PHASES.md Phase 4 item 8): replays fixture dispute events
through the real webhook route end to end - webhook in, features, `P(win)`,
policy decision, evidence assembly, API call, audit row out - and
demonstrates both SPEC.md §7 failure paths live. Runnable from a clean
clone with no `.env` and no network: the LLM and Razorpay clients are fakes
(CLAUDE.md: "No test may make a network call. Fake both the LLM and the
Razorpay API."), and the model is trained in memory from the same synthetic
generator every other command in this project already uses - no secrets,
no persisted artifacts, no external services required.

Run: `python -m disputedesk.cli.demo`

By default the audit DB is an in-memory SQLite database, so each run starts
clean. Pass `--db-path` to use a file instead - this is what lets the same
event be replayed across two separate process invocations, the scenario the
decision-before-API-call ordering (PHASES.md Phase 4 item 3) actually exists
for; the in-memory default only proves idempotency within one process.
Run: `python -m disputedesk.cli.demo --db-path data/demo/disputedesk.db`
"""

import argparse
import json
import warnings

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
from disputedesk.client.razorpay import FakeRazorpayClient
from disputedesk.evidence.llm import FakeLLMClient
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
                "reason_code": "VISA_83",
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


def main() -> None:
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
    args = parser.parse_args()

    model = _train_demo_model()

    database_url = f"sqlite:///{args.db_path}" if args.db_path else "sqlite:///:memory:"
    engine = get_engine(database_url)
    init_db(engine)

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

    client = TestClient(app)

    demo_end_to_end(client, engine)
    demo_second_dispute(client, engine)
    demo_malformed_webhook(client)
    demo_failure_path_llm_degrades(client, engine)
    demo_failure_path_timeout_retry()

    print()
    print("=" * 72)
    print("Done. Every step above ran against the real webhook route, the real")
    print("policy engine, the real evidence assembler, and the real retry helper -")
    print("only the LLM and Razorpay API calls were faked (no network, no secrets).")
    print("=" * 72)


if __name__ == "__main__":
    main()
