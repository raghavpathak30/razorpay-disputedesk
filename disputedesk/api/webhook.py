"""The FastAPI webhook (PHASES.md Phase 4 item 1): receives an `open`
dispute event, validates it, and runs it through
`disputedesk.api.pipeline.process_dispute_event`. Every dependency
(database session, LLM client, Razorpay client, model) is overridable via
FastAPI's `dependency_overrides` so tests never make a network call
(CLAUDE.md: "No test may make a network call. Fake both the LLM and the
Razorpay API.").
"""

from collections.abc import Iterator
from functools import lru_cache

import lightgbm as lgb
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sqlalchemy.orm import Session

from disputedesk.api.pipeline import ProcessResult, process_dispute_event
from disputedesk.api.schemas import DisputeWebhookEvent
from disputedesk.audit.db import get_engine, init_db, make_session_factory
from disputedesk.client.razorpay import RazorpayClient, RazorpayHttpClient
from disputedesk.evidence.llm import GroqHttpLLMClient, LLMClient
from disputedesk.model.registry import get_default_model_bundle

app = FastAPI(title="Dispute Desk webhook")


@lru_cache
def _engine():
    engine = get_engine()
    init_db(engine)
    return engine


def get_db_session() -> Iterator[Session]:
    session = make_session_factory(_engine())()
    try:
        yield session
    finally:
        session.close()


def get_llm_client() -> LLMClient:
    return GroqHttpLLMClient()


def get_razorpay_client() -> RazorpayClient:
    return RazorpayHttpClient()


def get_model() -> tuple[lgb.LGBMClassifier, str]:
    bundle = get_default_model_bundle()
    return bundle.model, bundle.version


class WebhookResponse(BaseModel):
    dispute_id: str
    already_processed: bool
    decision: str
    p_win: float
    human_review_required: bool
    api_outcome: str | None


def _to_response(result: ProcessResult) -> WebhookResponse:
    return WebhookResponse(
        dispute_id=result.dispute_id,
        already_processed=result.already_processed,
        decision=result.policy_decision.decision.value,
        p_win=result.policy_decision.p_win,
        human_review_required=result.decision_row.human_review_required,
        api_outcome=result.api_outcome.outcome if result.api_outcome else None,
    )


@app.post("/webhooks/disputes", response_model=WebhookResponse)
def receive_dispute_webhook(
    event: DisputeWebhookEvent,
    session: Session = Depends(get_db_session),
    llm_client: LLMClient = Depends(get_llm_client),
    razorpay_client: RazorpayClient = Depends(get_razorpay_client),
    model_and_version: tuple[lgb.LGBMClassifier, str] = Depends(get_model),
) -> WebhookResponse:
    model, model_version = model_and_version
    entity = event.payload.dispute.entity
    result = process_dispute_event(
        entity,
        session=session,
        llm_client=llm_client,
        razorpay_client=razorpay_client,
        model=model,
        model_version=model_version,
    )
    return _to_response(result)
