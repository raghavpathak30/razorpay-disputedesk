"""Processes one validated `open` dispute event end to end (SPEC.md §1):
score, decide, assemble evidence if contesting, persist the decision, then
file via the Razorpay client. Used by both the FastAPI route
(`disputedesk/api/webhook.py`) and the demo script
(`disputedesk/cli/demo.py`) so the two never drift - the webhook is a thin
HTTP wrapper around this function, not a second copy of it.
"""

from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd
from sqlalchemy.orm import Session

from disputedesk.api.schemas import DisputeEntity
from disputedesk.audit.log import (
    get_api_outcome,
    get_decision,
    record_api_outcome,
    record_decision,
)
from disputedesk.audit.models import ApiOutcome, DecisionRecord
from disputedesk.client.razorpay import RazorpayClient
from disputedesk.evidence.assembler import assemble_evidence_packet
from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.draft_letter import PROMPT_VERSION as _LETTER_PROMPT_VERSION
from disputedesk.evidence.letter import DraftedLetter
from disputedesk.evidence.llm import LLMClient
from disputedesk.evidence.published_reason_codes import is_supported_reason_code
from disputedesk.features.build import build_features
from disputedesk.model.predict import predict_proba
from disputedesk.policy.config import PolicyConfig
from disputedesk.policy.engine import Decision, PolicyDecision, decide

_PROMPT_VERSIONS_FOR_CONTEST = f"normalize_comms_log_v1,{_LETTER_PROMPT_VERSION}"


@dataclass(frozen=True)
class ProcessResult:
    dispute_id: str
    already_processed: bool
    policy_decision: PolicyDecision
    decision_row: DecisionRecord
    api_outcome: ApiOutcome | None


def _policy_decision_from_row(row: DecisionRecord) -> PolicyDecision:
    """Reconstruct the original `PolicyDecision` from a persisted row,
    rather than calling `decide()` again - `PolicyConfig` could have changed
    since the original decision was made, and the audit row, not a fresh
    recomputation, is the source of truth for what was actually decided.
    """
    return PolicyDecision(
        decision=Decision(row.policy_branch),
        p_win=row.p_win,
        amount=row.amount_inr,
        representment_cost_inr=row.representment_cost_inr,
        expected_value_inr=row.expected_value_inr,
        low_confidence=row.low_confidence,
    )


def _context_from_entity(entity: DisputeEntity) -> DisputeContext:
    return DisputeContext(
        reason_code=entity.reason_code,
        amount=entity.amount,
        avs_match=entity.avs_match,
        cvv_match=entity.cvv_match,
        device_fingerprint_known=entity.device_fingerprint_known,
        delivery_confirmed=entity.delivery_confirmed,
        prior_order_count=entity.prior_order_count,
    )


def _score(entity: DisputeEntity, model: lgb.LGBMClassifier) -> tuple[float, dict]:
    """`P(win)` and the exact feature dict it was computed from, for the
    audit row's `features_used`. Builds the feature row from `entity`'s own
    field names - `build_features` reads a dict, and a Pydantic model's
    `model_dump()` is one - the same function `eval/harness.py` uses.
    """
    features = build_features(entity.model_dump())
    X = pd.DataFrame([features])
    p_win = float(predict_proba(model, X)[0])
    return p_win, features


def _already_processed_result(session: Session, row: DecisionRecord) -> ProcessResult:
    return ProcessResult(
        dispute_id=row.dispute_id,
        already_processed=True,
        policy_decision=_policy_decision_from_row(row),
        decision_row=row,
        api_outcome=get_api_outcome(session, row.dispute_id),
    )


@dataclass(frozen=True)
class _EvidenceOutcome:
    prompt_version: str | None
    validation_result: str
    human_review_required: bool
    # The letter object, not its text: `_file_if_needed` needs its
    # `provenance` to decide whether it may be filed at all (defect 0.1).
    letter: DraftedLetter | None


_NO_EVIDENCE = _EvidenceOutcome(
    prompt_version=None,
    validation_result="not_applicable",
    human_review_required=False,
    letter=None,
)

# The documented fallback for a reason code this system has no evidence
# strategy for (2026-09-02, defect 0.3): the event is accepted, a full audit
# row is written with this tag on it, and the dispute is queued for a person.
# Nothing is filed in either direction - see `_file_if_needed`.
_UNRECOGNISED_REASON_CODE = _EvidenceOutcome(
    prompt_version=None,
    validation_result="reason_code_unrecognised",
    human_review_required=True,
    letter=None,
)


def _assemble_evidence_if_contesting(
    entity: DisputeEntity, policy_decision: PolicyDecision, llm_client: LLMClient
) -> _EvidenceOutcome:
    if not is_supported_reason_code(entity.reason_code):
        # Checked before the branch test, not after: an unrecognised code
        # stops the dispute regardless of what the policy engine decided,
        # including an ACCEPT, because accepting is irreversible and this
        # system does not know what it is accepting.
        return _UNRECOGNISED_REASON_CODE
    if policy_decision.decision != Decision.CONTEST:
        return _NO_EVIDENCE

    context = _context_from_entity(entity)
    packet = assemble_evidence_packet(context, entity.customer_communication_log, llm_client)
    return _EvidenceOutcome(
        prompt_version=_PROMPT_VERSIONS_FOR_CONTEST,
        validation_result="fallback_template_used" if packet.human_review_required else "validated",
        human_review_required=packet.human_review_required,
        letter=packet.explanation_letter,
    )


def _persist_decision(
    session: Session,
    entity: DisputeEntity,
    model_version: str,
    p_win: float,
    features: dict,
    policy_decision: PolicyDecision,
    evidence: _EvidenceOutcome,
) -> tuple[DecisionRecord, bool]:
    return record_decision(
        session,
        dispute_id=entity.id,
        reason_code=entity.reason_code,
        amount_inr=entity.amount,
        model_version=model_version,
        features=features,
        p_win=p_win,
        policy_branch=policy_decision.decision.value,
        expected_value_inr=policy_decision.expected_value_inr,
        representment_cost_inr=policy_decision.representment_cost_inr,
        low_confidence=policy_decision.low_confidence,
        prompt_version=evidence.prompt_version,
        validation_result=evidence.validation_result,
        human_review_required=evidence.human_review_required,
    )


def process_dispute_event(
    entity: DisputeEntity,
    *,
    session: Session,
    llm_client: LLMClient,
    razorpay_client: RazorpayClient,
    model: lgb.LGBMClassifier,
    model_version: str,
    policy_config: PolicyConfig | None = None,
) -> ProcessResult:
    """PHASES.md Phase 4's full pipeline. Idempotent: if `entity.id` already
    has a decision recorded (a replayed webhook), this makes no LLM call, no
    Razorpay API call, and no new database row - it returns the existing
    decision with `already_processed=True`. See
    `disputedesk/audit/log.py:record_decision` for why this is safe even
    under a race, not just in the common case this fast-path check covers.
    """
    dispute_id = entity.id

    existing_decision = get_decision(session, dispute_id)
    if existing_decision is not None:
        return _already_processed_result(session, existing_decision)

    p_win, features = _score(entity, model)
    policy_decision = decide(p_win, entity.amount, policy_config)
    evidence = _assemble_evidence_if_contesting(entity, policy_decision, llm_client)

    # PHASES.md Phase 4 item 3: persisted before the API call, so a crash mid-call
    # never loses the decision, and a retry after restart is still idempotent.
    decision_row, was_new = _persist_decision(
        session, entity, model_version, p_win, features, policy_decision, evidence
    )
    if not was_new:
        # Lost a race against a concurrent request for the same dispute -
        # the database constraint caught what the fast-path check above did not.
        return _already_processed_result(session, decision_row)

    api_outcome = _file_if_needed(
        session, razorpay_client, dispute_id, policy_decision, entity.amount, evidence
    )

    return ProcessResult(
        dispute_id=dispute_id,
        already_processed=False,
        policy_decision=policy_decision,
        decision_row=decision_row,
        api_outcome=api_outcome,
    )


def _file_if_needed(
    session: Session,
    razorpay_client: RazorpayClient,
    dispute_id: str,
    policy_decision: PolicyDecision,
    amount_inr: float,
    evidence: _EvidenceOutcome,
) -> ApiOutcome | None:
    action = _ACTION_BY_BRANCH.get(policy_decision.decision)
    if action is None:
        return None  # ESCALATE: no API call - a human decides. Nothing to file yet.

    if evidence.validation_result == "reason_code_unrecognised":
        return _withhold_for_review(
            session,
            dispute_id,
            action,
            "reason code is not one this system has an evidence strategy for",
        )

    if action == "contest":
        letter = evidence.letter
        if letter is None or not letter.submittable:
            provenance = letter.provenance.value if letter is not None else "missing"
            return _withhold_for_review(
                session,
                dispute_id,
                action,
                f"explanation letter provenance is {provenance!r}, not 'model'",
            )
        return _file(session, razorpay_client, dispute_id, "contest", amount_inr, letter)

    return _file(session, razorpay_client, dispute_id, "accept", amount_inr, None)


_ACTION_BY_BRANCH = {Decision.CONTEST: "contest", Decision.ACCEPT: "accept"}


def _withhold_for_review(session: Session, dispute_id: str, action: str, reason: str) -> ApiOutcome:
    """The policy engine reached a decision, but the packet is not fit to act
    on unsupervised - the letter is not the model's own validated output
    (defect 0.1), or the reason code is one this system has no strategy for
    (defect 0.3). Nothing is filed and no network call is made; the dispute is
    recorded as awaiting a person.

    Deliberately *not* re-decided as accept: accepting is irreversible
    (Razorpay's accept endpoint moves the dispute straight to "lost"), and
    neither a drafting failure nor an unknown reason code is evidence about
    whether this dispute is winnable. The policy branch on the decision row
    still reads what the policy engine decided - what changed is that the
    packet was not fit to file, which is a separate fact and is recorded as
    one. `action` records which filing was withheld.
    """
    return record_api_outcome(
        session,
        dispute_id=dispute_id,
        action=action,
        outcome="withheld_for_review",
        error=f"{reason}; queued for human review instead of being filed",
    )


def _file(
    session: Session,
    razorpay_client: RazorpayClient,
    dispute_id: str,
    action: str,
    amount_inr: float,
    letter: DraftedLetter | None,
) -> ApiOutcome:
    """Call the Razorpay client for `action` (retry/backoff on timeout or
    429 already happens inside the client - see
    `disputedesk/client/razorpay.py`) and record exactly one outcome row,
    success or failure. SPEC.md §7 failure path 1: if every retry inside the
    client is exhausted, the exception is caught here, not left to crash the
    request - the system degrades to a "failed" audit row a human can act
    on, and a later replay of the same webhook event is still safe (the
    `api_outcomes` UNIQUE constraint means a retry-by-replay can only ever
    produce this same row again, never a second filing).
    """
    try:
        if action == "contest":
            # `letter` is non-None and submittable here - `_file_if_needed`
            # withholds anything else before reaching this function.
            response = razorpay_client.contest(dispute_id, amount_inr, letter)
        else:
            response = razorpay_client.accept(dispute_id)
    except Exception as error:  # noqa: BLE001 - genuinely any failure must degrade, not crash
        return record_api_outcome(
            session, dispute_id=dispute_id, action=action, outcome="failed", error=str(error)
        )
    return record_api_outcome(
        session, dispute_id=dispute_id, action=action, outcome="success", response=response
    )
