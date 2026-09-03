"""Renders the evidence bundle a contest packet needs to upload: one small
document per required evidence type, stating only the facts this system
actually has.

Why this module exists (2026-09-04 scoped reopening). Razorpay's contest
endpoint requires at least one document id attached under `action="submit"`
(https://razorpay.com/docs/api/disputes/contest/, verified 2026-09-02). This
system had no document-upload pipeline, so every submit went out with zero -
a request the real API's documented contract would reject
(`tests/test_client_document_contract.py` proves it). This module is the
first stage of the fix: pure, deterministic, no I/O, no network - the upload
itself lives in `disputedesk/client/razorpay.py`, the only module allowed to
talk to the network (CLAUDE.md).

**What goes in each document, and why it is not invented.** This project has
no scanned receipts, no signature images, no tracking-number screenshots -
`docs/AI-SURFACE.md` §0.2 already names this: "There is only one
unstructured artifact per dispute" (the customer's own message). Rendering a
document that *looked* like a delivery receipt or an authentication log
would mean fabricating evidence for a fraud dispute, which is a different and
much worse problem than an unfinished upload pipeline. Every document below
states only fields already present on `DisputeContext` (or the raw
communication log, or the drafted letter) - the same discipline
`draft_letter.py`'s deterministic fallback already uses ("states only the
raw order-context facts, with no narrative framing"). `billing_proof` says
whether AVS/CVV matched; it does not claim a signature exists.

**One document per required evidence type**, matching
`reason_code_map.py`'s own stated reasoning for why each type is required:

- `billing_proof` - AVS/CVV match facts (the charge was authenticated).
- `access_activity_log` - device-fingerprint-known fact (the genuine account).
- `proof_of_service` - delivery-confirmed fact (the transaction was real).
- `customer_communication` - the raw communication log itself, verbatim.
- `explanation_letter` - the drafted letter's own text, verbatim.

**Format: a minimal, hand-rolled PDF, no new dependency.** Razorpay's
document-upload endpoint accepts only `image/jpg`, `image/jpeg`, `image/png`,
`application/pdf` (verified 2026-09-04) - plain text is not an accepted
`mime_type`, so a `.txt` file is not an option. Adding a PDF library
(reportlab, fpdf2, ...) was not done without asking (CLAUDE.md); a
single-page, one-font, left-aligned block of text is a small enough PDF
subset to write directly and verify byte-for-byte
(`tests/test_evidence_documents.py` checks the xref table's offsets are
self-consistent, not just that the bytes start with `%PDF-`).
"""

from dataclasses import dataclass

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.letter import DraftedLetter
from disputedesk.evidence.reason_code_map import REQUIRED_EVIDENCE_BY_REASON_CODE
from disputedesk.evidence.schemas import NormalizedCommunicationLog

_PAGE_WIDTH = 612  # US Letter, points (72/inch) - matches Razorpay's own
_PAGE_HEIGHT = 792  # sample document dimensions closely enough to be unremarkable
_FONT_SIZE = 10
_LINE_HEIGHT = 14
_MARGIN_LEFT = 50
_MARGIN_TOP = 60
_WRAP_CHARS = 92  # characters per line at 10pt Helvetica on a 612pt-wide page
_LINES_PER_PAGE = (int(_PAGE_HEIGHT) - _MARGIN_TOP - 40) // _LINE_HEIGHT


class EvidenceRenderError(ValueError):
    """Raised when a required evidence type has no renderer, or the bundle
    would otherwise be empty. Caught by `disputedesk.evidence.assembler`,
    which treats it exactly like a drafting or grounding failure - route to
    human review, never crash the dispute's whole pipeline run."""


@dataclass(frozen=True)
class EvidenceDocument:
    """One rendered document, ready to upload. `evidence_type` is the exact
    key Razorpay's contest body expects it filed under (SPEC.md §3)."""

    evidence_type: str
    filename: str
    content: bytes
    mime_type: str = "application/pdf"


def _wrap(text: str, width: int = _WRAP_CHARS) -> list[str]:
    """Plain word wrap - no hyphenation, no justification. "Keep it plain":
    this is a document, not a design exercise."""
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) <= width:
                current += " " + word
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _pdf_escape(text: str) -> str:
    """Escape a string for use inside a PDF literal-string operand `(...)`.

    Required for correctness, not polish: the customer communication log is
    attacker-influenced free text that can contain `(`, `)`, or `\\` -
    unescaped, any of those breaks out of the string literal and corrupts the
    content stream's syntax, which is the PDF-generation equivalent of a
    SQL/format-string injection. Every character outside printable ASCII is
    dropped rather than passed through, since the base-14 Helvetica font this
    renderer uses has no defined glyphs for anything else.
    """
    out = []
    for ch in text:
        if ch in ("(", ")", "\\"):
            out.append("\\" + ch)
        elif ch == "\n":
            out.append(" ")
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        # else: dropped - outside the standard font's encoding.
    return "".join(out)


def _page_content_stream(lines: list[str]) -> bytes:
    y = _PAGE_HEIGHT - _MARGIN_TOP
    ops = ["BT", f"/F1 {_FONT_SIZE} Tf", f"{_MARGIN_LEFT} {y} Td"]
    first = True
    for line in lines:
        if not first:
            ops.append(f"0 {-_LINE_HEIGHT} Td")
        ops.append(f"({_pdf_escape(line)}) Tj")
        first = False
    ops.append("ET")
    return ("\n".join(ops)).encode("latin-1")


def render_pdf(lines: list[str]) -> bytes:
    """A minimal, valid, single-font, multi-page-if-needed PDF containing
    `lines` as left-aligned text, one line per `Tj` operator. No external
    dependency - see this module's docstring for why.
    """
    pages = [
        lines[i : i + _LINES_PER_PAGE] for i in range(0, max(len(lines), 1), _LINES_PER_PAGE)
    ] or [[]]

    objects: list[bytes] = []
    # Object 1: Catalog, object 2: Pages (filled in after page objects are
    # known), object 3: the shared Helvetica font. Page and content-stream
    # objects follow, two per page, in order.
    page_obj_ids = [4 + 2 * i for i in range(len(pages))]
    content_obj_ids = [5 + 2 * i for i in range(len(pages))]

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page_lines, _page_id, content_id in zip(pages, page_obj_ids, content_obj_ids, strict=True):
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_PAGE_WIDTH} {_PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        stream = _page_content_stream(page_lines)
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")

    return _assemble_pdf(objects)


def _assemble_pdf(objects: list[bytes]) -> bytes:
    """Write the header, every numbered object, a byte-exact xref table, and
    the trailer. Object numbers are `objects`' list position + 1 (PDF objects
    are 1-indexed; object 0 is the reserved free-list head).
    """
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]  # offsets[0] is object 0's, the reserved free entry
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    n = len(objects) + 1
    out += f"xref\n0 {n}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    return bytes(out)


def _yes_no(value: bool, yes: str = "yes", no: str = "no") -> str:
    return yes if value else no


def _render_billing_proof(context: DisputeContext, **_ignored) -> list[str]:
    return [
        f"Billing authentication - reason code {context.reason_code}, "
        f"transaction INR {context.amount:.2f}.",
        f"Address Verification Service (AVS) match: {_yes_no(context.avs_match)}.",
        f"Card Verification Value (CVV) match: {_yes_no(context.cvv_match)}.",
    ]


def _render_access_activity_log(context: DisputeContext, **_ignored) -> list[str]:
    return [
        f"Access activity - reason code {context.reason_code}.",
        "Device recognized from this customer's prior activity: "
        f"{_yes_no(context.device_fingerprint_known)}.",
        f"This customer has {context.prior_order_count} prior order(s) on this account.",
    ]


def _render_proof_of_service(context: DisputeContext, **_ignored) -> list[str]:
    return [
        f"Proof of service - reason code {context.reason_code}.",
        "Delivery or fulfillment of this transaction: "
        f"{_yes_no(context.delivery_confirmed, 'confirmed', 'not confirmed')}.",
    ]


def _render_customer_communication(
    context: DisputeContext, raw_communication_log: str, **_ignored
) -> list[str]:
    header = [f"Customer communication - reason code {context.reason_code}.", ""]
    body = raw_communication_log.strip() or "(no message on file)"
    return header + _wrap(body)


def _render_explanation_letter(letter: DraftedLetter, **_ignored) -> list[str]:
    return _wrap(letter.letter_text)


_RENDERERS = {
    "billing_proof": (_render_billing_proof, "billing_proof.pdf"),
    "access_activity_log": (_render_access_activity_log, "access_activity_log.pdf"),
    "proof_of_service": (_render_proof_of_service, "proof_of_service.pdf"),
    "customer_communication": (_render_customer_communication, "customer_communication.pdf"),
    "explanation_letter": (_render_explanation_letter, "explanation_letter.pdf"),
}
"""Every evidence type this renderer knows how to produce. Deliberately not
derived from `REQUIRED_EVIDENCE_BY_REASON_CODE` - that would let a future
reason-code addition silently "work" by rendering nothing for a type this
module was never taught, rather than failing loudly via
`EvidenceRenderError`."""


def render_evidence_bundle(
    context: DisputeContext,
    normalized_comms: NormalizedCommunicationLog,
    raw_communication_log: str,
    letter: DraftedLetter,
    evidence_types: tuple[str, ...],
) -> tuple[EvidenceDocument, ...]:
    """One `EvidenceDocument` per entry in `evidence_types`, in order.

    Raises `EvidenceRenderError` if `evidence_types` is empty, or names a
    type this module has no renderer for - both are configuration errors
    (an evidence-type gap, or a reason code with no evidence at all), not
    runtime inputs to tolerate silently, matching
    `reason_code_map.required_evidence_types`'s own `KeyError` convention
    for the same class of problem.
    """
    if not evidence_types:
        raise EvidenceRenderError("evidence_types is empty - nothing to render")

    documents = []
    for evidence_type in evidence_types:
        entry = _RENDERERS.get(evidence_type)
        if entry is None:
            raise EvidenceRenderError(
                f"no renderer for evidence type {evidence_type!r} - "
                f"known types: {sorted(_RENDERERS)}"
            )
        renderer, filename = entry
        lines = renderer(
            context=context,
            normalized_comms=normalized_comms,
            raw_communication_log=raw_communication_log,
            letter=letter,
        )
        documents.append(
            EvidenceDocument(
                evidence_type=evidence_type,
                filename=filename,
                content=render_pdf(lines),
            )
        )

    if not documents:
        raise EvidenceRenderError("rendering produced an empty bundle")
    return tuple(documents)


assert set(_RENDERERS) >= set().union(*REQUIRED_EVIDENCE_BY_REASON_CODE.values()), (
    "every evidence type any supported reason code requires must have a renderer"
)
