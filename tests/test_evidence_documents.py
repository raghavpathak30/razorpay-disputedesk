"""The evidence-bundle renderer (`disputedesk/evidence/documents.py`): pure,
no I/O, no network. Pins that the hand-rolled PDF writer produces
byte-consistent output, that content never fabricates facts the record
doesn't have, and that every listed failure mode (empty bundle, unknown
evidence type) fails loudly rather than producing a malformed document.
"""

import re

import pytest

from disputedesk.evidence.context import DisputeContext
from disputedesk.evidence.documents import (
    EvidenceRenderError,
    render_evidence_bundle,
    render_pdf,
)
from disputedesk.evidence.letter import DraftedLetter, LetterProvenance
from disputedesk.evidence.reason_code_map import required_evidence_types
from disputedesk.evidence.schemas import NormalizedCommunicationLog

CONTEXT = DisputeContext(
    reason_code="MC_4837",
    amount=4999.0,
    avs_match=True,
    cvv_match=False,
    device_fingerprint_known=True,
    delivery_confirmed=True,
    prior_order_count=6,
)
COMMS = NormalizedCommunicationLog(
    claims_unauthorized_transaction=True,
    mentions_prior_bank_contact=False,
    mentions_shared_card_access=False,
    mentions_travel=False,
    tone="terse",
    is_substantive=True,
    summary="Customer disputes the charge.",
)
LETTER = DraftedLetter(
    letter_text="This letter responds to a chargeback. " * 3,
    cites_evidence_types=("billing_proof",),
    provenance=LetterProvenance.MODEL,
)


# --------------------------------------------------------------------------
# The hand-rolled PDF writer is byte-consistent, not just "starts with %PDF-"
# --------------------------------------------------------------------------


def _parse_xref_offsets(pdf: bytes) -> list[int]:
    """Extract the byte offsets the xref table itself claims, for
    cross-checking against where the objects actually sit."""
    xref_start = pdf.rindex(b"\nxref\n")
    xref_block = pdf[xref_start + len(b"\nxref\n") :]
    lines = xref_block.split(b"\n")
    offsets = []
    for line in lines[1:]:  # line 0 is "0 N" (start, count)
        if re.fullmatch(rb"\d{10} \d{5} [fn] ?", line):
            offsets.append(int(line[:10]))
        else:
            break
    return offsets


def test_render_pdf_starts_with_the_pdf_header_and_ends_with_eof():
    pdf = render_pdf(["hello"])
    assert pdf.startswith(b"%PDF-1.4\n")
    assert pdf.rstrip().endswith(b"%%EOF")


def test_render_pdf_xref_offsets_point_at_real_object_markers():
    """The class of bug a hand-rolled xref table is most likely to have: an
    off-by-one in a byte offset. Every offset the xref table claims for
    object N must land exactly on that object's "N 0 obj" marker - checked
    by re-deriving each offset from the file itself, not by trusting the
    writer's own arithmetic.
    """
    pdf = render_pdf(["first line", "second line", "third line"])

    offsets = _parse_xref_offsets(pdf)
    assert len(offsets) >= 1  # at least the free-list entry's siblings
    # Skip the reserved object-0 entry (always all-zero, "f" not "n").
    real_offsets = offsets[1:] if offsets and offsets[0] == 0 else offsets

    for object_number, offset in enumerate(real_offsets, start=1):
        marker = f"{object_number} 0 obj".encode()
        assert pdf[offset : offset + len(marker)] == marker, (
            f"xref claims object {object_number} is at byte {offset}, "
            f"but that byte is {pdf[offset : offset + 20]!r}"
        )


def test_render_pdf_startxref_points_at_the_real_xref_keyword():
    pdf = render_pdf(["one line"])
    startxref_pos = pdf.rindex(b"startxref\n")
    declared_offset = int(pdf[startxref_pos + len(b"startxref\n") :].split(b"\n")[0])
    assert pdf[declared_offset : declared_offset + 4] == b"xref"


def test_render_pdf_paginates_long_content_into_multiple_pages():
    many_lines = [f"line {i}" for i in range(200)]
    pdf = render_pdf(many_lines)

    assert pdf.count(b"/Type /Page") >= 2  # more than one page object
    offsets = _parse_xref_offsets(pdf)
    for object_number, offset in enumerate(offsets, start=0):
        if offset == 0:
            continue
        marker = f"{object_number} 0 obj".encode()
        assert pdf[offset : offset + len(marker)] == marker


def test_render_pdf_handles_zero_lines_without_producing_a_broken_file():
    pdf = render_pdf([])
    assert pdf.startswith(b"%PDF-1.4\n")
    offsets = _parse_xref_offsets(pdf)
    real_offsets = offsets[1:] if offsets and offsets[0] == 0 else offsets
    for object_number, offset in enumerate(real_offsets, start=1):
        marker = f"{object_number} 0 obj".encode()
        assert pdf[offset : offset + len(marker)] == marker


# --------------------------------------------------------------------------
# Escaping - the customer communication log is attacker-influenced free text
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dangerous",
    [
        "text with (parentheses) inside",
        "backslash \\ and (mixed) content",
        "unbalanced ((( parens",
        ") Tj ET BT (injected operator attempt",
    ],
)
def test_special_pdf_characters_do_not_break_the_document_structure(dangerous):
    """A raw customer message containing PDF-syntax-significant characters
    must not be able to escape the string literal it's placed inside and
    inject content-stream operators - the PDF-generation equivalent of an
    unescaped format string. The whole file must remain a well-formed,
    self-consistent PDF (same offset check as above), regardless of content.
    """
    pdf = render_pdf([dangerous, "a following line"])

    offsets = _parse_xref_offsets(pdf)
    real_offsets = offsets[1:] if offsets and offsets[0] == 0 else offsets
    for object_number, offset in enumerate(real_offsets, start=1):
        marker = f"{object_number} 0 obj".encode()
        assert pdf[offset : offset + len(marker)] == marker


def test_unbalanced_parentheses_are_escaped_not_left_to_unbalance_the_stream():
    from disputedesk.evidence.documents import _pdf_escape

    assert _pdf_escape("(((") == "\\(\\(\\("
    assert _pdf_escape("a)b(c\\d") == "a\\)b\\(c\\\\d"


def test_non_ascii_and_control_characters_are_dropped_not_passed_through():
    from disputedesk.evidence.documents import _pdf_escape

    assert _pdf_escape("café \x00 emoji 🎉 end") == "caf  emoji  end"


# --------------------------------------------------------------------------
# Content: only stated facts, nothing fabricated
# --------------------------------------------------------------------------


def _text_of(evidence_type: str) -> str:
    bundle = render_evidence_bundle(
        CONTEXT, COMMS, "I don't recognize this charge.", LETTER, (evidence_type,)
    )
    # The content stream is latin-1 text inside the PDF bytes - decode
    # loosely and check the facts landed, rather than re-parsing the PDF.
    return bundle[0].content.decode("latin-1")


def test_billing_proof_states_avs_and_cvv_facts_only():
    text = _text_of("billing_proof")
    assert "Address Verification" in text
    assert "yes" in text  # avs_match=True
    assert "Card Verification" in text
    assert "no" in text  # cvv_match=False
    assert "signature" not in text.lower()  # never fabricated


def test_access_activity_log_states_device_fingerprint_fact_only():
    text = _text_of("access_activity_log")
    assert "Device recognized" in text
    assert "yes" in text  # device_fingerprint_known=True


def test_proof_of_service_states_delivery_confirmed_fact_only():
    text = _text_of("proof_of_service")
    assert "confirmed" in text
    assert "tracking number" not in text.lower()  # never fabricated


def test_customer_communication_carries_the_raw_log_verbatim():
    bundle = render_evidence_bundle(
        CONTEXT,
        COMMS,
        "a very specific customer message right here",
        LETTER,
        ("customer_communication",),
    )
    text = bundle[0].content.decode("latin-1")
    assert "a very specific customer message right here" in text


def test_explanation_letter_carries_the_letter_text():
    text = _text_of("explanation_letter")
    assert "This letter responds to a chargeback." in text


def test_each_document_is_tagged_with_its_evidence_type_and_a_filename():
    bundle = render_evidence_bundle(
        CONTEXT, COMMS, "msg", LETTER, ("billing_proof", "proof_of_service")
    )
    assert [d.evidence_type for d in bundle] == ["billing_proof", "proof_of_service"]
    assert all(d.filename.endswith(".pdf") for d in bundle)
    assert all(d.mime_type == "application/pdf" for d in bundle)


def test_the_full_required_set_for_a_supported_reason_code_all_render():
    evidence_types = required_evidence_types(CONTEXT.reason_code)
    bundle = render_evidence_bundle(CONTEXT, COMMS, "msg", LETTER, evidence_types)
    assert len(bundle) == len(evidence_types)
    assert all(len(d.content) > 0 for d in bundle)


# --------------------------------------------------------------------------
# Failure modes fail loudly
# --------------------------------------------------------------------------


def test_an_empty_evidence_types_tuple_raises():
    with pytest.raises(EvidenceRenderError, match="empty"):
        render_evidence_bundle(CONTEXT, COMMS, "msg", LETTER, ())


def test_an_unknown_evidence_type_raises():
    with pytest.raises(EvidenceRenderError, match="no renderer"):
        render_evidence_bundle(CONTEXT, COMMS, "msg", LETTER, ("not_a_real_evidence_type",))
