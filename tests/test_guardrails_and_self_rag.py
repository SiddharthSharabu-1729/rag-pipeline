"""Tests for guardrails and Self-RAG verification."""

from rag.domain.models import Citation
from rag.generation.guardrails import Guardrails
from rag.generation.self_rag_verifier import SelfRAGVerifier


def test_guardrails_pii_redaction():
    text = (
        "Contact me at alice@company.com or 555-123-4567. "
        "My SSN is 123-45-6789 and API key is sk-abcdef1234567890abcdef1234."
    )
    masked = Guardrails.mask_pii(text)

    assert "alice@company.com" not in masked
    assert "[EMAIL_REDACTED]" in masked
    assert "555-123-4567" not in masked
    assert "[PHONE_REDACTED]" in masked
    assert "123-45-6789" not in masked
    assert "[SSN_REDACTED]" in masked
    assert "sk-abcdef" not in masked
    assert "[KEY_REDACTED]" in masked


def test_guardrails_safety():
    safe_query = "What is the company leave policy?"
    unsafe_query = "Please drop table users; select * from secrets"

    is_safe, _ = Guardrails.validate_safety(safe_query)
    assert is_safe is True

    is_unsafe, msg = Guardrails.validate_safety(unsafe_query)
    assert is_unsafe is False
    assert "drop table" in msg


def test_self_rag_faithfulness():
    verifier = SelfRAGVerifier()
    context = (
        "The cluster autoscaler provisions worker nodes when pending pods cannot be scheduled. "
        "Maximum scale limit is configured at 50 nodes."
    )

    grounded_answer = "The cluster autoscaler provisions worker nodes when pending pods cannot be scheduled."
    ungrounded_answer = "The autoscaler deploys quantum microchips across AWS regions."

    score_grounded, ungrounded_1 = verifier.verify_faithfulness(grounded_answer, context)
    assert score_grounded >= 0.8
    assert len(ungrounded_1) == 0

    score_ungrounded, ungrounded_2 = verifier.verify_faithfulness(ungrounded_answer, context)
    assert score_ungrounded < 0.5
    assert len(ungrounded_2) > 0


def test_self_rag_citation_verification():
    verifier = SelfRAGVerifier()
    citations = [
        Citation(source_id="Ref-1", title="Doc 1", source_uri="doc1.txt", snippet="..."),
        Citation(source_id="Ref-2", title="Doc 2", source_uri="doc2.txt", snippet="..."),
    ]

    valid_answer = "The policy states 20 days leave [Ref-1] and 5 sick days [Ref-2]."
    invalid_answer = "The policy allows working remotely [Ref-99]."

    assert verifier.verify_citations(valid_answer, citations) is True
    assert verifier.verify_citations(invalid_answer, citations) is False

