"""Input and output guardrails: PII redaction and safety checks."""

import re
from rag.config import settings


class Guardrails:
    """Enterprise guardrails for data privacy and output sanitization."""

    # PII Regular Expressions
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
    API_KEY_PATTERN = re.compile(r"\b(sk-[a-zA-Z0-9]{20,}|Bearer\s+[a-zA-Z0-9_\-\.]{20,})\b")

    @classmethod
    def mask_pii(cls, text: str) -> str:
        """Redact sensitive personally identifiable information (PII) from text."""
        if not settings.enable_pii_masking:
            return text

        masked = cls.EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
        masked = cls.PHONE_PATTERN.sub("[PHONE_REDACTED]", masked)
        masked = cls.SSN_PATTERN.sub("[SSN_REDACTED]", masked)
        masked = cls.CREDIT_CARD_PATTERN.sub("[CARD_REDACTED]", masked)
        masked = cls.API_KEY_PATTERN.sub("[KEY_REDACTED]", masked)
        return masked

    @classmethod
    def validate_safety(cls, text: str) -> tuple[bool, str]:
        """Verify prompt does not violate system safety boundaries."""
        # Simple heuristic check for malicious jailbreak vectors
        harmful_signals = [
            "drop table", "exec(", "eval(", "rm -rf", "delete from",
            "format c:", "/etc/shadow", "/etc/passwd"
        ]
        text_lower = text.lower()
        for signal in harmful_signals:
            if signal in text_lower:
                return False, f"Prohibited instruction detected: {signal}"
        return True, "Passed safety checks."

