"""
Input sanitization before LLM calls — prevents prompt injection,
exfiltration attempts, and dangerous payloads in user goals.
"""
import re
from typing import Optional


PROMPT_INJECTION_PATTERNS = [
    (r"(?i)(?:\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b)", "ignore-prior-instruction"),
    (r"(?i)(?:\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|commands)\b)", "ignore-prior-instruction"),
    (r"(?i)(?:\bforget\s+(?:all\s+)?(?:previous|prior|above)\s)", "forget-prior"),
    (r"(?i)(?:\boutput\s+(?:the\s+)?(?:system\s+)?prompt\b)", "extract-prompt"),
    (r"(?i)(?:\bshow\s+(?:me\s+)?(?:the\s+)?(?:system\s+)?prompt\b)", "extract-prompt"),
    (r"(?i)(?:\breveal\s+(?:the\s+)?(?:system\s+)?prompt\b)", "extract-prompt"),
    (r"(?i)(?:\bleak\s+(?:the\s+)?(?:system\s+)?prompt\b)", "extract-prompt"),
    (r"(?i)(?:\bprint\s+(?:the\s+)?(?:system\s+)?prompt\b)", "extract-prompt"),
    (r"(?i)(?:\breturn\s+(?:the\s+)?(?:system\s+)?instructions\b)", "extract-prompt"),
    (r"(?i)(?:\byou\s+(?:are\s+)?(?:now\s+)?(?:an?\s+)?(?:jailbreak|unconstrained|free|unbounded)\b)", "jailbreak"),
    (r"(?i)(?:\bpretend\s+(?:you\s+)?(?:are|were)\s+(?:an?\s+)?(?:jailbreak|unconstrained|free|unbounded)\b)", "jailbreak"),
    (r"(?i)(?:\bDAN\b)", "jailbreak-acronym"),
    (r"(?i)(?:\bdo\s+(?:not\s+)?(?:.+)?\bignore\b)", "inverted-ignore"),
    (r"(?i)(?:\bactual\s+(?:output|response|instruction|format)\b)", "exfiltration"),
    (r"(?i)(?:\byou\s+must\s+(?:start|begin|respond|output)\s+(?:with\s+)?)", "instruction-override"),
    (r"(?i)(?:\bSECRET\s+KEY\b|\bAPI[_\s]KEY\b|\bPASSWORD\b|\bTOKEN\b)", "secret-reference"),
    (r"(?i)(?:\bcurl\s+.*?\|\s*bash\b)", "pipe-to-bash"),
    (r"(?i)(?:\beval\s*\(.*?request.*?\))", "eval-request"),
]

SENSITIVE_DATA_PATTERNS = [
    (r"(?i)(?:sk-[A-Za-z0-9]{20,})", "openai-api-key"),
    (r"(?i)(?:ghp_[A-Za-z0-9]{36})", "github-token"),
    (r"(?i)(?:hf_[A-Za-z0-9]{20,})", "hf-token"),
    (r"(?i)(?:xox[bpras]-[A-Za-z0-9-]{10,})", "slack-token"),
    (r"(?i)(?:AKIA[0-9A-Z]{16})", "aws-access-key"),
    (r"(?i)(?:eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})", "jwt-token"),
]


def sanitize_user_input(text: str) -> tuple[bool, str, list[str]]:
    """
    Sanitize user input before sending to the LLM.
    Returns (is_safe, sanitized_text, warnings).
    """
    warnings = []
    for pattern, label in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text):
            warnings.append(f"Prompt injection risk: {label}")

    for pattern, label in SENSITIVE_DATA_PATTERNS:
        if re.search(pattern, text):
            warnings.append(f"Sensitive data detected: {label}")

    sanitized = text
    if warnings:
        sanitized = _redact_sensitive(text)

    return len(warnings) == 0, sanitized, warnings


def _redact_sensitive(text: str) -> str:
    for pattern, _ in SENSITIVE_DATA_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text


def validate_goal_format(goal: str) -> Optional[str]:
    """
    Check that the goal is a reasonable fine-tuning request.
    Returns an error message if invalid, None if OK.
    """
    if not goal or not goal.strip():
        return "Goal cannot be empty"

    if len(goal) > 5000:
        return f"Goal too long ({len(goal)} chars, max 5000)"

    if len(goal.split()) > 1000:
        return f"Goal too many words ({len(goal.split())}, max 1000)"

    return None


def strip_code_blocks(text: str) -> str:
    """Remove code blocks from user input — code should go through codegen only."""
    text = re.sub(r"```[\s\S]*?```", "[CODE BLOCK REMOVED]", text)
    text = re.sub(r"(?m)^( {4,}|\t+).*$", "", text)
    return text.strip()
