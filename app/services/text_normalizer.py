import re
from typing import Tuple, List

from app.config import settings
from app.services.safety import detect_prompt_injection, detect_toxicity

_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def normalize_whitespace(text: str) -> str:
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def remove_control_chars(text: str) -> str:
    return _CONTROL_CHARS_RE.sub("", text)


def hard_truncate(text: str, max_chars: int) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def normalize_text_pipeline(raw: str) -> Tuple[str, bool, List[str]]:
    """
    Базова нормалізація: control-символи, пробіли, обрізка довжини.
    Повертає: normalized_text, truncated, warning_codes
    """
    warnings: List[str] = []

    text = remove_control_chars(raw)
    text = normalize_whitespace(text)

    text, truncated = hard_truncate(text, settings.normalizer_max_chars)
    if truncated:
        warnings.append("TRUNCATED")

    return text, truncated, warnings


def run_full_normalization(raw: str):
    """
    Повна нормалізація + простий safety.
    Повертає: normalized_text, truncated, warning_codes, safe
    """
    norm_text, truncated, warnings = normalize_text_pipeline(raw)

    inj = detect_prompt_injection(norm_text)
    tox = detect_toxicity(norm_text)

    safe = not (inj or tox)
    if inj:
        warnings.append("PROMPT_INJECTION_SUSPECTED")
    if tox:
        warnings.append("POTENTIAL_TOXIC_CONTENT")

    return norm_text, truncated, warnings, safe
