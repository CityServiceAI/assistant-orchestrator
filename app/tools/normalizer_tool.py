import os
import re
from typing import List, Tuple

import unicodedata

from app.services.safety import detect_prompt_injection

NORMALIZER_INPUT_HARD_LIMIT = int(os.getenv('NORMALIZER_INPUT_HARD_LIMIT', 12000))
NORMALIZER_PRESERVE_NEWLINES = bool(os.getenv('NORMALIZER_PRESERVE_NEWLINES', True))
NORMALIZER_MAX_CHARS = bool(os.getenv('NORMALIZER_MAX_CHARS', 3000))

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_ZERO_WIDTH_RE = re.compile(
    "["
    "\u200b"  # ZERO WIDTH SPACE
    "\u200c"  # ZERO WIDTH NON-JOINER
    "\u200d"  # ZERO WIDTH JOINER
    "\ufeff"  # ZERO WIDTH NO-BREAK SPACE/BOM
    "]"
)
_SPACES_ONLY_RE = re.compile(r"[ \t\r\f\v]+")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")


def _remove_control_chars(text: str) -> str:
    return _CONTROL_CHARS_RE.sub("", text)


def _remove_zero_width(text: str) -> str:
    return _ZERO_WIDTH_RE.sub("", text)


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _normalize_whitespace(text: str, preserve_newlines: bool = True) -> str:
    text = text.strip()

    if not preserve_newlines:
        return re.sub(r"\s+", " ", text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES_ONLY_RE.sub(" ", text)
    text = _MULTIPLE_NEWLINES_RE.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def _hard_truncate(text: str, max_chars: int) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def normalize_text(raw: str) -> Tuple[str, bool, List[str], bool]:
    warnings: List[str] = []

    if raw is None:
        return "", False, ["EMPTY_INPUT"], True

    if len(raw) > NORMALIZER_INPUT_HARD_LIMIT:
        preview, _ = _hard_truncate(raw, 200)
        warnings.append("INPUT_TOO_LONG")
        return preview, False, warnings, False

    text = _normalize_unicode(raw)
    text = _remove_control_chars(text)
    text = _remove_zero_width(text)
    text = _normalize_whitespace(
        text, preserve_newlines=NORMALIZER_PRESERVE_NEWLINES
    )
    text, truncated = _hard_truncate(text, NORMALIZER_MAX_CHARS)

    if truncated:
        warnings.append("TRUNCATED")

    inj = detect_prompt_injection(text)
    if inj:
        warnings.append("PROMPT_INJECTION_SUSPECTED")

    safe = not inj and "INPUT_TOO_LONG" not in warnings

    return text, truncated, warnings, safe
