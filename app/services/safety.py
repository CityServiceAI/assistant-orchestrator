import re
from typing import Iterable


PROMPT_INJECTION_PATTERNS: Iterable[re.Pattern] = [
    re.compile(r"ігноруй.{0,40}інструкц", re.IGNORECASE | re.DOTALL),
    re.compile(r"ігноруй.{0,40}правил", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"забудь.{0,40}(попередн(і|ю)|інструкц|контекст)", re.IGNORECASE | re.DOTALL
    ),
    re.compile(
        r"відключ(и|іть).{0,40}(обмежен|фільтр|модерац)", re.IGNORECASE | re.DOTALL
    ),
    re.compile(
        r"ignore.{0,40}(instructions|previous|system)", re.IGNORECASE | re.DOTALL
    ),
    re.compile(
        r"disregard.{0,40}(rules|instructions|previous)", re.IGNORECASE | re.DOTALL
    ),
    re.compile(
        r"forget.{0,40}(previous|everything|all that)", re.IGNORECASE | re.DOTALL
    ),
    re.compile(
        r"you are now.{0,60}(unfiltered|unsafe|jailbreak)", re.IGNORECASE | re.DOTALL
    ),
    re.compile(r"игнорируй.{0,40}(инструкц|правил)", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"забудь.{0,40}(предыдущие|инструкц|контекст)", re.IGNORECASE | re.DOTALL
    ),
    re.compile(r"отключи.{0,40}(ограничен|фильтр|модерац)", re.IGNORECASE | re.DOTALL),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"hidden instructions?", re.IGNORECASE),
    re.compile(r'"role"\s*:\s*"system"', re.IGNORECASE),
    re.compile(r"покажи.{0,40}системн(і|ые).{0,40}інструкц", re.IGNORECASE | re.DOTALL),
    re.compile(r"\b(jailbreak|dan[\s-]?mode|do anything now)\b", re.IGNORECASE),
]


def detect_prompt_injection(text: str) -> bool:
    lowered = text.lower()

    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(lowered):
            return True

    return False
