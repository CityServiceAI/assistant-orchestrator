import re


TOXIC_PATTERNS = [
    r"\b(хуй|пизд|бляд|сука|ебан)\w*",
]


def detect_toxicity(text: str) -> bool:
    lowered = text.lower()
    for pattern in TOXIC_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return True
    return False


def detect_prompt_injection(text: str) -> bool:
    """
    Дуже простий rule-based детектор prompt-injection:
    ловимо пари слів типу "ігноруй" + "інструкц", "забудь" + "правил" і т.д.
    """
    lowered = text.lower()

    dangerous_pairs = [
        ("ігноруй", "інструкц"),
        ("забудь", "правил"),
        ("відключи", "обмежен"),
        ("ignore", "instructions"),
        ("disregard", "rules"),
        ("forget", "previous"),
    ]

    for w1, w2 in dangerous_pairs:
        if w1 in lowered and w2 in lowered:
            return True

    return False
