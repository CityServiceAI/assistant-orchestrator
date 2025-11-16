from app.schemas.agents import (
    NormalizerRequest,
    NormalizerResponse,
    NormalizerWarning,
)
from app.services.text_normalizer import run_full_normalization
from app.agents.base import Agent


class NormalizerAgent(Agent):
    name = "normalizer"

    async def run(self, payload: NormalizerRequest) -> NormalizerResponse:
        raw = payload.text
        original_len = len(raw)

        normalized_text, truncated, warning_codes, safe = run_full_normalization(raw)

        warnings = [
            NormalizerWarning(code=code, message=_warning_message(code))
            for code in warning_codes
        ]

        return NormalizerResponse(
            normalized_text=normalized_text,
            original_length=original_len,
            normalized_length=len(normalized_text),
            truncated=truncated,
            safe=safe,
            warnings=warnings,
        )


def _warning_message(code: str) -> str:
    MESSAGES = {
        "TRUNCATED": "Текст було скорочено до максимального дозволеного розміру.",
        "PROMPT_INJECTION_SUSPECTED": "Виявлено спробу змінити інструкції моделі або обійти обмеження.",
        "POTENTIAL_TOXIC_CONTENT": "В тексті знайдено потенційно образливі вирази.",
    }
    return MESSAGES.get(code, code)
