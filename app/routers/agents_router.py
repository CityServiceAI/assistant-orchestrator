from fastapi import APIRouter

from app.schemas.agents import (
    NormalizerRequest,
    NormalizerResponse,
    NormalizerWarning,
)
from app.tools.normalizer_tool import normalize_text

router = APIRouter(prefix="/agents", tags=["agents"])


def _warning_message(code: str) -> str:
    messages = {
        "TRUNCATED": "Текст було скорочено до максимального дозволеного розміру.",
        "PROMPT_INJECTION_SUSPECTED": "Виявлено спробу змінити інструкції моделі або обійти обмеження.",
    }
    return messages.get(code, code)


@router.post("/normalize", response_model=NormalizerResponse)
async def normalize_text_endpoint(body: NormalizerRequest) -> NormalizerResponse:
    normalized_text, truncated, warning_codes, safe = normalize_text(body.text)

    warnings = [
        NormalizerWarning(code=code, message=_warning_message(code))
        for code in warning_codes
    ]

    return NormalizerResponse(
        normalized_text=normalized_text,
        original_length=len(body.text),
        normalized_length=len(normalized_text),
        truncated=truncated,
        safe=safe,
        warnings=warnings,
    )
