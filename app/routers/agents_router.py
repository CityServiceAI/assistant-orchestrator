from fastapi import APIRouter

from app.schemas.agents import (
    NormalizerRequest,
    NormalizerResponse,
    NormalizerWarning,
)
from app.tools.normalizer_tool import normalize_text
from app.schemas.agents import (
    CategoryDetectionRequest,
    CategoryDetectionResponse,
)
from app.agents.category_classifier import CategoryClassifierAgent

router = APIRouter(prefix="/agents", tags=["agents"])
category_agent = CategoryClassifierAgent()


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


@router.post("/category", response_model=CategoryDetectionResponse)
def detect_category(req: CategoryDetectionRequest) -> CategoryDetectionResponse:
    result = category_agent.run(req.text)

    return CategoryDetectionResponse(
        category=result.category,
        confidence=result.confidence,
        need_clarification=result.need_clarification,
        clarification_question=result.clarification_question,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
    )
