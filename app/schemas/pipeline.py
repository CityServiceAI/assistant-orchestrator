from pydantic import BaseModel, Field
from typing import Optional


class LLMCallInfo(BaseModel):
    id: Optional[str] = Field(
        default=None,
        description="Необов'язковий внутрішній ID виклику (може дорівнювати conversation_step_id або бути окремим uuid).",
    )
    agent: str = Field(
        ...,
        description="Назва агента / ноди графа, наприклад: 'normalizer', 'language_cleanup', 'category_classifier'.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Назва моделі, яку реально викликали (наприклад, 'gpt-4.1-mini').",
    )
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
