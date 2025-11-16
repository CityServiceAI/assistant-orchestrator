from pydantic import BaseModel, Field
from typing import List


class NormalizerRequest(BaseModel):
    text: str = Field(
        ..., description="Сирий текст від користувача (скарга/опис проблеми)"
    )


class NormalizerWarning(BaseModel):
    code: str
    message: str


class NormalizerResponse(BaseModel):
    normalized_text: str
    original_length: int
    normalized_length: int
    truncated: bool
    safe: bool
    warnings: List[NormalizerWarning] = []
