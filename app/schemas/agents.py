from pydantic import BaseModel, Field
from typing import List
from enum import Enum


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


class CategoryCode(str, Enum):
    WATER_SUPPLY = "WATER_SUPPLY"
    HEATING = "HEATING"
    ELECTRICITY = "ELECTRICITY"
    ROAD_INFRASTRUCTURE = "ROAD_INFRASTRUCTURE"
    WASTE = "WASTE"
    YARD_TERRITORY = "YARD_TERRITORY"
    PUBLIC_TRANSPORT = "PUBLIC_TRANSPORT"
    OTHER = "OTHER"
    NOT_MUNICIPAL = "NOT_MUNICIPAL"


class CategoryDetectionRequest(BaseModel):
    text: str = Field(..., description="Очищений український текст звернення.")


class CategoryDetectionResponse(BaseModel):
    category: CategoryCode | None = Field(
        None, description="Код категорії або None, якщо не вдалося визначити."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Впевненість [0,1].")
    need_clarification: bool = Field(..., description="true, якщо потрібно уточнення.")
    clarification_question: str | None = Field(
        None,
        description="Уточнююче запитання користувачу, якщо need_clarification=true.",
    )
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
