from enum import Enum
import json
from typing import Optional

from pydantic import BaseModel, Field

from app.deps.litellm_client import chat_completion, LLMCallConfig
from app.config import settings


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


class CategoryDetectionResult(BaseModel):
    category: CategoryCode | None = Field(
        ...,
        description="Код категорії або None, якщо її неможливо визначити.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Впевненість у класифікації [0,1].",
    )
    need_clarification: bool = Field(
        ...,
        description="Чи потрібно уточнення від користувача.",
    )
    clarification_question: Optional[str] = Field(
        None,
        description="Уточнююче запитання, якщо need_clarification=true, інакше null.",
    )
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


CATEGORIES = [
    {
        "code": "WATER_SUPPLY",
        "title_uk": "Проблеми з водопостачанням",
        "description_uk": "Відсутність води, низький тиск, іржава вода, прориви труб, витік води тощо.",
    },
    {
        "code": "HEATING",
        "title_uk": "Опалення",
        "description_uk": "Відсутність опалення, холодні батареї, протікання системи опалення.",
    },
    {
        "code": "ELECTRICITY",
        "title_uk": "Електропостачання",
        "description_uk": "Відключення світла, аварійна електропроводка, мерехтіння світла.",
    },
    {
        "code": "ROAD_INFRASTRUCTURE",
        "title_uk": "Дороги та тротуари",
        "description_uk": "Ями, погане покриття, зруйновані тротуари.",
    },
    {
        "code": "WASTE",
        "title_uk": "Сміття та прибирання",
        "description_uk": "Переповнені баки, несвоєчасне вивезення сміття, стихійні сміттєзвалища.",
    },
    {
        "code": "YARD_TERRITORY",
        "title_uk": "Прибудинкова територія",
        "description_uk": "Освітлення двору, дитячі майданчики, дерева, лавочки тощо.",
    },
    {
        "code": "PUBLIC_TRANSPORT",
        "title_uk": "Громадський транспорт",
        "description_uk": "Маршрути, графік руху, зупинки, стан транспорту.",
    },
    {
        "code": "OTHER",
        "title_uk": "Інше (комунальна тема)",
        "description_uk": "Комунальні проблеми, які не підпадають під інші категорії.",
    },
    {
        "code": "NOT_MUNICIPAL",
        "title_uk": "Не стосується комунальних послуг",
        "description_uk": "Текст не про комунальні послуги або міську інфраструктуру.",
    },
]

CATEGORIES_JSON = json.dumps(CATEGORIES, ensure_ascii=False, indent=2)


SYSTEM_CATEGORY_PROMPT = f"""
Ви — класифікатор звернень до міських комунальних служб.

Вхід: офіційний, нейтральний текст звернення українською мовою.

Доступні категорії (code):
{CATEGORIES_JSON}

Правила:
1) Виберіть одну категорію "code" з наведених.
2) Якщо звернення не стосується комунальних послуг або міської інфраструктури —
   використайте "NOT_MUNICIPAL".
3) Якщо інформації недостатньо, але тема комунальна —
   оберіть найбільш ймовірну категорію, встановіть need_clarification=true
   і сформуйте коротке уточнююче запитання.
4) Якщо текст зовсім незрозумілий — category = null, confidence = 0.0, need_clarification = true.

Формат відповіді СТРОГО:
Поверніть ОДИН JSON-об'єкт з такими полями:
- "category": один із code або null
- "confidence": число від 0 до 1
- "need_clarification": true або false
- "clarification_question": рядок або null

НЕ додавайте жодних пояснень поза JSON.
"""


CATEGORY_CLASSIFIER_CONFIG = LLMCallConfig(
    model=getattr(
        settings, "category_classifier_model", settings.language_cleanup_model
    ),
    temperature=0.0,
    max_tokens=200,
)


class CategoryClassifierAgent:
    name = "category_classifier"

    def run(self, cleaned_text_uk: str) -> CategoryDetectionResult:
        """
        Вхід: вже очищений та нормалізований український текст (мовним агентом).
        Вихід: CategoryDetectionResult з полями category/confidence/need_clarification/clarification_question.
        """

        messages = [
            {"role": "system", "content": SYSTEM_CATEGORY_PROMPT},
            {"role": "user", "content": cleaned_text_uk},
        ]
        response = chat_completion(CATEGORY_CLASSIFIER_CONFIG, messages)
        msg = response.choices[0].message
        content = getattr(msg, "content", None)

        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                text_part = getattr(part, "text", None)
                if isinstance(text_part, str):
                    parts.append(text_part)
                elif isinstance(part, dict):
                    t = part.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            content = "".join(parts)

        if content is None:
            content = ""

        result = CategoryDetectionResult.model_validate_json(content)
        usage = getattr(response, "usage", None)
        result.model = CATEGORY_CLASSIFIER_CONFIG.model
        result.prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        result.completion_tokens = (
            getattr(usage, "completion_tokens", None) if usage else None
        )
        result.total_tokens = getattr(usage, "total_tokens", None) if usage else None

        return result
