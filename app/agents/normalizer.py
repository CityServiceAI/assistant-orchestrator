from app.agents.base import Agent
from app.config import settings
from app.deps.litellm_client import chat_completion
from app.schemas.agents import (
    NormalizerRequest,
    NormalizerResponse,
    NormalizerWarning,
)
from app.services.text_normalizer import run_full_normalization


class PlainNormalizerAgent(Agent):
    name = "normalizer"

    async def run(self, payload: NormalizerRequest) -> NormalizerResponse:
        raw = payload.text
        normalized_text, truncated, warning_codes, safe = run_full_normalization(raw)

        warnings = [
            NormalizerWarning(code=code, message=_warning_message(code))
            for code in warning_codes
        ]

        return NormalizerResponse(
            normalized_text=normalized_text,
            original_length=len(raw),
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


class LLMNormalizerAgent(Agent):
    SYSTEM_PROMPT = """
    Ви — висококваліфікований редактор української мови. Ваше завдання — нормалізувати наданий користувачем текст, дотримуючись суворих правил форматування та стилю.
    
    Дотримуйтесь наступних інструкцій без винятків:
    1. Видаліть зайві пробіли та виправте пунктуацію.
    2. Замініть будь-яку ненормативну лексику, матюки або образливі слова на загальноприйняті, нейтральні українські відповідники.
    3. Замініть розмовний сленг на нейтральні відповідники.
    4. Мова: українська, Тон: офіційний, нейтральний та професійний.
    5. Надайте лише кінцевий, виправлений текст. Жодних додаткових коментарів чи пояснень.
    """

    name = "normalizer"

    async def run(self, text):
        messages = [
            {"role": "system", "content": LLMNormalizerAgent.SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        return chat_completion(
            model=settings.default_model,  # ToDo replace model if needed
            messages=messages,
            temperature=0.2,
        )
