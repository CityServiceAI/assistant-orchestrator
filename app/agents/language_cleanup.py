from pydantic import BaseModel

from app.deps.litellm_client import chat_completion, LLMCallConfig
from app.config import settings


SYSTEM_LANGUAGE_CLEANUP_PROMPT = """
Ви — редактор офіційних звернень до міських служб.

Ваше завдання:
1) Якщо текст не українською — перекладіть його українською.
2) Усуньте лайку, образливі вислови, погрози, жаргон та надмірно емоційні фрази — замініть їх на коректні й нейтральні формулювання.
3) Збережіть усі факти, дати, адреси, конкретні проблеми та вимоги заявника.
4) НІЧОГО не вигадуйте: не додавайте нових фактів, імен, контактів чи вимог, яких не було в оригіналі.
5) Стиль: офіційно-діловий, нейтральний, без емоцій, без просторіччя.
6) Скоротіть текст: приберіть повтори, "воду" та другорядні деталі, які не впливають на суть звернення.
7) Відповідь має бути чіткою та лаконічною: бажано 3–6 речень, 1–2 абзаци.
8) Відповідайте ТІЛЬКИ очищеним текстом українською: без пояснень, без лапок, без привітань і підписів, без додаткових коментарів.
"""


class LanguageCleanupResult(BaseModel):
    cleaned_text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


LANGUAGE_CLEANUP_CONFIG = LLMCallConfig(
    model=settings.language_cleanup_model,
    temperature=0.0,
    max_tokens=220,
)


class LanguageCleanupAgent:
    name = "language_cleanup"

    def run(self, text: str) -> LanguageCleanupResult:
        messages = [
            {"role": "system", "content": SYSTEM_LANGUAGE_CLEANUP_PROMPT},
            {"role": "user", "content": text},
        ]

        response = chat_completion(LANGUAGE_CLEANUP_CONFIG, messages)
        msg = response.choices[0].message
        content = msg.content

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

        usage = getattr(response, "usage", None)

        return LanguageCleanupResult(
            cleaned_text=content.strip(),
            model=LANGUAGE_CLEANUP_CONFIG.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            total_tokens=getattr(usage, "total_tokens", None) if usage else None,
        )
