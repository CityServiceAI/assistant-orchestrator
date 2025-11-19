from pydantic import BaseModel

from app.deps.litellm_client import client
from app.tools.response import get_content_as_str

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

LLM_MODEL= 'gpt-4.1-mini'
NORMALIZER_MAX_CHARS=3000
NORMALIZER_INPUT_HARD_LIMIT=12000
NORMALIZER_PRESERVE_NEWLINES=True

class LanguageCleanupAgent:
    name = "language_cleanup"

    def run(self, text: str):
        messages = [
            {"role": "system", "content": SYSTEM_LANGUAGE_CLEANUP_PROMPT},
            {"role": "user", "content": text},
        ]

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=220,
        )

        return {
            "assistant_response": get_content_as_str(response.choices[0].message.content),
            "model": LLM_MODEL,
            "usage": response.usage
        }
