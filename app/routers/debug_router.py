from fastapi import APIRouter

from app.schemas.agents import NormalizerRequest
from app.agents.normalizer import NormalizerAgent
from app.deps.litellm_client import chat_completion
from app.config import settings

router = APIRouter(prefix="/debug", tags=["debug"])
SYSTEM_NORMALIZATION_PROMPT = """
    Ви — висококваліфікований редактор української мови. Ваше завдання — нормалізувати наданий користувачем текст, дотримуючись суворих правил форматування та стилю.
    
    Дотримуйтесь наступних інструкцій без винятків:
    1. Видаліть зайві пробіли та виправте пунктуацію.
    2. Замініть будь-яку ненормативну лексику, матюки або образливі слова на загальноприйняті, нейтральні українські відповідники.
    3. Замініть розмовний сленг на нейтральні відповідники.
    4. Мова: українська, Тон: офіційний, нейтральний та професійний.
    5. Надайте лише кінцевий, виправлений текст. Жодних додаткових коментарів чи пояснень.
"""

normalizer = NormalizerAgent()


@router.post("/normalize-and-llm")
async def normalize_and_llm(body: NormalizerRequest):
    # 1) нормалізація
    norm_resp = await normalizer.run(body)

    if not norm_resp.safe:
        print("⚠️ SAFETY WARNING:", norm_resp.warnings)

    # 2) виклик LLM через твій proxy
    messages = [
        {
            "role": "system",
            "content": SYSTEM_NORMALIZATION_PROMPT,
        },
        {
            "role": "user",
            "content": norm_resp.normalized_text,
        },
    ]

    llm_response = chat_completion(
        model=settings.default_model,
        messages=messages,
        temperature=0.2,
    )

    reply = llm_response.choices[0].message.content
    usage = getattr(llm_response, "usage", None)

    return {
        "normalized_text": norm_resp.normalized_text,
        "safe": norm_resp.safe,
        "warnings": [w.model_dump() for w in norm_resp.warnings],
        "llm_reply": reply,
        "token_usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None)
            if usage
            else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        },
    }
