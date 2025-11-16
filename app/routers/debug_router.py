from fastapi import APIRouter

from app.schemas.agents import NormalizerRequest
from app.agents.normalizer import PlainNormalizerAgent, LLMNormalizerAgent

router = APIRouter(prefix="/debug", tags=["debug"])


@router.post("/normalize-and-llm")
async def normalize_and_llm(body: NormalizerRequest):
    norm_resp = await PlainNormalizerAgent().run(body)

    if not norm_resp.safe:
        print("⚠️ SAFETY WARNING:", norm_resp.warnings)

    llm_response = await LLMNormalizerAgent().run(norm_resp.normalized_text)

    reply = llm_response.choices[0].message.content
    usage = getattr(llm_response, "usage", None)

    return {
        "normalized_text": norm_resp.normalized_text,
        "safe": norm_resp.safe,
        "warnings": [w.model_dump() for w in norm_resp.warnings],
        "llm_reply": reply,
        "token_usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        },
    }
