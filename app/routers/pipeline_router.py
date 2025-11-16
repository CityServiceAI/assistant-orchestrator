from fastapi import APIRouter

from app.schemas.agents import NormalizerRequest
from app.pipeline.complaint_pipeline import run_complaint_pipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/complaint")
async def complaint_pipeline_endpoint(body: NormalizerRequest):
    ctx = run_complaint_pipeline(body.text)

    return {
        "raw_text": ctx.raw_text,
        "normalized_text": ctx.normalized_text,
        "normalizer_safe": ctx.normalizer_safe,
        "normalizer_warnings": ctx.normalizer_warnings,
        "cleaned_text_uk": ctx.cleaned_text_uk,
        "llm_calls": [
            {
                "agent": c.agent,
                "model": c.model,
                "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens,
                "total_tokens": c.total_tokens,
            }
            for c in ctx.llm_calls
        ],
    }
