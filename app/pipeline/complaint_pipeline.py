from app.pipeline.context import PipelineContext, LLMCallRecord
from app.tools.normalizer_tool import normalize_text
from app.agents.language_cleanup import LanguageCleanupAgent

language_agent = LanguageCleanupAgent()


def run_complaint_pipeline(raw_text: str) -> PipelineContext:
    ctx = PipelineContext(raw_text=raw_text)

    normalized_text, _truncated, warning_codes, safe = normalize_text(raw_text)
    ctx.normalized_text = normalized_text
    ctx.normalizer_safe = safe
    ctx.normalizer_warnings = warning_codes

    if not safe:
        return ctx

    result = language_agent.run(normalized_text)
    ctx.cleaned_text_uk = result.cleaned_text

    ctx.add_llm_call(
        LLMCallRecord(
            agent=language_agent.name,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        )
    )

    return ctx
