from app.pipeline.state import ComplaintState
from app.tools.normalizer_tool import normalize_text
from app.agents.language_cleanup import LanguageCleanupAgent
from app.agents.category_classifier import CategoryClassifierAgent
from app.pipeline.context import LLMCallRecord
from langgraph.graph import END

language_agent = LanguageCleanupAgent()
category_agent = CategoryClassifierAgent()


def normalize_node(state: ComplaintState) -> ComplaintState:
    raw_text = state["raw_text"]

    normalized_text, _truncated, warning_codes, safe = normalize_text(raw_text)

    state["normalized_text"] = normalized_text
    state["normalizer_safe"] = safe
    state["normalizer_warnings"] = warning_codes

    return state


def language_cleanup_node(state: ComplaintState) -> ComplaintState:
    if not state.get("normalizer_safe", True):
        return state

    agent = LanguageCleanupAgent()
    result = agent.run(state["raw_text"])

    state["cleaned_text_uk"] = result.cleaned_text

    llm_calls = state.get("llm_calls", [])
    llm_calls.append(
        {
            "agent": "language_cleanup",
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        }
    )
    state["llm_calls"] = llm_calls

    return state


def category_node(state: ComplaintState) -> ComplaintState:
    if not state.get("cleaned_text_uk"):
        return state

    agent = CategoryClassifierAgent()
    result = agent.run(state["cleaned_text_uk"])

    state["category"] = result.category.value if result.category else None
    state["category_confidence"] = result.confidence
    state["category_need_clarification"] = result.need_clarification
    state["category_clarification_question"] = result.clarification_question

    llm_calls = state.get("llm_calls", [])
    llm_calls.append(
        {
            "agent": "category_classifier",
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        }
    )
    state["llm_calls"] = llm_calls

    return state


def router_node(state: ComplaintState) -> str:
    if state.get("normalizer_safe") is False:
        return END

    if state.get("category_need_clarification"):
        return END

    return END
