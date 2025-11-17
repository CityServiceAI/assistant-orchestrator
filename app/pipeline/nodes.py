from langgraph.graph import END

from app.agents.category_classifier import CategoryClassifierAgent
from app.agents.language_cleanup import LanguageCleanupAgent
from app.pipeline.conversation_graph_state import ConversationGraphState
from app.tools.normalizer_tool import normalize_text
import json

language_agent = LanguageCleanupAgent()
category_agent = CategoryClassifierAgent()


def normalize_node(state: ConversationGraphState) -> ConversationGraphState:
    # ToDo only for role=user messages
    user_input = state["messages"][-1]['content']

    normalized_text, _truncated, warning_codes, safe = normalize_text(user_input)

    return {
        "message": normalized_text,
        "debug": [{
            "node": "normalize_node",
            "input_text": user_input,
            "output_text": normalized_text,
            "normalizer_safe": safe,
            "normalizer_warnings": warning_codes
        }]
    }


def language_cleanup_node(state: ConversationGraphState):
    agent = LanguageCleanupAgent()
    result = agent.run(state['message'])

    debug = {
        "agent": "language_cleanup",
        "model": result.model,
        "input_text": state['message'],
        "output_text": result.cleaned_text,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
    }

    return {
        "message": result.cleaned_text,
        "debug": [debug]
    }


def category_node(state: ConversationGraphState) -> ConversationGraphState:
    if "category" in state:
        return {}

    if "message" not in state:
        raise Exception("No message in stage")

    agent = CategoryClassifierAgent()
    result = agent.run(state['messages'])
    return {
        "category": result["category"],
        "category_confidence": result["confidence"],
        "category_need_clarification": result["need_clarification"],
        "messages":  [{"role": "assistant", "content": result["clarification_question"]}] if "clarification_question" in result else [],
        "debug": [{**result, "agent": "category_classifier"}]
    }


def router_node(state: ConversationGraphState) -> str:
    if state.get("normalizer_safe") is False:
        return END

    if state.get("category_need_clarification"):
        return END

    return END
