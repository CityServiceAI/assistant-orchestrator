import logging

from langgraph.graph import END

from app.agents.category_classifier import CategoryClassifierAgent
from app.agents.language_cleanup import LanguageCleanupAgent
from app.pipeline.conversation_graph_state import ConversationGraphState
from app.tools.normalizer_tool import normalize_text

language_agent = LanguageCleanupAgent()
category_agent = CategoryClassifierAgent()


def normalize_node(state: ConversationGraphState) -> ConversationGraphState:
    # ToDo only for role=user messages
    user_input = state["messages"][-1]['content']
    logging.info(f"User input {user_input}")

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

    return {
        "message": result['assistant_response'],
        "debug": [{
            "agent": "language_cleanup",
            "input_text": state['message'],
            **result
        }]
    }


def category_node(state: ConversationGraphState) -> ConversationGraphState:
    if "message" not in state:
        raise Exception("No message in stage")

    agent = CategoryClassifierAgent()
    result = agent.run(state['messages'], state['message'])

    if result.get("clarification_question") is not None:
        assistant_message = [{"role": "assistant", "content": result["clarification_question"]}]
    else:
        assistant_message = []

    return {
        "category": result["category"],
        "category_confidence": result["confidence"],
        "category_need_clarification": result["need_clarification"],
        "messages": assistant_message,
        "debug": [{
            **result,
            "agent": "category-classifier",
            "input_text": state['message']
        }]
    }


def router_node(state: ConversationGraphState) -> str:
    if state.get("normalizer_safe") is False:
        return END

    if state.get("category_need_clarification"):
        return END

    return END
