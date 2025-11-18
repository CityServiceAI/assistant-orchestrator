import logging

from app.agents.category_classifier import CategoryClassifierAgent
from app.agents.language_cleanup import LanguageCleanupAgent
from app.agents.service_agent import ServiceAgent
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
        "trace": [{
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
        "trace": [{
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
        "trace": [{
            **result,
            "agent": "category-classifier",
            "input_text": state['message']
        }]
    }

def search_service_node(state: ConversationGraphState) -> ConversationGraphState:
    logging.info(f"Search service node. State: {state}")

    result = ServiceAgent().run("H1.1.1")

    return {
        "trace": [{
            **result,
            "agent": "service"
        }]
    }

def ask_clarification_node(state: ConversationGraphState):
    logging.info("Ask clarification node.")
    return state

def generate_appeal_node(state: ConversationGraphState):
    logging.info(f"Generate appeal node: State: {state}")
    return {
        "messages": [{"role": "assistant", "content": "Ну що я можу сказати, беріть відро та черпайте"}]
    }


def route_after_classification(state: ConversationGraphState):
    logging.info(f"Route after classification. State {state}")
    category = state.get("category")
    category_confidence = state.get("category_confidence")
    clarification_count = state.get("clarification_count", 0)

    if category is None and category_confidence < 90 :
        if clarification_count > 10:
            logging.info("Route to handle_failure")
            return "handle_failure"

        logging.info("Route to ask_clarification")
        return "ask_clarification"

    logging.info("Route to service_search")
    return "service_search"


def route_after_service_search(state: ConversationGraphState):
    logging.info(f"Route after search. State: {state}")

    if state.get('category') == "YARD_TERRITORY":
        return "generate_appeal"

    return "handle_failure"


def handle_failure_node(state: ConversationGraphState):
    return {
        "messages": [
            {"role": "assistant", "content": "Вибачте, нажаль я не можу визначити відповідальну службу за вашу проблему"},
            {"role": "assistant", 'content': 'Зверніться за номером 1551 або створіть звернення на сайті контактного центру міста Києва https://1551.gov.ua'}
        ]
    }