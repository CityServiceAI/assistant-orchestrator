import logging

from app.agents.category_classifier import CategoryClassifierAgent
from app.agents.category_rules import pre_classification_rules
from app.agents.classifier_v2 import ClassifierV2
from app.agents.service_agent import ServiceAgent
from app.data.rag import rag_search_categories
from app.pipeline.state import ConversationGraphState
from app.tools.normalizer_tool import normalize_text
from app.tools.response import assistant_msg

category_agent = CategoryClassifierAgent()


def normalize_node(state: ConversationGraphState) -> ConversationGraphState:
    last_user_msg = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    )
    logging.info(f"User input {last_user_msg}")

    normalized_text, _truncated, warning_codes, safe = normalize_text(last_user_msg)

    base_issue_text = state.get("issue_text") or ""
    need_clar_prev = state.get("need_clarification", False)

    if not base_issue_text:
        new_issue_text = normalized_text
    elif need_clar_prev:
        new_issue_text = f"{base_issue_text}. {normalized_text}"
    else:
        new_issue_text = base_issue_text

    return {
        "message": normalized_text,
        "issue_text": new_issue_text,
        "trace": [
            {
                "node": "normalize_node",
                "input_text": last_user_msg,
                "output_text": normalized_text,
                "normalizer_safe": safe,
                "normalizer_warnings": warning_codes,
            }
        ],
    }


def increase_clarification_count(state: ConversationGraphState, result):
    if bool(result.get("need_clarification")):
        return state.get("clarification_count", 0) + 1

    return state.get("clarification_count")


def category_node(state: ConversationGraphState) -> ConversationGraphState:
    if "message" not in state:
        raise Exception("No message in state for category_node")

    if state.get("summary") is not None and state.get("summary").get('normalized_description'):
        text = state.get("summary").get('normalized_description') + " " + state["message"]
    else:
        text = state["message"]

    logging.info(f"Category node. Input text: {text}")

    rules_result = pre_classification_rules(text)

    if rules_result is not None:
        logging.info(f"Rules matched: {rules_result}")
        result = {
            "category": rules_result["category"],
            "confidence": rules_result["confidence"],
            "need_clarification": rules_result["need_clarification"],
            "clarification_question": rules_result["clarification_question"],
            "usage": None,
            "model": "rules-engine",
        }
    else:
        candidates = rag_search_categories(text, k=5)
        logging.info(f"RAG candidates: {[c.l3_code for c in candidates]}")

        result = category_agent.run(
            text=text,
            candidates=candidates,
            is_clarification=state.get("need_clarification"),
            previous_summary=state.get("summary")
        )

    return {
        **result,
        "category": result.get("category"),
        "category_confidence": float(result.get("confidence", 0.0)),
        "clarification_count": increase_clarification_count(state, result),
        "messages": [get_clarification_message(result)] if get_clarification_message(result) else [],
        "trace": [
            {
                **result,
                "agent": "category-classifier",
                "input_text": text,
            }
        ],
    }


def get_clarification_message(result):
    if result.get("need_clarification") and result.get("clarification_question"):
        return assistant_msg(result.get("clarification_question"), "category_classifier")

    return None


def search_service_node(state: ConversationGraphState) -> ConversationGraphState:
    logging.info(f"Search service node. State: {state}")

    result = ServiceAgent().run(state)

    problem = state.get("problem")
    return {
        "messages": [assistant_msg(
            f'Ваша проблема {problem.get("code")}: {problem.get("description")}, {problem.get("category_name")}')],
        "trace": [
            {
                **result,
                "agent": "service",
            }
        ]
    }


def ask_clarification_node(state: ConversationGraphState):
    logging.info("Ask clarification node.")
    return {}


def generate_appeal_node(state: ConversationGraphState):
    logging.info(f"Generate appeal node: State: {state}")
    return {
        "messages": [assistant_msg("Дякуємо за звернення")]
    }


def route_after_classification(state: ConversationGraphState):
    logging.info(f"Route after classification. State {state}")

    category = state.get("category")
    confidence = state.get("category_confidence", 0)
    need_clarification = state.get("need_clarification", False)
    clarification_count = state.get("clarification_count", 0)
    emergency_score = state.get("emergency_score", 0)

    if emergency_score > 0 and need_clarification:
        return "ask_clarification"

    if emergency_score >= 0.9 and not need_clarification:
        return "emergency"

    if isinstance(category, str) and category.startswith("Z."):
        if confidence >= 0.8:
            logging.info(
                "Category is NOT_MUNICIPAL with high confidence → handle_failure"
            )
            return "handle_failure"

        if need_clarification:
            logging.info("NOT_MUNICIPAL but low confidence → ask_clarification")
            return "ask_clarification"
        logging.info("NOT_MUNICIPAL low confidence w/o clarification → handle_failure")
        return "handle_failure"

    if category is None:
        if need_clarification:
            logging.info("No category, need clarification → ask_clarification")
            return "ask_clarification"
        logging.info("No category and no clarification → handle_failure")
        return "handle_failure"

    if confidence >= 0.9:
        logging.info("Have category with high confidence ≥ 0.9 → service_search")
        return "service_search"

    logging.info(
        f"Have category={category} but confidence={confidence} < 0.9 → ask_clarification "
        f"(clarification_count={clarification_count})"
    )
    return "ask_clarification"


def route_after_service_search(state: ConversationGraphState):
    logging.info(f"Route after search. State: {state}")
    return "generate_appeal"


def handle_failure_node(state: ConversationGraphState):
    return {
        "messages": [
            assistant_msg("Вибачте, нажаль я не можу визначити відповідальну службу за вашу проблему"),
            assistant_msg(
                "Зверніться за номером 1551 або створіть звернення на сайті контактного центру міста Києва https://1551.gov.ua")
        ]
    }


def classifier_node(state: ConversationGraphState) -> ConversationGraphState:
    logging.info(f"Classifier node: request {state}")
    result = ClassifierV2().run(
        state["messages"][-1]['content'],
        state.get("need_clarification"),
        state.get('summary')
    )

    if result.get("need_clarification", True):
        assistant_message = [
            assistant_msg(result.get("clarification_question"))
        ]
    else:
        assistant_message = []

    return {
        **result,
        "category": get_problem_code(result.get("problem")),
        "category_confidence": result.get("confidence", 0),

        "messages": assistant_message,
        "trace": [
            {
                **result,
                "agent": "category-classifier",
                "input_text": state.get("message"),
            }
        ],
    }


def emergency_node(state: ConversationGraphState) -> ConversationGraphState:
    logging.info(f"Emergency node: request {state}")

    responsible_entity_type = state.get("problem", {}).get("responsible_entity_type")
    logging.info(f"Responsible entity type:  {responsible_entity_type}")

    match responsible_entity_type:
        case "MunicipalUtility_GasService":
            message = """
            ❗️ УВАГА: Загроза вибуху!
            Ми зафіксували вашу скаргу про запах газу. 
            Будь ласка, негайно виконайте наступні дії:
            - Перекрийте вентилі на газових приладах та на вході в квартиру/будинок.
            - Відчиніть вікна для провітрювання.
            - Не вмикайте/не вимикайте світло та будь-які електроприлади!
            - Негайно зателефонуйте до аварійної служби газу за номером 104 (з мобільного чи стаціонарного телефону).
            - Залиште небезпечне приміщення.
            """
            return {
                "messages": [assistant_msg(message)],
            }
        case 'MunicipalUtility_Electricity':
            message = """
            ❗️ УВАГА: Небезпека ураження струмом!
            Ми зафіксували вашу скаргу про обрив електропроводів/іскріння. Це вкрай небезпечно!
            Не наближайтесь до місця обриву ближче ніж на 8 метрів.
            Не торкайтесь проводів.
            Аварійна служба РЕМ (Район електричних мереж) вже повідомлена і прямує на місце події. Будьте обережні.
            """
            return {
                "messages": [assistant_msg(message)],
            }
        case 'MunicipalUtility_Water':
            message = """
            ❗️ УВАГА: Аварія на зовнішніх мережах водопостачання!
            Ми отримали ваше повідомлення про витік води (прорив труби / гідранта) на вулиці. Цю скаргу класифіковано як екстрену аварію.
            Бригада аварійно-відновлювальних робіт Вже прямує на місце події для локалізації та усунення витоку.
            Ваші дії:
            Будь ласка, тримайтеся на безпечній відстані від місця прориву.
            Не намагайтеся самостійно перекрити гідрант або трубу.
            Якщо поруч є відкриті електропроводи, попередьте перехожих про небезпеку.
            """
            return {
                "messages": [assistant_msg(message)],
            }
        case 'MunicipalUtility_GreeneryService':
            message = """
            ❗️ УВАГА: Загроза безпеці!
            Дякуємо за повідомлення про повалене дерево, яке загрожує життю/майну. Ми класифікували це як екстрену ситуацію.
            Будь ласка, тримайтеся на безпечній відстані від небезпечного місця.
            Чергова бригада відповідної комунальної служби вже отримала заявку.
            """
            return {
                "messages": [assistant_msg(message)],
            }
        case _:  # Default case (wildcard)\
            message = """
            ❗️ УВАГА: Загроза безпеці!
            Зверніться за номером 112
            """
            return {
                "messages": [assistant_msg(message)],
            }


def get_problem_code(problem):
    if problem is not None:
        return problem.get("code")

    return None
