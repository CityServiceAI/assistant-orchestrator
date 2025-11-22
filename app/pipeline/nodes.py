import logging
from typing import Optional, Any

from app.agents.category_classifier import CategoryClassifierAgent
from app.agents.category_rules import pre_classification_rules
from app.agents.classifier_v2 import ClassifierV2
from app.agents.classifier_v3 import ClassifierV3
from app.agents.service_agent import ServiceAgent
from app.agents.location_agent import LocationAgent
from app.data.rag import rag_search_categories
from app.pipeline.state import ConversationGraphState
from app.tools.normalizer_tool import normalize_text
from app.tools.response import assistant_msg
from app.services.guardrails import get_guardrails_service
from app.services.langfuse_service import get_langfuse_service

category_agent = CategoryClassifierAgent()


def _trace_node(node_name: str):
    def decorator(func):
        def wrapper(state: ConversationGraphState):
            langfuse_service = get_langfuse_service()
            span = None

            if langfuse_service.enabled and langfuse_service.langfuse:
                try:
                    # Отримуємо останнє повідомлення користувача для input (якщо є)
                    last_user_message = next(
                        (
                            m.get("content", "")
                            for m in reversed(state.get("messages", []))
                            if m.get("role") == "user"
                        ),
                        None,
                    )

                    input_data = {
                        "messages_count": len(state.get("messages", [])),
                        "has_summary": bool(state.get("summary")),
                        "category": state.get("category"),
                    }

                    # Додаємо текст повідомлення для нод, які обробляють текст
                    if last_user_message and node_name in (
                        "normalize",
                        "category-classifier",
                        "location",
                    ):
                        input_data["user_message"] = last_user_message[
                            :200
                        ]  # Обмежуємо довжину

                    # Визначаємо тип: agent або node
                    agent_nodes = ("category-classifier", "service", "location")
                    node_type = "agent" if node_name in agent_nodes else "node"

                    span = langfuse_service.langfuse.start_span(
                        name=node_name,
                        input=input_data,
                        metadata={
                            "node_type": "langgraph_node",
                            "component_type": node_type,
                        },
                    )
                except Exception as e:
                    logging.debug(
                        f"Не вдалося створити Langfuse span для ноди {node_name}: {e}"
                    )

            try:
                result = func(state)

                if span:
                    try:
                        span.update(
                            output={
                                "messages_count": len(result.get("messages", [])),
                                "has_summary": bool(result.get("summary")),
                                "category": result.get("category"),
                                "need_clarification": result.get("need_clarification"),
                            },
                            metadata={"node_completed": True},
                        )
                        span.end()
                    except Exception as e:
                        logging.debug(
                            f"Не вдалося оновити Langfuse span для ноди {node_name}: {e}"
                        )

                return result
            except Exception as e:
                if span:
                    try:
                        span.update(level="ERROR", status_message=str(e))
                        span.end()
                    except Exception:
                        pass
                raise

        return wrapper

    return decorator


def _create_guardrail_blocked_response(
    node_name: str, message: str, violation_type: Optional[str] = None
) -> ConversationGraphState:
    trace_data = {
        "node": node_name,
        "guardrail_blocked": True,
        "guardrail_message": message,
    }
    if violation_type:
        trace_data["guardrail_violation_type"] = violation_type

    return {
        "messages": [
            assistant_msg(
                "Вибачте, ваш запит не може бути оброблений через політики безпеки. "
                "Будь ласка, сформулюйте ваше звернення коректно."
            )
        ],
        "guardrail_blocked": True,
        "trace": [trace_data],
        "message": None,
        "issue_text": None,
    }


def run_guardrails(message) -> ConversationGraphState | None:
    try:
        guardrails = get_guardrails_service()
        if guardrails.enabled and message:
            guardrail_check = guardrails.check_user_input(message)

            if guardrail_check.is_blocked:
                logging.warning(
                    f"Guardrail заблокував запит: {guardrail_check.message}"
                )
                violation_type = (
                    guardrail_check.violation_type.value
                    if guardrail_check.violation_type
                    else None
                )
                return _create_guardrail_blocked_response(
                    "normalize_node", guardrail_check.message, violation_type
                )

            if guardrail_check.action.value == "UNKNOWN":
                is_timeout = "Таймаут" in (guardrail_check.message or "")
                if not is_timeout:
                    logging.error(
                        f"Guardrails API помилка (fail-safe блокування): {guardrail_check.message}"
                    )
                    return {
                        "messages": [
                            assistant_msg(
                                "Вибачте, система перевірки безпеки тимчасово недоступна. "
                                "Будь ласка, спробуйте пізніше."
                            )
                        ],
                        "guardrail_blocked": True,
                        "trace": [
                            {
                                "node": "normalize_node",
                                "guardrail_error": True,
                                "guardrail_message": guardrail_check.message,
                            }
                        ],
                        "message": None,
                        "issue_text": None,
                    }
    except Exception as e:
        logging.error(f"Помилка перевірки Guardrails: {e}", exc_info=True)

    return None


@_trace_node("normalize")
def normalize_node(state: ConversationGraphState) -> ConversationGraphState:
    last_user_msg = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    )
    logging.info(f"User input {last_user_msg}")

    try:
        from app.services.guardrails import get_guardrails_service

        guardrails = get_guardrails_service()
        if guardrails.enabled and last_user_msg:
            guardrail_check = guardrails.check_user_input(last_user_msg)

            if guardrail_check.is_blocked:
                logging.warning(
                    f"Guardrail заблокував запит: {guardrail_check.message}"
                )
                violation_type = (
                    guardrail_check.violation_type.value
                    if guardrail_check.violation_type
                    else None
                )
                return _create_guardrail_blocked_response(
                    "normalize_node", guardrail_check.message, violation_type
                )

            if guardrail_check.action.value == "UNKNOWN":
                is_timeout = "Таймаут" in (guardrail_check.message or "")
                if not is_timeout:
                    logging.error(
                        f"Guardrails API помилка (fail-safe блокування): {guardrail_check.message}"
                    )
                    return {
                        "messages": [
                            assistant_msg(
                                "Вибачте, система перевірки безпеки тимчасово недоступна. "
                                "Будь ласка, спробуйте пізніше."
                            )
                        ],
                        "guardrail_blocked": True,
                        "trace": [
                            {
                                "node": "normalize_node",
                                "guardrail_error": True,
                                "guardrail_message": guardrail_check.message,
                            }
                        ],
                        "message": None,
                        "issue_text": None,
                    }
        elif not guardrails.enabled:
            logging.warning("Guardrails вимкнено - пропускаємо перевірку безпеки")
    except Exception as e:
        logging.error(f"Помилка перевірки Guardrails: {e}", exc_info=True)
        return {
            "messages": [
                assistant_msg(
                    "Вибачте, система перевірки безпеки тимчасово недоступна. "
                    "Будь ласка, спробуйте пізніше."
                )
            ],
            "guardrail_blocked": True,
            "trace": [
                {
                    "node": "normalize_node",
                    "guardrail_error": True,
                    "guardrail_message": str(e),
                }
            ],
            "message": None,
            "issue_text": None,
        }

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


@_trace_node("category-classifier")
def category_node(state: ConversationGraphState) -> ConversationGraphState:
    if "message" not in state:
        raise Exception("No message in state for category_node")

    if state.get("summary") is not None and state.get("summary").get(
        "normalized_description"
    ):
        text = (
            state.get("summary").get("normalized_description") + " " + state["message"]
        )
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

        try:
            result = category_agent.run(
                text=text,
                candidates=candidates,
                is_clarification=state.get("need_clarification"),
                previous_summary=state.get("summary"),
            )
        except ValueError as e:
            if "Guardrails" in str(e) or "заблоковано" in str(e):
                logging.warning(f"Guardrails заблокував запит: {e}")
                return _create_guardrail_blocked_response("category_node", str(e))
            raise

    return {
        **result,
        "category": result.get("category"),
        "category_confidence": float(result.get("confidence", 0.0)),
        "clarification_count": increase_clarification_count(state, result),
        "messages": [get_clarification_message(result)]
        if get_clarification_message(result)
        else [],
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
        return assistant_msg(
            result.get("clarification_question"), "category_classifier"
        )

    return None


@_trace_node("service")
def search_service_node(state: ConversationGraphState) -> ConversationGraphState:
    logging.info(f"Search service node. State: {state}")

    results = ServiceAgent().run(state.get("summary"), state.get("problems"), state.get("location_type"), state.get("location_details"))

    return {
        **results,
        "trace": [
            {
                **results,
                "agent": "service",
            }
        ],
    }


@_trace_node("ask_clarification")
def ask_clarification_node(state: ConversationGraphState):
    logging.info("Ask clarification node.")
    return {}


@_trace_node("generate_appeal")
def generate_appeal_node(state: ConversationGraphState):
    logging.info(f"Generate appeal node: State: {state}")
    problems = list(map(to_assistant_message, state.get("problems", [])))
    return {
        "messages": problems + [assistant_msg("Дякуємо за звернення")]
    }

def to_assistant_message(x):
    return assistant_msg(
        f"Ваша проблема {x.get('code')}: {x.get('description')}, {x.get('category_name')}"
    )

def route_after_normalize(state: ConversationGraphState):
    if state.get("guardrail_blocked"):
        logging.debug("Guardrails заблокував запит - завершуємо виконання")
        return "end"
    return "category"


def route_after_classification(state: ConversationGraphState):
    logging.info(f"Route after classification. State {state}")

    problems = state.get("problems", [])
    confidence = state.get("category_confidence", 0)
    need_clarification = state.get("need_clarification", False)
    clarification_count = state.get("clarification_count", 0)
    emergency_score = state.get("emergency_score", 0)

    if emergency_score > 0 and need_clarification:
        return "ask_clarification"

    if emergency_score >= 0.9 and not need_clarification:
        return "emergency"

    if need_clarification:
        logging.info("No category, need clarification → ask_clarification")
        return "ask_clarification"

    if len(problems) < 1:
        logging.info("No category and no clarification → handle_failure")
        return "handle_failure"

    logging.info("Redirect to location")
    return "location"


def route_after_service_search(state: ConversationGraphState):
    return "generate_appeal"


@_trace_node("handle_failure")
def handle_failure_node(state: ConversationGraphState):
    return {
        "messages": [
            assistant_msg(
                "Вибачте, нажаль я не можу визначити відповідальну службу за вашу проблему"
            ),
            assistant_msg(
                "Зверніться за номером 1551 або створіть звернення на сайті контактного центру міста Києва https://1551.gov.ua"
            ),
        ]
    }


@_trace_node("category-classifier")
def classifier_node(state: ConversationGraphState) -> ConversationGraphState:
    try:
        result = ClassifierV2().run(
            state["messages"][-1]["content"],
            state.get("need_clarification"),
            state.get("summary"),
        )
    except ValueError as e:
        if "Guardrails" in str(e) or "заблоковано" in str(e):
            logging.warning(f"Guardrails заблокував запит: {e}")
            return _create_guardrail_blocked_response("classifier_node", str(e))
        raise

    if result.get("need_clarification", True):
        assistant_message = [assistant_msg(result.get("clarification_question"))]
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


@_trace_node("category-classifier")
def classifier_node_3(state: ConversationGraphState) -> ConversationGraphState:
    logging.info(f"Classifier node: request {state}")
    result = ClassifierV3().run(state)

    if result.get("need_clarification", True):
        assistant_message = [assistant_msg(result.get("clarification_question"))]
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


@_trace_node("emergency")
def emergency_node(state: ConversationGraphState) -> ConversationGraphState:
    messages = []
    for problem in state.get("problems", []):
        messages.append(to_assistant_message(problem))
        messages.append(assistant_msg(get_emergency_message(problem)))
    return {"messages": messages}


def get_emergency_message(responsible_entity_type):
    match responsible_entity_type:
        case "MunicipalUtility_GasService":
            return """
            ❗️ УВАГА: Загроза вибуху!
            Ми зафіксували вашу скаргу про запах газу. 
            Будь ласка, негайно виконайте наступні дії:
            - Перекрийте вентилі на газових приладах та на вході в квартиру/будинок.
            - Відчиніть вікна для провітрювання.
            - Не вмикайте/не вимикайте світло та будь-які електроприлади!
            - Негайно зателефонуйте до аварійної служби газу за номером 104 (з мобільного чи стаціонарного телефону).
            - Залиште небезпечне приміщення.
            """
        case "MunicipalUtility_Electricity":
            return """
            ❗️ УВАГА: Небезпека ураження струмом!
            Ми зафіксували вашу скаргу про обрив електропроводів/іскріння. Це вкрай небезпечно!
            Не наближайтесь до місця обриву ближче ніж на 8 метрів.
            Не торкайтесь проводів.
            Аварійна служба РЕМ (Район електричних мереж) вже повідомлена і прямує на місце події. Будьте обережні.
            """
        case "MunicipalUtility_Water":
            return """
            ❗️ УВАГА: Аварія на зовнішніх мережах водопостачання!
            Ми отримали ваше повідомлення про витік води (прорив труби / гідранта) на вулиці. Цю скаргу класифіковано як екстрену аварію.
            Бригада аварійно-відновлювальних робіт Вже прямує на місце події для локалізації та усунення витоку.
            Ваші дії:
            Будь ласка, тримайтеся на безпечній відстані від місця прориву.
            Не намагайтеся самостійно перекрити гідрант або трубу.
            Якщо поруч є відкриті електропроводи, попередьте перехожих про небезпеку.
            """
        case "MunicipalUtility_GreeneryService":
            return """
            ❗️ УВАГА: Загроза безпеці!
            Дякуємо за повідомлення про повалене дерево, яке загрожує життю/майну. Ми класифікували це як екстрену ситуацію.
            Будь ласка, тримайтеся на безпечній відстані від небезпечного місця.
            Чергова бригада відповідної комунальної служби вже отримала заявку.
            """
        case _:
            return """
            ❗️ УВАГА: Загроза безпеці!
            Зверніться за номером 112
            """


def get_problem_code(problem):
    if problem is not None:
        return problem.get("code")

    return None


@_trace_node("location")
def location(state: ConversationGraphState) -> ConversationGraphState:
    try:
        result = LocationAgent().run(
            state.get("summary"), state["messages"][-1]["content"]
        )
    except ValueError as e:
        if "Guardrails" in str(e) or "заблоковано" in str(e):
            logging.warning(f"Guardrails заблокував запит: {e}")
            return _create_guardrail_blocked_response("location", str(e))
        raise

    if result.get("need_clarification", False):
        assistant_messages = [assistant_msg(result.get("clarification_question"))]
    else:
        assistant_messages = []

    return {**result, "messages": assistant_messages, "trace": [{**result}]}


def route_after_location(state: ConversationGraphState):
    if state.get("need_clarification"):
        return "ask_clarification"

    return "service_search"
