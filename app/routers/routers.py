import logging
from fastapi import APIRouter
from app.pipeline.graph import build_complaint_graph
from app.schemas.conversation import Conversation
from app.pipeline.state import ConversationGraphState
from app.services.langfuse_service import get_langfuse_service

router = APIRouter(prefix="/conversations", tags=["conversation"])
graph = build_complaint_graph()


@router.post("/", response_model=Conversation)
async def conversation(body: Conversation) -> Conversation:
    graph_state = ConversationGraphState(body.model_dump())
    conversation_id = body.id or graph_state.get("conversation_id", "unknown")
    langfuse_service = get_langfuse_service()
    langfuse_trace = None

    if langfuse_service.enabled and langfuse_service.langfuse:
        try:
            last_user_message = next(
                (
                    m.get("content", "")
                    for m in reversed(graph_state.get("messages", []))
                    if m.get("role") == "user"
                ),
                "",
            )

            langfuse_trace = langfuse_service.langfuse.start_span(
                name="conversation_flow",
                input={
                    "conversation_id": conversation_id,
                    "message_count": len(graph_state.get("messages", [])),
                    "last_user_message": last_user_message[:200]
                    if last_user_message
                    else None,  # Обмежуємо довжину
                },
                metadata={
                    "conversation_id": conversation_id,
                    "message_count": len(graph_state.get("messages", [])),
                },
            )
            langfuse_trace.update_trace(
                user_id=conversation_id,
                session_id=conversation_id,
            )
            logging.info(f"Langfuse trace створено: conversation_id={conversation_id}")
        except Exception as e:
            logging.error(f"Помилка створення Langfuse trace: {e}", exc_info=True)
            langfuse_trace = None

    config = {}
    if langfuse_service.enabled:
        callback_handler = langfuse_service.get_callback_handler()
        if callback_handler:
            config = {"callbacks": [callback_handler]}

    try:
        state = graph.invoke(graph_state, config=config if config else None)

        if langfuse_trace:
            try:
                langfuse_trace.update(
                    output={
                        "messages_count": len(state.get("messages", [])),
                        "has_summary": bool(state.get("summary")),
                        "category": state.get("category"),
                    }
                )
                langfuse_trace.end()
            except Exception as e:
                logging.debug(f"Помилка оновлення Langfuse trace: {e}")
    except Exception as e:
        if langfuse_trace:
            try:
                langfuse_trace.update(level="ERROR", status_message=str(e))
                langfuse_trace.end()
            except Exception:
                pass
        logging.error(f"Помилка виклику графа: {e}", exc_info=True)
        raise

    if langfuse_service.enabled and langfuse_service.langfuse:
        try:
            langfuse_service.langfuse.flush()
        except Exception as e:
            logging.debug(f"Помилка flush Langfuse: {e}")

    return Conversation(**state)
