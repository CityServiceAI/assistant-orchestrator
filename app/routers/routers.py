from fastapi import APIRouter
from app.pipeline.graph import build_complaint_graph
from app.schemas.conversation import Conversation
from app.pipeline.conversation_graph_state import ConversationGraphState
from app.storage import complaint_state_store as store

router = APIRouter(prefix="/conversations", tags=["conversation"])

graph = build_complaint_graph()


def message_as_json(message):
    return {"role": message["role"], "content": message["content"]}


def as_model(state):
    return {
        "id": state.get("conversation_id", "123"),
        "messages": [message_as_json(m) for m in state.get("messages", [])],
        "trace": state.get("trace", []),
        "context": {
            "issue_category": state.get("category"),
            "issue_address": "",
            "issue_category_level": "",
            "issuer_name": "",
            "service_provider_name": "",
            "service_provider_email": "",
            "service_provider_phones": None,
            "emergency": "",
        },
    }


@router.post("/", response_model=Conversation)
async def conversation(body: Conversation) -> Conversation:
    prev_state = store.get(body.id) or {}

    incoming_messages = [
        {
            "role": m.role,
            "content": m.content,
            "agent": m.agent,
        }
        for m in body.messages
    ]

    state: ConversationGraphState = {
        **prev_state,
        "conversation_id": body.id,
        "messages": incoming_messages,
    }

    state = graph.invoke(state)
    store.set_(body.id, state)
    payload = as_model(state)
    return Conversation(**payload)
