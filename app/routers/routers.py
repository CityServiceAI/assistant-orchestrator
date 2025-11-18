from fastapi import APIRouter
from app.pipeline.graph import build_complaint_graph
from app.schemas.conversation import Conversation, ConversationContext, ConversationMessage
from app.pipeline.conversation_graph_state import ConversationGraphState
from uuid import uuid4

router = APIRouter(prefix="/conversations", tags=["conversation"])

graph = build_complaint_graph()


def message_as_json(message):
    return {
        "role": message["role"],
        "content": message["content"]
    }


def as_model(state):
    return {
        "id": "123",
        "messages": list(map(lambda x: message_as_json(x), state["messages"])),
        "trace": state['trace'],
        "context": {
            "issue_category": state["category"],
            "issue_address": "",
            "issue_category_level": "",
            "issuer_name": "",
            "service_provider_name": "",
            "service_provider_email": "",
            "service_provider_phones": None,
            "emergency": "",
        }
    }


@router.post("/", response_model=Conversation)
async def conversation(body: Conversation) -> Conversation:
    state = ConversationGraphState()
    state['messages'] = list(map(lambda x: {"role": x.role, "content": x.content}, body.messages))
    state = graph.invoke(state)

    return as_model(state)
