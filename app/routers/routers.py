from fastapi import APIRouter
from app.pipeline.graph import build_complaint_graph
from app.schemas.conversation import Conversation, ConversationContext, ConversationMessage
from app.pipeline.state import ConversationGraphState
from uuid import uuid4

router = APIRouter(prefix="/conversations", tags=["conversation"])

graph = build_complaint_graph()

@router.post("/", response_model=Conversation)
async def conversation(body: Conversation) -> Conversation:
    graph_state = ConversationGraphState(body.model_dump())
    state = graph.invoke(graph_state)

    return Conversation(**state)
