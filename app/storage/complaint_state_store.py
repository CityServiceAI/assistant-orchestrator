from typing import Dict
from app.pipeline.conversation_graph_state import ConversationGraphState

_store: Dict[str, ConversationGraphState] = {}


def get(conversation_id: str) -> ConversationGraphState | None:
    return _store.get(conversation_id)


def set_(conversation_id: str, state: ConversationGraphState) -> None:
    _store[conversation_id] = state
