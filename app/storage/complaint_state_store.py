from typing import Dict
from app.pipeline.state import ComplaintState

_store: Dict[str, ComplaintState] = {}


def get(conversation_id: str) -> ComplaintState | None:
    return _store.get(conversation_id)


def set_(conversation_id: str, state: ComplaintState) -> None:
    _store[conversation_id] = state
