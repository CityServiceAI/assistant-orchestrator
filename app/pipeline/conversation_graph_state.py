from doctest import debug
from typing import List, Optional
import operator
from langgraph.graph import add_messages
from typing_extensions import TypedDict, Annotated
from app.pipeline.context import LLMCallRecord
from dataclasses import dataclass, asdict


class ConversationGraphState(TypedDict, total=False):
    messages: Annotated[list, operator.add]

    #ToDo Не накращий спосіб, мені здається має бути кращі варіанти
    message: str
    debug: Annotated[list, operator.add]

    conversation_id: str
    aggregated_raw_text: str
    user_turns: List[str]
    raw_text: str
    normalized_text: Optional[str]
    normalizer_safe: Optional[bool]
    normalizer_warnings: List[str]
    category: Optional[str]
    category_confidence: Optional[float]
    category_need_clarification: bool
    category_clarification_question: Optional[str]
    llm_calls: List[LLMCallRecord]