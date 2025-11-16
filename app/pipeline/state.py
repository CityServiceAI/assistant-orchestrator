from typing import TypedDict, List, Optional
from app.pipeline.context import LLMCallRecord


class ComplaintState(TypedDict, total=False):
    conversation_id: str
    aggregated_raw_text: str
    user_turns: List[str]
    raw_text: str
    normalized_text: Optional[str]
    normalizer_safe: Optional[bool]
    normalizer_warnings: List[str]
    cleaned_text_uk: Optional[str]
    category: Optional[str]
    category_confidence: Optional[float]
    category_need_clarification: bool
    category_clarification_question: Optional[str]
    llm_calls: List[LLMCallRecord]
