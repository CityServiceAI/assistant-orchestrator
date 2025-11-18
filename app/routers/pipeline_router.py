from uuid import uuid4
from enum import Enum
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.pipeline.graph import build_complaint_graph
from app.pipeline.conversation_graph_state import ConversationGraphState
from app.storage import complaint_state_store
from app.schemas.pipeline import LLMCallInfo

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
complaint_graph = build_complaint_graph()


class StepType(str, Enum):
    ASK_USER = "ASK_USER"
    CATEGORY_READY = "CATEGORY_READY"
    BLOCKED_BY_NORMALIZER = "BLOCKED_BY_NORMALIZER"


class ComplaintStepRequest(BaseModel):
    text: str
    conversation_id: str | None = None


class ComplaintStepResponse(BaseModel):
    conversation_id: str
    step_type: StepType

    cleaned_text_uk: str | None = None
    category: str | None = None
    confidence: float | None = None
    clarification_question: str | None = None

    normalizer_warnings: List[str] = Field(default_factory=list)
    llm_calls: List[LLMCallInfo] = Field(default_factory=list)


@router.post("/complaints/step", response_model=ComplaintStepResponse)
def complaint_step(req: ComplaintStepRequest) -> ComplaintStepResponse:
    if req.conversation_id:
        state = complaint_state_store.get(req.conversation_id)
        if state is None:
            conversation_id = req.conversation_id
            state = ConversationGraphState(
                conversation_id=conversation_id,
                aggregated_raw_text="",
                user_turns=[],
                llm_calls=[],
            )
        else:
            conversation_id = req.conversation_id
    else:
        conversation_id = str(uuid4())
        state = ConversationGraphState(
            conversation_id=conversation_id,
            aggregated_raw_text="",
            user_turns=[],
            llm_calls=[],
        )

    prev = state.get("aggregated_raw_text", "")
    if prev:
        agg = prev + "\n\nУточнення: " + req.text
    else:
        agg = req.text

    state["aggregated_raw_text"] = agg
    turns = state.get("user_turns", [])
    turns.append(req.text)
    state["user_turns"] = turns
    state["raw_text"] = state["aggregated_raw_text"]
    final_state = complaint_graph.invoke(state)

    complaint_state_store.set_(conversation_id, final_state)

    raw_llm_calls = final_state.get("llm_calls", []) or []
    llm_calls: List[LLMCallInfo] = []
    for item in raw_llm_calls:
        if isinstance(item, dict):
            try:
                llm_calls.append(LLMCallInfo(**item))
            except Exception:
                continue

    normalizer_warnings = final_state.get("normalizer_warnings", []) or []

    if final_state.get("normalizer_safe") is False:
        return ComplaintStepResponse(
            conversation_id=conversation_id,
            step_type=StepType.BLOCKED_BY_NORMALIZER,
            normalizer_warnings=normalizer_warnings,
            llm_calls=llm_calls,
        )

    if final_state.get("category_need_clarification"):
        return ComplaintStepResponse(
            conversation_id=conversation_id,
            step_type=StepType.ASK_USER,
            cleaned_text_uk=final_state.get("cleaned_text_uk"),
            category=final_state.get("category"),
            confidence=final_state.get("category_confidence"),
            clarification_question=final_state.get("category_clarification_question"),
            normalizer_warnings=normalizer_warnings,
            llm_calls=llm_calls,
        )

    return ComplaintStepResponse(
        conversation_id=conversation_id,
        step_type=StepType.CATEGORY_READY,
        cleaned_text_uk=final_state.get("cleaned_text_uk"),
        category=final_state.get("category"),
        confidence=final_state.get("category_confidence"),
        clarification_question=None,
        normalizer_warnings=normalizer_warnings,
        llm_calls=llm_calls,
    )
