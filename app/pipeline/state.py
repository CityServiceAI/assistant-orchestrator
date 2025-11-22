import operator
from typing import Optional, NotRequired

from typing_extensions import Annotated, TypedDict


class ProblemDescriptor(TypedDict, total=False):
    code: Optional[str]
    description: Optional[str]
    responsible_entity_type: Optional[str]
    category_name: Optional[str]

class ProblemSummary(TypedDict, total=False):
    normalized_description: Optional[str]
    context_notes: Optional[str]

class LocationDetails(TypedDict, total=False):
    city: Optional[str]
    street: Optional[str]
    building_number: Optional[str]
    apartment: Optional[str]

class ConversationGraphState(TypedDict, total=False):
    messages: Annotated[list, operator.add]
    #ToDo Check
    issue_text: str
    # ToDo Не накращий спосіб, мені здається має бути кращі варіанти
    message: str
    trace: Annotated[list, operator.add]

    conversation_id: str
    category: Optional[str]
    category_confidence: Optional[float]
    need_clarification: bool
    clarification_count: Optional[int]
    problem: Optional[ProblemDescriptor]
    summary: Optional[ProblemSummary]
    emergency_score: Optional[float]
    problems: NotRequired[list[ProblemDescriptor]]
    guardrail_blocked: Optional[bool]
    location_type: Optional[str]
    location_details: Optional[LocationDetails]
