from typing import Optional
import operator
from typing_extensions import Annotated, TypedDict


class ConversationGraphState(TypedDict, total=False):
    messages: Annotated[list, operator.add]
    issue_text: str
    # ToDo Не накращий спосіб, мені здається має бути кращі варіанти
    message: str
    trace: Annotated[list, operator.add]

    conversation_id: str
    category: Optional[str]
    category_confidence: Optional[float]
    category_need_clarification: bool
    clarification_count: Optional[int]
