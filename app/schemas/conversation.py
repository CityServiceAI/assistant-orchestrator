from typing import List, Optional
from pydantic import BaseModel, Field
import datetime
from uuid import uuid4


class ConversationMessage(BaseModel):
    role: str
    content: str
    agent: Optional[str] = None

class ProblemDescriptor(BaseModel):
    code: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    responsible_entity_type: Optional[str] = Field(None)
    category_name: Optional[str] = Field(None)

class ProblemSummary(BaseModel):
    normalized_description: Optional[str] = Field(None)
    context_notes: Optional[str] = Field(None)

class ConversationContext(BaseModel):
    issue_category: Optional[str] = Field(None, example="WATER_SUPPLY")
    issue_address: Optional[str] = Field(
        None, example="м. Київ, вул. Григоренка 16, кв. 47"
    )
    issue_category_level: Optional[str] = Field(None, example="LOCAL")
    issuer_name: Optional[str] = Field(None, example="Остап Остапенко")
    service_provider_name: Optional[str] = Field(None, example="")
    service_provider_email: Optional[str] = Field(None, example="")
    service_provider_phones: Optional[List[str]] = Field(
        None, example=["0960000000000", "0440000000000000"]
    )
    emergency: Optional[str] = Field(None, example="LOW")


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    context: Optional[ConversationContext] = None
    messages: List[ConversationMessage] = Field([])
    summary: Optional[ProblemSummary] = None
    trace: Optional[List[dict]] = Field([])

    class Config:
        json_schema_extra = {
            "example": {
                "id": "c2c8331e-4716-4d71-a010-206f70274121",
                "context": {
                    "issue_category": "WATER_SUPPLY",
                    "issue_address": "м. Київ, вул. Григоренка 16, кв. 47",
                    "issue_category_level": "LOCAL",
                    "issuer_name": "Остап Остапенко",
                    "service_provider_name": "",
                    "service_provider_email": "",
                    "service_provider_phones": ["0960000000000", "0440000000000000"],
                    "emergency": "LOW",
                },
                "messages": [
                    {
                        "role": "user",
                        "content": "Протікає вода з крану за адресою, м. Київ, вул. Григоренка 16, кв. 47",
                    },
                    {
                        "role": "assistant",
                        "content": "Зверніться до служби за телефоном",
                        "agent": "agent-13132",
                    },
                    {"role": "user", "content": "Дякую відправив"},
                ],
            }
        }
