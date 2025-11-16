from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMCallRecord:
    agent: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass
class PipelineContext:
    raw_text: str

    normalized_text: Optional[str] = None
    normalizer_safe: bool = True
    normalizer_warnings: List[str] = field(default_factory=list)

    cleaned_text_uk: Optional[str] = None

    llm_calls: List[LLMCallRecord] = field(default_factory=list)

    def add_llm_call(self, record: LLMCallRecord):
        self.llm_calls.append(record)
