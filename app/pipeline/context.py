from dataclasses import dataclass, field


@dataclass
class LLMCallRecord:
    agent: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class PipelineContext:
    raw_text: str
    normalized_text: str | None = None
    normalizer_safe: bool | None = None
    normalizer_warnings: list[str] = field(default_factory=list)
    cleaned_text_uk: str | None = None
    category: str | None = None
    category_confidence: float | None = None
    category_need_clarification: bool = False
    category_clarification_question: str | None = None
    llm_calls: list[LLMCallRecord] = field(default_factory=list)

    def add_llm_call(self, record: LLMCallRecord) -> None:
        self.llm_calls.append(record)
