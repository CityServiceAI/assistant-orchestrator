from openai import AzureOpenAI
from pydantic import BaseModel
from app.config import settings

class LLMCallConfig(BaseModel):
    model: str
    temperature: float = 0.2
    max_tokens: int | None = None


client = AzureOpenAI(
    api_key=settings.llm_key,
    azure_endpoint=settings.llm_url,
    api_version="2024-02-01",
)


def chat_completion(config: LLMCallConfig, messages: list[dict]):
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return response
