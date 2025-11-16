from openai import AzureOpenAI
from app.config import settings

client = AzureOpenAI(
    api_key=settings.llm_key,
    azure_endpoint=settings.llm_url,
    api_version="2024-02-01",
)


def chat_completion(model: str, messages: list[dict], **kwargs):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
