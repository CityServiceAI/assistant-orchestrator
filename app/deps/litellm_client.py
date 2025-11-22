from openai import AzureOpenAI
import os

LLM_PROXY_SERVICE_API_URL = os.getenv("LLM_PROXY_SERVICE_API_URL")
LLM_PROXY_SERVICE_API_KEY = os.getenv("LLM_PROXY_SERVICE_API_KEY")

if not LLM_PROXY_SERVICE_API_URL or not LLM_PROXY_SERVICE_API_KEY:
    raise RuntimeError("LLM credentials is missing in .env. ")

_base_client = AzureOpenAI(
    api_key=LLM_PROXY_SERVICE_API_KEY,
    azure_endpoint=LLM_PROXY_SERVICE_API_URL,
    api_version="2024-02-01",
)

try:
    from app.deps.guarded_llm_client import get_guarded_client

    client = get_guarded_client()
except Exception as e:
    import logging

    logging.warning(
        f"Не вдалося завантажити guarded клієнт: {e}. "
        f"Використовується базовий клієнт без Guardrails."
    )
    client = _base_client
