from openai import AzureOpenAI
import os

LLM_PROXY_SERVICE_API_URL = os.getenv('LLM_PROXY_SERVICE_API_URL')
LLM_PROXY_SERVICE_API_KEY = os.getenv('LLM_PROXY_SERVICE_API_KEY')

client = AzureOpenAI(
    api_key=LLM_PROXY_SERVICE_API_KEY,
    azure_endpoint=LLM_PROXY_SERVICE_API_URL,
    api_version="2024-02-01",
)


