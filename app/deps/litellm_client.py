from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

LLM_PROXY_SERVICE_API_URL = os.getenv("LLM_PROXY_SERVICE_API_URL")
LLM_PROXY_SERVICE_API_KEY = os.getenv("LLM_PROXY_SERVICE_API_KEY")

if not LLM_PROXY_SERVICE_API_URL or not LLM_PROXY_SERVICE_API_KEY:
    raise RuntimeError("LLM credentials is missing in .env. ")

client = AzureOpenAI(
    api_key=LLM_PROXY_SERVICE_API_KEY,
    azure_endpoint=LLM_PROXY_SERVICE_API_URL,
    api_version="2024-02-01",
)
