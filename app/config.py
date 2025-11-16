from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_url: str = Field(alias="LLM_PROXY_SERVICE_API_URL")
    llm_key: str = Field(alias="LLM_PROXY_SERVICE_API_KEY")
    default_model: str = "gpt-4.1-mini"
    normalizer_max_chars: int = 3000


settings = AppSettings()
