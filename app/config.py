from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_url: str = Field(alias="LLM_PROXY_SERVICE_API_URL")
    llm_key: str = Field(alias="LLM_PROXY_SERVICE_API_KEY")
    language_cleanup_model: str = Field(
        default="gpt-4.1-mini",
        alias="LANGUAGE_CLEANUP_MODEL",
    )
    normalizer_max_chars: int = 3000
    normalizer_input_hard_limit: int = 12000
    normalizer_preserve_newlines: bool = True


settings = AppSettings()
