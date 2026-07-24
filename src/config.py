from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AEG_",
        extra="ignore",
    )

    llm_provider: Literal["openai", "anthropic"] = "openai"
    model_name: str = "gpt-4o-mini"
    api_key: SecretStr
    embedding_model_name: str = "text-embedding-3-small"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
