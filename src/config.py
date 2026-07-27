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
    # The only browser origin allowed to call the API cross-origin (see src/api/main.py's
    # CORSMiddleware). Defaults to the dashboard's local dev port; override in production
    # once the dashboard's real deployed origin is known (Day 6 / Step 4).
    dashboard_origin: str = "http://localhost:8501"
    # Dated snapshot, not the rolling "gpt-4o-mini" alias - judge model/version must be
    # pinned explicitly and never silently move out from under the gate (see Run Result
    # Schema's judge_model field, and internal/mentoring_notes.md Day 2 / Step 5).
    judge_model: str = "gpt-4o-mini-2024-07-18"


@lru_cache
def get_settings() -> Settings:
    return Settings()
