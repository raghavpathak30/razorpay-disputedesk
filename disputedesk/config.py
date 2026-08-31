"""Single source of truth for environment-dependent settings. Nothing else reads os.environ."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    razorpay_key_id: str
    razorpay_key_secret: str
    llm_api_key: str
    database_url: str


@lru_cache
def get_settings() -> Settings:
    """Load settings from the environment, failing loudly if a required variable is missing."""
    return Settings()
