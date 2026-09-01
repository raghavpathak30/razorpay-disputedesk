"""Single source of truth for environment-dependent settings. Nothing else reads os.environ."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    razorpay_key_id: str
    razorpay_key_secret: str
    # Not a separate host for test vs. live mode - Razorpay determines mode from
    # which key pair (test vs. live prefix) is configured, not the URL. Has a
    # default so existing .env files without this line keep working.
    razorpay_api_base_url: str = "https://api.razorpay.com/v1"
    llm_api_key: str
    llm_api_url: str
    llm_model: str
    database_url: str


@lru_cache
def get_settings() -> Settings:
    """Load settings from the environment, failing loudly if a required variable is missing."""
    return Settings()
