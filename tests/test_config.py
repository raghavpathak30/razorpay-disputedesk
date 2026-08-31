"""Config module must fail loudly when required env vars are missing, and load when present."""

import pytest
from pydantic import ValidationError

from disputedesk.config import Settings, get_settings

REQUIRED_VARS = {
    "RAZORPAY_KEY_ID": "rzp_test_id",
    "RAZORPAY_KEY_SECRET": "rzp_test_secret",
    "LLM_API_KEY": "llm_test_key",
    "DATABASE_URL": "sqlite:///./test.db",
}


@pytest.fixture(autouse=True)
def _clear_env_and_cache(monkeypatch):
    for name in REQUIRED_VARS:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_missing_required_variable_fails_loudly(monkeypatch):
    for name, value in REQUIRED_VARS.items():
        if name != "DATABASE_URL":
            monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert "database_url" in str(exc_info.value).lower()


def test_loads_correctly_when_all_variables_present(monkeypatch):
    for name, value in REQUIRED_VARS.items():
        monkeypatch.setenv(name, value)

    settings = get_settings()

    assert settings.razorpay_key_id == REQUIRED_VARS["RAZORPAY_KEY_ID"]
    assert settings.razorpay_key_secret == REQUIRED_VARS["RAZORPAY_KEY_SECRET"]
    assert settings.llm_api_key == REQUIRED_VARS["LLM_API_KEY"]
    assert settings.database_url == REQUIRED_VARS["DATABASE_URL"]
