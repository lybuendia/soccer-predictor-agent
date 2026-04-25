import os

from soccer_forecast_agent.config import Config


def test_config_defaults_embedding_provider_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "fd")
    monkeypatch.setenv("ODDS_API_KEY", "odds")
    monkeypatch.setenv("SEARCH_API_KEY", "search")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "user")
    monkeypatch.setenv("SMTP_PASSWORD", "pass")
    monkeypatch.setenv("ALERT_EMAIL", "alerts@example.com")
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    config = Config.from_env()

    assert config.embedding_provider == "huggingface"
    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
