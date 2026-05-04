"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """All runtime settings. Load via Config.from_env() — never instantiate with literals in production."""

    llm_provider: str
    llm_model: str
    embedding_provider: str
    openai_api_key: str
    anthropic_api_key: str
    football_data_api_key: str
    odds_api_key: str
    search_api_key: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    alert_email: str
    db_path: str
    chroma_path: str
    embedding_model: str
    min_edge_threshold: float = 0.05
    min_confidence_threshold: float = 0.60
    home_advantage_boost: float = 0.05
    react_max_steps: int = 8
    min_evidence_count: int = 3
    spam_window_hours: int = 6
    base_sensitivity: float = 0.15

    @classmethod
    def from_env(cls) -> "Config":
        """Build Config from environment variables. Raises KeyError for any missing required var."""
        return cls(
            llm_provider=os.environ["LLM_PROVIDER"],
            llm_model=os.environ.get("LLM_MODEL", "gpt-4o"),
            embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "huggingface"),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            football_data_api_key=os.environ["FOOTBALL_DATA_API_KEY"],
            odds_api_key=os.environ["ODDS_API_KEY"],
            search_api_key=os.environ["SEARCH_API_KEY"],
            smtp_host=os.environ["SMTP_HOST"],
            smtp_port=int(os.environ.get("SMTP_PORT", "587")),
            smtp_user=os.environ["SMTP_USER"],
            smtp_password=os.environ["SMTP_PASSWORD"],
            alert_email=os.environ["ALERT_EMAIL"],
            db_path=os.environ.get("DB_PATH", "soccer_forecast.db"),
            chroma_path=os.environ.get("CHROMA_PATH", ".chroma"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        )
