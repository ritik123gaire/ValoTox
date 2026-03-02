"""Centralised configuration loaded from .env and sensible defaults."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATED_DIR = DATA_DIR / "annotated"
MODEL_DIR = ROOT_DIR / "models"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, ANNOTATED_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """App-wide settings hydrated from .env."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Reddit
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "ValoTox Research Dataset v1.0"

    # Twitter
    twitter_bearer_token: str = ""

    # OpenAI
    openai_api_key: str = ""

    # Label Studio
    label_studio_url: str = "http://localhost:8080"
    label_studio_api_key: str = ""

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "valotox"

    # Model
    model_dir: str = str(MODEL_DIR)
    best_model: str = str(MODEL_DIR / "roberta-valotox")


settings = Settings()
