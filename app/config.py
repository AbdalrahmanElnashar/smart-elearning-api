from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_DIR / ".env"), extra="ignore")

    gemini_api_key: str = ""
    model_dir: Path = PROJECT_DIR / "models"

    gemini_model: str = "gemini-3.5-flash"
    gemini_timeout_seconds: float = 12.0
    feedback_cache_ttl_seconds: int = 600
    feedback_rate_limit: str = "20/minute"


settings = Settings()
