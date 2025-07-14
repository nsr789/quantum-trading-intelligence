"""Centralised runtime configuration.

Loads environment variables from `.env` (via pydantic-settings) and exposes a
singleton `settings` object so every module does:

    from src.config.settings import settings
    api_key = settings.OPENAI_API_KEY
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# project root = two parents up from this file  (.../src/config/settings.py)
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
ENV_PATH: Path = ROOT_DIR / ".env"


class Settings(BaseSettings):
    # ───── LLM providers ────────────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # ───── Finance / data APIs ─────────────────────────────────────────────
    NEWS_API_KEY: Optional[str] = None
    FRED_API_KEY: Optional[str] = None
    ALPHAVANTAGE_API_KEY: Optional[str] = None
    FINNHUB_API_KEY: Optional[str] = None
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "quantum-trading-intel/0.1"

    # ───── General settings ────────────────────────────────────────────────
    CACHE_DIR: Path = ROOT_DIR / ".cache"
    DEFAULT_TIMEZONE: str = "US/Eastern"

    # new-style config (replaces inner `class Config`)
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # helpful helper for printing redacted keys in logs / reports
    def redacted(self) -> Dict[str, Any]:
        def _mask(v: Optional[str]) -> str:
            return "<set>" if v else "<empty>"

        return {
            "OPENAI_API_KEY": _mask(self.OPENAI_API_KEY),
            "GROQ_API_KEY": _mask(self.GROQ_API_KEY),
            "NEWS_API_KEY": _mask(self.NEWS_API_KEY),
            "FRED_API_KEY": _mask(self.FRED_API_KEY),
            "ALPHAVANTAGE_API_KEY": _mask(self.ALPHAVANTAGE_API_KEY),
            "FINNHUB_API_KEY": _mask(self.FINNHUB_API_KEY),
            "REDDIT": _mask(self.REDDIT_CLIENT_ID),
        }


# expose singleton
settings = Settings()
settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)  # ensure cache folder
