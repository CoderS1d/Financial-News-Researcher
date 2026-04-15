from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./output"))

    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    MAX_TOKENS: int = 4096

    # HTTP
    HTTP_TIMEOUT: float = 30.0
    HTTP_MAX_RETRIES: int = 3

    # RSS feeds for financial news
    RSS_FEEDS: list[str] = [
        "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.investing.com/rss/news.rss",
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    ]

    NEWSAPI_BASE_URL: str = "https://newsapi.org/v2"

    @classmethod
    def ensure_output_dir(cls) -> None:
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
