"""FetcherAgent – gathers NewsArticles from RSS feeds and NewsAPI."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from agents.base_agent import BaseAgent
from config import config
from models.schemas import NewsArticle
from tools.news_api import fetch_newsapi
from tools.rss_reader import fetch_rss

logger = logging.getLogger(__name__)

_MAX_ARTICLES = 20
_LOOKBACK_HOURS = 48

_DEFAULT_RSS_SOURCES: list[str] = [
    # MarketWatch real-time headlines
    "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    # CNBC Finance
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    # CNBC Markets
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    # Seeking Alpha markets
    "https://seekingalpha.com/market_currents.xml",
    # Investing.com
    "https://www.investing.com/rss/news.rss",
]


class FetcherAgent(BaseAgent):
    """
    Concurrently fetches financial news from RSS feeds and NewsAPI,
    deduplicates by URL, filters to the last 48 hours, and returns
    up to 20 articles sorted by ``published_at`` descending.
    """

    def __init__(self) -> None:
        super().__init__()

    async def run(  # type: ignore[override]
        self,
        query: str,
        sources: list[str] | None = None,
        **kwargs,
    ) -> list[NewsArticle]:
        """
        Fetch articles for *query*.

        Parameters
        ----------
        query:
            Financial topic or keywords used for the NewsAPI search.
        sources:
            Optional list of RSS feed URLs.  Defaults to
            ``_DEFAULT_RSS_SOURCES`` when *None*.
        """
        rss_sources = sources or _DEFAULT_RSS_SOURCES
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=_LOOKBACK_HOURS)

        # Build tasks: one per RSS feed + one NewsAPI call
        rss_tasks = [fetch_rss(url) for url in rss_sources]
        newsapi_task = fetch_newsapi(query, api_key=config.NEWSAPI_KEY)

        all_results = await asyncio.gather(*rss_tasks, newsapi_task, return_exceptions=True)

        all_articles: list[NewsArticle] = []
        source_count = 0
        for result in all_results:
            if isinstance(result, BaseException):
                self._logger.warning("A source fetch failed: %s", result)
                continue
            all_articles.extend(result)  # type: ignore[arg-type]
            source_count += 1

        # Deduplicate by URL
        seen: set[str] = set()
        unique: list[NewsArticle] = []
        for article in all_articles:
            if article.url not in seen:
                seen.add(article.url)
                unique.append(article)

        # Filter to last 48 hours (skip articles with no date)
        def _is_recent(a: NewsArticle) -> bool:
            if a.published_at is None:
                return False
            dt = a.published_at
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt >= cutoff

        recent = [a for a in unique if _is_recent(a)]

        # Sort by published_at desc, cap at _MAX_ARTICLES
        recent.sort(
            key=lambda a: (
                a.published_at.replace(tzinfo=timezone.utc)
                if a.published_at.tzinfo is None
                else a.published_at
            ),
            reverse=True,
        )
        final = recent[:_MAX_ARTICLES]

        self._logger.info(
            "Fetched %d articles from %d sources", len(final), source_count
        )
        return final
