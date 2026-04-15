"""RSS feed reader using feedparser + httpx for async fetching."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import httpx

from config import config
from models.schemas import NewsArticle, NewsSource, RawArticle

logger = logging.getLogger(__name__)


def _parse_published(entry: feedparser.FeedParserDict) -> datetime | None:
    """Best-effort datetime parsing from a feedparser entry."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    if hasattr(entry, "published") and entry.published:
        try:
            return parsedate_to_datetime(entry.published)
        except Exception:
            pass
    return None


def _extract_content(entry: feedparser.FeedParserDict) -> str:
    """Pull the best available text content from a feed entry."""
    if hasattr(entry, "content") and entry.content:
        return entry.content[0].get("value", "")
    if hasattr(entry, "summary"):
        return entry.summary
    return ""


async def fetch_rss_feed(
    url: str,
    client: httpx.AsyncClient,
) -> list[RawArticle]:
    """Fetch and parse a single RSS feed, returning a list of RawArticles."""
    try:
        response = await client.get(url, timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()
        feed = feedparser.parse(response.text)
        articles: list[RawArticle] = []
        for entry in feed.entries:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            if not title or not link:
                continue
            articles.append(
                RawArticle(
                    title=title,
                    url=link,
                    source=NewsSource.RSS,
                    published_at=_parse_published(entry),
                    content=_extract_content(entry),
                    author=getattr(entry, "author", None),
                    source_name=feed.feed.get("title"),
                )
            )
        logger.info("Fetched %d articles from %s", len(articles), url)
        return articles
    except Exception as exc:
        logger.warning("Failed to fetch RSS feed %s: %s", url, exc)
        return []


async def fetch_all_rss_feeds(
    feeds: list[str] | None = None,
) -> list[RawArticle]:
    """Fetch all configured RSS feeds concurrently."""
    urls = feeds or config.RSS_FEEDS
    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": "financial-news-researcher/1.0"},
    ) as client:
        results = await asyncio.gather(
            *[fetch_rss_feed(url, client) for url in urls], return_exceptions=False
        )
    articles = [article for batch in results for article in batch]
    logger.info("Total RSS articles fetched: %d", len(articles))
    return articles


async def fetch_rss(url: str) -> list[NewsArticle]:
    """
    Fetch and parse a single RSS feed URL, returning ``NewsArticle`` objects.

    Uses ``httpx.AsyncClient`` for the HTTP request and ``feedparser`` for
    parsing.  Extracts title, link, summary/content, and published date.
    Returns an empty list on any error so callers can always safely unpack.
    """
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": "financial-news-researcher/1.0"},
            timeout=config.HTTP_TIMEOUT,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        feed = feedparser.parse(response.text)
        feed_name: str = feed.feed.get("title") or url
        articles: list[NewsArticle] = []

        for entry in feed.entries:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            if not title or not link:
                continue

            published_at = _parse_published(entry)
            if published_at is None:
                published_at = datetime.now(tz=timezone.utc)

            raw_text = _extract_content(entry).strip()

            articles.append(
                NewsArticle(
                    title=title,
                    url=link,
                    source=feed_name,
                    published_at=published_at,
                    raw_text=raw_text,
                )
            )

        logger.info("fetch_rss: %d articles from %s", len(articles), url)
        return articles

    except Exception as exc:
        logger.warning("fetch_rss failed for %s: %s", url, exc)
        return []
