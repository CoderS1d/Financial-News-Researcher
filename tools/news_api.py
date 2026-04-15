"""NewsAPI.org client built on httpx."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from config import config
from models.schemas import NewsArticle, NewsSource, RawArticle

logger = logging.getLogger(__name__)

_LANGUAGE = "en"
_DEFAULT_PAGE_SIZE = 20


async def search_news(
    query: str,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    page_size: int = _DEFAULT_PAGE_SIZE,
    sort_by: str = "relevancy",
) -> list[RawArticle]:
    """
    Query NewsAPI /v2/everything for ``query`` and return RawArticles.

    Returns an empty list if NEWSAPI_KEY is not configured.
    """
    if not config.NEWSAPI_KEY:
        logger.warning("NEWSAPI_KEY not set – skipping NewsAPI fetch.")
        return []

    if from_date is None:
        from_date = datetime.now(tz=timezone.utc) - timedelta(days=7)
    if to_date is None:
        to_date = datetime.now(tz=timezone.utc)

    params: dict[str, str | int] = {
        "q": query,
        "language": _LANGUAGE,
        "sortBy": sort_by,
        "pageSize": min(page_size, 100),
        "from": from_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "to": to_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "apiKey": config.NEWSAPI_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
            response = await client.get(
                f"{config.NEWSAPI_BASE_URL}/everything",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        if data.get("status") != "ok":
            logger.error("NewsAPI error: %s", data.get("message", "unknown"))
            return []

        articles: list[RawArticle] = []
        for item in data.get("articles", []):
            url = item.get("url", "")
            title = item.get("title", "")
            if not url or not title or url == "https://removed.com":
                continue

            published_at: datetime | None = None
            raw_date = item.get("publishedAt")
            if raw_date:
                try:
                    published_at = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                except ValueError:
                    pass

            articles.append(
                RawArticle(
                    title=title,
                    url=url,
                    source=NewsSource.NEWSAPI,
                    published_at=published_at,
                    content=item.get("content") or item.get("description") or "",
                    author=item.get("author"),
                    source_name=item.get("source", {}).get("name"),
                )
            )

        logger.info("NewsAPI returned %d articles for query '%s'", len(articles), query)
        return articles

    except httpx.HTTPStatusError as exc:
        logger.error("NewsAPI HTTP error %s: %s", exc.response.status_code, exc)
        return []
    except Exception as exc:
        logger.error("NewsAPI unexpected error: %s", exc)
        return []


async def fetch_newsapi(
    query: str,
    api_key: str,
    page_size: int = 10,
) -> list[NewsArticle]:
    """
    Fetch articles from NewsAPI ``/v2/everything`` for *query*.

    Parameters
    ----------
    query:
        Search terms (passed as ``q``).
    api_key:
        NewsAPI key.  Returns an empty list immediately if blank.
    page_size:
        Number of articles to request (capped at 100 by NewsAPI).

    Results are sorted by ``publishedAt`` descending and filtered to
    English language only.
    """
    if not api_key:
        logger.warning("fetch_newsapi: no api_key provided, skipping.")
        return []

    params: dict[str, str | int] = {
        "q": query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": min(page_size, 100),
        "apiKey": api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
            response = await client.get(
                f"{config.NEWSAPI_BASE_URL}/everything",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        if data.get("status") != "ok":
            logger.error("fetch_newsapi error: %s", data.get("message", "unknown"))
            return []

        articles: list[NewsArticle] = []
        for item in data.get("articles", []):
            url = item.get("url", "")
            title = item.get("title", "")
            if not url or not title or url == "https://removed.com":
                continue

            published_at: datetime = datetime.now(tz=timezone.utc)
            raw_date = item.get("publishedAt")
            if raw_date:
                try:
                    published_at = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                except ValueError:
                    pass

            raw_text = item.get("content") or item.get("description") or ""
            source_name: str = item.get("source", {}).get("name") or "NewsAPI"

            articles.append(
                NewsArticle(
                    title=title,
                    url=url,
                    source=source_name,
                    published_at=published_at,
                    raw_text=raw_text,
                )
            )

        logger.info("fetch_newsapi: %d articles for query '%s'", len(articles), query)
        return articles

    except httpx.HTTPStatusError as exc:
        logger.error("fetch_newsapi HTTP %s: %s", exc.response.status_code, exc)
        return []
    except Exception as exc:
        logger.error("fetch_newsapi unexpected error: %s", exc)
        return []


async def get_top_headlines(
    category: str = "business",
    country: str = "us",
    page_size: int = 20,
) -> list[RawArticle]:
    """Fetch top business headlines from NewsAPI."""
    if not config.NEWSAPI_KEY:
        logger.warning("NEWSAPI_KEY not set – skipping top headlines fetch.")
        return []

    params: dict[str, str | int] = {
        "category": category,
        "country": country,
        "pageSize": min(page_size, 100),
        "apiKey": config.NEWSAPI_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
            response = await client.get(
                f"{config.NEWSAPI_BASE_URL}/top-headlines",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        articles: list[RawArticle] = []
        for item in data.get("articles", []):
            url = item.get("url", "")
            title = item.get("title", "")
            if not url or not title or url == "https://removed.com":
                continue

            published_at = None
            raw_date = item.get("publishedAt")
            if raw_date:
                try:
                    published_at = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                except ValueError:
                    pass

            articles.append(
                RawArticle(
                    title=title,
                    url=url,
                    source=NewsSource.NEWSAPI,
                    published_at=published_at,
                    content=item.get("content") or item.get("description") or "",
                    author=item.get("author"),
                    source_name=item.get("source", {}).get("name"),
                )
            )

        logger.info("NewsAPI top headlines: %d articles", len(articles))
        return articles

    except Exception as exc:
        logger.error("NewsAPI top-headlines error: %s", exc)
        return []
