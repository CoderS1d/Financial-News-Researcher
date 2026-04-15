"""Lightweight async web scraper using httpx.

Extracts readable text from HTML pages without requiring a headless browser.
Uses a simple heuristic: strip script/style tags, then grab paragraph text.
"""
from __future__ import annotations

import asyncio
import logging
import re
from html.parser import HTMLParser

import httpx

from config import config
from models.schemas import NewsSource, RawArticle

logger = logging.getLogger(__name__)

_SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "nav", "footer", "header", "aside", "form"}
)
_BLOCK_TAGS = frozenset({"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "blockquote"})


class _TextExtractor(HTMLParser):
    """Minimal HTML → plain text converter."""

    def __init__(self) -> None:
        super().__init__()
        self._in_skip: int = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_TAGS:
            self._in_skip += 1
        elif tag in _BLOCK_TAGS and self._parts and self._parts[-1] != "\n":
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._in_skip = max(0, self._in_skip - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_skip:
            return
        text = data.strip()
        if text:
            self._parts.append(text + " ")

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse excessive whitespace / blank lines
        raw = re.sub(r" {2,}", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


async def scrape_url(
    url: str,
    client: httpx.AsyncClient,
    title: str = "",
) -> RawArticle | None:
    """
    Scrape a single URL and return a RawArticle.
    Returns None on failure.
    """
    try:
        response = await client.get(url, timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            logger.debug("Skipping non-HTML content at %s (%s)", url, content_type)
            return None

        text = _html_to_text(response.text)
        if not text:
            return None

        return RawArticle(
            title=title or url,
            url=url,
            source=NewsSource.SCRAPED,
            content=text[:8000],  # cap to avoid oversized LLM context
        )
    except httpx.HTTPStatusError as exc:
        logger.warning("HTTP %s scraping %s", exc.response.status_code, url)
        return None
    except Exception as exc:
        logger.warning("Failed to scrape %s: %s", url, exc)
        return None


async def scrape_urls(urls: list[str], concurrency: int = 5) -> list[RawArticle]:
    """
    Scrape multiple URLs concurrently, respecting a concurrency cap.
    Returns successfully scraped articles only.
    """
    semaphore = asyncio.Semaphore(concurrency)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; financial-news-researcher/1.0; "
            "+https://github.com/example/financial-news-researcher)"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }

    async def _bounded(url: str) -> RawArticle | None:
        async with semaphore:
            async with httpx.AsyncClient(
                follow_redirects=True,
                headers=headers,
                timeout=config.HTTP_TIMEOUT,
            ) as client:
                return await scrape_url(url, client)

    results = await asyncio.gather(*[_bounded(url) for url in urls], return_exceptions=False)
    articles = [r for r in results if r is not None]
    logger.info("Scraped %d/%d URLs successfully", len(articles), len(urls))
    return articles


async def enrich_articles(articles: list[RawArticle], min_content_length: int = 200) -> list[RawArticle]:
    """
    For articles that have little content (e.g. RSS summaries), attempt to
    scrape the full article body from their URL.
    """
    to_enrich = [a for a in articles if len(a.content) < min_content_length and a.url]
    if not to_enrich:
        return articles

    logger.info("Enriching %d articles via web scraping", len(to_enrich))
    url_to_scraped: dict[str, RawArticle] = {}

    semaphore = asyncio.Semaphore(5)
    headers = {"User-Agent": "financial-news-researcher/1.0"}

    async def _scrape_one(article: RawArticle) -> tuple[str, RawArticle | None]:
        async with semaphore:
            async with httpx.AsyncClient(
                follow_redirects=True, headers=headers, timeout=config.HTTP_TIMEOUT
            ) as client:
                scraped = await scrape_url(article.url, client, title=article.title)
                return article.url, scraped

    pairs = await asyncio.gather(*[_scrape_one(a) for a in to_enrich])
    for url, scraped in pairs:
        if scraped:
            url_to_scraped[url] = scraped

    enriched: list[RawArticle] = []
    for article in articles:
        if article.url in url_to_scraped:
            scraped = url_to_scraped[article.url]
            # Merge: keep metadata from original, use scraped content
            enriched.append(
                article.model_copy(update={"content": scraped.content})
            )
        else:
            enriched.append(article)

    return enriched
