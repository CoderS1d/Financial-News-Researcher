"""ExtractorAgent – extracts structured financial events from NewsArticles."""
from __future__ import annotations

import asyncio
import logging
from itertools import islice
from uuid import UUID

from agents.base_agent import BaseAgent
from models.schemas import (
    Entity,
    ExtractedArticle,
    ExtractedEvent,
    KeyFact,
    NewsArticle,
    RawArticle,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a financial event extraction specialist. Given a news article, identify
every meaningful financial event it describes and extract structured data for
each one using the `extract_financial_event` tool.

Rules:
- Call the tool once per distinct event found in the article.
- If no financial event is present, do not call the tool.
- event_type must be one of: earnings, merger, rate_decision, regulation, macro, other.
- magnitude must be one of: low, medium, high.
- entities: named companies, people, indices, or organisations mentioned.
- tickers: stock/crypto ticker symbols (uppercase, e.g. AAPL, BTC).
"""

_EXTRACT_TOOL: dict = {
    "name": "extract_financial_event",
    "description": (
        "Extract a single structured financial event from the article. "
        "Call this tool once for each distinct event."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "event_type": {
                "type": "string",
                "enum": ["earnings", "merger", "rate_decision", "regulation", "macro", "other"],
                "description": "Category of the financial event.",
            },
            "description": {
                "type": "string",
                "description": "One or two sentence description of the event.",
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Named entities involved (companies, people, indices).",
            },
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Relevant stock or crypto ticker symbols in uppercase.",
            },
            "magnitude": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "How market-moving is this event.",
            },
        },
        "required": ["event_type", "description", "magnitude"],
    },
}

_BATCH_SIZE = 5


class ExtractorAgent(BaseAgent):
    """
    Extracts structured ``ExtractedEvent`` objects from ``NewsArticle`` items
    using Claude's tool_use feature.

    Articles are processed in batches of ``_BATCH_SIZE`` concurrently.
    Uses a local Ollama model for cost-free high-volume extraction.
    """

    def __init__(self) -> None:
        super().__init__(model="qwen2.5:7b", max_tokens=1024)

    async def run(  # type: ignore[override]
        self,
        articles: list[NewsArticle] = None,
        **kwargs,
    ) -> list[ExtractedEvent]:
        """
        Extract financial events from all *articles*.

        Articles are processed ``_BATCH_SIZE`` at a time with
        ``asyncio.gather`` for concurrency.  Returns a flat, deduplicated
        list of ``ExtractedEvent`` objects sorted by article order.
        """
        if not articles:
            return []

        # Split into batches of _BATCH_SIZE
        batches = list(_batched(articles, _BATCH_SIZE))
        self._logger.info(
            "ExtractorAgent processing %d articles in %d batches",
            len(articles),
            len(batches),
        )

        batch_results = await asyncio.gather(
            *[self._process_batch(batch) for batch in batches],
            return_exceptions=False,
        )

        # Flatten and deduplicate by (article_id, description)
        seen: set[tuple] = set()
        events: list[ExtractedEvent] = []
        for batch_events in batch_results:
            for event in batch_events:
                key = (event.article_id, event.description)
                if key not in seen:
                    seen.add(key)
                    events.append(event)

        self._logger.info("ExtractorAgent produced %d events total", len(events))
        return events

    async def _process_batch(
        self, articles: list[NewsArticle]
    ) -> list[ExtractedEvent]:
        """Process a single batch concurrently."""
        results = await asyncio.gather(
            *[self._extract_article(a) for a in articles],
            return_exceptions=False,
        )
        return [event for article_events in results for event in article_events]

    async def _extract_article(self, article: NewsArticle) -> list[ExtractedEvent]:
        """Call Claude with tool_use for a single article; parse tool_use blocks."""
        body = (article.raw_text or "")[:3000]
        user_message = (
            f"Title: {article.title}\n"
            f"Source: {article.source}\n"
            f"Published: {article.published_at.isoformat() if article.published_at else 'unknown'}\n\n"
            f"{body}"
        )

        try:
            content_blocks = await self._call_llm_raw(
                _SYSTEM_PROMPT,
                [{"role": "user", "content": user_message}],
                tools=[_EXTRACT_TOOL],
            )
        except Exception as exc:
            self._logger.warning(
                "LLM call failed for article '%s': %s", article.title[:60], exc
            )
            return []

        events: list[ExtractedEvent] = []
        tool_calls_found = 0
        for block in content_blocks:
            if getattr(block, "type", None) != "tool_use":
                continue
            if getattr(block, "name", None) != "extract_financial_event":
                continue
            tool_calls_found += 1
            inp: dict = block.input or {}
            try:
                events.append(
                    ExtractedEvent(
                        article_id=article.id,
                        event_type=inp.get("event_type", "other"),
                        description=inp.get("description", ""),
                        entities=inp.get("entities") or [],
                        tickers=[t.upper() for t in (inp.get("tickers") or [])],
                        magnitude=inp.get("magnitude", "low"),
                    )
                )
            except Exception as exc:
                self._logger.warning(
                    "Failed to parse tool call for '%s': %s", article.title[:60], exc
                )

        if tool_calls_found == 0:
            self._logger.warning(
                "No tool_use blocks returned for article: '%s'", article.title[:60]
            )

        return events

    # ------------------------------------------------------------------
    # Legacy interface used by the orchestrator (RawArticle pipeline)
    # ------------------------------------------------------------------

    async def run_batch(self, articles: list[RawArticle]) -> list[ExtractedArticle]:
        """Legacy extraction path returning ExtractedArticle objects."""
        results: list[ExtractedArticle] = []
        for i, article in enumerate(articles):
            self._logger.info("Extracting (%d/%d): %s", i + 1, len(articles), article.title[:60])
            try:
                results.append(await self._extract_raw(article))
            except Exception as exc:
                self._logger.warning("Extraction failed for '%s': %s", article.title[:60], exc)
                results.append(
                    ExtractedArticle(
                        raw=article,
                        summary=article.content[:300] or article.title,
                        key_facts=[],
                        entities=[],
                        topics=[],
                    )
                )
        return results

    async def _extract_raw(self, article: RawArticle) -> ExtractedArticle:
        """Single-article extraction returning an ExtractedArticle (legacy)."""
        _LEGACY_SYSTEM = (
            "You are a financial news extraction specialist. Extract structured "
            "information as valid JSON with keys: summary, key_facts "
            "(list of {fact, confidence}), entities (list of {name, entity_type, relevance}), "
            "topics (list of strings)."
        )
        content_snippet = article.content[:3000] if article.content else "(no content)"
        user_message = (
            f"Title: {article.title}\n\n"
            f"Source: {article.source_name or article.source.value}\n\n"
            f"Content:\n{content_snippet}"
        )
        text = await self._call_llm(_LEGACY_SYSTEM, [{"role": "user", "content": user_message}])
        data = self._extract_json(text)
        return ExtractedArticle(
            raw=article,
            summary=data.get("summary", ""),
            key_facts=[
                KeyFact(fact=kf["fact"], confidence=float(kf.get("confidence", 1.0)))
                for kf in data.get("key_facts", [])
                if kf.get("fact")
            ],
            entities=[
                Entity(
                    name=e["name"],
                    entity_type=e.get("entity_type", "other"),
                    relevance=float(e.get("relevance", 1.0)),
                )
                for e in data.get("entities", [])
                if e.get("name")
            ],
            topics=[t for t in data.get("topics", []) if t],
        )


def _batched(iterable, n: int):
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk
