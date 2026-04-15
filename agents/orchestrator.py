"""Orchestrator – coordinates all agents in sequence."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from agents.base_agent import BaseAgent
from agents.debate_agent import DebateAgent
from agents.extractor_agent import ExtractorAgent
from agents.fetcher_agent import FetcherAgent
from agents.sentiment_agent import SentimentAgent
from agents.synthesizer_agent import SynthesizerAgent
from config import config
from models.schemas import (
    MarketBrief,
    NewsArticle,
    NewsSource,
    RawArticle,
    ResearchReport,
)

logger = logging.getLogger(__name__)


def _to_raw_articles(articles: list[NewsArticle]) -> list[RawArticle]:
    """Convert NewsArticle objects to RawArticle for the legacy extraction pipeline."""
    return [
        RawArticle(
            title=a.title,
            url=a.url,
            source=NewsSource.RSS,
            published_at=a.published_at,
            content=a.raw_text or "",
            source_name=a.source,
        )
        for a in articles
    ]


# ---------------------------------------------------------------------------
# New typed orchestrator (BaseAgent subclass)
# ---------------------------------------------------------------------------


class OrchestratorAgent(BaseAgent):
    """
    Orchestrates the full multi-agent research pipeline and returns a MarketBrief.

    Pipeline:
      FetcherAgent
        → ExtractorAgent ‖ SentimentAgent (preliminary, no events)
        → SentimentAgent (with events)
        → DebateAgent → get_verdict
        → SynthesizerAgent → save
    """

    def __init__(self) -> None:
        # Model / tokens / temperature are not used by the orchestrator itself;
        # each sub-agent manages its own configuration.
        super().__init__(model="qwen2.5:7b", max_tokens=256)
        self._fetcher = FetcherAgent()
        self._extractor = ExtractorAgent()
        self._sentiment = SentimentAgent()
        self._debate = DebateAgent()
        self._synthesizer = SynthesizerAgent()

    async def run(  # type: ignore[override]
        self,
        query: str = "",
        debate_rounds: int = 3,
        **kwargs: Any,
    ) -> MarketBrief:
        """Run the full pipeline for *query* and return a ``MarketBrief``."""
        t0 = time.perf_counter()
        logger.info("Starting research pipeline for: %s", query)

        # ── Step 1: Fetch ────────────────────────────────────────────────────
        logger.info("[1/6] Fetching articles…")
        articles: list[NewsArticle] = await self._fetcher.run(query=query)
        logger.info("[1/6] Fetched %d articles", len(articles))

        if not articles:
            logger.warning("No articles fetched — check RSS feeds and NEWSAPI_KEY. Aborting pipeline.")
            from models.schemas import MarketBrief
            return MarketBrief(
                query=query,
                articles_analyzed=0,
                final_verdict="neutral",
                executive_summary="No articles could be fetched for this query. Please check your RSS feed configuration and NEWSAPI_KEY.",
            )

        # ── Step 2: Extract + preliminary sentiment concurrently ─────────────
        logger.info("[2/6] Extracting events and preliminary sentiment concurrently…")
        events_result, prelim_sentiment = await asyncio.gather(
            self._extractor.run(articles),
            self._sentiment.run(articles=articles, events=[]),
        )
        logger.info(
            "[2/6] Extracted %d events, %d preliminary sentiment results",
            len(events_result),
            len(prelim_sentiment),
        )

        # ── Step 3: Re-run sentiment with events ─────────────────────────────
        logger.info("[3/6] Re-scoring sentiment with extracted events…")
        sentiment_results = await self._sentiment.run(
            articles=articles, events=events_result
        )
        logger.info("[3/6] Final sentiment: %d results", len(sentiment_results))

        # ── Step 4: Debate ───────────────────────────────────────────────────
        logger.info("[4/6] Running %d-round debate…", debate_rounds)
        debate_rounds_result = await self._debate.run(
            events=events_result,
            sentiment_results=sentiment_results,
            rounds=debate_rounds,
        )
        verdict = await self._debate.get_verdict(debate_rounds_result)
        logger.info("[4/6] Debate complete – verdict: %s", verdict)

        # ── Step 5: Synthesise ───────────────────────────────────────────────
        logger.info("[5/6] Synthesising MarketBrief…")
        brief = await self._synthesizer.run(
            query=query,
            articles=articles,
            events=events_result,
            sentiment_results=sentiment_results,
            debate_rounds=debate_rounds_result,
            verdict=verdict,
        )

        # ── Step 6: Save ─────────────────────────────────────────────────────
        logger.info("[6/6] Saving outputs to %s", config.OUTPUT_DIR)
        SynthesizerAgent.save(brief, str(config.OUTPUT_DIR))

        elapsed = time.perf_counter() - t0
        logger.info(
            "Pipeline complete in %.1fs | articles=%d events=%d sentiment=%d rounds=%d verdict=%s",
            elapsed,
            len(articles),
            len(events_result),
            len(sentiment_results),
            len(debate_rounds_result),
            verdict,
        )
        return brief


# ---------------------------------------------------------------------------
# Legacy orchestrator (kept for backward compatibility)
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    max_articles: int = 30
    enrich_articles: bool = True
    rss_feeds: list[str] | None = None
    verbose: bool = False


class Orchestrator:
    """
    Legacy orchestrator returning ResearchReport.
    Kept for backward compatibility; new code should use OrchestratorAgent.
    """

    def __init__(self, cfg: OrchestratorConfig | None = None) -> None:
        self._cfg = cfg or OrchestratorConfig()
        self._fetcher = FetcherAgent()
        self._extractor = ExtractorAgent()
        self._sentiment = SentimentAgent()
        self._debate = DebateAgent()
        self._synthesizer = SynthesizerAgent()

    def run(self, query: str) -> ResearchReport:
        """Synchronous entry point – runs the async pipeline on a new event loop."""
        return asyncio.run(self._run_async(query))

    async def _run_async(self, query: str) -> ResearchReport:
        t0 = time.perf_counter()
        logger.info("=== Orchestrator (legacy) starting for query: '%s' ===", query)

        news_articles = await self._fetcher.run(query=query, sources=self._cfg.rss_feeds)
        if not news_articles:
            raise RuntimeError("No articles fetched – check API keys and network connectivity.")
        articles_raw = _to_raw_articles(news_articles)

        extracted = await self._extractor.run_batch(articles_raw)
        sentiment = await self._sentiment.run_legacy(articles=extracted)
        debate = await self._debate.run_legacy(query=query, articles=extracted, sentiment=sentiment)
        report = await self._synthesizer.run_legacy(
            query=query, articles=extracted, sentiment=sentiment, debate=debate
        )

        elapsed = time.perf_counter() - t0
        logger.info("=== Orchestrator (legacy) done in %.1fs ===", elapsed)
        return report

    def _log_step(self, message: str) -> None:
        logger.info(message)
        if self._cfg.verbose:
            print(f"  {message}")
