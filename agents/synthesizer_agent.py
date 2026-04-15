"""SynthesizerAgent – produces the final MarketBrief."""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from agents.base_agent import BaseAgent
from models.schemas import (
    DebateReport,
    DebateRound,
    Entity,
    ExtractedArticle,
    ExtractedEvent,
    MarketBrief,
    NewsArticle,
    ResearchReport,
    SentimentResult,
    SentimentSummary,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior financial journalist writing a concise, high-signal market \
intelligence brief. Write in tight, active prose. No bullet-point padding. \
Lead with the most market-moving fact.

Write exactly 3 paragraphs as the executive_summary (max 250 words total):
  Paragraph 1 – key facts and hard data from the news.
  Paragraph 2 – bull vs bear tension: what optimists and pessimists are arguing.
  Paragraph 3 – forward-looking implication: what market participants should watch.

Respond ONLY with valid JSON:
{"executive_summary": "<3 paragraphs, max 250 words>"}
"""

_MAGNITUDE_ORDER = {"high": 0, "medium": 1, "low": 2}


def _slugify(text: str, max_length: int = 50) -> str:
    slug = re.sub(r"[^\w\s]", "", text.lower())
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug[:max_length].rstrip("_")


class SynthesizerAgent(BaseAgent):
    """Synthesises all pipeline outputs into a final MarketBrief."""

    def __init__(self) -> None:
        super().__init__(model="qwen2.5:7b", max_tokens=600)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    async def run(  # type: ignore[override]
        self,
        query: str = "",
        articles: list[NewsArticle] = None,
        events: list[ExtractedEvent] = None,
        sentiment_results: list[SentimentResult] = None,
        debate_rounds: list[DebateRound] = None,
        verdict: str = "neutral",
        **kwargs,
    ) -> MarketBrief:
        """Synthesise all pipeline outputs into a MarketBrief."""
        self._logger.info("SynthesizerAgent synthesising brief for '%s'", query)

        articles = articles or []
        events = events or []
        sentiment_results = sentiment_results or []
        debate_rounds = debate_rounds or []

        # Top 5 events by magnitude
        top_events = sorted(events, key=lambda e: _MAGNITUDE_ORDER[e.magnitude])[:5]

        # Sentiment distribution
        sentiment_dist: dict[Literal["bullish", "bearish", "neutral"], int] = {
            "bullish": 0, "bearish": 0, "neutral": 0
        }
        for r in sentiment_results:
            sentiment_dist[r.label] = sentiment_dist.get(r.label, 0) + 1  # type: ignore[literal-required]

        user_message = self._build_context(
            query, top_events, sentiment_dist, debate_rounds, verdict
        )

        text = await self._call_llm(
            _SYSTEM_PROMPT,
            [{"role": "user", "content": user_message}],
        )

        executive_summary = self._parse_summary(text)

        source_urls = list(dict.fromkeys(a.url for a in articles if a.url))

        return MarketBrief(
            query=query,
            articles_analyzed=len(articles),
            top_events=top_events,
            sentiment_summary=sentiment_dist,
            debate_rounds=debate_rounds,
            final_verdict=verdict if verdict in ("bullish", "bearish", "neutral") else "neutral",  # type: ignore[arg-type]
            executive_summary=executive_summary,
            source_urls=source_urls,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        query: str,
        top_events: list[ExtractedEvent],
        sentiment_dist: dict,
        debate_rounds: list[DebateRound],
        verdict: str,
    ) -> str:
        parts: list[str] = [f"## Research Query\n{query}\n"]

        parts.append("## Top Events (by market impact)")
        for ev in top_events:
            tickers = ", ".join(ev.tickers) if ev.tickers else "—"
            parts.append(
                f"- [{ev.magnitude.upper()} | {ev.event_type}] {ev.description} "
                f"(tickers: {tickers})"
            )

        bulls = sentiment_dist.get("bullish", 0)
        bears = sentiment_dist.get("bearish", 0)
        neutrals = sentiment_dist.get("neutral", 0)
        total = bulls + bears + neutrals or 1
        parts.append(
            f"\n## Sentiment Distribution\n"
            f"Bullish: {bulls} ({bulls/total:.0%})  "
            f"Bearish: {bears} ({bears/total:.0%})  "
            f"Neutral: {neutrals} ({neutrals/total:.0%})"
        )

        if debate_rounds:
            parts.append("\n## Debate Summary (moderator notes)")
            for rnd in debate_rounds:
                parts.append(f"Round {rnd.round_number}: {rnd.moderator_note}")

        parts.append(f"\n## Final Verdict\n{verdict.upper()}")

        return "\n".join(parts)

    @staticmethod
    def _parse_summary(text: str) -> str:
        """Extract executive_summary from JSON, with regex fallback."""
        try:
            data = json.loads(text)
            return data.get("executive_summary", text)
        except json.JSONDecodeError:
            # Try fenced JSON block
            fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if fence:
                try:
                    data = json.loads(fence.group(1))
                    return data.get("executive_summary", text)
                except json.JSONDecodeError:
                    pass
            # Last resort: grab value of "executive_summary" key directly
            match = re.search(r'"executive_summary"\s*:\s*"(.*?)"(?:\s*[,}])', text, re.DOTALL)
            if match:
                return match.group(1).replace("\\n", "\n")
        return text

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @staticmethod
    def save(brief: MarketBrief, output_dir: str) -> None:
        """
        Persist *brief* to *output_dir* as both JSON and Markdown.

        Files are named ``{query_slug}_{timestamp}.{json|md}``.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = brief.generated_at.strftime("%Y%m%d_%H%M%S")
        slug = _slugify(brief.query)
        stem = f"{slug}_{timestamp}"

        json_path = Path(output_dir) / f"{stem}.json"
        md_path = Path(output_dir) / f"{stem}.md"

        json_path.write_text(brief.model_dump_json(indent=2), encoding="utf-8")
        md_path.write_text(brief.to_markdown(), encoding="utf-8")

        logger.info("MarketBrief saved → %s  |  %s", json_path, md_path)

    # ------------------------------------------------------------------
    # Legacy interface (orchestrator ExtractedArticle pipeline)
    # ------------------------------------------------------------------

    async def run_legacy(
        self,
        query: str = "",
        articles: list[ExtractedArticle] = None,
        sentiment: SentimentSummary = None,
        debate: DebateReport = None,
        **kwargs,
    ) -> ResearchReport:
        """Original synthesis path returning a ResearchReport (legacy orchestrator)."""
        from collections import Counter

        self._logger.info("SynthesizerAgent (legacy) synthesising for '%s'", query)
        articles = articles or []

        entity_counter: Counter[str] = Counter()
        entity_map: dict[str, Entity] = {}
        for article in articles:
            for entity in article.entities:
                entity_counter[entity.name] += 1
                if entity.name not in entity_map:
                    entity_map[entity.name] = entity

        top_entities = [
            entity_map[name]
            for name, _ in entity_counter.most_common(15)
            if name in entity_map
        ]
        sources = list({a.raw.url for a in articles if a.raw.url})

        _LEGACY_SYSTEM = (
            "You are a senior financial research analyst. Respond ONLY with valid JSON: "
            '{"executive_summary": str, "key_findings": [str], "risk_factors": [str]}'
        )
        context = self._build_legacy_context(articles, sentiment, debate)
        user_message = (
            f"Research query: {query}\n\n"
            f"Sentiment: {sentiment.overall_sentiment.value} "
            f"(bullish={sentiment.bullish_count}, bearish={sentiment.bearish_count}, "
            f"neutral={sentiment.neutral_count})\n\n"
            f"Debate verdict: {debate.verdict}\n\nEvidence:\n{context}"
        )

        text = await self._call_llm(
            _LEGACY_SYSTEM,
            [{"role": "user", "content": user_message}],
            max_tokens=2048,
        )
        data = self._extract_json(text)

        return ResearchReport(
            query=query,
            article_count=len(articles),
            executive_summary=data.get("executive_summary", ""),
            sentiment_summary=sentiment,
            debate=debate,
            top_entities=top_entities,
            key_findings=data.get("key_findings", []),
            risk_factors=data.get("risk_factors", []),
            sources=sources,
            raw_articles=[a.raw for a in articles],
        )

    def _build_legacy_context(
        self,
        articles: list[ExtractedArticle],
        sentiment: SentimentSummary,
        debate: DebateReport,
    ) -> str:
        parts: list[str] = ["## Article Summaries"]
        for i, article in enumerate(articles[:15], start=1):
            parts.append(f"{i}. **{article.raw.title}**")
            parts.append(f"   {article.summary[:300]}")
        parts.append("\n## Bull Case Highlights")
        for arg in debate.bull_case[:3]:
            parts.append(f"- {arg.point}")
        parts.append("\n## Bear Case Highlights")
        for arg in debate.bear_case[:3]:
            parts.append(f"- {arg.point}")
        return "\n".join(parts)

