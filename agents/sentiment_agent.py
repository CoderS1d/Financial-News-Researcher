"""SentimentAgent – classifies financial sentiment per NewsArticle using ExtractedEvents."""
from __future__ import annotations

import asyncio
import json
import logging
import re

from agents.base_agent import BaseAgent
from models.schemas import (
    ExtractedArticle,
    ExtractedEvent,
    NewsArticle,
    Sentiment,
    SentimentResult,
    SentimentScore,
    SentimentSummary,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a quantitative financial analyst specializing in market sentiment. Analyze \
news articles and extracted events to determine market sentiment. Be precise, \
evidence-based, and cite specific phrases.

Respond ONLY with a valid JSON object matching this schema exactly:
{
  "label": "bullish" | "bearish" | "neutral",
  "score": <float 0.0–1.0, confidence in the label>,
  "reasoning": "<one or two sentences explaining the sentiment, citing specific evidence>",
  "key_phrases": ["<exact phrase from the article>", ...]
}
"""


class SentimentAgent(BaseAgent):
    """Analyses the financial sentiment of NewsArticles using associated ExtractedEvents."""

    def __init__(self) -> None:
        super().__init__(model="qwen2.5:7b", max_tokens=512, temperature=0.1)

    async def run(  # type: ignore[override]
        self,
        articles: list[NewsArticle] = None,
        events: list[ExtractedEvent] = None,
        **kwargs,
    ) -> list[SentimentResult]:
        """Score all articles concurrently and return a SentimentResult per article."""
        if not articles:
            return []
        events = events or []

        # Build a lookup: article_id -> list[ExtractedEvent]
        events_by_article: dict = {}
        for ev in events:
            events_by_article.setdefault(ev.article_id, []).append(ev)

        return list(
            await asyncio.gather(
                *(self._score_article(a, events_by_article.get(a.id, [])) for a in articles)
            )
        )

    # ------------------------------------------------------------------

    async def _score_article(
        self, article: NewsArticle, related_events: list[ExtractedEvent]
    ) -> SentimentResult:
        body = (article.raw_text or "")[:3000]
        events_section = ""
        if related_events:
            lines = []
            for ev in related_events:
                entities = ", ".join(ev.entities) if ev.entities else "—"
                tickers = ", ".join(ev.tickers) if ev.tickers else "—"
                lines.append(
                    f"  - [{ev.event_type.upper()}] {ev.description} "
                    f"(entities: {entities}; tickers: {tickers}; magnitude: {ev.magnitude})"
                )
            events_section = "\n\nExtracted events:\n" + "\n".join(lines)

        user_message = (
            f"Title: {article.title}\n"
            f"Source: {article.source}\n"
            f"Published: {article.published_at.isoformat() if article.published_at else 'unknown'}\n\n"
            f"{body}"
            f"{events_section}"
        )

        try:
            text = await self._call_llm(
                _SYSTEM_PROMPT, [{"role": "user", "content": user_message}]
            )
            data = self._parse_json(text)
            return SentimentResult(
                article_id=article.id,
                label=data.get("label", "neutral"),
                score=float(data.get("score", 0.5)),
                reasoning=data.get("reasoning", ""),
                key_phrases=data.get("key_phrases") or [],
            )
        except Exception as exc:
            self._logger.warning(
                "Sentiment scoring failed for '%s': %s", article.title[:60], exc
            )
            return SentimentResult(
                article_id=article.id,
                label="neutral",
                score=0.0,
                reasoning="Extraction failed.",
                key_phrases=[],
            )

    @staticmethod
    def _parse_json(text: str) -> dict:
        """json.loads with a regex fallback to extract the first JSON object."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {}

    @staticmethod
    def summarize(results: list[SentimentResult]) -> dict:
        """Return label counts and the weighted-average score across all results."""
        counts: dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}
        total_score = 0.0
        for r in results:
            counts[r.label] = counts.get(r.label, 0) + 1
            total_score += r.score
        avg_score = round(total_score / len(results), 4) if results else 0.0
        return {**counts, "average_score": avg_score, "total": len(results)}

    # ------------------------------------------------------------------
    # Legacy interface used by the orchestrator (ExtractedArticle pipeline)
    # ------------------------------------------------------------------

    async def run_legacy(
        self, articles: list[ExtractedArticle] = None, **kwargs
    ) -> SentimentSummary:
        """Original aggregation path returning a SentimentSummary (legacy orchestrator)."""
        scores = await asyncio.gather(*[self._score_article_legacy(a) for a in (articles or [])])
        return self._aggregate(list(scores))

    async def _score_article_legacy(self, article: ExtractedArticle) -> SentimentScore:
        _LEGACY_SYSTEM = (
            "You are a seasoned financial analyst specialising in market sentiment analysis. "
            "Respond ONLY with valid JSON: "
            '{"sentiment": "bullish"|"bearish"|"neutral", "confidence": float, '
            '"reasoning": str, "impact_magnitude": float}'
        )
        facts_text = "\n".join(f"- {kf.fact}" for kf in article.key_facts[:10])
        user_message = (
            f"Article title: {article.raw.title}\n\n"
            f"Summary:\n{article.summary}\n\n"
            f"Key facts:\n{facts_text or '(none extracted)'}"
        )
        try:
            text = await self._call_llm(_LEGACY_SYSTEM, [{"role": "user", "content": user_message}])
            data = self._extract_json(text)
            sentiment_str = data.get("sentiment", "neutral").lower()
            sentiment = (
                Sentiment(sentiment_str)
                if sentiment_str in Sentiment._value2member_map_
                else Sentiment.NEUTRAL
            )
            return SentimentScore(
                article_url=article.raw.url,
                title=article.raw.title,
                sentiment=sentiment,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                impact_magnitude=float(data.get("impact_magnitude", 0.5)),
            )
        except Exception as exc:
            self._logger.warning(
                "Legacy sentiment scoring failed for '%s': %s", article.raw.title[:60], exc
            )
            return SentimentScore(
                article_url=article.raw.url,
                title=article.raw.title,
                sentiment=Sentiment.NEUTRAL,
                confidence=0.0,
                reasoning="Extraction failed.",
                impact_magnitude=0.0,
            )

    def _aggregate(self, scores: list[SentimentScore]) -> SentimentSummary:
        from collections import Counter

        if not scores:
            return SentimentSummary(
                overall_sentiment=Sentiment.NEUTRAL,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                average_confidence=0.0,
                scores=[],
            )

        counts = Counter(s.sentiment for s in scores)
        avg_confidence = sum(s.confidence for s in scores) / len(scores)

        weighted: dict[Sentiment, float] = {s: 0.0 for s in Sentiment}
        for score in scores:
            weighted[score.sentiment] += score.impact_magnitude * score.confidence
        overall = max(weighted, key=lambda s: weighted[s])

        sorted_scores = sorted(scores, key=lambda s: s.impact_magnitude, reverse=True)
        dominant_topics: list[str] = []
        for s in sorted_scores[: max(1, len(sorted_scores) // 2)]:
            words = [w for w in s.title.split() if len(w) > 4 and w[0].isupper()]
            dominant_topics.extend(words[:2])
        seen: set[str] = set()
        deduped_topics: list[str] = []
        for t in dominant_topics:
            if t not in seen:
                seen.add(t)
                deduped_topics.append(t)

        return SentimentSummary(
            overall_sentiment=overall,
            bullish_count=counts.get(Sentiment.BULLISH, 0),
            bearish_count=counts.get(Sentiment.BEARISH, 0),
            neutral_count=counts.get(Sentiment.NEUTRAL, 0),
            average_confidence=round(avg_confidence, 3),
            dominant_topics=deduped_topics[:10],
            scores=scores,
        )

