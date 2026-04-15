from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_serializer, field_validator


class Sentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class NewsSource(str, Enum):
    RSS = "rss"
    NEWSAPI = "newsapi"
    SCRAPED = "scraped"


# ---------------------------------------------------------------------------
# Raw article coming from any source
# ---------------------------------------------------------------------------

class RawArticle(BaseModel):
    title: str
    url: str
    source: NewsSource
    published_at: datetime | None = None
    content: str = ""
    author: str | None = None
    source_name: str | None = None

    @field_validator("title", "content", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip() if v else v


# ---------------------------------------------------------------------------
# Extracted structured facts from an article
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    name: str
    entity_type: str = Field(
        description="e.g. 'company', 'person', 'index', 'currency', 'commodity'"
    )
    relevance: float = Field(ge=0.0, le=1.0, default=1.0)


class KeyFact(BaseModel):
    fact: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class ExtractedArticle(BaseModel):
    raw: RawArticle
    summary: str
    key_facts: list[KeyFact] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Sentiment analysis result
# ---------------------------------------------------------------------------

class SentimentScore(BaseModel):
    article_url: str
    title: str
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    impact_magnitude: float = Field(
        ge=0.0,
        le=1.0,
        description="How significant is this news (0 = trivial, 1 = market-moving)",
    )


class SentimentSummary(BaseModel):
    overall_sentiment: Sentiment
    bullish_count: int
    bearish_count: int
    neutral_count: int
    average_confidence: float
    dominant_topics: list[str] = Field(default_factory=list)
    scores: list[SentimentScore] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Debate agent output
# ---------------------------------------------------------------------------

class Argument(BaseModel):
    point: str
    supporting_evidence: list[str] = Field(default_factory=list)
    strength: float = Field(ge=0.0, le=1.0, description="How compelling is this argument")


class DebateReport(BaseModel):
    topic: str
    bull_case: list[Argument] = Field(default_factory=list)
    bear_case: list[Argument] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    key_opportunities: list[str] = Field(default_factory=list)
    verdict: str = Field(description="Balanced one-paragraph conclusion")


# ---------------------------------------------------------------------------
# Final research report
# ---------------------------------------------------------------------------

class ResearchReport(BaseModel):
    query: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    article_count: int
    executive_summary: str
    sentiment_summary: SentimentSummary
    debate: DebateReport
    top_entities: list[Entity] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    raw_articles: list[RawArticle] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"# Financial Research Report: {self.query}",
            f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}*",
            f"*Articles analysed: {self.article_count}*",
            "",
            "---",
            "",
            "## Executive Summary",
            self.executive_summary,
            "",
            "## Overall Sentiment",
            f"**{self.sentiment_summary.overall_sentiment.value.upper()}**  "
            f"(🟢 {self.sentiment_summary.bullish_count} bullish | "
            f"🔴 {self.sentiment_summary.bearish_count} bearish | "
            f"⚪ {self.sentiment_summary.neutral_count} neutral)",
            "",
            "## Key Findings",
        ]
        for finding in self.key_findings:
            lines.append(f"- {finding}")

        lines += [
            "",
            "## Bull Case",
        ]
        for arg in self.debate.bull_case:
            lines.append(f"- **{arg.point}** *(strength: {arg.strength:.0%})*")
            for ev in arg.supporting_evidence:
                lines.append(f"  - {ev}")

        lines += ["", "## Bear Case"]
        for arg in self.debate.bear_case:
            lines.append(f"- **{arg.point}** *(strength: {arg.strength:.0%})*")
            for ev in arg.supporting_evidence:
                lines.append(f"  - {ev}")

        lines += ["", "## Risk Factors"]
        for risk in self.risk_factors:
            lines.append(f"- {risk}")

        lines += ["", "## Verdict", self.debate.verdict, "", "## Sources"]
        for src in self.sources:
            lines.append(f"- {src}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# New typed models
# ---------------------------------------------------------------------------

class NewsArticle(BaseModel):
    model_config = ConfigDict()

    id: UUID = Field(default_factory=uuid4)
    title: str
    url: str
    source: str
    published_at: datetime
    raw_text: str
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("published_at", "fetched_at")
    def _serialize_dt(self, v: datetime) -> str:
        return v.isoformat()


class ExtractedEvent(BaseModel):
    model_config = ConfigDict()

    article_id: UUID
    event_type: Literal["earnings", "merger", "rate_decision", "regulation", "macro", "other"]
    description: str
    entities: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    magnitude: Literal["low", "medium", "high"]


class SentimentResult(BaseModel):
    model_config = ConfigDict()

    article_id: UUID
    label: Literal["bullish", "bearish", "neutral"]
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_phrases: list[str] = Field(default_factory=list)


class DebateRound(BaseModel):
    model_config = ConfigDict()

    round_number: int
    bull_argument: str
    bear_argument: str
    moderator_note: str


class MarketBrief(BaseModel):
    model_config = ConfigDict()

    query: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("generated_at")
    def _serialize_dt(self, v: datetime) -> str:
        return v.isoformat()
    articles_analyzed: int
    top_events: list[ExtractedEvent] = Field(default_factory=list)
    sentiment_summary: dict[Literal["bullish", "bearish", "neutral"], int] = Field(
        default_factory=lambda: {"bullish": 0, "bearish": 0, "neutral": 0}
    )
    debate_rounds: list[DebateRound] = Field(default_factory=list)
    final_verdict: Literal["bullish", "bearish", "neutral"]
    executive_summary: str
    source_urls: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        verdict_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(
            self.final_verdict, "⚪"
        )
        bullish = self.sentiment_summary.get("bullish", 0)
        bearish = self.sentiment_summary.get("bearish", 0)
        neutral = self.sentiment_summary.get("neutral", 0)

        lines: list[str] = [
            f"# 📰 Market Brief: {self.query}",
            f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}*  ",
            f"*Articles analysed: {self.articles_analyzed}*",
            "",
            "---",
            "",
            "## 📋 Executive Summary",
            self.executive_summary,
            "",
            "## 📊 Sentiment Overview",
            f"🟢 **Bullish:** {bullish}  |  🔴 **Bearish:** {bearish}  |  ⚪ **Neutral:** {neutral}",
            "",
            "## 🗞️ Top Events",
        ]

        for event in self.top_events:
            mag_icon = {"low": "🔵", "medium": "🟡", "high": "🔥"}.get(event.magnitude, "•")
            lines.append(
                f"- {mag_icon} **[{event.event_type.replace('_', ' ').title()}]** {event.description}"
            )
            if event.tickers:
                lines.append(f"  - Tickers: `{'` `'.join(event.tickers)}`")
            if event.entities:
                lines.append(f"  - Entities: {', '.join(event.entities)}")

        lines += ["", "## 🥊 Debate Rounds"]
        for rnd in self.debate_rounds:
            lines += [
                f"### Round {rnd.round_number}",
                f"🟢 **Bull:** {rnd.bull_argument}",
                "",
                f"🔴 **Bear:** {rnd.bear_argument}",
                "",
                f"⚖️ **Moderator:** {rnd.moderator_note}",
                "",
            ]

        lines += [
            "## 🏁 Final Verdict",
            f"{verdict_emoji} **{self.final_verdict.upper()}**",
            "",
            "## 🔗 Sources",
        ]
        for url in self.source_urls:
            lines.append(f"- {url}")

        return "\n".join(lines)

