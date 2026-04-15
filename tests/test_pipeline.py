"""Tests for the research pipeline and individual agents.

Run with:  pytest tests/ -v
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.schemas import (
    Argument,
    DebateReport,
    Entity,
    ExtractedArticle,
    KeyFact,
    NewsArticle,
    NewsSource,
    RawArticle,
    ResearchReport,
    Sentiment,
    SentimentScore,
    SentimentSummary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_raw_article() -> RawArticle:
    return RawArticle(
        title="Fed Raises Rates by 25bps",
        url="https://example.com/article/1",
        source=NewsSource.RSS,
        published_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        content="The Federal Reserve raised interest rates by 25 basis points today.",
        author="Jane Doe",
        source_name="Financial Times",
    )


@pytest.fixture
def sample_extracted(sample_raw_article: RawArticle) -> ExtractedArticle:
    return ExtractedArticle(
        raw=sample_raw_article,
        summary="The Fed raised rates by 25bps, signalling continued hawkish policy.",
        key_facts=[
            KeyFact(fact="Rate increase: 25 basis points", confidence=0.99),
            KeyFact(fact="Decision unanimous among FOMC members", confidence=0.85),
        ],
        entities=[
            Entity(name="Federal Reserve", entity_type="company", relevance=1.0),
            Entity(name="Jerome Powell", entity_type="person", relevance=0.9),
        ],
        topics=["interest rates", "monetary policy", "inflation"],
    )


@pytest.fixture
def sample_sentiment_score(sample_raw_article: RawArticle) -> SentimentScore:
    return SentimentScore(
        article_url=sample_raw_article.url,
        title=sample_raw_article.title,
        sentiment=Sentiment.BEARISH,
        confidence=0.82,
        reasoning="Rate hikes increase borrowing costs, pressuring equities.",
        impact_magnitude=0.75,
    )


@pytest.fixture
def sample_sentiment_summary(sample_sentiment_score: SentimentScore) -> SentimentSummary:
    return SentimentSummary(
        overall_sentiment=Sentiment.BEARISH,
        bullish_count=1,
        bearish_count=3,
        neutral_count=1,
        average_confidence=0.78,
        dominant_topics=["Federal", "Reserve", "Rates"],
        scores=[sample_sentiment_score],
    )


@pytest.fixture
def sample_debate() -> DebateReport:
    return DebateReport(
        topic="Federal Reserve interest rate policy",
        bull_case=[
            Argument(
                point="Rate hikes signal confidence in the economy.",
                supporting_evidence=["Unemployment remains at 3.7%."],
                strength=0.65,
            )
        ],
        bear_case=[
            Argument(
                point="Higher rates compress corporate margins.",
                supporting_evidence=["Loan costs rise across sectors."],
                strength=0.80,
            )
        ],
        key_risks=["Recession risk increases with each hike."],
        key_opportunities=["Fixed-income assets become more attractive."],
        verdict="The rate environment remains challenging for equities while benefiting savers.",
    )


# ---------------------------------------------------------------------------
# Schema / model tests
# ---------------------------------------------------------------------------

class TestRawArticle:
    def test_strips_whitespace(self) -> None:
        article = RawArticle(
            title="  Fed News  ",
            url="https://example.com",
            source=NewsSource.NEWSAPI,
            content="  Some content  ",
        )
        assert article.title == "Fed News"
        assert article.content == "Some content"

    def test_optional_fields_default_to_none(self) -> None:
        article = RawArticle(
            title="Test",
            url="https://example.com",
            source=NewsSource.RSS,
        )
        assert article.published_at is None
        assert article.author is None
        assert article.source_name is None


class TestSentimentSummary:
    def test_counts(self, sample_sentiment_summary: SentimentSummary) -> None:
        assert sample_sentiment_summary.bullish_count == 1
        assert sample_sentiment_summary.bearish_count == 3
        assert sample_sentiment_summary.neutral_count == 1

    def test_overall_sentiment(self, sample_sentiment_summary: SentimentSummary) -> None:
        assert sample_sentiment_summary.overall_sentiment == Sentiment.BEARISH


class TestResearchReport:
    def test_to_markdown_contains_query(
        self,
        sample_extracted: ExtractedArticle,
        sample_sentiment_summary: SentimentSummary,
        sample_debate: DebateReport,
    ) -> None:
        report = ResearchReport(
            query="Federal Reserve interest rate policy",
            article_count=1,
            executive_summary="The Fed raised rates.",
            sentiment_summary=sample_sentiment_summary,
            debate=sample_debate,
            key_findings=["25bps rate hike implemented"],
            risk_factors=["Recession risk"],
            sources=["https://example.com/article/1"],
        )
        md = report.to_markdown()
        assert "Federal Reserve interest rate policy" in md
        assert "Executive Summary" in md
        assert "Bull Case" in md
        assert "Bear Case" in md

    def test_model_dump_json_roundtrip(
        self,
        sample_sentiment_summary: SentimentSummary,
        sample_debate: DebateReport,
    ) -> None:
        report = ResearchReport(
            query="Test query",
            article_count=5,
            executive_summary="Summary.",
            sentiment_summary=sample_sentiment_summary,
            debate=sample_debate,
        )
        dumped = report.model_dump_json()
        parsed = json.loads(dumped)
        assert parsed["query"] == "Test query"
        assert parsed["article_count"] == 5


# ---------------------------------------------------------------------------
# ExtractorAgent unit test (mocked LLM)
# ---------------------------------------------------------------------------

class TestExtractorAgent:
    def test_extract_parses_llm_response(self, sample_raw_article: RawArticle) -> None:
        from agents.extractor_agent import ExtractorAgent

        # Build a fake ToolUseBlock-like object the way the Anthropic SDK returns it.
        fake_block = MagicMock()
        fake_block.type = "tool_use"
        fake_block.name = "extract_financial_event"
        fake_block.input = {
            "event_type": "rate_decision",
            "description": "The Fed raised rates by 25bps.",
            "entities": ["Federal Reserve"],
            "tickers": ["SPY"],
            "magnitude": "high",
        }

        news_article = NewsArticle(
            title=sample_raw_article.title,
            url=sample_raw_article.url,
            source="Financial Times",
            published_at=sample_raw_article.published_at,
            raw_text=sample_raw_article.content,
        )

        agent = ExtractorAgent()
        with patch.object(agent, "_call_llm_raw", new=AsyncMock(return_value=[fake_block])):
            results = asyncio.run(agent.run([news_article]))

        assert len(results) == 1
        event = results[0]
        assert event.article_id == news_article.id
        assert event.event_type == "rate_decision"
        assert event.description == "The Fed raised rates by 25bps."
        assert "Federal Reserve" in event.entities
        assert "SPY" in event.tickers
        assert event.magnitude == "high"


# ---------------------------------------------------------------------------
# SentimentAgent unit test (mocked LLM)
# ---------------------------------------------------------------------------

class TestSentimentAgent:
    def test_run_returns_results(self, sample_raw_article: RawArticle) -> None:
        from agents.sentiment_agent import SentimentAgent

        mock_response = json.dumps({
            "label": "bearish",
            "score": 0.85,
            "reasoning": "Rate hikes pressure equities.",
            "key_phrases": ["raised interest rates", "25 basis points"],
        })

        news_article = NewsArticle(
            title=sample_raw_article.title,
            url=sample_raw_article.url,
            source="Financial Times",
            published_at=sample_raw_article.published_at,
            raw_text=sample_raw_article.content,
        )

        agent = SentimentAgent()
        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            results = asyncio.run(agent.run(articles=[news_article], events=[]))

        assert len(results) == 1
        result = results[0]
        assert result.article_id == news_article.id
        assert result.label == "bearish"
        assert result.score == 0.85
        assert "raised interest rates" in result.key_phrases

    def test_summarize(self) -> None:
        from agents.sentiment_agent import SentimentAgent
        from models.schemas import SentimentResult
        from uuid import uuid4

        results = [
            SentimentResult(article_id=uuid4(), label="bearish", score=0.9, reasoning="r1", key_phrases=[]),
            SentimentResult(article_id=uuid4(), label="bullish", score=0.7, reasoning="r2", key_phrases=[]),
            SentimentResult(article_id=uuid4(), label="bearish", score=0.8, reasoning="r3", key_phrases=[]),
        ]
        summary = SentimentAgent.summarize(results)
        assert summary["bearish"] == 2
        assert summary["bullish"] == 1
        assert summary["neutral"] == 0
        assert summary["total"] == 3
        assert abs(summary["average_score"] - round((0.9 + 0.7 + 0.8) / 3, 4)) < 1e-6


# ---------------------------------------------------------------------------
# DebateAgent unit tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestDebateAgent:
    def test_run_returns_debate_rounds(self) -> None:
        from agents.debate_agent import DebateAgent
        from models.schemas import ExtractedEvent, SentimentResult
        from uuid import uuid4

        art_id = uuid4()
        events = [
            ExtractedEvent(
                article_id=art_id,
                event_type="rate_decision",
                description="Fed raised rates by 25bps.",
                entities=["Federal Reserve"],
                tickers=["SPY"],
                magnitude="high",
            )
        ]
        results = [
            SentimentResult(
                article_id=art_id,
                label="bearish",
                score=0.8,
                reasoning="Rate hikes are negative for equities.",
                key_phrases=["raised rates"],
            )
        ]

        agent = DebateAgent()
        # Each round makes 3 LLM calls; side_effect cycles through them
        call_responses = ["Bull argument text.", "Bear rebuttal text.", "Moderator note text."]
        with patch.object(agent, "_call_llm", new=AsyncMock(side_effect=call_responses * 2)):
            rounds = asyncio.run(agent.run(events=events, sentiment_results=results, rounds=1))

        assert len(rounds) == 1
        r = rounds[0]
        assert r.round_number == 1
        assert r.bull_argument == "Bull argument text."
        assert r.bear_argument == "Bear rebuttal text."
        assert r.moderator_note == "Moderator note text."

    def test_get_verdict_parses_json(self) -> None:
        from agents.debate_agent import DebateAgent
        from models.schemas import DebateRound

        rounds = [
            DebateRound(
                round_number=1,
                bull_argument="Markets will rally.",
                bear_argument="Recession is coming.",
                moderator_note="Bear argument was more evidence-based.",
            )
        ]

        agent = DebateAgent()
        verdict_json = '{"verdict": "bearish", "reasoning": "Bear side cited more concrete data."}'
        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=verdict_json)):
            verdict = asyncio.run(agent.get_verdict(rounds))

        assert verdict == "bearish"

    def test_get_verdict_empty_rounds(self) -> None:
        from agents.debate_agent import DebateAgent

        agent = DebateAgent()
        verdict = asyncio.run(agent.get_verdict([]))
        assert verdict == "neutral"


# ---------------------------------------------------------------------------
# SynthesizerAgent unit tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestSynthesizerAgent:
    def test_run_returns_market_brief(self, sample_raw_article: RawArticle) -> None:
        import tempfile
        from agents.synthesizer_agent import SynthesizerAgent
        from models.schemas import DebateRound, ExtractedEvent, SentimentResult
        from uuid import uuid4

        art_id = uuid4()
        news_article = NewsArticle(
            title=sample_raw_article.title,
            url=sample_raw_article.url,
            source="Financial Times",
            published_at=sample_raw_article.published_at,
            raw_text=sample_raw_article.content,
        )
        event = ExtractedEvent(
            article_id=art_id,
            event_type="rate_decision",
            description="Fed raised rates by 25bps.",
            entities=["Federal Reserve"],
            tickers=["SPY"],
            magnitude="high",
        )
        sentiment_result = SentimentResult(
            article_id=art_id,
            label="bearish",
            score=0.8,
            reasoning="Rate hikes pressure equities.",
            key_phrases=["raised rates"],
        )
        debate_round = DebateRound(
            round_number=1,
            bull_argument="Economy remains resilient.",
            bear_argument="Higher rates will slow growth.",
            moderator_note="Bear argument was more evidence-based.",
        )

        mock_response = json.dumps({
            "executive_summary": (
                "The Federal Reserve raised rates by 25bps, signalling continued hawkish policy. "
                "Bulls argue the economy remains resilient while bears point to slowing growth. "
                "Markets should watch the next FOMC meeting closely."
            )
        })

        agent = SynthesizerAgent()
        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            brief = asyncio.run(agent.run(
                query="Federal Reserve rate policy",
                articles=[news_article],
                events=[event],
                sentiment_results=[sentiment_result],
                debate_rounds=[debate_round],
                verdict="bearish",
            ))

        assert brief.query == "Federal Reserve rate policy"
        assert brief.articles_analyzed == 1
        assert brief.final_verdict == "bearish"
        assert brief.sentiment_summary["bearish"] == 1
        assert brief.sentiment_summary["bullish"] == 0
        assert len(brief.top_events) == 1
        assert brief.top_events[0].event_type == "rate_decision"
        assert news_article.url in brief.source_urls
        assert "Federal Reserve" in brief.executive_summary

    def test_save_creates_files(self, tmp_path) -> None:
        from agents.synthesizer_agent import SynthesizerAgent
        from models.schemas import MarketBrief
        from datetime import datetime, timezone

        brief = MarketBrief(
            query="Test query for saving",
            generated_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            articles_analyzed=3,
            final_verdict="neutral",
            executive_summary="Para one. Para two. Para three.",
        )

        SynthesizerAgent.save(brief, str(tmp_path))

        files = list(tmp_path.iterdir())
        assert len(files) == 2
        extensions = {f.suffix for f in files}
        assert ".json" in extensions
        assert ".md" in extensions

        json_file = next(f for f in files if f.suffix == ".json")
        data = json.loads(json_file.read_text())
        assert data["query"] == "Test query for saving"
        assert data["final_verdict"] == "neutral"

        md_file = next(f for f in files if f.suffix == ".md")
        md_text = md_file.read_text(encoding="utf-8")
        assert "Test query for saving" in md_text
        assert "NEUTRAL" in md_text


# ---------------------------------------------------------------------------
# OrchestratorAgent + ResearchPipeline tests (fully mocked)
# ---------------------------------------------------------------------------

class TestOrchestratorAgent:
    def test_run_returns_market_brief(self, sample_raw_article: RawArticle) -> None:
        import json as _json
        from datetime import datetime, timezone
        from unittest.mock import MagicMock
        from uuid import uuid4
        from agents.orchestrator import OrchestratorAgent
        from models.schemas import (
            DebateRound, ExtractedEvent, MarketBrief, NewsArticle, SentimentResult
        )

        art_id = uuid4()
        fake_article = NewsArticle(
            title=sample_raw_article.title,
            url=sample_raw_article.url,
            source="Reuters",
            published_at=sample_raw_article.published_at,
            raw_text=sample_raw_article.content,
        )
        fake_event = ExtractedEvent(
            article_id=art_id,
            event_type="rate_decision",
            description="Fed raised rates.",
            entities=["Federal Reserve"],
            tickers=["SPY"],
            magnitude="high",
        )
        fake_sentiment = SentimentResult(
            article_id=art_id, label="bearish", score=0.8,
            reasoning="Rates up.", key_phrases=["raised rates"],
        )
        fake_round = DebateRound(
            round_number=1,
            bull_argument="Economy is strong.",
            bear_argument="Rates will slow growth.",
            moderator_note="Bear more evidence-based.",
        )
        fake_brief = MarketBrief(
            query="Fed policy",
            generated_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
            articles_analyzed=1,
            final_verdict="bearish",
            executive_summary="Rates rising. Bears win. Watch FOMC.",
            sentiment_summary={"bullish": 0, "bearish": 1, "neutral": 0},
            top_events=[fake_event],
            debate_rounds=[fake_round],
            source_urls=[fake_article.url],
        )

        agent = OrchestratorAgent()

        # Patch each sub-agent's run method
        with patch.object(agent._fetcher, "run", new=AsyncMock(return_value=[fake_article])), \
             patch.object(agent._extractor, "run", new=AsyncMock(return_value=[fake_event])), \
             patch.object(agent._sentiment, "run", new=AsyncMock(return_value=[fake_sentiment])), \
             patch.object(agent._debate, "run", new=AsyncMock(return_value=[fake_round])), \
             patch.object(agent._debate, "get_verdict", new=AsyncMock(return_value="bearish")), \
             patch.object(agent._synthesizer, "run", new=AsyncMock(return_value=fake_brief)), \
             patch("agents.synthesizer_agent.SynthesizerAgent.save"):
            result = asyncio.run(agent.run(query="Fed policy", debate_rounds=1))

        assert isinstance(result, MarketBrief)
        assert result.final_verdict == "bearish"
        assert result.articles_analyzed == 1


class TestResearchPipeline:
    def test_run_returns_brief(self, sample_raw_article: RawArticle) -> None:
        from datetime import datetime, timezone
        from uuid import uuid4
        from pipelines.research_pipeline import ResearchPipeline
        from models.schemas import MarketBrief

        fake_brief = MarketBrief(
            query="rate hike",
            generated_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
            articles_analyzed=2,
            final_verdict="neutral",
            executive_summary="Rates steady. Both sides argue. Watch yields.",
        )

        pipeline = ResearchPipeline(debate_rounds=1)
        with patch.object(pipeline._orchestrator, "run", new=AsyncMock(return_value=fake_brief)):
            result = asyncio.run(pipeline.run("rate hike"))

        assert result.query == "rate hike"
        assert result.final_verdict == "neutral"

    def test_run_wraps_api_error(self) -> None:
        import anthropic as _anthropic
        from pipelines.research_pipeline import PipelineError, ResearchPipeline

        pipeline = ResearchPipeline(debate_rounds=1)

        async def _raise(*a, **kw):
            raise _anthropic.APIStatusError(
                "overloaded",
                response=MagicMock(status_code=529, headers={}),
                body={"error": {"type": "overloaded_error"}},
            )

        with patch.object(pipeline._orchestrator, "run", new=_raise):
            with pytest.raises(PipelineError, match="Pipeline failed"):
                asyncio.run(pipeline.run("some query"))

    def test_run_batch_returns_successful_briefs(self) -> None:
        from datetime import datetime, timezone
        from pipelines.research_pipeline import PipelineError, ResearchPipeline
        from models.schemas import MarketBrief
        import anthropic as _anthropic

        brief = MarketBrief(
            query="q1",
            generated_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
            articles_analyzed=1,
            final_verdict="bullish",
            executive_summary="Up. Up. Up.",
        )

        pipeline = ResearchPipeline(debate_rounds=1)

        call_count = 0

        async def _run(query, **kwargs):
            nonlocal call_count
            call_count += 1
            if query == "bad query":
                raise PipelineError("forced failure")
            return brief

        with patch.object(pipeline, "run", new=_run):
            results = asyncio.run(pipeline.run_batch(["q1", "bad query", "q2"]))

        # Only the two successful queries should be returned
        assert len(results) == 2
        assert all(isinstance(r, MarketBrief) for r in results)


# ---------------------------------------------------------------------------
# Pipeline slug utility
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_basic_slug(self) -> None:
        from pipelines.research_pipeline import _slugify
        assert _slugify("Federal Reserve Rates") == "federal_reserve_rates"

    def test_special_chars_removed(self) -> None:
        from pipelines.research_pipeline import _slugify
        assert _slugify("Apple: Q1 2025!") == "apple_q1_2025"

    def test_max_length(self) -> None:
        from pipelines.research_pipeline import _slugify
        long_query = "a " * 100
        result = _slugify(long_query, max_length=20)
        assert len(result) <= 20


# ---------------------------------------------------------------------------
# pytest-asyncio tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_articles() -> list[NewsArticle]:
    now = datetime.now(tz=timezone.utc)
    return [
        NewsArticle(
            title=f"Article {i}",
            url=f"https://example.com/article/{i}",
            source="Reuters",
            published_at=now,
            raw_text=f"Content of article {i}.",
        )
        for i in range(1, 4)
    ]


@pytest.mark.asyncio
async def test_fetcher_deduplication() -> None:
    """FetcherAgent deduplicates articles that share the same URL."""
    from agents.fetcher_agent import FetcherAgent

    now = datetime.now(tz=timezone.utc)
    dup_url = "https://example.com/duplicate-story"
    shared = NewsArticle(
        title="Duplicate Story",
        url=dup_url,
        source="Reuters",
        published_at=now,
        raw_text="Body of the duplicate story.",
    )

    # RSS returns the article twice; NewsAPI also returns it — 3 total, all same URL.
    with (
        patch("agents.fetcher_agent.fetch_rss", new=AsyncMock(return_value=[shared, shared])),
        patch("agents.fetcher_agent.fetch_newsapi", new=AsyncMock(return_value=[shared])),
    ):
        result = await FetcherAgent().run("test query", sources=["https://rss.test/feed"])

    assert len(result) == 1
    assert result[0].url == dup_url


@pytest.mark.asyncio
async def test_extractor_batch_processing(mock_articles: list[NewsArticle]) -> None:
    """ExtractorAgent processes every article and returns one event per article."""
    from agents.extractor_agent import ExtractorAgent

    fake_block = MagicMock()
    fake_block.type = "tool_use"
    fake_block.name = "extract_financial_event"
    fake_block.input = {
        "event_type": "earnings",
        "description": "Q3 earnings beat expectations.",
        "entities": ["Apple Inc"],
        "tickers": ["AAPL"],
        "magnitude": "medium",
    }

    agent = ExtractorAgent()
    with patch.object(agent, "_call_llm_raw", new=AsyncMock(return_value=[fake_block])):
        events = await agent.run(mock_articles)

    assert len(events) == len(mock_articles)
    for event in events:
        assert event.event_type == "earnings"
        assert "AAPL" in event.tickers
        assert event.magnitude == "medium"


@pytest.mark.asyncio
async def test_sentiment_scoring(mock_articles: list[NewsArticle]) -> None:
    """SentimentAgent scores each article and returns correctly parsed SentimentResults."""
    from agents.sentiment_agent import SentimentAgent

    mock_response = json.dumps({
        "label": "bullish",
        "score": 0.9,
        "reasoning": "Strong earnings beat analyst estimates.",
        "key_phrases": ["beat expectations", "record revenue"],
    })

    agent = SentimentAgent()
    with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
        results = await agent.run(articles=mock_articles, events=[])

    assert len(results) == len(mock_articles)
    for r in results:
        assert r.label == "bullish"
        assert r.score == 0.9
        assert "beat expectations" in r.key_phrases
        assert "record revenue" in r.key_phrases


@pytest.mark.asyncio
async def test_debate_rounds(mock_articles: list[NewsArticle]) -> None:
    """DebateAgent runs exactly the requested number of rounds with correct structure."""
    from agents.debate_agent import DebateAgent
    from models.schemas import ExtractedEvent, SentimentResult

    art_id = mock_articles[0].id
    events = [
        ExtractedEvent(
            article_id=art_id,
            event_type="earnings",
            description="Record Q3 results.",
            entities=["Apple Inc"],
            tickers=["AAPL"],
            magnitude="high",
        )
    ]
    sentiments = [
        SentimentResult(
            article_id=art_id,
            label="bullish",
            score=0.85,
            reasoning="Earnings beat analyst expectations.",
            key_phrases=["beat expectations"],
        )
    ]

    # 3 rounds × 3 LLM calls each (bull, bear, moderator)
    side_effects = [
        "Bull argument 1.", "Bear argument 1.", "Moderator note 1.",
        "Bull argument 2.", "Bear argument 2.", "Moderator note 2.",
        "Bull argument 3.", "Bear argument 3.", "Moderator note 3.",
    ]

    agent = DebateAgent()
    with patch.object(agent, "_call_llm", new=AsyncMock(side_effect=side_effects)):
        rounds = await agent.run(events=events, sentiment_results=sentiments, rounds=3)

    assert len(rounds) == 3
    for i, r in enumerate(rounds):
        assert r.round_number == i + 1
        assert r.bull_argument == f"Bull argument {i + 1}."
        assert r.bear_argument == f"Bear argument {i + 1}."
        assert r.moderator_note == f"Moderator note {i + 1}."


def test_market_brief_to_markdown() -> None:
    """MarketBrief.to_markdown() includes executive summary and ticker symbols."""
    from uuid import uuid4
    from models.schemas import ExtractedEvent, MarketBrief

    event = ExtractedEvent(
        article_id=uuid4(),
        event_type="earnings",
        description="Apple reported record Q3 results, beating all estimates.",
        entities=["Apple Inc"],
        tickers=["AAPL", "NVDA"],
        magnitude="high",
    )
    brief = MarketBrief(
        query="Apple Q3 earnings",
        articles_analyzed=3,
        top_events=[event],
        sentiment_summary={"bullish": 2, "bearish": 1, "neutral": 0},
        final_verdict="bullish",
        executive_summary="Apple posted record revenue, decisively beating all estimates.",
        source_urls=["https://example.com/aapl"],
    )

    md = brief.to_markdown()

    assert "Apple Q3 earnings" in md
    assert "Apple posted record revenue" in md
    assert "AAPL" in md
    assert "NVDA" in md
    assert "BULLISH" in md


@pytest.mark.asyncio
async def test_full_pipeline_integration() -> None:
    """OrchestratorAgent wires all sub-agents together and returns a MarketBrief."""
    from uuid import uuid4
    from agents.orchestrator import OrchestratorAgent
    from agents.synthesizer_agent import SynthesizerAgent
    from models.schemas import (
        DebateRound,
        ExtractedEvent,
        MarketBrief,
        SentimentResult,
    )

    now = datetime.now(tz=timezone.utc)
    art_id = uuid4()
    fake_article = NewsArticle(
        title="Test Article",
        url="https://example.com/test",
        source="Reuters",
        published_at=now,
        raw_text="Test content body.",
    )
    fake_event = ExtractedEvent(
        article_id=art_id,
        event_type="macro",
        description="Federal Reserve holds rates steady.",
        entities=["Federal Reserve"],
        tickers=["SPY"],
        magnitude="medium",
    )
    fake_sentiment = SentimentResult(
        article_id=art_id,
        label="neutral",
        score=0.5,
        reasoning="Mixed signals from the Fed statement.",
        key_phrases=["mixed signals"],
    )
    fake_round = DebateRound(
        round_number=1,
        bull_argument="Steady rates support equity valuations.",
        bear_argument="No rate cuts means continued pressure on growth stocks.",
        moderator_note="Both sides made evidence-based points.",
    )
    expected_brief = MarketBrief(
        query="test query",
        articles_analyzed=1,
        top_events=[fake_event],
        sentiment_summary={"bullish": 0, "bearish": 0, "neutral": 1},
        debate_rounds=[fake_round],
        final_verdict="neutral",
        executive_summary="Integration test: the pipeline ran end-to-end successfully.",
        source_urls=["https://example.com/test"],
    )

    agent = OrchestratorAgent()
    with (
        patch.object(agent._fetcher, "run", new=AsyncMock(return_value=[fake_article])),
        patch.object(agent._extractor, "run", new=AsyncMock(return_value=[fake_event])),
        patch.object(agent._sentiment, "run", new=AsyncMock(return_value=[fake_sentiment])),
        patch.object(agent._debate, "run", new=AsyncMock(return_value=[fake_round])),
        patch.object(agent._debate, "get_verdict", new=AsyncMock(return_value="neutral")),
        patch.object(agent._synthesizer, "run", new=AsyncMock(return_value=expected_brief)),
        patch.object(SynthesizerAgent, "save"),
    ):
        result = await agent.run("test query", debate_rounds=1)

    assert isinstance(result, MarketBrief)
    assert result.query == "test query"
    assert result.final_verdict == "neutral"
    assert result.executive_summary != ""
    assert len(result.debate_rounds) == 1
