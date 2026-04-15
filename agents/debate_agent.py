"""DebateAgent – multi-persona bull/bear debate using sequential Claude calls."""
from __future__ import annotations

import logging
from typing import Literal

from agents.base_agent import BaseAgent
from models.schemas import (
    Argument,
    DebateReport,
    DebateRound,
    ExtractedArticle,
    ExtractedEvent,
    SentimentResult,
    SentimentSummary,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persona system prompts
# ---------------------------------------------------------------------------

BULL_SYSTEM = """\
You are an aggressive bull-case financial analyst. Your job is to argue WHY the \
market/asset is poised for gains. Cite specific data from the news. Be persuasive, \
concrete, and cite tickers and numbers."""

BEAR_SYSTEM = """\
You are a cautious bear-case financial analyst. Your job is to argue WHY there is \
downside risk. Challenge the bull's optimism with counter-evidence from the news. \
Be analytical and risk-focused."""

MODERATOR_SYSTEM = """\
You are a neutral senior portfolio manager. After each debate round, summarize the \
strongest point from each side in one sentence each, then state which argument was \
more evidence-based this round."""

_VERDICT_SYSTEM = """\
You are a senior portfolio manager reviewing a completed bull/bear debate. Read all \
debate rounds below and return a JSON object with exactly two fields:
  "verdict": one of "bullish", "bearish", or "neutral"
  "reasoning": one sentence explaining the verdict

Respond ONLY with valid JSON."""

# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _format_events(events: list[ExtractedEvent]) -> str:
    if not events:
        return "(no events extracted)"
    lines: list[str] = []
    for ev in events[:20]:
        tickers = ", ".join(ev.tickers) if ev.tickers else "—"
        entities = ", ".join(ev.entities) if ev.entities else "—"
        lines.append(
            f"  [{ev.event_type.upper()} | {ev.magnitude}] {ev.description} "
            f"(tickers: {tickers}; entities: {entities})"
        )
    return "\n".join(lines)


def _format_sentiment(results: list[SentimentResult]) -> str:
    if not results:
        return "(no sentiment data)"
    counts: dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}
    total_score = 0.0
    for r in results:
        counts[r.label] = counts.get(r.label, 0) + 1
        total_score += r.score
    avg = total_score / len(results) if results else 0.0
    summary = (
        f"Bullish: {counts['bullish']}  Bearish: {counts['bearish']}  "
        f"Neutral: {counts['neutral']}  Avg confidence: {avg:.2f}"
    )
    # Include top key phrases for grounding
    phrases: list[str] = []
    for r in sorted(results, key=lambda r: r.score, reverse=True)[:5]:
        phrases.extend(r.key_phrases[:2])
    if phrases:
        summary += "\nKey phrases: " + "; ".join(list(dict.fromkeys(phrases))[:8])
    return summary


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DebateAgent(BaseAgent):
    """
    Multi-persona bull/bear debate agent.

    Runs *rounds* sequential debate cycles.  Each cycle makes three LLM calls:
      1. Bull persona  (max_tokens=400)
      2. Bear persona  (max_tokens=400, sees bull argument)
      3. Moderator     (max_tokens=200, sees both arguments)

    A final ``get_verdict`` call reads all rounds and returns a single label.
    """

    def __init__(self) -> None:
        super().__init__(model="qwen2.5:7b", max_tokens=400, temperature=0.7)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    async def run(  # type: ignore[override]
        self,
        events: list[ExtractedEvent] = None,
        sentiment_results: list[SentimentResult] = None,
        rounds: int = 3,
        **kwargs,
    ) -> list[DebateRound]:
        """Run *rounds* sequential debate cycles and return the full history."""
        events = events or []
        sentiment_results = sentiment_results or []

        base_context = (
            "## Extracted Financial Events\n"
            + _format_events(events)
            + "\n\n## Sentiment Overview\n"
            + _format_sentiment(sentiment_results)
        )

        history: list[DebateRound] = []
        for round_num in range(1, rounds + 1):
            self._logger.info("DebateAgent: starting round %d/%d", round_num, rounds)

            prior_note = ""
            if history:
                prior_note = (
                    "\n\n## Previous Round Moderator Summary\n"
                    + history[-1].moderator_note
                )

            context = base_context + prior_note

            # 1 — Bull argument
            bull_text = await self._call_llm(
                BULL_SYSTEM,
                [{"role": "user", "content": context}],
                max_tokens=400,
            )

            # 2 — Bear rebuttal (sees bull argument in conversation history)
            bear_text = await self._call_llm(
                BEAR_SYSTEM,
                [
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": bull_text},
                    {"role": "user", "content": "Now provide your bear-case rebuttal to the above argument."},
                ],
                max_tokens=400,
            )

            # 3 — Moderator summary
            mod_text = await self._call_llm(
                MODERATOR_SYSTEM,
                [
                    {
                        "role": "user",
                        "content": (
                            f"BULL ARGUMENT (Round {round_num}):\n{bull_text}"
                            f"\n\nBEAR ARGUMENT (Round {round_num}):\n{bear_text}"
                            "\n\nProvide your moderator summary."
                        ),
                    }
                ],
                max_tokens=200,
            )

            history.append(
                DebateRound(
                    round_number=round_num,
                    bull_argument=bull_text,
                    bear_argument=bear_text,
                    moderator_note=mod_text,
                )
            )

        return history

    async def get_verdict(
        self, rounds: list[DebateRound]
    ) -> Literal["bullish", "bearish", "neutral"]:
        """Ask Claude to synthesise all debate rounds into a single verdict label."""
        if not rounds:
            return "neutral"

        rounds_text = "\n\n".join(
            f"--- Round {r.round_number} ---\n"
            f"BULL: {r.bull_argument}\n\n"
            f"BEAR: {r.bear_argument}\n\n"
            f"MODERATOR: {r.moderator_note}"
            for r in rounds
        )

        text = await self._call_llm(
            _VERDICT_SYSTEM,
            [{"role": "user", "content": rounds_text}],
            max_tokens=200,
        )

        try:
            data = self._extract_json(text)
            verdict = str(data.get("verdict", "neutral")).lower()
            if verdict in ("bullish", "bearish", "neutral"):
                return verdict  # type: ignore[return-value]
        except Exception:
            pass

        # Fallback: scan text for first matching label
        lower = text.lower()
        for label in ("bullish", "bearish", "neutral"):
            if label in lower:
                return label  # type: ignore[return-value]
        return "neutral"

    # ------------------------------------------------------------------
    # Legacy interface used by the orchestrator (ExtractedArticle pipeline)
    # ------------------------------------------------------------------

    async def run_legacy(  # type: ignore[override]
        self,
        query: str = "",
        articles: list[ExtractedArticle] = None,
        sentiment: SentimentSummary = None,
        **kwargs,
    ) -> DebateReport:
        """Original single-call DebateReport path (legacy orchestrator)."""
        self._logger.info("DebateAgent (legacy) running for query: '%s'", query)

        _LEGACY_SYSTEM = (
            "You are an experienced financial debate moderator. Given financial news "
            "and sentiment data, construct a balanced bull vs bear debate. "
            "Respond ONLY with valid JSON matching this schema: "
            '{"bull_case": [{"point": str, "supporting_evidence": [str], "strength": float}], '
            '"bear_case": [...], "key_risks": [str], "key_opportunities": [str], "verdict": str}'
        )

        context = self._build_legacy_context(articles or [], sentiment)
        user_message = (
            f"Research topic: {query}\n\n"
            f"Overall sentiment: {sentiment.overall_sentiment.value} "
            f"({sentiment.bullish_count} bullish, {sentiment.bearish_count} bearish, "
            f"{sentiment.neutral_count} neutral)\n\n"
            f"Evidence from {len(articles or [])} articles:\n\n{context}"
        )

        text = await self._call_llm(
            _LEGACY_SYSTEM,
            [{"role": "user", "content": user_message}],
            max_tokens=3000,
        )
        data = self._extract_json(text)

        def _parse_args(raw: list[dict]) -> list[Argument]:
            return [
                Argument(
                    point=a.get("point", ""),
                    supporting_evidence=a.get("supporting_evidence", []),
                    strength=float(a.get("strength", 0.5)),
                )
                for a in raw
                if a.get("point")
            ]

        return DebateReport(
            topic=query,
            bull_case=_parse_args(data.get("bull_case", [])),
            bear_case=_parse_args(data.get("bear_case", [])),
            key_risks=data.get("key_risks", []),
            key_opportunities=data.get("key_opportunities", []),
            verdict=data.get("verdict", ""),
        )

    def _build_legacy_context(
        self,
        articles: list[ExtractedArticle],
        sentiment: SentimentSummary,
    ) -> str:
        lines: list[str] = []
        url_to_impact = {s.article_url: s.impact_magnitude for s in sentiment.scores}
        sorted_articles = sorted(
            articles,
            key=lambda a: url_to_impact.get(a.raw.url, 0.0),
            reverse=True,
        )
        for i, article in enumerate(sorted_articles[:20], start=1):
            impact = url_to_impact.get(article.raw.url, 0.0)
            lines.append(f"### Article {i}: {article.raw.title}")
            lines.append(f"Source: {article.raw.source_name or article.raw.source.value}")
            lines.append(f"Impact magnitude: {impact:.2f}")
            lines.append(f"Summary: {article.summary}")
            if article.key_facts:
                lines.append("Key facts:")
                for kf in article.key_facts[:5]:
                    lines.append(f"  - {kf.fact}")
            lines.append("")
        return "\n".join(lines)
