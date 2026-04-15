"""High-level research pipeline with error handling and batch support."""
from __future__ import annotations

import asyncio
import logging
import re

import anthropic

from agents.orchestrator import Orchestrator, OrchestratorAgent, OrchestratorConfig
from config import config
from models.schemas import MarketBrief, ResearchReport

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when the research pipeline encounters an unrecoverable API error."""


class ResearchPipeline:
    """
    End-to-end async research pipeline.

    Uses ``OrchestratorAgent`` to run the full multi-agent pipeline and returns
    a ``MarketBrief``.  Wraps Anthropic API errors as ``PipelineError``.
    """

    def __init__(self, debate_rounds: int = 3) -> None:
        self._orchestrator = OrchestratorAgent()
        self._debate_rounds = debate_rounds

    async def run(self, query: str, **kwargs) -> MarketBrief:
        """
        Execute the research pipeline for *query*.

        Raises
        ------
        PipelineError
            If an Anthropic API error occurs during the pipeline run.
        """
        logger.info("ResearchPipeline.run('%s')", query)
        try:
            return await self._orchestrator.run(
                query=query,
                debate_rounds=self._debate_rounds,
                **kwargs,
            )
        except anthropic.APIError as exc:
            logger.error("Anthropic API error for query '%s': %s", query, exc)
            raise PipelineError(f"Pipeline failed for '{query}': {exc}") from exc

    async def run_batch(self, queries: list[str]) -> list[MarketBrief]:
        """
        Run the pipeline concurrently for every query in *queries*.

        Failed queries are logged and omitted from the returned list so that
        successful results are always returned.
        """
        logger.info("ResearchPipeline.run_batch: %d queries", len(queries))
        results = await asyncio.gather(
            *(self.run(q) for q in queries),
            return_exceptions=True,
        )
        briefs: list[MarketBrief] = []
        for query, result in zip(queries, results):
            if isinstance(result, BaseException):
                logger.error("Batch query '%s' failed: %s", query, result)
            else:
                briefs.append(result)
        return briefs


def _slugify(text: str, max_length: int = 50) -> str:
    """Convert a query string to a safe filename slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "_", slug)
    slug = slug.strip("_")
    return slug[:max_length]
