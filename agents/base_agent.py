"""Abstract base class shared by all agents."""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import openai
from openai import OpenAI

from config import config
from models.schemas import NewsArticle

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


@dataclass
class _ToolUseBlock:
    """Adapts OpenAI tool-call objects to the Anthropic-style interface expected by ExtractorAgent."""
    type: str
    name: str
    input: dict


def _to_openai_tool(tool: dict) -> dict:
    """Convert an Anthropic-format tool definition to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


class BaseAgent(ABC):
    """
    Abstract base for all research agents.

    Each subclass must implement ``async def run(**kwargs)``.
    LLM calls go through ``_call_llm`` which handles rate-limit backoff,
    structured logging, and optional tool definitions.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """Execute the agent's primary task."""

    # ------------------------------------------------------------------
    # Core LLM call with rate-limit backoff
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Call ``client.messages.create`` asynchronously via a thread executor.

        Retries up to ``_MAX_RETRIES`` times on ``RateLimitError``, waiting
        2^attempt seconds between attempts (1 s, 2 s, 4 s).

        Logs agent class name and token usage at INFO level on success.

        Returns the text content of the first content block.

        Pass ``max_tokens`` to override the instance default for this call only.
        """
        all_messages = [{"role": "system", "content": system}] + list(messages)
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            temperature=self._temperature,
            messages=all_messages,
        )
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda kw=kwargs: self._client.chat.completions.create(**kw),
                )
                usage = response.usage
                self._logger.info(
                    "[%s] LLM call succeeded — prompt_tokens=%d completion_tokens=%d",
                    self.__class__.__name__,
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )
                return response.choices[0].message.content or ""

            except openai.RateLimitError as exc:
                last_exc = exc
                wait = 2 ** attempt
                self._logger.warning(
                    "[%s] Rate limit hit (attempt %d/%d). Retrying in %ds…",
                    self.__class__.__name__,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"[{self.__class__.__name__}] LLM call failed after {_MAX_RETRIES} retries."
        ) from last_exc
    async def _call_llm_raw(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> list:
        """
        Same as ``_call_llm`` but returns the full ``response.content`` list
        of SDK content-block objects (``TextBlock``, ``ToolUseBlock``, etc.)
        instead of extracting just the first text string.

        Use this when the response may contain ``tool_use`` blocks.
        """
        all_messages = [{"role": "system", "content": system}] + list(messages)
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=all_messages,
        )
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda kw=kwargs: self._client.chat.completions.create(**kw),
                )
                usage = response.usage
                self._logger.info(
                    "[%s] LLM raw call succeeded — prompt_tokens=%d completion_tokens=%d",
                    self.__class__.__name__,
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )
                tool_calls = response.choices[0].message.tool_calls or []
                return [
                    _ToolUseBlock(
                        type="tool_use",
                        name=tc.function.name,
                        input=json.loads(tc.function.arguments),
                    )
                    for tc in tool_calls
                ]

            except openai.RateLimitError as exc:
                last_exc = exc
                wait = 2 ** attempt
                self._logger.warning(
                    "[%s] Rate limit hit (attempt %d/%d). Retrying in %ds\u2026",
                    self.__class__.__name__,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"[{self.__class__.__name__}] LLM raw call failed after {_MAX_RETRIES} retries."
        ) from last_exc
    # ------------------------------------------------------------------
    # JSON extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Any:
        """
        Extract the first JSON object/array from ``text``.
        Tries a ```json fence first, then falls back to brace-matching.
        """
        fence_start = text.find("```json")
        if fence_start != -1:
            fence_end = text.find("```", fence_start + 7)
            if fence_end != -1:
                json_str = text[fence_start + 7 : fence_end].strip()
                return json.loads(json_str)

        for start_char, end_char in [("{", "}"), ("[", "]")]:
            idx = text.find(start_char)
            if idx != -1:
                depth = 0
                in_str = False
                escape = False
                for i, ch in enumerate(text[idx:], start=idx):
                    if escape:
                        escape = False
                        continue
                    if ch == "\\" and in_str:
                        escape = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if ch == start_char:
                        depth += 1
                    elif ch == end_char:
                        depth -= 1
                        if depth == 0:
                            return json.loads(text[idx : i + 1])

        raise ValueError(f"No JSON found in response:\n{text[:500]}")

    # ------------------------------------------------------------------
    # Article formatting helper
    # ------------------------------------------------------------------

    @staticmethod
    def _format_articles(articles: list[NewsArticle]) -> str:
        """
        Format a list of ``NewsArticle`` objects into a numbered XML-like
        block suitable for injection into an LLM prompt.

        Example output::

            <articles>
              <article index="1">
                <title>Fed Raises Rates</title>
                <source>Reuters</source>
                <published_at>2025-01-15T12:00:00</published_at>
                <url>https://example.com/article/1</url>
                <body>Full article text…</body>
              </article>
              …
            </articles>
        """
        parts: list[str] = ["<articles>"]
        for i, article in enumerate(articles, start=1):
            published = (
                article.published_at.isoformat() if article.published_at else "unknown"
            )
            # Truncate very long bodies to avoid blowing the context window
            body = article.raw_text[:4000] if article.raw_text else ""
            parts.append(
                f'  <article index="{i}">\n'
                f"    <title>{_escape_xml(article.title)}</title>\n"
                f"    <source>{_escape_xml(article.source)}</source>\n"
                f"    <published_at>{published}</published_at>\n"
                f"    <url>{article.url}</url>\n"
                f"    <body>{_escape_xml(body)}</body>\n"
                f"  </article>"
            )
        parts.append("</articles>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Module-level helper — not part of the public API
# ---------------------------------------------------------------------------

def _escape_xml(text: str) -> str:
    """Minimal XML escaping to keep the prompt well-formed."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )
