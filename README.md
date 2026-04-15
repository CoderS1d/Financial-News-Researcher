<div align="center">

# 📰 financial-news-researcher

### Agentic multi-agent financial intelligence — powered by Ollama (local LLMs, free)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?style=flat-square&logo=ollama&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-27%20passing-22c55e?style=flat-square&logo=pytest&logoColor=white)

</div>

---

## Overview

`financial-news-researcher` is a fully autonomous, multi-agent pipeline that turns a plain-English financial query into a structured intelligence brief — without any human in the loop. Given a topic such as *"Fed rate decision impact on tech stocks"*, it concurrently fetches articles from RSS feeds and NewsAPI, uses the OpenAI-compatible tool-calling API to extract structured financial events, and scores per-article sentiment before assembling all findings into a coherent `MarketBrief` with JSON and Markdown output. The entire pipeline is orchestrated by a single `OrchestratorAgent` that coordinates five specialist agents and writes results to disk automatically. **All LLM inference runs locally via [Ollama](https://ollama.com) — no API key and no cost.**

The standout feature is the **multi-agent Bull vs Bear debate**: after extraction and sentiment scoring, two adversarial LLM instances argue opposite sides of the market thesis for a configurable number of rounds, moderated by a third impartial call. A final verdict (`bullish` / `bearish` / `neutral`) is then drawn from the accumulated debate transcript. This deliberative reasoning step produces substantially richer, more balanced executive summaries than single-pass LLM analysis could achieve on its own.

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │           OrchestratorAgent             │
                         └────────────────┬────────────────────────┘
                                          │
                                    [1] Query
                                          │
                              ┌───────────▼───────────┐
                              │      FetcherAgent      │
                              │  RSS feeds + NewsAPI   │
                              │  dedup · filter 48 h   │
                              └───────────┬───────────┘
                                          │  list[NewsArticle]
                           ┌──────────────┴──────────────┐
                           │  [2] Parallel               │
                  ┌────────▼────────┐         ┌──────────▼──────────┐
                  │ ExtractorAgent  │         │   SentimentAgent    │
                  │  tool_use API   │         │  preliminary pass   │
                  │ → ExtractedEvent│         │ → SentimentResult   │
                  └────────┬────────┘         └──────────┬──────────┘
                           └──────────────┬──────────────┘
                                          │
                              ┌───────────▼───────────┐
                              │   SentimentAgent  [3] │
                              │   re-score with events│
                              └───────────┬───────────┘
                                          │
                              ┌───────────▼───────────┐
                              │      DebateAgent  [4] │
                              │  🟢 Bull  vs  🔴 Bear  │
                              │   N rounds · verdict  │
                              └───────────┬───────────┘
                                          │
                              ┌───────────▼───────────┐
                              │   SynthesizerAgent[5] │
                              │  executive summary    │
                              └───────────┬───────────┘
                                          │
                              ┌───────────▼───────────┐
                              │       MarketBrief  [6]│
                              │   output/.json + .md  │
                              └───────────────────────┘
```

---

## Features

- **Multi-source news fetching** — pulls from configurable RSS feeds (MarketWatch, CNBC, Seeking Alpha, Investing.com) and NewsAPI in parallel, with automatic URL-based deduplication and a 48-hour recency filter.
- **Structured event extraction via tool use** — `ExtractorAgent` uses the OpenAI-compatible function-calling API to extract typed `ExtractedEvent` objects (earnings, rate decisions, mergers, macro events, regulation) with entity lists, ticker symbols, and impact magnitude.
- **Per-article sentiment scoring** — `SentimentAgent` runs a two-pass analysis: a preliminary pass on raw articles, then a refined pass once events are known, returning `bullish` / `bearish` / `neutral` labels with confidence scores and key phrases.
- **Multi-agent Bull vs Bear debate** — the flagship feature. `DebateAgent` runs N configurable rounds with independent bull, bear, and moderator LLM calls per round, then distils a final investment verdict.
- **Synthesised executive brief** — `SynthesizerAgent` assembles all upstream outputs into a single `MarketBrief` with an AI-written executive summary and structured `sentiment_summary` dict.
- **JSON + Markdown output** — every brief is written to `./output/` as a validated Pydantic v2 JSON file and a richly formatted Markdown report, both timestamped.
- **100% local & free** — all LLM calls go to a local Ollama server. No cloud API key, no per-token cost.

---

## Quickstart

### Prerequisites

**1. Install Ollama**

Download from [ollama.com](https://ollama.com) and run the installer. Ollama starts a local server on `http://localhost:11434` automatically.

**2. Pull a model**

```bash
ollama pull qwen2.5:7b
```

> [!TIP]
> `qwen2.5:7b` is the default and a good balance of speed and quality at ~4 GB. For better results on complex queries try `qwen2.5:14b` or `llama3.1:8b`. Any model that supports function calling will work — set it via the `OLLAMA_MODEL` env var.

### Install & run

```bash
# 1. Clone
git clone https://github.com/your-org/financial-news-researcher.git
cd financial-news-researcher

# 2. Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure (no API key needed)
cp .env.example .env
# Optionally add a free NEWSAPI_KEY for broader article coverage

# 5. Run your first research query
python main.py research "Fed rate decision impact on tech stocks"
```

> [!TIP]
> Add `--rounds 5` to the `research` command for a more thorough debate and a higher-quality verdict. Three rounds is the default and a good balance for most queries.

For batch processing, pass a plain-text file with one query per line:

```bash
python main.py batch queries.txt
```

---

## Output Example

### JSON (`output/fed_rate_decision_impact_on_tech_stocks_20260414_093012.json`)

```json
{
  "query": "Fed rate decision impact on tech stocks",
  "generated_at": "2026-04-14T09:30:12+00:00",
  "articles_analyzed": 17,
  "final_verdict": "bearish",
  "executive_summary": "The Federal Reserve held rates at 5.25–5.50 % for the third consecutive meeting but signalled a higher-for-longer stance that rattled rate-sensitive technology names. Analysts at Goldman Sachs and JPMorgan flagged valuation compression risk for high-multiple growth stocks. Bull-side arguments centred on resilient Q1 earnings for mega-cap tech; bear-side arguments pointed to refinancing pressure and declining multiple expansion headroom...",
  "sentiment_summary": {
    "bullish": 5,
    "bearish": 10,
    "neutral": 2
  },
  "top_events": [
    {
      "article_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "event_type": "rate_decision",
      "description": "Federal Reserve holds benchmark rate at 5.25–5.50 %, signals prolonged restrictive stance.",
      "entities": ["Federal Reserve", "Jerome Powell"],
      "tickers": ["SPY", "QQQ", "NVDA", "MSFT"],
      "magnitude": "high"
    }
  ],
  "debate_rounds": [
    {
      "round_number": 1,
      "bull_argument": "Mega-cap tech balance sheets are fortress-like; higher rates barely dent FCF yield for MSFT or GOOG.",
      "bear_argument": "The real risk is multiple compression. At 30× forward earnings, a 50 bps re-rating wipes 15 % off valuations.",
      "moderator_note": "Both sides agree on earnings resilience but diverge sharply on valuation risk — the bear case is better quantified."
    }
  ],
  "source_urls": [
    "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "https://www.cnbc.com/2026/04/14/fed-holds-rates-tech-selloff.html"
  ]
}
```

### Markdown excerpt

```markdown
# 📰 Market Brief: Fed rate decision impact on tech stocks
*Generated: 2026-04-14 09:30 UTC*
*Articles analysed: 17*

---

## 📋 Executive Summary
The Federal Reserve held rates at 5.25–5.50 % for the third consecutive meeting...

## 📊 Sentiment Overview
🟢 **Bullish:** 5  |  🔴 **Bearish:** 10  |  ⚪ **Neutral:** 2

## 🗞️ Top Events
- 🔥 **[Rate Decision]** Federal Reserve holds benchmark rate, signals prolonged restrictive stance.
  - Tickers: `SPY` `QQQ` `NVDA` `MSFT`
  - Entities: Federal Reserve, Jerome Powell

## 🥊 Debate Rounds
### Round 1
🟢 **Bull:** Mega-cap tech balance sheets are fortress-like...

🔴 **Bear:** The real risk is multiple compression...

⚖️ **Moderator:** Both sides agree on earnings resilience but diverge sharply on valuation risk...

## 🏁 Final Verdict
🔴 **BEARISH**
```

---

## Configuration

All configuration is read from a `.env` file at startup (via `python-dotenv`).

| Variable | Required | Default | Description |
|---|:---:|---|---|
| `OLLAMA_BASE_URL` | ☑️ | `http://localhost:11434/v1` | Base URL of the Ollama OpenAI-compatible endpoint. Change this if Ollama runs on a different host or port. |
| `OLLAMA_MODEL` | ☑️ | `qwen2.5:7b` | Model name to use for all agents. Must be pulled locally first (`ollama pull <model>`). |
| `NEWSAPI_KEY` | ☑️ | — | [NewsAPI.org](https://newsapi.org) key. Free tier allows 100 req/day. Without this key the pipeline falls back to RSS-only mode. |
| `OUTPUT_DIR` | ☑️ | `./output` | Directory where `.json` and `.md` reports are written. Created automatically if absent. |

> [!TIP]
> No API key is required to run this project. `OLLAMA_BASE_URL` and `OLLAMA_MODEL` both have sensible defaults so an empty `.env` file is enough to get started.

The following constants live in `config.py` and can be adjusted directly without overriding env vars:

| Constant | Default | Description |
|---|---|---|
| `MAX_TOKENS` | `4096` | Default token budget per LLM call. |
| `HTTP_TIMEOUT` | `30.0` s | Timeout for outbound HTTP requests. |
| `HTTP_MAX_RETRIES` | `3` | Retry attempts on transient failures. |

---

## Extending

Adding a new specialist agent takes about ten lines of code. Create a Python file under `agents/`, define a class that inherits from `BaseAgent`, and implement the single `async def run(self, **kwargs)` abstract method. Inside `run`, use `self._call_llm(system, messages, max_tokens=…)` to get a plain string response or `self._call_llm_raw(system, messages, tools=[…])` to receive parsed tool-call objects for function-calling workflows. Rate-limit backoff, structured logging, and the Ollama client are all managed by the base class — your agent only needs to contain its domain logic. Once ready, instantiate the agent inside `OrchestratorAgent.__init__` and wire its `run` call into the appropriate step of `OrchestratorAgent.run`.

> [!TIP]
> All agents are fully unit-testable in isolation. Use `unittest.mock.AsyncMock` to patch `_call_llm` or `_call_llm_raw` on the agent instance, then call `await agent.run(…)` directly — no running Ollama server required. See `tests/test_pipeline.py` for examples.

---

## License

[MIT](LICENSE) — free for personal and commercial use.

---

## Overview

`financial-news-researcher` is a fully autonomous, multi-agent pipeline that turns a plain-English financial query into a structured intelligence brief — without any human in the loop. Given a topic such as *"Fed rate decision impact on tech stocks"*, it concurrently fetches articles from RSS feeds and NewsAPI, uses Claude's tool-use API to extract structured financial events, and scores per-article sentiment before assembling all findings into a coherent `MarketBrief` with JSON and Markdown output. The entire pipeline is orchestrated by a single `OrchestratorAgent` that coordinates five specialist agents and writes results to disk automatically.

The standout feature is the **multi-agent Bull vs Bear debate**: after extraction and sentiment scoring, two adversarial Claude instances argue opposite sides of the market thesis for a configurable number of rounds, moderated by a third impartial Claude call. A final verdict (`bullish` / `bearish` / `neutral`) is then drawn from the accumulated debate transcript. This deliberative reasoning step produces substantially richer, more balanced executive summaries than single-pass LLM analysis could achieve on its own.

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │           OrchestratorAgent             │
                         └────────────────┬────────────────────────┘
                                          │
                                    [1] Query
                                          │
                              ┌───────────▼───────────┐
                              │      FetcherAgent      │
                              │  RSS feeds + NewsAPI   │
                              │  dedup · filter 48 h   │
                              └───────────┬───────────┘
                                          │  list[NewsArticle]
                           ┌──────────────┴──────────────┐
                           │  [2] Parallel               │
                  ┌────────▼────────┐         ┌──────────▼──────────┐
                  │ ExtractorAgent  │         │   SentimentAgent    │
                  │  tool_use API   │         │  preliminary pass   │
                  │ → ExtractedEvent│         │ → SentimentResult   │
                  └────────┬────────┘         └──────────┬──────────┘
                           └──────────────┬──────────────┘
                                          │
                              ┌───────────▼───────────┐
                              │   SentimentAgent  [3] │
                              │   re-score with events│
                              └───────────┬───────────┘
                                          │
                              ┌───────────▼───────────┐
                              │      DebateAgent  [4] │
                              │  🟢 Bull  vs  🔴 Bear  │
                              │   N rounds · verdict  │
                              └───────────┬───────────┘
                                          │
                              ┌───────────▼───────────┐
                              │   SynthesizerAgent[5] │
                              │  executive summary    │
                              └───────────┬───────────┘
                                          │
                              ┌───────────▼───────────┐
                              │       MarketBrief  [6]│
                              │   output/.json + .md  │
                              └───────────────────────┘
```

---

## Features

- **Multi-source news fetching** — pulls from configurable RSS feeds (MarketWatch, Bloomberg, Yahoo Finance, CNBC, Investing.com) and NewsAPI in parallel, with automatic URL-based deduplication and a 48-hour recency filter.
- **Structured event extraction via tool use** — `ExtractorAgent` leverages Claude's native tool-use API to extract typed `ExtractedEvent` objects (earnings, rate decisions, mergers, macro events, regulation) with entity lists, ticker symbols, and impact magnitude.
- **Per-article sentiment scoring** — `SentimentAgent` runs a two-pass analysis: a preliminary pass on raw articles, then a refined pass once events are known, returning `bullish` / `bearish` / `neutral` labels with confidence scores and key phrases.
- **Multi-agent Bull vs Bear debate** — the flagship feature. `DebateAgent` runs N configurable rounds with independent bull, bear, and moderator Claude calls per round, then distils a final investment verdict.
- **Synthesised executive brief** — `SynthesizerAgent` assembles all upstream outputs into a single `MarketBrief` with an AI-written executive summary and structured `sentiment_summary` dict.
- **JSON + Markdown output** — every brief is written to `./output/` as a validated Pydantic v2 JSON file and a richly formatted Markdown report, both timestamped.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/your-org/financial-news-researcher.git
cd financial-news-researcher

# 2. Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Open .env and fill in ANTHROPIC_API_KEY (required) and NEWSAPI_KEY (optional)

# 5. Run your first research query
python main.py research "Fed rate decision impact on tech stocks"
```

> [!TIP]
> Add `--rounds 5` to the `research` command for a more thorough debate and a higher-quality verdict. Three rounds is the default and a good balance for most queries.

For batch processing, pass a plain-text file with one query per line:

```bash
python main.py batch queries.txt
```

---

## Output Example

### JSON (`output/fed_rate_decision_impact_on_tech_stocks_20260414_093012.json`)

```json
{
  "query": "Fed rate decision impact on tech stocks",
  "generated_at": "2026-04-14T09:30:12+00:00",
  "articles_analyzed": 17,
  "final_verdict": "bearish",
  "executive_summary": "The Federal Reserve held rates at 5.25–5.50 % for the third consecutive meeting but signalled a higher-for-longer stance that rattled rate-sensitive technology names. Analysts at Goldman Sachs and JPMorgan flagged valuation compression risk for high-multiple growth stocks. Bull-side arguments centred on resilient Q1 earnings for mega-cap tech; bear-side arguments pointed to refinancing pressure and declining multiple expansion headroom...",
  "sentiment_summary": {
    "bullish": 5,
    "bearish": 10,
    "neutral": 2
  },
  "top_events": [
    {
      "article_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "event_type": "rate_decision",
      "description": "Federal Reserve holds benchmark rate at 5.25–5.50 %, signals prolonged restrictive stance.",
      "entities": ["Federal Reserve", "Jerome Powell"],
      "tickers": ["SPY", "QQQ", "NVDA", "MSFT"],
      "magnitude": "high"
    }
  ],
  "debate_rounds": [
    {
      "round_number": 1,
      "bull_argument": "Mega-cap tech balance sheets are fortress-like; higher rates barely dent FCF yield for MSFT or GOOG.",
      "bear_argument": "The real risk is multiple compression. At 30× forward earnings, a 50 bps re-rating wipes 15 % off valuations.",
      "moderator_note": "Both sides agree on earnings resilience but diverge sharply on valuation risk — the bear case is better quantified."
    }
  ],
  "source_urls": [
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://www.cnbc.com/2026/04/14/fed-holds-rates-tech-selloff.html"
  ]
}
```

### Markdown excerpt

```markdown
# 📰 Market Brief: Fed rate decision impact on tech stocks
*Generated: 2026-04-14 09:30 UTC*
*Articles analysed: 17*

---

## 📋 Executive Summary
The Federal Reserve held rates at 5.25–5.50 % for the third consecutive meeting...

## 📊 Sentiment Overview
🟢 **Bullish:** 5  |  🔴 **Bearish:** 10  |  ⚪ **Neutral:** 2

## 🗞️ Top Events
- 🔥 **[Rate Decision]** Federal Reserve holds benchmark rate, signals prolonged restrictive stance.
  - Tickers: `SPY` `QQQ` `NVDA` `MSFT`
  - Entities: Federal Reserve, Jerome Powell

## 🥊 Debate Rounds
### Round 1
🟢 **Bull:** Mega-cap tech balance sheets are fortress-like...

🔴 **Bear:** The real risk is multiple compression...

⚖️ **Moderator:** Both sides agree on earnings resilience but diverge sharply on valuation risk...

## 🏁 Final Verdict
🔴 **BEARISH**
```

---

## Configuration

All configuration is read from a `.env` file at startup (via `python-dotenv`).

| Variable | Required | Default | Description |
|---|:---:|---|---|
| `ANTHROPIC_API_KEY` | ✅ | — | Your [Anthropic API key](https://console.anthropic.com/). All LLM calls use this. |
| `NEWSAPI_KEY` | ☑️ | — | [NewsAPI.org](https://newsapi.org) key. Free tier allows 100 req/day. Without this key the pipeline falls back to RSS-only mode. |
| `OUTPUT_DIR` | ☑️ | `./output` | Directory where `.json` and `.md` reports are written. Created automatically if absent. |

> [!TIP]
> `ANTHROPIC_API_KEY` is the only hard requirement. The pipeline runs in RSS-only mode when `NEWSAPI_KEY` is absent, which is sufficient for most queries.

The following constants live in `config.py` and can be adjusted directly without overriding env vars:

| Constant | Default | Description |
|---|---|---|
| `MODEL` | `claude-opus-4-5` | Anthropic model used by all agents. |
| `MAX_TOKENS` | `4096` | Default token budget per LLM call. |
| `HTTP_TIMEOUT` | `30.0` s | Timeout for outbound HTTP requests. |
| `HTTP_MAX_RETRIES` | `3` | Retry attempts on transient failures. |

---

## Extending

Adding a new specialist agent takes about ten lines of code. Create a Python file under `agents/`, define a class that inherits from `BaseAgent`, and implement the single `async def run(self, **kwargs)` abstract method. Inside `run`, use `self._call_llm(messages, max_tokens=…)` to get a plain string response or `self._call_llm_raw(messages, tools=[…])` to access raw `ContentBlock` objects for tool-use parsing. Rate-limit backoff, structured logging, and Anthropic client management are all handled by the base class — your agent only needs to contain its domain logic. Once ready, instantiate the agent inside `OrchestratorAgent.__init__` and wire its `run` call into the appropriate step of `OrchestratorAgent.run`.

> [!TIP]
> All agents are fully unit-testable in isolation. Use `unittest.mock.AsyncMock` to patch `_call_llm` or `_call_llm_raw` on the agent instance, then call `await agent.run(…)` directly — no network or API key required. See `tests/test_pipeline.py` for examples.

---

## License

[MIT](LICENSE) — free for personal and commercial use.

#### Options

```
usage: financial-news-researcher [-h] [--max-articles N] [--no-enrich] [--no-save] [--verbose] query

positional arguments:
  query                Financial topic or question to research.

options:
  --max-articles N     Maximum number of articles to analyse (default: 30).
  --no-enrich          Skip full-body article enrichment via web scraping.
  --no-save            Do not save the report to disk.
  --verbose, -v        Enable debug logging.
```

### Examples

```bash
# Research Apple earnings
python main.py "Apple earnings Q1 2025" --max-articles 20

# Quick scan without saving
python main.py "Bitcoin price rally" --no-save --no-enrich

# Verbose debug output
python main.py "Oil prices OPEC" -v
```

## Output

Each run produces two files in `./output/`:

- `<slug>_<timestamp>.md` — Human-readable Markdown report
- `<slug>_<timestamp>.json` — Machine-readable JSON (Pydantic-serialised)

### Report sections

1. **Executive Summary** — 3-5 paragraph overview
2. **Overall Sentiment** — Bullish / Bearish / Neutral with counts
3. **Key Findings** — Bullet-point facts extracted from articles
4. **Bull Case** — Arguments with supporting evidence and strength score
5. **Bear Case** — Counter-arguments with supporting evidence
6. **Risk Factors** — Downside risks identified
7. **Verdict** — Balanced one-paragraph conclusion
8. **Sources** — All article URLs used

## Project structure

```
financial-news-researcher/
├── agents/
│   ├── base_agent.py        # Anthropic client + JSON extraction helpers
│   ├── orchestrator.py      # Coordinates all agents in sequence
│   ├── fetcher_agent.py     # Gathers articles from RSS / NewsAPI / web
│   ├── extractor_agent.py   # Extracts facts, entities, topics
│   ├── sentiment_agent.py   # Classifies financial sentiment
│   ├── debate_agent.py      # Generates bull/bear debate
│   └── synthesizer_agent.py # Produces final report
├── pipelines/
│   └── research_pipeline.py # End-to-end pipeline + file saving
├── models/
│   └── schemas.py           # All Pydantic v2 data models
├── tools/
│   ├── rss_reader.py        # feedparser + httpx async RSS fetcher
│   ├── news_api.py          # NewsAPI.org httpx client
│   └── web_scraper.py       # Lightweight HTML → text scraper
├── output/                  # Generated reports land here
├── tests/
│   └── test_pipeline.py     # Pytest unit tests
├── main.py                  # CLI entry point
├── config.py                # Settings loaded from .env
└── requirements.txt
```

## Running tests

```bash
pytest tests/ -v
```

Tests mock all LLM and HTTP calls so no API keys are needed.

## Technology choices

| Library | Purpose |
|---|---|
| `anthropic` | All LLM inference (Claude) |
| `pydantic v2` | Data models, validation, serialisation |
| `httpx` | Async HTTP for RSS, NewsAPI, scraping |
| `feedparser` | RSS / Atom feed parsing |
| `python-dotenv` | `.env` file loading |
| `pytest` | Test framework |

## License

MIT
