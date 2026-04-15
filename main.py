#!/usr/bin/env python3
"""
financial-news-researcher
=========================
CLI entry point.

Usage
-----
    python main.py research "NVIDIA earnings impact"
    python main.py research "NVIDIA earnings impact" --rounds 5
    python main.py research "NVIDIA earnings impact" --output-format json
    python main.py batch queries.txt
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

from models.schemas import MarketBrief
from pipelines.research_pipeline import PipelineError, ResearchPipeline

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )
    for lib in ("httpx", "httpcore", "anthropic", "feedparser"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Rich UI helpers
# ---------------------------------------------------------------------------

def _print_header(query: str) -> None:
    console.print()
    console.print(Panel(
        f"[bold cyan]{query}[/bold cyan]",
        title="[bold white]📰 Financial News Researcher[/bold white]",
        border_style="cyan",
        expand=False,
    ))
    console.print()


def _print_brief(brief: MarketBrief, output_format: str) -> None:
    # Sentiment bar
    bull = brief.sentiment_summary.get("bullish", 0)
    bear = brief.sentiment_summary.get("bearish", 0)
    neut = brief.sentiment_summary.get("neutral", 0)
    console.print(
        f"  [green]📈 Bullish: {bull}[/green]  "
        f"[yellow]😐 Neutral: {neut}[/yellow]  "
        f"[red]📉 Bearish: {bear}[/red]  "
        f"  Verdict: [bold]{brief.final_verdict.upper()}[/bold]"
    )
    console.print()

    # Top events table
    if brief.top_events:
        table = Table(
            title="Top Events",
            box=box.ROUNDED,
            border_style="dim",
            show_lines=False,
        )
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Event Type", style="magenta")
        table.add_column("Magnitude", justify="center")
        table.add_column("Description", overflow="fold")

        mag_style = {"high": "bold red", "medium": "yellow", "low": "dim"}
        for ev in brief.top_events[:3]:
            tickers = ", ".join(ev.tickers) if ev.tickers else "—"
            style = mag_style.get(ev.magnitude, "")
            table.add_row(
                tickers,
                ev.event_type.replace("_", " ").title(),
                f"[{style}]{ev.magnitude.upper()}[/{style}]",
                ev.description[:120],
            )
        console.print(table)
        console.print()

    # Executive summary panel
    console.print(Panel(
        brief.executive_summary,
        title="[bold white]📊 Executive Summary[/bold white]",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()

    # Output format
    if output_format in ("json", "both"):
        console.print(Panel(
            brief.model_dump_json(indent=2),
            title="[bold white]JSON[/bold white]",
            border_style="blue",
        ))
    if output_format in ("markdown", "both"):
        console.print(brief.to_markdown())


# ---------------------------------------------------------------------------
# Pipeline runner with spinner
# ---------------------------------------------------------------------------

async def _run_research(
    query: str,
    rounds: int,
    output_format: str,
) -> int:
    _print_header(query)

    pipeline = ResearchPipeline(debate_rounds=rounds)

    phases = [
        "Fetching articles",
        "Extracting events & sentiment",
        "Re-scoring sentiment with events",
        "Running bull/bear debate",
        "Synthesising brief",
        "Saving outputs",
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task(phases[0], total=len(phases))

        # Monkey-patch the orchestrator to advance the spinner at each step.
        # We wrap OrchestratorAgent.run and intercept sub-agent calls via
        # logging; the simplest approach is to hook into the existing log
        # messages by installing a handler that advances the task.
        phase_idx = [0]

        class _PhaseHandler(logging.Handler):
            _TRIGGERS = [
                ("[1/6]", 0),
                ("[2/6]", 1),
                ("[3/6]", 2),
                ("[4/6]", 3),
                ("[5/6]", 4),
                ("[6/6]", 5),
            ]

            def emit(self, record: logging.LogRecord) -> None:
                msg = record.getMessage()
                for trigger, idx in self._TRIGGERS:
                    if trigger in msg and idx > phase_idx[0]:
                        phase_idx[0] = idx
                        label = phases[min(idx, len(phases) - 1)]
                        progress.update(task, description=label, advance=1)
                        break

        handler = _PhaseHandler()
        logging.getLogger().addHandler(handler)
        # Temporarily set root logger to INFO so our handler sees the messages
        root_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)

        try:
            brief = await pipeline.run(query)
        except PipelineError as exc:
            console.print(f"[bold red]Pipeline error:[/bold red] {exc}")
            return 1
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            return 130
        finally:
            logging.getLogger().removeHandler(handler)
            logging.getLogger().setLevel(root_level)

    _print_brief(brief, output_format)
    return 0


async def _run_batch(filepath: str, rounds: int, output_format: str) -> int:
    path = Path(filepath)
    if not path.exists():
        console.print(f"[bold red]File not found:[/bold red] {filepath}")
        return 1

    queries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not queries:
        console.print("[yellow]No queries found in file.[/yellow]")
        return 0

    console.print(Panel(
        "\n".join(f"  {i+1}. {q}" for i, q in enumerate(queries)),
        title=f"[bold white]Batch: {len(queries)} queries[/bold white]",
        border_style="cyan",
    ))
    console.print()

    pipeline = ResearchPipeline(debate_rounds=rounds)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task(f"Running {len(queries)} queries…", total=len(queries))

        briefs: list[MarketBrief] = []
        for i, query in enumerate(queries, 1):
            progress.update(task, description=f"[{i}/{len(queries)}] {query[:60]}…")
            try:
                brief = await pipeline.run(query)
                briefs.append(brief)
            except PipelineError as exc:
                console.print(f"[red]✗[/red] '{query}': {exc}")
            progress.advance(task)

    console.print(f"\n[bold green]✓ Completed {len(briefs)}/{len(queries)} queries[/bold green]\n")
    for brief in briefs:
        _print_brief(brief, output_format)

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Multi-agent financial news research pipeline powered by Claude.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    sub = parser.add_subparsers(dest="command", required=True)

    # research subcommand
    research = sub.add_parser("research", help="Run the pipeline for a single query.")
    research.add_argument("query", help="Financial topic or question to research.")
    research.add_argument(
        "--rounds",
        type=int,
        default=3,
        metavar="N",
        help="Number of debate rounds (default: 3).",
    )
    research.add_argument(
        "--output-format",
        choices=["json", "markdown", "both"],
        default="markdown",
        dest="output_format",
        help="Controls terminal output format (default: markdown).",
    )

    # batch subcommand
    batch = sub.add_parser("batch", help="Run the pipeline for each query in a file.")
    batch.add_argument("file", help="Path to a text file with one query per line.")
    batch.add_argument(
        "--rounds",
        type=int,
        default=3,
        metavar="N",
        help="Number of debate rounds per query (default: 3).",
    )
    batch.add_argument(
        "--output-format",
        choices=["json", "markdown", "both"],
        default="markdown",
        dest="output_format",
        help="Controls terminal output format (default: markdown).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.command == "research":
        return await _run_research(args.query, args.rounds, args.output_format)
    elif args.command == "batch":
        return await _run_batch(args.file, args.rounds, args.output_format)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
