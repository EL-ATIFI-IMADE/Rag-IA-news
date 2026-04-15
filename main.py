"""
main.py — Interactive CLI for the AI News RAG system.

Usage:
    python main.py              # interactive Q&A loop
    python main.py --ingest     # refresh news then start Q&A
    python main.py --auto       # ingest every 60 min in background + Q&A
"""

import argparse
import threading
import time

import schedule
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from config import LLM_MODEL, EMBED_MODEL, TOP_K_RESULTS
from ingest import run_ingestion
from rag_pipeline import ask

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════╗
║          🤖  AI News RAG  — Powered by Ollama        ║
║   Ask anything about the latest AI news!             ║
╚══════════════════════════════════════════════════════╝
"""

EXAMPLE_QUERIES = [
    "What are the latest AI news today?",
    "What has OpenAI or Google announced recently?",
    "What are the recent breakthroughs in large language models?",
    "Are there any new AI regulations or policies?",
    "What AI hardware or chip news is there?",
    "What startups have raised funding in AI recently?",
]


def print_sources(sources: list[dict]):
    """Print sources as a Rich table."""
    if not sources:
        return
    table = Table(
        title="📎 Sources",
        box=box.ROUNDED,
        show_lines=True,
        style="dim",
    )
    table.add_column("#", style="cyan", width=3)
    table.add_column("Publication", style="yellow", width=20)
    table.add_column("Title", style="white", width=45)
    table.add_column("Published", style="green", width=15)

    for i, src in enumerate(sources, 1):
        table.add_row(
            str(i),
            src["source"],
            src["title"][:44],
            src.get("published", "")[:10],
        )
    console.print(table)
    console.print()
    for i, src in enumerate(sources, 1):
        if src["url"]:
            console.print(f"  [dim][{i}][/dim] [link={src['url']}]{src['url']}[/link]")
    console.print()


def auto_refresh_loop():
    """Background thread: re-ingest every 60 minutes."""
    schedule.every(60).minutes.do(run_ingestion)
    console.print("[dim]⏰ Auto-refresh scheduled every 60 minutes.[/dim]\n")
    while True:
        schedule.run_pending()
        time.sleep(30)


def interactive_loop():
    """Main REPL: read question → retrieve → answer → print sources."""
    console.print(Panel(BANNER.strip(), style="bold cyan"))

    # Show config summary
    cfg_table = Table(box=box.SIMPLE, show_header=False)
    cfg_table.add_column("Key", style="dim")
    cfg_table.add_column("Value", style="cyan")
    cfg_table.add_row("LLM", LLM_MODEL)
    cfg_table.add_row("Embeddings", EMBED_MODEL)
    cfg_table.add_row("Retrieved chunks", str(TOP_K_RESULTS))
    console.print(cfg_table)

    # Show example queries
    console.print(Rule("[dim]Example queries[/dim]"))
    for q in EXAMPLE_QUERIES:
        console.print(f"  [dim]›[/dim] {q}")
    console.print(Rule())
    console.print('[dim]Type "exit" or press Ctrl-C to quit.[/dim]\n')

    while True:
        try:
            query = console.input("[bold green]You ▸[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q", "bye"}:
            console.print("[dim]Goodbye![/dim]")
            break

        console.print()
        console.print(Rule(f"[cyan]{query}[/cyan]"))
        console.print("[bold yellow]🤖 Answer[/bold yellow]\n")

        result = ask(query, stream=True)   # streams to stdout

        console.print()
        console.print(Rule("[dim]Sources[/dim]"))
        print_sources(result["sources"])
        console.print(Rule())
        console.print()


def main():
    parser = argparse.ArgumentParser(
        description="AI News RAG — ask questions about the latest AI news"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Fetch & index the latest news before starting Q&A",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-refresh news every 60 minutes (runs in background)",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only run ingestion, then exit",
    )
    args = parser.parse_args()

    if args.ingest or args.ingest_only:
        run_ingestion()

    if args.ingest_only:
        return

    if args.auto:
        t = threading.Thread(target=auto_refresh_loop, daemon=True)
        t.start()

    interactive_loop()


if __name__ == "__main__":
    main()
