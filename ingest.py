"""
ingest.py — Fetches AI news from RSS feeds, cleans the text, splits it into
chunks, generates embeddings via Ollama, and stores everything in ChromaDB.

Run directly:
    python ingest.py
"""

import hashlib
import re
import time
from datetime import datetime
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    RSS_FEEDS,
    MAX_ARTICLES_PER_FEED,
    REQUEST_TIMEOUT,
)

console = Console()


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_doc_id(url: str, chunk_index: int) -> str:
    """Create a stable, unique ID for each text chunk."""
    raw = f"{url}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def clean_html(raw_html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    soup = BeautifulSoup(raw_html, "lxml")
    text = soup.get_text(separator=" ")
    # Collapse multiple spaces / newlines into single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_full_article(url: str) -> Optional[str]:
    """
    Try to download the full article body from its URL.
    Returns cleaned plain text, or None on failure.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AINewsRAG/1.0)"}
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove nav, ads, footers, scripts, styles
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "figure", "noscript"]):
            tag.decompose()

        # Try common article containers first
        for selector in ["article", "main", ".article-body",
                          ".post-content", ".entry-content", "#content"]:
            block = soup.select_one(selector)
            if block:
                text = block.get_text(separator=" ")
                return re.sub(r"\s+", " ", text).strip()

        # Fallback: all paragraphs
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        return re.sub(r"\s+", " ", text).strip()

    except Exception:
        return None


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping character-based chunks.
    Simple but effective for RAG without requiring tiktoken.
    """
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ── RSS Ingestion ──────────────────────────────────────────────────────────────

def parse_feed(feed_cfg: dict) -> list[dict]:
    """
    Parse one RSS feed and return a list of article dicts:
      { title, url, summary, published, source }
    """
    articles = []
    try:
        feed = feedparser.parse(feed_cfg["url"])
        entries = feed.entries[:MAX_ARTICLES_PER_FEED]
        for entry in entries:
            title = entry.get("title", "No title")
            url = entry.get("link", "")
            summary = clean_html(
                entry.get("summary", "")
                or entry.get("description", "")
            )
            published = entry.get("published", str(datetime.now().date()))

            articles.append({
                "title": title,
                "url": url,
                "summary": summary,
                "published": published,
                "source": feed_cfg["name"],
            })
    except Exception as e:
        console.print(f"[red]  ✗ Error parsing {feed_cfg['name']}: {e}[/red]")
    return articles


# ── Vector Store ───────────────────────────────────────────────────────────────

def get_collection():
    """Initialise ChromaDB client and return (or create) our collection."""
    embed_fn = OllamaEmbeddingFunction(
        url=f"{OLLAMA_BASE_URL}/api/embeddings",
        model_name=EMBED_MODEL,
    )
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def store_articles(articles: list[dict], collection) -> int:
    """
    For each article:
      1. Fetch full content (or fall back to summary)
      2. Chunk the text
      3. Upsert chunks + metadata into ChromaDB
    Returns total number of chunks stored.
    """
    total_chunks = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding articles…", total=len(articles))

        for article in articles:
            progress.update(
                task,
                description=f"Processing: {article['title'][:55]}…",
            )

            # Use full article text if possible, else fall back to summary
            full_text = fetch_full_article(article["url"])
            body = full_text if full_text and len(full_text) > 200 else article["summary"]
            if not body:
                progress.advance(task)
                continue

            # Prepend title so every chunk knows what article it belongs to
            combined = f"Title: {article['title']}\n\n{body}"
            chunks = chunk_text(combined)

            ids, texts, metadatas = [], [], []
            for i, chunk in enumerate(chunks):
                doc_id = make_doc_id(article["url"], i)
                ids.append(doc_id)
                texts.append(chunk)
                metadatas.append({
                    "title": article["title"],
                    "url": article["url"],
                    "source": article["source"],
                    "published": article["published"],
                    "chunk_index": i,
                })

            try:
                collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
                total_chunks += len(chunks)
            except Exception as e:
                console.print(f"\n[red]  ✗ Upsert failed for {article['url']}: {e}[/red]")

            progress.advance(task)
            time.sleep(0.1)  # polite pause between HTTP requests

    return total_chunks


# ── Main Entry Point ───────────────────────────────────────────────────────────

def run_ingestion():
    console.rule("[bold cyan]AI News RAG — Ingestion[/bold cyan]")
    console.print(f"[dim]Embedding model : {EMBED_MODEL}[/dim]")
    console.print(f"[dim]Vector store    : {CHROMA_DB_DIR}[/dim]\n")

    collection = get_collection()

    all_articles = []
    for feed_cfg in RSS_FEEDS:
        console.print(f"[yellow]📡 Fetching:[/yellow] {feed_cfg['name']}")
        articles = parse_feed(feed_cfg)
        console.print(f"   → {len(articles)} articles found")
        all_articles.extend(articles)

    console.print(f"\n[green]Total articles fetched: {len(all_articles)}[/green]\n")

    if not all_articles:
        console.print("[red]No articles found. Check your internet connection.[/red]")
        return

    chunks_stored = store_articles(all_articles, collection)

    console.print(
        f"\n[bold green]✓ Done![/bold green] "
        f"Stored [cyan]{chunks_stored}[/cyan] chunks from "
        f"[cyan]{len(all_articles)}[/cyan] articles into ChromaDB.\n"
    )


if __name__ == "__main__":
    run_ingestion()
