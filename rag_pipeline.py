"""
rag_pipeline.py — Retrieval-Augmented Generation pipeline.

Steps:
  1. Embed the user's question with Ollama (nomic-embed-text)
  2. Query ChromaDB for the most relevant chunks
  3. Build a prompt that includes those chunks as context
  4. Send the prompt to the Ollama LLM and stream the answer
  5. Return the answer + deduplicated source links
"""

from typing import Generator

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import requests
from rich.console import Console

from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    TOP_K_RESULTS,
)

console = Console()


# ── Vector Store ───────────────────────────────────────────────────────────────

def get_collection():
    """Open the existing ChromaDB collection (must have run ingest.py first)."""
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


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve(query: str, collection, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for the query.
    Returns a list of dicts: { text, title, url, source, published, distance }
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append({
            "text": doc,
            "title": meta.get("title", "Unknown"),
            "url": meta.get("url", ""),
            "source": meta.get("source", "Unknown"),
            "published": meta.get("published", ""),
            "distance": dist,
        })
    return chunks


# ── Prompt Building ────────────────────────────────────────────────────────────

def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Assemble a RAG prompt.  The LLM sees the retrieved context and must
    answer ONLY from that context.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i}] {chunk['source']} — {chunk['title']}\n"
            f"Published: {chunk['published']}\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are an AI news analyst. Use ONLY the provided context to answer the question.
If the context does not contain enough information, say so honestly.
Do not fabricate facts.

=== CONTEXT ===
{context}

=== QUESTION ===
{query}

=== INSTRUCTIONS ===
- Provide a clear, concise answer based on the context above.
- Highlight the most important/recent developments.
- If multiple sources cover the same event, synthesise them.
- End your answer with a brief bullet-list of key takeaways.

=== ANSWER ==="""
    return prompt


# ── LLM Query ─────────────────────────────────────────────────────────────────

def query_ollama(prompt: str) -> Generator[str, None, None]:
    """
    Stream a response from the local Ollama LLM.
    Yields text chunks as they arrive.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.3,     # Low temp → more factual
            "num_predict": 1024,    # Max tokens in the answer
        },
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    token = data.get("response", "")
                    yield token
                    if data.get("done"):
                        break
    except requests.exceptions.ConnectionError:
        yield (
            "\n\n[ERROR] Cannot connect to Ollama. "
            "Make sure it is running: `ollama serve`"
        )
    except Exception as e:
        yield f"\n\n[ERROR] {e}"


# ── Public API ─────────────────────────────────────────────────────────────────

def ask(query: str, stream: bool = True) -> dict:
    """
    Full RAG pipeline.

    Args:
        query:  The user's question.
        stream: If True, prints the answer token-by-token to stdout.

    Returns:
        {
          "answer":  str,
          "sources": [{"title", "url", "source", "published"}, …],
          "chunks":  list[dict]   # raw retrieved chunks (for debugging)
        }
    """
    collection = get_collection()

    # 1. Retrieve relevant chunks
    chunks = retrieve(query, collection)
    if not chunks:
        return {
            "answer": "No relevant articles found. Try running `python ingest.py` first.",
            "sources": [],
            "chunks": [],
        }

    # 2. Build prompt
    prompt = build_prompt(query, chunks)

    # 3. Generate answer (streaming)
    full_answer = ""
    if stream:
        for token in query_ollama(prompt):
            print(token, end="", flush=True)
            full_answer += token
        print()  # newline after streaming
    else:
        for token in query_ollama(prompt):
            full_answer += token

    # 4. Deduplicate sources
    seen_urls: set[str] = set()
    sources = []
    for chunk in chunks:
        if chunk["url"] not in seen_urls:
            seen_urls.add(chunk["url"])
            sources.append({
                "title": chunk["title"],
                "url": chunk["url"],
                "source": chunk["source"],
                "published": chunk["published"],
            })

    return {"answer": full_answer, "sources": sources, "chunks": chunks}


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = ask("What are the latest developments in AI today?")
    print("\n\n── Sources ──")
    for s in result["sources"]:
        print(f"  • {s['source']} — {s['title']}")
        print(f"    {s['url']}")
