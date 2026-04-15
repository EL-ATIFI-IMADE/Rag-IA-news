"""
config.py — Central configuration for the AI News RAG system.
Edit this file to customize models, sources, and chunking settings.
"""

# ── Ollama Models ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"

# LLM used for answering questions (choose one you have pulled)
# Options: "llama3", "mistral", "phi3", "gemma2", "llama3.2"
LLM_MODEL = "llama3.2"

# Embedding model for vectorising text chunks
# "nomic-embed-text" is small, fast, and great for RAG
EMBED_MODEL = "nomic-embed-text"

# ── Vector Database ────────────────────────────────────────────────────────────
CHROMA_DB_DIR = "./chroma_db"          # Where ChromaDB stores data on disk
COLLECTION_NAME = "ai_news"           # Name of the ChromaDB collection

# ── Text Chunking ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 800        # Characters per chunk (not tokens — simpler & faster)
CHUNK_OVERLAP = 150     # Overlap between consecutive chunks to preserve context

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_RESULTS = 5       # Number of chunks to retrieve per query

# ── Ingestion ─────────────────────────────────────────────────────────────────
MAX_ARTICLES_PER_FEED = 15    # Cap per RSS feed to avoid overloading
REQUEST_TIMEOUT = 10          # Seconds before HTTP request times out

# ── RSS Feed Sources ──────────────────────────────────────────────────────────
RSS_FEEDS = [
    {
        "name": "NY Times Technology",
        "url": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    },
    {
        "name": "VentureBeat AI",
        "url": "https://feeds.feedburner.com/venturebeat/SZYF",
    },
    {
        "name": "The Verge – AI",
        "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    },
    {
        "name": "MIT Technology Review",
        "url": "https://www.technologyreview.com/feed/",
    },
    {
        "name": "Wired AI",
        "url": "https://www.wired.com/feed/tag/ai/latest/rss",
    },
    {
        "name": "TechCrunch AI",
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
    },
]
