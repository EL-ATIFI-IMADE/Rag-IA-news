# 🤖 AI News RAG — Local, Free, Fully Private

A complete **Retrieval-Augmented Generation** system that fetches the latest AI news from RSS feeds, indexes it locally with ChromaDB, and lets you ask natural-language questions — all powered by **Ollama** (no cloud APIs, no cost).

---

## 📁 Project Structure

```
ai-news-rag/
├── config.py          # All settings (models, feeds, chunking)
├── ingest.py          # Fetch → clean → chunk → embed → store
├── rag_pipeline.py    # Retrieve → prompt → LLM → answer
├── main.py            # Interactive CLI
├── app.py             # Streamlit web UI
├── requirements.txt
└── setup.sh           # One-shot setup script
```

---

## ⚙️ Prerequisites

| Tool | Install |
|------|---------|
| Python ≥ 3.10 | [python.org](https://python.org) |
| Ollama | [ollama.com](https://ollama.com) |

---

## 🚀 Installation (Step-by-Step)

### Step 1 — Install & start Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama serve          # start the server (keep this terminal open)
```

### Step 2 — Pull the required models

```bash
ollama pull llama3.2          # LLM for answering (~2 GB)
ollama pull nomic-embed-text  # Embedding model (~300 MB)
```

> **Alternative LLMs** (if you have less RAM):
> - `ollama pull phi3` (lighter, ~2.3 GB)
> - `ollama pull mistral` (~4 GB)
> - Edit `LLM_MODEL` in `config.py` to match.

### Step 3 — Set up Python environment

```bash
cd ai-news-rag
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4 — Ingest the latest AI news

```bash
python ingest.py
```

This will:
- Fetch articles from 6 RSS feeds
- Download full article text
- Split into 800-character chunks
- Generate embeddings via Ollama
- Store everything in ChromaDB (./chroma_db/)

Takes **~3–8 minutes** depending on your machine.

---

## 💻 How to Run

### Option A — Interactive CLI

```bash
python main.py
```

```
You ▸ What are the latest AI news today?
```

Flags:
```bash
python main.py --ingest         # refresh news then start Q&A
python main.py --ingest-only    # just refresh, no Q&A
python main.py --auto           # auto-refresh every 60 min
```

### Option B — Streamlit Web UI

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🧪 Example Queries

```
What are the latest AI news today?
What has OpenAI or Google announced recently?
What are the recent breakthroughs in large language models?
Are there any new AI regulations or government policies?
What AI chip or hardware news is there?
Which AI startups have raised funding recently?
What is the latest on autonomous agents?
Any news about AI safety or alignment?
```

---

## 🔧 Configuration

Edit `config.py` to customise:

```python
LLM_MODEL = "llama3.2"          # Change to mistral, phi3, etc.
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800                 # Characters per chunk
TOP_K_RESULTS = 5                # Chunks retrieved per query
MAX_ARTICLES_PER_FEED = 15       # Articles per RSS source
```

### Adding custom RSS feeds

```python
RSS_FEEDS = [
    {"name": "My Source", "url": "https://example.com/feed.xml"},
    # ... existing feeds
]
```

---

## 🏗️ Architecture

```
User question
     │
     ▼
[Ollama Embeddings]  ← nomic-embed-text
     │
     ▼
[ChromaDB Query]  →  Top-5 relevant chunks
     │
     ▼
[Prompt Builder]  →  Context + Question
     │
     ▼
[Ollama LLM]  ←  llama3.2 / mistral / phi3
     │
     ▼
Answer + Sources
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot connect to Ollama` | Run `ollama serve` in a terminal |
| `Model not found` | Run `ollama pull llama3.2` |
| `No relevant articles` | Run `python ingest.py` first |
| Slow responses | Use a smaller model like `phi3` |
| Out of memory | Reduce `MAX_ARTICLES_PER_FEED` in config.py |

---

## 📝 License

MIT — free to use, modify, and distribute.
