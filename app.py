"""
app.py — Streamlit web UI for the AI News RAG system.

Run with:
    streamlit run app.py
"""

import time
import threading

import streamlit as st

from config import LLM_MODEL, EMBED_MODEL, TOP_K_RESULTS, RSS_FEEDS
from ingest import run_ingestion
from rag_pipeline import ask, get_collection


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI News RAG",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-top: 0;
    }
    .source-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 3px solid #667eea;
    }
    .answer-box {
        background: #0d1117;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #30363d;
        font-size: 1rem;
        line-height: 1.7;
    }
    .badge {
        display: inline-block;
        background: #667eea22;
        color: #667eea;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🤖 AI News RAG</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ask anything about the latest AI news — '
    'powered by local Ollama models</p>',
    unsafe_allow_html=True,
)
st.divider()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown(f"**LLM model:** `{LLM_MODEL}`")
    st.markdown(f"**Embedding model:** `{EMBED_MODEL}`")
    st.markdown(f"**Top-K chunks:** `{TOP_K_RESULTS}`")

    st.divider()
    st.subheader("📡 News Sources")
    for feed in RSS_FEEDS:
        st.markdown(f"• {feed['name']}")

    st.divider()
    st.subheader("🔄 Data Refresh")
    if st.button("🚀 Fetch & Index Latest News", use_container_width=True):
        with st.spinner("Ingesting news… this may take a few minutes."):
            # Run ingestion in the current thread (simple approach)
            import io
            from contextlib import redirect_stdout

            run_ingestion()
        st.success("✅ News indexed successfully!")

    st.caption(
        "Tip: Run `python main.py --ingest` once before opening this UI "
        "to pre-populate the database."
    )

    # Show DB stats
    st.divider()
    st.subheader("📊 Database Stats")
    try:
        col = get_collection()
        count = col.count()
        st.metric("Chunks in DB", count)
    except Exception:
        st.warning("DB not yet initialised — run ingestion first.")


# ── Example queries ────────────────────────────────────────────────────────────
EXAMPLES = [
    "What are the latest AI news today?",
    "What has OpenAI or Google announced recently?",
    "Any new AI regulations or government policies?",
    "Latest breakthroughs in large language models?",
    "AI chip or hardware news?",
    "Which AI startups have raised funding recently?",
]

st.subheader("💬 Ask a question")

# Quick-fill buttons
cols = st.columns(3)
for i, example in enumerate(EXAMPLES):
    if cols[i % 3].button(example, key=f"ex_{i}", use_container_width=True):
        st.session_state["query_input"] = example

# Text input
query = st.text_input(
    "Your question:",
    key="query_input",
    placeholder="What are the latest developments in AI?",
)

ask_btn = st.button("🔍 Ask", type="primary", use_container_width=False)

# ── Answer ─────────────────────────────────────────────────────────────────────
if ask_btn and query:
    st.divider()
    st.subheader("🤖 Answer")

    answer_placeholder = st.empty()
    sources_placeholder = st.empty()
    status = st.status("Retrieving relevant news…", expanded=True)

    # Collect streamed answer
    full_answer = ""
    with status:
        st.write("🔎 Querying vector database…")
        result = ask(query, stream=False)   # non-streaming for Streamlit
        full_answer = result["answer"]
        sources = result["sources"]
        st.write(f"✅ Retrieved {len(result['chunks'])} chunks from {len(sources)} articles.")
    status.update(label="Done!", state="complete")

    # Display answer
    answer_placeholder.markdown(
        f'<div class="answer-box">{full_answer}</div>',
        unsafe_allow_html=True,
    )

    # Display sources
    if sources:
        st.subheader(f"📎 Sources ({len(sources)})")
        for i, src in enumerate(sources, 1):
            with st.container():
                st.markdown(
                    f"""<div class="source-card">
                        <strong>[{i}] {src['title']}</strong><br>
                        <span class="badge">{src['source']}</span>
                        <span class="badge">📅 {src.get('published','')[:10]}</span><br>
                        <a href="{src['url']}" target="_blank" style="color:#667eea;font-size:0.85rem;">
                            {src['url'][:90]}{'…' if len(src['url'])>90 else ''}
                        </a>
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No sources found. Make sure you have ingested news first.")

elif ask_btn and not query:
    st.warning("Please enter a question.")
