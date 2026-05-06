"""
Streamlit application for the bilingual RAG system.
Provides a user-friendly interface for querying documents in English and French.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_engine import BilingualRAGEngine


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def load_rag_engine():
    """Load or initialize the RAG engine."""
    if st.session_state.rag_engine is None:
        with st.spinner("Loading RAG engine... This may take a moment."):
            try:
                st.session_state.rag_engine = BilingualRAGEngine(
                    db_path="db/chroma_db",
                    llm_model="llama3.2:3b-instruct-q4_K_M",
                    embedding_model="nomic-embed-text-v2-moe",
                    similarity_top_k=5,
                )
                st.success("RAG engine loaded successfully!")
            except Exception as e:
                st.error(f"Error loading RAG engine: {e}")
                st.info(
                    "Make sure you have run the ingestion script and Ollama is running."
                )
                return None
    return st.session_state.rag_engine


def _get_language_names():
    """Safely import LANGUAGE_NAMES from translator module."""
    try:
        from translator import LANGUAGE_NAMES
        return LANGUAGE_NAMES
    except Exception:
        return {}


def _render_translation_expander(orig_lang, lang_names, original_query,
                                  translated_query, english_response, final_response):
    """Render the translation detail expander inside an assistant message."""
    lang_name = lang_names.get(orig_lang, orig_lang.upper())
    with st.expander(f"🌐 Translation Applied — {lang_name} detected"):
        st.write(f"**Original query ({lang_name}):** {original_query}")
        st.write(f"**Translated to English:** {translated_query}")
        st.write(f"**English answer:** {english_response}")
        st.write(f"**Final answer translated back to {lang_name}:**")
        st.success(final_response)


def _render_sources_expander(sources):
    """Render the source documents expander."""
    with st.expander("📚 View Sources"):
        for j, source in enumerate(sources):
            st.markdown(f"**Source {j + 1}** (Score: {source['score']:.3f})")
            st.markdown(
                f"*Language: {source['metadata'].get('language', 'unknown')}*"
            )
            st.text(source["text"])
            st.divider()


def main():
    """Main application function."""
    st.set_page_config(
        page_title="LUFA Collective Agreement - Bilingual RAG",
        page_icon="🌍",
        layout="wide",
    )

    initialize_session_state()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🌍 LUFA Collective Agreement - Bilingual RAG (EN/FR)")
    st.markdown(
        """
        Ask questions about the **Laurentian University Faculty Association collective agreement**
        in **English** or **French**, and get answers from the document collection.

        **Features:**
        - 🔍 Cross-lingual retrieval with nomic-embed-text-v2-moe
        - 🌐 Auto-detect query language (EN / FR / other languages via translation)
        - 💬 Responds in your language
        - 📚 Top 5 most relevant chunks
        """
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        language = st.selectbox(
            "Language / Langue", ["English", "Français"], key="ui_language"
        )

        st.subheader("System Information")
        st.info(
            """
            **Models:**
            - LLM: llama3.2:3b-instruct-q4_K_M
            - Embeddings: nomic-embed-text-v2-moe

            **Database:**
            - ChromaDB (db/chroma_db)

            **Retrieval:**
            - Top 5 chunks
            - 0.7 similarity threshold
            - 5 retries on failure
            - hybrid retrieval by (dense vector+rank-bm25)
