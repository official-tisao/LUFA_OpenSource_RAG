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
                    llm_model="mistral:7b",
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
            - LLM: mistral:7b
            - Embeddings: nomic-embed-text-v2-moe

            **Database:**
            - ChromaDB (db/chroma_db)

            **Retrieval:**
            - Top 5 chunks
            - 0.7 similarity threshold
            - 5 retries on failure
            - hybrid retrieval by (dense vector+rank-bm25)
            - fused via Reciprocal Rank Fusion(RRF)
            - Clause-based chunking
            """
        )

        with st.expander("Advanced Settings"):
            top_k = st.slider("Number of retrieved documents", 1, 10, 5)
            show_sources = st.checkbox("Show source documents", value=False)

        st.divider()

        st.subheader("📖 Instructions")
        if language == "English":
            st.markdown(
                """
                1. Make sure Ollama is running
                2. Ensure documents are ingested
                3. Type your question below
                4. Get your answer!

                **Example queries:**
                - "What are the office hours requirements?"
                - "What is the policy on academic freedom?"
                """
            )
        else:
            st.markdown(
                """
                1. Assurez-vous qu'Ollama fonctionne
                2. Assurez-vous que les documents sont ingérés
                3. Tapez votre question ci-dessous
                4. Obtenez votre réponse!

                **Exemples de questions:**
                - "Quelles sont les exigences concernant les heures de bureau?"
                - "Quelle est la politique sur la liberté académique?"
                """
            )

        if st.button("🔄 Reset Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Main content ─────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Chat Interface")

        rag_engine = load_rag_engine()
        if rag_engine is None:
            st.warning("⚠️ RAG engine not loaded. Please check the configuration.")
            return

        if hasattr(rag_engine, "set_similarity_top_k"):
            rag_engine.set_similarity_top_k(top_k)

        lang_names = _get_language_names()

        # ── Render chat history ───────────────────────────────────────────────
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["query"])

            with st.chat_message("assistant"):
                st.write(chat["response"])

                # Translation expander (history)
                if chat.get("translation_applied"):
                    _render_translation_expander(
                        orig_lang=chat.get("original_language", "unknown"),
                        lang_names=lang_names,
                        original_query=chat.get("original_query", chat["query"]),
                        translated_query=chat.get("translated_query", ""),
                        english_response=chat.get("english_response", ""),
                        final_response=chat["response"],
                    )

                # Language caption
                lang_emoji = (
                    "🇬🇧" if chat.get("language") == "en"
                    else "🇫🇷" if chat.get("language") == "fr"
                    else "🌐"
                )
                st.caption(f"{lang_emoji} Detected language: {chat.get('language', 'unknown')}")

                # Sources expander (history)
                if show_sources and chat.get("sources"):
                    _render_sources_expander(chat["sources"])

        # ── New query input ───────────────────────────────────────────────────
        placeholder = (
            "Ask a question / Posez une question"
            if language == "English"
            else "Posez une question / Ask a question"
        )
        query = st.chat_input(placeholder)

        if query:
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = rag_engine.query(query, return_sources=show_sources)

                        # ── Primary response ──────────────────────────────────
                        st.write(result["response"])

                        # ── Translation expander (new query) ──────────────────
                        if result.get("translation_applied"):
                            _render_translation_expander(
                                orig_lang=result.get("original_language", "unknown"),
                                lang_names=lang_names,
                                original_query=result.get("original_query", query),
                                translated_query=result.get("translated_query", ""),
                                english_response=result.get("english_response", ""),
                                final_response=result["response"],
                            )

                        # ── Language caption ──────────────────────────────────
                        detected = result.get("detected_language", "unknown")
                        lang_emoji = (
                            "🇬🇧" if detected == "en"
                            else "🇫🇷" if detected == "fr"
                            else "🌐"
                        )
                        st.caption(f"{lang_emoji} Detected language: {detected}")

                        # ── Sources expander (new query) ──────────────────────
                        if show_sources and result.get("sources"):
                            _render_sources_expander(result["sources"])

                        # ── Persist to session state ──────────────────────────
                        st.session_state.chat_history.append({
                            "query": query,
                            "response": result["response"],
                            "language": detected,
                            "sources": result.get("sources", []),
                            "translation_applied": result.get("translation_applied", False),
                            "original_language": result.get("original_language"),
                            "original_query": result.get("original_query"),
                            "translated_query": result.get("translated_query"),
                            "english_response": result.get("english_response"),
                        })

                    except Exception as e:
                        st.error(f"Error processing query: {e}")
                        st.info(
                            "Make sure Ollama is running and the models are available."
                        )

    # ── Statistics panel ──────────────────────────────────────────────────────
    with col2:
        st.header("📊 Statistics")

        total = len(st.session_state.chat_history)
        st.metric("Total Questions Asked", total)

        if total:
            languages = [c.get("language", "unknown") for c in st.session_state.chat_history]
            en_count = languages.count("en")
            fr_count = languages.count("fr")
            other_count = total - en_count - fr_count

            st.subheader("Language Distribution")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("🇬🇧 EN", en_count)
            with col_b:
                st.metric("🇫🇷 FR", fr_count)
            with col_c:
                st.metric("🌐 Other", other_count)

            # Translation usage metric
            translated_count = sum(
                1 for c in st.session_state.chat_history if c.get("translation_applied")
            )
            if translated_count:
                st.metric("🌐 Queries Translated", translated_count)

            st.subheader("Recent Queries")
            for chat in reversed(st.session_state.chat_history[-5:]):
                lang = chat.get("language", "unknown")
                lang_emoji = "🇬🇧" if lang == "en" else "🇫🇷" if lang == "fr" else "🌐"
                preview = chat["query"][:50]
                ellipsis = "..." if len(chat["query"]) > 50 else ""
                with st.expander(f"{lang_emoji} {preview}{ellipsis}"):
                    st.write(f"**Query:** {chat['query']}")
                    if chat.get("translation_applied"):
                        orig_lang = chat.get("original_language", "?")
                        st.write(f"**Translated from:** {orig_lang.upper()}")
                    st.write(f"**Response:** {chat['response'][:200]}...")


if __name__ == "__main__":
    main()