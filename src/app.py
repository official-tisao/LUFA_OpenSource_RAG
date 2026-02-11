"""
Streamlit application for the bilingual RAG system.
Provides a user-friendly interface for querying documents in English and French.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_engine import BilingualRAGEngine


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def load_rag_engine():
    """Load or initialize the RAG engine."""
    if st.session_state.rag_engine is None:
        with st.spinner("Loading RAG engine... This may take a moment."):
            try:
                st.session_state.rag_engine = BilingualRAGEngine(
                    db_path="db/chroma_db",
                    llm_model="llama3.2:latest",
                    embedding_model="bge-m3:latest",
                    similarity_top_k=3
                )
                st.success("RAG engine loaded successfully!")
            except Exception as e:
                st.error(f"Error loading RAG engine: {e}")
                st.info("Make sure you have run the ingestion script and Ollama is running.")
                return None
    return st.session_state.rag_engine


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Bilingual RAG System",
        page_icon="🌍",
        layout="wide",
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🌍 Bilingual RAG System (EN/FR)")
    st.markdown("""
    Welcome to the bilingual RAG system! Ask questions in **English** or **French**, 
    and get answers from your document collection.
    
    **Features:**
    - 🔍 Cross-lingual retrieval
    - 🌐 Auto-detect query language
    - 💬 Responds in your language
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Display system info
        st.subheader("System Information")
        st.info("""
        **Models:**
        - LLM: llama3.2:latest
        - Embeddings: bge-m3:latest
        
        **Database:**
        - ChromaDB (db/chroma_db)
        """)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            top_k = st.slider("Number of retrieved documents", 1, 10, 3)
            show_sources = st.checkbox("Show source documents", value=False)
        
        st.divider()
        
        # Instructions
        st.subheader("📖 Instructions")
        st.markdown("""
        1. Make sure Ollama is running
        2. Ensure documents are ingested
        3. Type your question below
        4. Get your answer!
        
        **Example queries:**
        - *English:* "What is this about?"
        - *French:* "De quoi s'agit-il?"
        """)
        
        # Reset button
        if st.button("🔄 Reset Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Load RAG engine
        rag_engine = load_rag_engine()
        
        if rag_engine is None:
            st.warning("⚠️ RAG engine not loaded. Please check the configuration.")
            return
        
        # Update top_k if changed
        if hasattr(rag_engine, 'set_similarity_top_k'):
            rag_engine.set_similarity_top_k(top_k)
        
        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(chat['query'])
            with st.chat_message("assistant"):
                st.write(chat['response'])
                if show_sources and 'sources' in chat:
                    with st.expander("📚 View Sources"):
                        for j, source in enumerate(chat['sources']):
                            st.markdown(f"**Source {j+1}** (Score: {source['score']:.3f})")
                            st.markdown(f"*Language: {source['metadata'].get('language', 'unknown')}*")
                            st.text(source['text'])
                            st.divider()
        
        # Query input
        query = st.chat_input("Ask a question in English or French...")
        
        if query:
            # Add user query to chat
            with st.chat_message("user"):
                st.write(query)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = rag_engine.query(query, return_sources=show_sources)
                        st.write(result['response'])
                        
                        # Show language detection
                        lang_emoji = "🇬🇧" if result['detected_language'] == 'en' else "🇫🇷"
                        st.caption(f"{lang_emoji} Detected language: {result['detected_language']}")
                        
                        # Show sources if enabled
                        if show_sources and 'sources' in result:
                            with st.expander("📚 View Sources"):
                                for j, source in enumerate(result['sources']):
                                    st.markdown(f"**Source {j+1}** (Score: {source['score']:.3f})")
                                    st.markdown(f"*Language: {source['metadata'].get('language', 'unknown')}*")
                                    st.text(source['text'])
                                    st.divider()
                        
                        # Save to chat history
                        st.session_state.chat_history.append({
                            'query': query,
                            'response': result['response'],
                            'language': result['detected_language'],
                            'sources': result.get('sources', [])
                        })
                    
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
                        st.info("Make sure Ollama is running and the models are available.")
    
    with col2:
        st.header("📊 Statistics")
        
        # Chat statistics
        st.metric("Total Questions Asked", len(st.session_state.chat_history))
        
        if st.session_state.chat_history:
            # Language distribution
            languages = [chat['language'] for chat in st.session_state.chat_history]
            en_count = languages.count('en')
            fr_count = languages.count('fr')
            
            st.subheader("Language Distribution")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🇬🇧 English", en_count)
            with col_b:
                st.metric("🇫🇷 French", fr_count)
        
        # Recent queries
        if st.session_state.chat_history:
            st.subheader("Recent Queries")
            for chat in reversed(st.session_state.chat_history[-5:]):
                lang_emoji = "🇬🇧" if chat['language'] == 'en' else "🇫🇷"
                with st.expander(f"{lang_emoji} {chat['query'][:50]}..."):
                    st.write(f"**Query:** {chat['query']}")
                    st.write(f"**Response:** {chat['response'][:200]}...")


if __name__ == "__main__":
    main()
