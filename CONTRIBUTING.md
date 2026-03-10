# Contributing to LUFA_OpenSource_RAG

Thank you for your interest in contributing to the Bilingual RAG System!

## Development Setup

1. Fork and clone the repository
2. Run the bootstrap script to set up your environment:
   ```bash
   ./bootstrap.sh
   ```
3. Activate the virtual environment:
   ```bash
   conda activate LUFA_OpenSource_RAG
   ```

## Code Structure

- `src/ingestion.py` - Document processing and indexing
- `src/rag_engine.py` - RAG query engine
- `src/app.py` - Streamlit web interface
- `data/` - Document storage
