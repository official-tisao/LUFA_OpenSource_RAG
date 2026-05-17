# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bilingual (EN/FR) agentic RAG system for Laurentian University Faculty Association collective agreements. Combines Ollama local LLMs with optional GitHub frontier models, clause-aware PDF chunking, hybrid RRF retrieval, and a LlamaIndex + ChromaDB stack.

## Build & Run Commands

```bash
# Install deps (assume conda env already created)
pip install -r requirements.txt

# Start Ollama (prerequisite — must be running on :11434)
ollama serve
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull nomic-embed-text-v2-moe

# Ingest PDFs into ChromaDB
python src/ingestion.py

# Streamlit web UI
streamlit run src/app.py

# FastAPI REST server (port 8000)
python src/api.py

# Batch simulation (all test questions → tests/lufa_out_data.csv)
python src/run_simulation.py --mode local
python src/run_simulation.py --mode api            # requires running server
python src/run_simulation.py --mode frontier --model gpt-4o

# Evaluation metrics + dashboard
python src/evaluate.py
python src/evaluate.py --no_llm_judge  # skip Ollama judge (faster)

# Repair source IDs in simulation output
python src/repair_lufa_out.py

# Open dashboard
open dashboard/index.html
```

**API endpoints** (after `python src/api.py`):
- `GET  /health` — liveness + loaded models
- `POST /query` — single-pass RAG
- `POST /agentic-query` — 3-pass agentic loop (60–180s, set high timeout)
- `POST /copilot-query` — frontier model generation (needs `GITHUB_TOKEN`)

Config lives in `config/config.yaml` (or falls back to defaults in `src/config_template.py`).

## Architecture

### Data flow (ingestion → retrieval → generation)

```
data/english/*.pdf ─�
                     ├─→ ClauseBoundaryChunker (clause_chunker.py) ─→ TextNode list
data/french/*.pdf  ─�                                                     │
                                                                          ▼
                                                          VectorStoreIndex (LlamaIndex)
                                                                          │
                                                          ChromaDB PersistentClient
                                                          db/chroma_db / collection "multilingual_docs"
