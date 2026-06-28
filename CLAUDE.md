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

# Model proxy — Gemini & Claude CLIs via OpenAI-compatible API (port 9090)
# No API keys needed — uses your account subscriptions via installed CLIs
# Gemini goes through antigravity (Google Antigravity IDE with built-in Gemini)
# Set GEMINI_FALLBACK_CLI=1 to use headless gemini CLI instead of antigravity
python src/model_proxy.py
MODEL_PROXY_PORT=8080 python src/model_proxy.py  # custom port
```

**API endpoints** (after `python src/api.py`):
- `GET  /health` — liveness + loaded models
- `POST /query` — single-pass RAG
- `POST /agentic-query` — 3-pass agentic loop (60–180s, set high timeout)
- `POST /copilot-query` — frontier model generation (needs `GITHUB_TOKEN`)

**Model proxy endpoints** (after `python src/model_proxy.py` on port 9090):
- `GET  /health` — liveness + CLI availability
- `GET  /v1/models` — list available Gemini/Claude models
- `POST /v1/chat/completions` — OpenAI-compatible chat completion (routes by model prefix via CLI sub-process)
- Requires `claude` CLI (Claude Code) and/or `antigravity` (Google Antigravity with built-in Gemini) installed and authenticated — uses your standard account subscription, no API keys needed
- Set `GEMINI_FALLBACK_CLI=1` to use the headless `gemini` CLI (`npm i -g @google/gemini-cli`) instead of antigravity for non-GUI environments

**MODEL_API_AUTH**: All model API credentials are resolved from `config/config.yaml` `model_api_auth` section (per-model `api_key`/`api_base` entries, with `default` fallback). Runtime overrides via env vars `MODEL_API_KEY_<NAME>` / `MODEL_API_BASE_<NAME>`, or `LUFA_*` env vars for any config key. See `src/model_api_auth.py` and `src/config_loader.py`.

Config lives in `config/config.yaml`. All modules load settings via `src/config_loader.py` (dotted-key access: `cfg("models.llm.name")`). Legacy `config.py` / `config_template.py` are no longer imported.

## Architecture

### Data flow (ingestion → retrieval → generation)

```
data/english/*.pdf ─ 
                     ├─→ ClauseBoundaryChunker (clause_chunker.py) ─→ TextNode list
data/french/*.pdf  ─                                                      │
                                                                          ▼
                                                          VectorStoreIndex (LlamaIndex)
                                                                          │
                                                          ChromaDB PersistentClient
                                                          db/chroma_db / collection "multilingual_docs"
                                                                          │
user query ──→ BilingualRAGEngine (rag_engine.py) ───────────────────────┘
                     │
                     ├─ language_detector.py → detect 'en' / 'fr' / other
                     ├─ translator.py       → non-EN/FR queries translated to EN, answer translated back
                     ├─ query_handler.py    → prompt augmentation (auto-appends year range if missing)
                     ├─ query_rewriter.py   → retrieval-optimized rewrite (agentic only)
                     ├─ _retrieve_nodes()   → hybrid RRF: dense (cosine) + sparse (BM25Okapi)
                     └─ reflector.py        → "GROUNDED" / "UNGROUNDED" check → re-retrieval loop
```

### Key design decisions

- **Clause-boundary chunking** (not fixed-size): `ClauseBoundaryChunker` splits PDFs at ARTICLE/NUMBERED-CLAUSE headers using regex, merges short clauses (<30 tokens), splits long ones (>512 tokens) at sentence boundaries. Each chunk gets metadata: `article_number`, `clause_id`, `section_title`, `language`, `page_no`, `end_year`, `recency_weight`.
- **Hybrid retrieval via RRF**: `_retrieve_nodes()` sorts dense vector results by recency within tie-buckets (0.02 threshold), then fuses with BM25 scores via Reciprocal Rank Fusion (k=60). Direct ChromaDB access to avoid LlamaIndex's ZeroDivisionError.
- **Recency weighting**: `recency_reranker.py` computes weights from filenames' years (1998–2026 range), linearly scaled to 0.30–1.00, used as tie-breaker so newer clauses win over identical older ones.
- **Translation bridge**: Only non-EN/FR queries go through translation. The agentic pipeline operates entirely in English when translated, then translates the final answer back.
- **Agentic loop**: Up to `max_retries` passes. Each pass rewrites the query (with hinting on retries 2+), retrieves with widening top_k, generates, then reflects. Stops early if `reflect()` returns GROUNDED.
- **Copilot/frontier mode**: Local ChromaDB retrieval + OpenAI-compatible GitHub Models generation (`copilot_engine.py`). Requires `GITHUB_TOKEN` env var or `config.yaml:copilot.github_token`.

### Evaluation pipeline

```
tests/combined_test_data_and_ground_truth.csv
        │
        ▼
run_simulation.py  (row-by-row, crash-resumable, appends to lufa_out_data.csv)
        │
        ▼
evaluate.py
  ├─ generation metrics: token F1, BLEU, ROUGE-1/2/L, METEOR
  ├─ retrieval metrics: MRR, NDCG@5, Recall@1/3/5
  ├─ LLM-as-judge (Ollama): answer_relevance, faithfulness, context_precision
  └─ repair hooks: repair_lufa_out.py fixes missing source IDs inline
        │
        ▼
tests/evaluation_results.csv  +  dashboard/index.html (Chart.js)
```

`evaluate.py` supports resumption (reads existing `evaluation_results.csv`), runs inline simulation for questions not yet in `lufa_out_data.csv`, and regenerates the dashboard after every row.

## File state notes (branch: test-data)

The git status shows many files as deleted from the working tree. Active files in `src/` and relevant to normal operation:
- **Active**: `app.py`, `api.py`, `rag_engine.py`, `ingestion.py`, `clause_chunker.py`, `translator.py`, `query_handler.py`, `language_detector.py`, `query_rewriter.py`, `reflector.py`, `recency_reranker.py`, `copilot_engine.py`, `evaluate.py`, `run_simulation.py`, `find_ground_truth.py`, `generate_test_question.py`, `repair_lufa_out.py`, `repair_evaluation.py`, `get_lufa_stats.py`, `pdf_ocr_converter.py`, `bilingual_pdf_splitter.py`, `side_by_side_clause_chunker.py`, `config_loader.py`, `model_api_auth.py`, `model_proxy.py`
- **Deleted from working tree** (still in git history): `test_basic.py`, `test_integration.py`, `test_reflector.py`, `bootstrap-backup.sh`, `pdf_ocr_converter.py` (old location), `src/ingestion.py` (old version), `src/bilingual_pdf_splitter.py` (old version)

PDFs under `data/english/` and `data/french/` are also deleted from the working tree — they must be re-added before ingestion.

## Conventions

- All `src/` modules insert the parent dir onto `sys.path` so they can be run directly (`python src/ingestion.py`).
- `TextNode.excluded_embed_metadata_keys` strips `token_count`, `chunk_index`, `doc_source`, `page_no`, `end_year`, `recency_weight` from embedding — only `article_number`, `clause_id`, `section_title`, `language` go into the vector.
- The `DEFAULT_AGREEMENT_YEAR_RANGE` constant ("2020 - 2025") is auto-appended to queries that lack a 4-digit year, improving retrieval precision for time-scoped questions.
- `run_simulation.py` and `evaluate.py` both write row-by-row with `mode="a"` so crashes are never catastrophic — just re-run and they resume.
