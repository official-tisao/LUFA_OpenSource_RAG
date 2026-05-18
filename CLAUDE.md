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
