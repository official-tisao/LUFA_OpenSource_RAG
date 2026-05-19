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
- **Active**: `app.py`, `api.py`, `rag_engine.py`, `ingestion.py`, `clause_chunker.py`, `translator.py`, `query_handler.py`, `language_detector.py`, `query_rewriter.py`, `reflector.py`, `recency_reranker.py`, `copilot_engine.py`, `evaluate.py`, `run_simulation.py`, `find_ground_truth.py`, `generate_test_question.py`, `repair_lufa_out.py`, `repair_evaluation.py`, `get_lufa_stats.py`, `pdf_ocr_converter.py`, `bilingual_pdf_splitter.py`, `side_by_side_clause_chunker.py`, `config_template.py`
- **Deleted from working tree** (still in git history): `test_basic.py`, `test_integration.py`, `test_reflector.py`, `bootstrap-backup.sh`, `pdf_ocr_converter.py` (old location), `src/ingestion.py` (old version), `src/bilingual_pdf_splitter.py` (old version)

PDFs under `data/english/` and `data/french/` are also deleted from the working tree — they must be re-added before ingestion.

## Conventions

- All `src/` modules insert the parent dir onto `sys.path` so they can be run directly (`python src/ingestion.py`).
- `TextNode.excluded_embed_metadata_keys` strips `token_count`, `chunk_index`, `doc_source`, `page_no`, `end_year`, `recency_weight` from embedding — only `article_number`, `clause_id`, `section_title`, `language` go into the vector.
- The `DEFAULT_AGREEMENT_YEAR_RANGE` constant ("2020 - 2025") is auto-appended to queries that lack a 4-digit year, improving retrieval precision for time-scoped questions.
- `run_simulation.py` and `evaluate.py` both write row-by-row with `mode="a"` so crashes are never catastrophic — just re-run and they resume.
