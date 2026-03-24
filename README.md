ollama pull mxbai-embed-large

# Optional: Frontier model simulators (for local testing)
ollama pull mistral
```


**Multilingual Embedding Model**:[^1]

1. **nomic-embed-text-v2-moe** : Supports 100+ languages with strong French/English performance[^2]

```bash
ollama pull nomic-embed-text-v2-moe
```


### Phase 3: Enhanced Project Structure

```
LUFA_OpenSource_RAG/
│
├── src/                           # Core Python modules
│   ├── app.py                     # Streamlit web interface (see below)
│   ├── api.py                     # FastAPI REST server
│   ├── rag_engine.py              # BilingualRAGEngine orchestrator
│   ├── ingestion.py               # Document loading & indexing
│   ├── clause_chunker.py          # Clause-aware PDF parsing
│   ├── side_by_side_clause_chunker.py  # Bilingual PDF extraction
│   ├── query_handler.py           # Query language detection
│   ├── language_detector.py       # Language identification utilities
│   ├── query_rewriter.py          # Query expansion (for agentic mode)
│   ├── translator.py              # Inter-language translation
│   ├── reflector.py               # Self-reflection step (agentic)
│   ├── recency_reranker.py        # Time-weighted ranking
│   ├── copilot_engine.py          # GitHub Models integration
│   ├── run_simulation.py          # Batch evaluation runner
│   ├── evaluate.py                # RAGAS metrics + dashboard
│   ├── find_ground_truth.py       # Ground truth linking
│   ├── generate_test_question.py  # Test question generation
│   ├── pdf_ocr_converter.py       # OCR for scanned PDFs
│   └── test_*.py                  # Unit tests
│
├── data/                          # Document corpus
│   ├── english/                   # English PDFs
│   ├── french/                    # French PDFs
│   ├── english_and_french/        # Bilingual side-by-side PDFs
│   ├── processed/                 # Extracted text cache
│   └── metadata.json              # Document tracking
│
├── db/
│   └── chroma_db/                 # Vector store (persistent)
│
├── config/
│   ├── config.yaml                # Main configuration
│   └── config_template.py         # Configuration template
│
├── tests/
│   ├── combined_test_data.csv     # Evaluation test set
│   └── *.json                     # Ground truth labels
│
├── dashboard/                     # Evaluation results
│   └── index.html                 # Interactive results dashboard
│
├── requirements.txt               # Python dependencies
├── bootstrap.sh                   # Setup automation script
├── ARCHITECTURE.md                # Detailed architecture docs
├── QUICKSTART.md                  # 5-minute getting started
├── TROUBLESHOOTING.md             # Common issues & fixes
└── README.md                      # This file
```


### Phase 4: Updated Dependencies

```
llama-index
llama-index-llms-ollama
