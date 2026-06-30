# LUFA_OpenSource_RAG

A production-grade **Bilingual (English/French) Agentic RAG System** for querying the Laurentian University Faculty Association (LUFA) collective agreements. Combines local LLMs with frontier model support, clause-aware chunking, and multi-pass agentic retrieval for accurate legal document processing.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Core Python Modules](#core-python-modules)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [Evaluation & Testing](#evaluation--testing)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**LUFA_OpenSource_RAG** is an advanced retrieval-augmented generation (RAG) system purpose-built for university faculty collective agreements. The system:

- ✅ **Bilingual Support**: Native English/French query handling with automatic language detection
- ✅ **Clause-Aware Chunking**: Extracts semantic clause boundaries from PDFs (not naive fixed-size chunks)
- ✅ **Agentic Retrieval**: Multi-pass query rewriting, reflection, and re-retrieval for improved accuracy
- ✅ **Frontier Model Integration**: Optional use of GitHub Copilot models (GPT-5, Claude, Grok, Gemini) alongside local Ollama
- ✅ **Production API**: FastAPI REST interface with health checks, streaming, and timeout management
- ✅ **Evaluation Dashboard**: Automated RAGAS-style evaluation with interactive HTML dashboard
- ✅ **Local-First**: No cloud dependencies—runs entirely on-premise with Ollama + ChromaDB

**Target Users**: Legal researchers, faculty representatives, HR departments, and university administrators needing semantic search over bilingual policy documents.

---

## 🏗️ System Architecture
┌─────────────────────────────────────────────────────────────────┐ 
│ USER INTERFACES │ ├──────────────────────┬──────────────────┬──────────────────────┤ 
│ Streamlit Web UI │ REST API (Port │ CLI / Python SDK │ │ (app.py) │ 8000, api.py) │ 
│ └──────────────────────┴──────────────────┴──────────────────────┘ 
│ ┌───────────┼────────────┐ │ │ │ ┌───────▼─────┐ ┌──▼────────┐ ┌─▼──────────┐ 
│ Standard │ │ Agentic │ │ Frontier │ │ RAG │ │ RAG │ │ Model │ │ (1-pass) │ │ (3-pass) │ │ (GitHub) │
└───────┬─────┘ └──┬────────┘ └─┬──────────┘ │ │ │ └───────────┼────────────┘ │ 
┌───────────────────▼────────────────────┐ │ BilingualRAGEngine (rag_engine.py) │ │ - Language detection │ 
│ - Query translation/rewriting │ │ - Reflection/re-retrieval │ │ - Recency-weighted ranking │ 
└───────────────────┬────────────────────┘ │ ┌───────────────────▼────────────────────┐ 
│ ChromaDB Vector Store (db/chroma_db) │ │ - Embedding: BGE-M3 or Nomic │ 
│ - Collection: multilingual_docs │ │ - Query: Filtered by language │
└───────────────────┬────────────────────┘ │ ┌───────────────────▼────────────────────┐ 
│ Ingestion Pipeline (ingestion.py) │ │ - ClauseBoundaryChunker │ │ - Language-tagged metadata │ 
│ - Recency ranking │ └───────────────────┬────────────────────┘ │ ┌───────────────────▼────────────────────┐ 
│ Source Documents (data/ dirs) │ │ - English PDFs: data/english/ │ │ - French PDFs: data/french/ │ 
│ - Bilingual: data/english_and_... │ └────────────────────────────────────────┘

LUFA_OpenSource_RAG


## Refined Bilingual Open-Source RAG Technical Implementation Plan

## 💻 Installation & Setup

### Prerequisites

- **Python**: 3.11+
- **RAM**: 16GB minimum (32GB for faster inference)
- **GPU**: NVIDIA (6GB+ VRAM) or Apple Silicon (M1+) — or CPU (slower)
- **Environment**: Linux, macOS, or WSL2 on Windows

### Step 1: Clone & Environment

```bash
git clone https://github.com/your-org/LUFA_OpenSource_RAG.git
cd LUFA_OpenSource_RAG

# Create Conda environment
conda create -n lufa_rag python=3.11 -y
conda activate lufa_rag
```

**Hardware Requirements:**

- Minimum 16GB RAM (32GB recommended for multilingual models)
- NVIDIA GPU with 6GB+ VRAM (OR M1/M2/M3 Mac)


---

## 💻 Installation & Setup

### Prerequisites

- **Python**: 3.11+
- **RAM**: 16GB minimum (32GB for faster inference)
- **GPU**: NVIDIA (6GB+ VRAM) or Apple Silicon (M1+) — or CPU (slower)
- **Environment**: Linux, macOS, or WSL2 on Windows

### Step 1: Clone & Environment

```bash
git clone https://github.com/your-org/LUFA_OpenSource_RAG.git
cd LUFA_OpenSource_RAG

# Create Conda environment
conda create -n lufa_rag python=3.11 -y
conda activate lufa_rag
```


### Phase 2: Bilingual Model Selection


**Multilingual LLM** :[^1]

1. **Llama 3.2:3b-instruct-q4_K_M** : Officially supports French, English, and 6 other languages[^4]

# LLM for generation (3B parameter model, fits most GPUs)
ollama pull llama3.2:3b-instruct-q4_K_M

# Multilingual embedding (100+ language support)
ollama pull nomic-embed-text-v2-moe

# Optional: Alternative embedding (higher quality but larger)
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
llama-index-embeddings-ollama
llama-index-vector-stores-chroma
chromadb
streamlit
pypdf
langdetect                # Language detection
pycountry                 # Language code handling
```


### Phase 5: Bilingual Ingestion Strategy

**Key Features:**[^5][^1]

1. **Language Detection**: Automatically detect whether each PDF is English or French using `langdetect`
2. **Metadata Tagging**: Store language metadata with each chunk
3. **Unified Vector Store**: Both languages in same ChromaDB using multilingual embeddings[^6]
4. **Document Structure**:
    - Read both English and French collective agreements from respective folders
    - Chunk with 1024 tokens, 200 overlap
    - Tag each chunk with: `{language: "en/fr", source_doc: "filename", page: N}`

### Phase 6: Multilingual RAG Engine

**Core Capabilities:**[^1]

1. **Query Language Detection**: Detect if user asks in English or French
2. **Cross-Lingual Retrieval**: Multilingual embeddings enable:
    - English query → retrieves relevant French documents[^6]
    - French query → retrieves relevant English documents
    - Same-language retrieval
3. **Response Generation**: LLM responds in the same language as the query[^4]
4. **Retrieval Settings**: Top 5 chunks with similarity threshold 0.7

**System Prompts:**

```python
SYSTEM_PROMPTS = {
    "en": "You are a helpful assistant answering questions about the Laurentian University Faculty Association collective agreement. Respond in English.",
    "fr": "Tu es un assistant utile qui répond aux questions sur la convention collective de l'Association des professeur(e)s de l'Université Laurentienne. Réponds en français."
}
```


### Phase 7: Bilingual User Interface

**Streamlit Features:**[^1]

- Language toggle (EN/FR) for UI labels
- Automatic query language detection
- Display retrieved chunks with language tags
- Show source document and page numbers
- Option to translate the answer into another language

**UI Elements:**

```python
# Sidebar: Language preference selector
language = st.sidebar.selectbox("Language / Langue", ["English", "Français"])

# Main chat: Question input (accepts EN/FR)
query = st.chat_input("Ask a question / Posez une question")

# Response area: Answer + source citations
st.write(response)
st.caption(f"Sources: {source_docs}")

# Metadata display: Show which language documents were retrieved
st.info(f"Retrieved {len(chunks)} chunks ({lang_counts})")
```


### Phase 8: Evaluation Framework

**Bilingual Test Dataset:**[^1]

1. Create 20 English Q\&A pairs from English collective agreements
2. Create 20 French Q\&A pairs from French collective agreements
3. Test cross-lingual scenarios (English query, French source)

**Metrics:**

- Retrieval accuracy per language
- Answer quality (manual evaluation)
- Cross-lingual retrieval performance
- Response time per language

**Optional: RAGAS Evaluation** with local model as judge[^1]

### Phase 9: Technical Documentation Structure

Your thesis documentation should cover:

1. **Architecture Overview**
    - System diagram showing bilingual data flow
    - Component interactions
2. **Multilingual Challenges**
    - Embedding space alignment for EN/FR
    - Query-document language mismatch handling
    - Token counting differences between languages
3. **Implementation Details**
    - Model selection rationale (why nomic-embed-text-v2-moe or Llama 3.2)
    - Chunking strategy for French vs English text
    - Performance optimization for local deployment
4. **Evaluation Results**
    - Monolingual performance (EN→EN, FR→FR)
    - Cross-lingual performance (EN→FR, FR→EN)
    - Comparison with baseline approaches

### Immediate Next Steps

1. Install Ollama and pull bilingual models:

```bash
ollama pull llama3.2:3b-instruct-q4
ollama pull nomic-embed-text-v2-moe  # or mxbai-embed-large
```

2. Set up enhanced project structure with language folders

```bash
conda create -n LUFA_OpenSource_RAG python=3.11 -y
conda init
```

3. Implement `language_detector.py` for automatic language detection
4. Modify `ingestion.py` to handle both English and French PDFs with metadata tagging

This refined plan maintains your open-source approach while adding robust bilingual capabilities. The key advantage is that multilingual embedding models map semantically similar content across languages to nearby vectors, enabling true cross-lingual retrieval without translation overhead.[^7][^8]

### No documents found
- Ensure documents are placed in `data/english/` or `data/french/`
- Run ingestion: `python src/ingestion.py`
- Check for error messages during ingestion

### Import errors
```bash
# Activate virtual environment
conda activate LUFA_OpenSource_RAG

conda activate LUFA_OpenSource_RAG


# Reinstall dependencies
pip install -r requirements.txt
```
### Other Quick runs 
pip install fastapi uvicorn httpx openai nltk rouge-score pyyaml tqdm pandas numpy
pip install pdfplumber langdetect llama-index-core
pip install llama-index-vector-stores-chroma
pip install llama-index-embeddings-huggingface chromadb
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"


1. python src/find_ground_truth.py          # adds ground_truth_source_ids to combined_test_data.csv
2. python src/api.py                        # (optional) start REST API server
3. python src/run_simulation.py             # runs all questions → lufa_out_data.csv
4. python src/evaluate.py                   # metrics → evaluation_results.csv + dashboard/index.html
5. open dashboard/index.html                # view results in browser

# Health check
curl http://localhost:8000/health

# Standard RAG
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the salary grid for 2024?","return_sources":true}' \
  --max-time 300

# Agentic RAG (set --max-time high — agent loop takes 60–180s)
curl -X POST http://localhost:8000/agentic-query \
  -H "Content-Type: application/json" \
  -d '{"query":"Can a part-time faculty member defer a merit review?","return_sources":true,"max_retries":3}' \
  --max-time 600

# Frontier model (requires GITHUB_TOKEN)
curl -X POST http://localhost:8000/copilot-query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the academic freedom provisions?","model":"claude-sonnet-4-5"}' \
  --max-time 120

# Run full simulation
python src/run_simulation.py --mode local
python src/run_simulation.py --mode frontier --model gpt-4o

# Find ground truth IDs
python src/find_ground_truth.py

# Generate evaluation + dashboard
python src/evaluate.py
python src/evaluate.py --judge_llm   # faster, skips Ollama judge

# Health check
curl http://localhost:8000/health

# Standard RAG
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the salary grid for 2024?","return_sources":true}' \
  --max-time 300

# Agentic RAG (set --max-time high — agent loop takes 60–180s)
curl -X POST http://localhost:8000/agentic-query \
  -H "Content-Type: application/json" \
  -d '{"query":"Can a part-time faculty member defer a merit review?","return_sources":true,"max_retries":3}' \
  --max-time 600

# Frontier model (requires GITHUB_TOKEN)
curl -X POST http://localhost:8000/copilot-query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the academic freedom provisions?","model":"claude-sonnet-4-5"}' \
  --max-time 120

# Run full simulation
python src/run_simulation.py --mode local
python src/run_simulation.py --mode frontier --model gpt-4o

# Find ground truth IDs
python src/find_ground_truth.py

# Generate evaluation + dashboard
python src/evaluate.py
python src/evaluate.py --judge_llm   # faster, skips Ollama judge



## 📄 License

This project is open source and available under the terms specified in the LICENSE file.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

Built with ❤️ using Ollama, LlamaIndex, ChromaDB, and Streamlit

---

## Footnotes

[^1]: This approach leverages the inherent multilingual capabilities of modern embedding and LLM models to provide seamless bilingual support without requiring separate pipelines or translation services.

[^2]: nomic-embed-text-v2-moe (BAAI General Embedding - Multilingual, Multifunctionality, Multi-Granularity) is specifically designed for cross-lingual retrieval tasks and has been shown to perform well on French and English document pairs.

[^4]: Llama 3.2 officially supports 8 languages, including English and French, making it suitable for generating responses in either language while maintaining context and accuracy.

[^5]: Language detection and metadata tagging ensure that the system can track document provenance while still enabling cross-lingual retrieval through shared embedding space.

[^6]: Using a single unified vector store with multilingual embeddings is more efficient than maintaining separate stores per language and naturally enables cross-lingual retrieval.

[^7]: Cross-lingual retrieval allows users to query in one language (e.g., English) and retrieve relevant documents in another language (e.g., French) based on semantic similarity.

[^8]: Multilingual embedding models are trained to map semantically similar phrases across languages to nearby points in the embedding space, enabling natural cross-lingual information retrieval without explicit translation.

---
