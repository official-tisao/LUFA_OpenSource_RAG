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
