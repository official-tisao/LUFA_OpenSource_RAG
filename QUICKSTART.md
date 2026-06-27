# Quick Start Guide

## Prerequisites Check

Before starting, ensure you have:
- [ ] Python 3.8+ installed
- [ ] Anaconda/Miniconda installation
- [ ] Ollama installed and running
- [ ] Required models pulled (`llama3.2` and `nomic-embed-text-v2-moe`)

## Setup (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/official-tisao/LUFA_OpenSource_RAG.git
cd LUFA_OpenSource_RAG

# 2. Run bootstrap
./bootstrap.sh

# 3. Pull Ollama models (if not already done)
ollama pull llama3.2
ollama pull nomic-embed-text-v2-moe

conda create -n lufa_rag python=3.11 -y
conda init
```

## Using the System (3 steps)

### Step 1: Add Your Documents
```bash
# Copy your files
cp your-english-doc.pdf data/english/
cp your-french-doc.pdf data/french/
```

### Step 2: Ingest Documents
```bash
conda activate lufa_rag
# source venv/bin/activate # Uncomment this to use venv over conda
python src/ingestion.py
```

### Step 3: Start the App
```bash
streamlit run src/app.py
```

That's it! Open your browser to http://localhost:8501

## Testing Your Setup

```bash
# Basic tests (requires dependencies installed)
python src/test_basic.py

# Full tests (after setup)
source venv/bin/activate  # or conda activate lufa_rag
python src/test_integration.py
```

## Common Issues

### "Ollama not running"
```bash
# Start Ollama
ollama serve
```

### "Model not found"
```bash
# Pull the models
ollama pull llama3.2
ollama pull nomic-embed-text-v2-moe
```

### "No documents found"
```bash
# Make sure documents are in the right place
ls data/english/
ls data/french/
```

## Example Queries

Try these in the app:

**English:**
- "What are the main topics in these documents?"
- "Summarize the key findings"
- "What is artificial intelligence?"

**French:**
- "Quels sont les principaux sujets de ces documents?"
- "Résume les principales conclusions"
- "Qu'est-ce que l'intelligence artificielle?"

## Next Steps

- Add more documents to `data/english/` and `data/french/`
- Run ingestion again to update the index
- Customize the UI in `src/app.py`
- Adjust retrieval parameters in `src/rag_engine.py`

Need help? Check the full [README.md](README.md) or open an issue!
