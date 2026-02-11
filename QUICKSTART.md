# Quick Start Guide

## Prerequisites Check

Before starting, ensure you have:
- [ ] Python 3.8+ installed
- [ ] Ollama installed and running
- [ ] Required models pulled (`llama3.2` and `bge-m3`)

## Setup (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/official-tisao/LUFA_OpenSource_RAG.git
cd LUFA_OpenSource_RAG

# 2. Run bootstrap
./bootstrap.sh

# 3. Pull Ollama models (if not already done)
ollama pull llama3.2
ollama pull bge-m3
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
source venv/bin/activate
python src/ingestion.py
```

### Step 3: Start the App
```bash
streamlit run src/app.py
```

That's it! Open your browser to http://localhost:8501

## Testing Your Setup

```bash
# Basic tests (no dependencies needed)
python test_basic.py

# Full tests (after setup)
source venv/bin/activate
python test_integration.py
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
ollama pull bge-m3
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
