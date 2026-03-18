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

conda create -n LUFA_OpenSource_RAG python=3.11 -y
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
conda activate LUFA_OpenSource_RAG
# conda activate LUFA_OpenSource_RAG # Uncomment this to use venv over conda
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

conda activate LUFA_OpenSource_RAG


python test_integration.py
```

## Common Issues

### "Ollama not running"
```bash
# Start Ollama
ollama serve
```

### "Model not found"
