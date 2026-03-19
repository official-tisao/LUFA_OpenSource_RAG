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

# LUFA Agentic RAG System - Execution Workflow

This document outlines the correct execution order for the system scripts to ensure clean data generation, accurate metric scoring, and dashboard compilation.

### Step 1: Establish Ground Truth (One-Time Setup)
Before running any simulations, the master dataset must have its expected answers linked to the exact ChromaDB vector identifiers.
Command: python src/find_ground_truth.py
Input: data/combined_test_data.csv
Output: Appends UUIDs to ground_source_truth_id column in the same file.

### Step 2: Run The RAG Simulation
This script feeds the test questions into the local LLM and embedding layers. It performs the agentic looping and hybrid retrieval, saving the raw answers and retrieved chunks to an output log.
