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
