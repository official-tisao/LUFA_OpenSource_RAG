  --max-time 120

# Run full simulation
python src/run_simulation.py --mode local
python src/run_simulation.py --mode frontier --model gpt-4o

# Find ground truth IDs
python src/find_ground_truth.py

# Generate evaluation + dashboard
python src/evaluate.py
python src/evaluate.py --no_llm_judge   # faster, skips Ollama judge

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
python src/evaluate.py --no_llm_judge   # faster, skips Ollama judge



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
