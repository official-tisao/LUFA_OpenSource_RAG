# Configuration file for Bilingual RAG System
# Copy this file to config.py and customize as needed

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:latest"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
OLLAMA_REQUEST_TIMEOUT = 120.0

# Database Configuration
DB_PATH = "db/chroma_db"
COLLECTION_NAME = "multilingual_docs"
