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

# Document Directories
ENGLISH_DOCS_DIR = "data/english"
FRENCH_DOCS_DIR = "data/french"

# Chunking Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval Configuration
SIMILARITY_TOP_K = 3
RESPONSE_MODE = "compact"  # Options: "compact", "refine", "tree_summarize"

# Language Detection
DEFAULT_LANGUAGE = "en"  # Fallback if detection fails

# Agreement Configuration
DEFAULT_AGREEMENT_YEAR_RANGE = "2020 - 2025"

# Streamlit Configuration
STREAMLIT_PAGE_TITLE = "Bilingual RAG System"
STREAMLIT_PAGE_ICON = "🌍"
STREAMLIT_LAYOUT = "wide"

# Feature Flags
ENABLE_SOURCE_DISPLAY = True
ENABLE_LANGUAGE_STATS = True
ENABLE_CHAT_HISTORY = True

# Model API Authentication
# Per-model (api_key, api_base) entries.  Unknown models fall back to "default".
# Use env vars MODEL_API_KEY_<NAME> / MODEL_API_BASE_<NAME> to override at runtime.
MODEL_API_AUTH = {
    "default":
        {
            "api_key": "",
            "api_base": "http://localhost:11434/api",
        },
    "claude-3-opus-20240229":
        {
            "api_key": "",
            "api_base": "https://api.anthropic.com/v1",
        },
    "openrouter/owl-alpha":
        {
            "api_key": "",
            "api_base": "https://api.openrouter.ai/v1",
        },
    # ── Gemini / Claude via local model proxy (src/model_proxy.py — uses CLI, no API keys) ──
    "gemini-2.5-pro":
        {
            "api_key": "",
            "api_base": "http://localhost:9090/v1",
        },
    "claude-sonnet-4-5":
        {
            "api_key": "",
            "api_base": "http://localhost:9090/v1",
        },
}
