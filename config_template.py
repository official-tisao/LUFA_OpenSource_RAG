# Configuration file for Bilingual RAG System
# Copy this file to config.py and customize as needed

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b-instruct-q4_K_M"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe"
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
SIMILARITY_TOP_K = 5
RESPONSE_MODE = "compact"  # Options: "compact", "refine", "tree_summarize"

# Language Detection
DEFAULT_LANGUAGE = "en"  # Fallback if detection fails

# Agreement Configuration
DEFAULT_AGREEMENT_YEAR_RANGE = "2020 - 2025"

# Streamlit Configuration
STREAMLIT_PAGE_TITLE = "LUFA Collective Agreement - Bilingual RAG"
STREAMLIT_PAGE_ICON = "🌍"
STREAMLIT_LAYOUT = "wide"

# Feature Flags
ENABLE_SOURCE_DISPLAY = True
ENABLE_LANGUAGE_STATS = True
ENABLE_CHAT_HISTORY = True
SHOW_SOURCES_BY_DEFAULT = True
