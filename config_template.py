# Configuration file for Bilingual RAG System
# Copy this file to config.py and customize as needed

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:latest"
EMBEDDING_MODEL = "bge-m3:latest"
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

# Streamlit Configuration
STREAMLIT_PAGE_TITLE = "Bilingual RAG System"
STREAMLIT_PAGE_ICON = "🌍"
STREAMLIT_LAYOUT = "wide"

# Feature Flags
ENABLE_SOURCE_DISPLAY = True
ENABLE_LANGUAGE_STATS = True
ENABLE_CHAT_HISTORY = True
