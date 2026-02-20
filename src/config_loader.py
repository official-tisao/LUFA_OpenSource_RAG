"""
Configuration loader for the bilingual RAG system.
Reads settings from config/config.yaml and exposes them as a simple namespace object.
"""

import yaml
from pathlib import Path
from types import SimpleNamespace


def _load_config() -> SimpleNamespace:
    """Load configuration from config/config.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Ensure config/config.yaml exists in the project root."
        )

    try:
        ns = SimpleNamespace()

        # Ollama / Model configuration
        ns.OLLAMA_BASE_URL = raw["models"]["llm"]["base_url"]
        ns.LLM_MODEL = raw["models"]["llm"]["name"]
        ns.EMBEDDING_MODEL = raw["models"]["embedding"]["name"]
        ns.OLLAMA_REQUEST_TIMEOUT = raw["models"]["llm"]["request_timeout"]

        # Database configuration
        ns.DB_PATH = raw["database"]["path"]
        ns.COLLECTION_NAME = raw["database"]["collection_name"]

        # Document directories
        ns.ENGLISH_DOCS_DIR = raw["data"]["english_dir"]
        ns.FRENCH_DOCS_DIR = raw["data"]["french_dir"]

        # Chunking configuration
        ns.CHUNK_SIZE = raw["ingestion"]["chunk_size"]
        ns.CHUNK_OVERLAP = raw["ingestion"]["chunk_overlap"]

        # Retrieval configuration
        ns.SIMILARITY_TOP_K = raw["retrieval"]["top_k"]
        ns.RESPONSE_MODE = raw["retrieval"]["response_mode"]

        # Language detection
        ns.DEFAULT_LANGUAGE = raw["languages"]["default"]

        # Agreement configuration
        ns.DEFAULT_AGREEMENT_YEAR_RANGE = raw["agreement"]["year_range"]

        # Streamlit UI configuration
        ns.STREAMLIT_PAGE_TITLE = raw["ui"]["page_title"]
        ns.STREAMLIT_PAGE_ICON = raw["ui"]["page_icon"]
        ns.STREAMLIT_LAYOUT = raw["ui"]["layout"]

        # Feature flags
        ns.ENABLE_SOURCE_DISPLAY = raw["features"]["enable_source_display"]
        ns.ENABLE_LANGUAGE_STATS = raw["features"]["enable_language_stats"]
        ns.ENABLE_CHAT_HISTORY = raw["features"]["enable_chat_history"]
        ns.SHOW_SOURCES_BY_DEFAULT = raw["features"]["show_sources_by_default"]
    except KeyError as e:
        raise KeyError(
            f"Missing required configuration key {e} in {config_path}. "
            "Please ensure config/config.yaml contains all required settings."
        ) from e

    return ns


cfg = _load_config()
