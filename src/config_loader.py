"""
Unified configuration loader for the LUFA Bilingual RAG System.

Single source of truth: config/config.yaml.
All modules should import from this file instead of directly from
config.py, config_template.py, or ad-hoc load_config() calls.

Usage:
    from config_loader import cfg

    # Access any setting:
    cfg("models.llm.name")              # → "llama3.2:3b-instruct-q4_K_M"
    cfg("database.path")                # → "db/chroma_db"
    cfg("agreement.default_year_range") # → "2020 - 2025"

    # With default fallback:
    cfg("retrieval.top_k", 5)           # → 3 (from yaml) or 5 if key missing

    # Direct flat access for commonly-used values:
    LLM_MODEL, DB_PATH, etc. (module-level constants)

Environment variable overrides:
    LUFA_<KEY> overrides any yaml value, where <KEY> is the dotted path
    uppercased with dots→underscores, e.g. LUFA_MODELS_LLM_NAME=gemini-2.5-pro

The config is loaded once at import time and cached as a module-level dict.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# Automatically loads the .env file into the system environment
load_dotenv()
# ── Path to config.yaml ──────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

# ── Default fallback values (used when config.yaml is missing keys) ──────────
_DEFAULTS = {
    "models.judge_llm.name": "tensortemplar/prometheus2:8x7b-Q4_K_S",
    "models.llm.name": "llama3.2:3b",
    "models.llm.base_url": "http://localhost:11434",
    "models.llm.request_timeout": 240.0,
    "models.embedding.name": "nomic-embed-text-v2-moe",
    "models.embedding.base_url": "http://localhost:11434",
    "database.path": "db/chroma_db",
    "database.collection_name": "multilingual_docs",
    "ingestion.chunk_size": 512,
    "ingestion.chunk_overlap": 50,
    "retrieval.top_k": 5,
    "retrieval.similarity_threshold": 0.7,
    "retrieval.response_mode": "compact",
    "data.english_dir": "data/english",
    "data.french_dir": "data/french",
    "languages.default": "en",
    "agreement.default_year_range": "2020 - 2025",
    "ui.page_title": "LUFA Collective Agreement - Bilingual RAG",
    "ui.page_icon": "🌍",
    "ui.layout": "wide",
    "features.source_display": True,
    "features.language_stats": True,
    "features.chat_history": True,
    "copilot.github_token": "",
    "copilot.github_models_endpoint": "https://models.inference.ai.azure.com",
}


def _load_yaml(path: Path) -> Dict:
    """Load and parse the YAML config file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        print(f"[config_loader] Warning: {path} not found, using defaults.")
        return {}
    except yaml.YAMLError as e:
        print(f"[config_loader] Warning: error parsing {path}: {e}, using defaults.")
        return {}


def _get_nested(data: Dict, dotted_key: str, default: Any = None) -> Any:
    """
    Traverse a nested dict using a dotted key path.
    E.g. _get_nested(d, "models.llm.name") → d["models"]["llm"]["name"]
    """
    keys = dotted_key.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _env_override(dotted_key: str, value: Any) -> Any:
    """
    Check for an environment variable override.
    Dotted key "models.llm.name" → env var "LUFA_MODELS_LLM_NAME"
    """
    env_key = dotted_key.upper().replace(".", "_")
    env_val = os.environ.get(env_key)
    if env_val is not None:
        # Try to cast to the same type as the original value
        if isinstance(value, bool):
            return env_val.lower() in ("true", "1", "yes")
        if isinstance(value, int):
            return int(env_val)
        if isinstance(value, float):
            return float(env_val)
        return env_val
    return value


# ── Load config once ─────────────────────────────────────────────────────────
_raw_config = _load_yaml(_CONFIG_PATH)


def cfg(dotted_key: str, default: Any = None) -> Any:
    """
    Get a config value by dotted key path.

    Lookup order:
      1. Environment variable override (LUFA_<KEY>)
      2. YAML config value
      3. Default argument, or built-in default

    Args:
        dotted_key:  Dot-separated path, e.g. "models.llm.name"
        default:     Fallback if key not found in YAML or built-in defaults.
    """
    # 1. Get from YAML
    value = _get_nested(_raw_config, dotted_key)

    # 2. If not found, use built-in default, then argument default
    if value is None:
        value = _DEFAULTS.get(dotted_key, default)

    # 3. Apply env var override
    return _env_override(dotted_key, value)


def cfg_raw() -> Dict:
    """Return the raw parsed YAML config dict (for advanced use)."""
    return dict(_raw_config)


# ── MODEL_API_AUTH accessor ──────────────────────────────────────────────────

def get_model_api_auth() -> Dict[str, Dict[str, str]]:
    """
    Return the model_api_auth mapping from YAML config.

    Falls back to a minimal default if not configured.
    """
    auth = _get_nested(_raw_config, "model_api_auth")
    if isinstance(auth, dict) and auth:
        return auth
    return {"default": {"api_key": "", "api_base": "http://localhost:11434/api"}}


# ── Module-level constants for backward compatibility ────────────────────────
# These provide the same names as the old config.py / config_template.py
# so existing `from config import X` can be replaced with `from config_loader import X`.

# Ollama / Model
OLLAMA_BASE_URL       = cfg("models.llm.base_url")
LLM_MODEL             = cfg("models.llm.name")
EMBEDDING_MODEL       = cfg("models.embedding.name")
OLLAMA_REQUEST_TIMEOUT = cfg("models.llm.request_timeout")

# Database
DB_PATH               = cfg("database.path")
COLLECTION_NAME       = cfg("database.collection_name")

# Document Directories
ENGLISH_DOCS_DIR      = cfg("data.english_dir")
FRENCH_DOCS_DIR       = cfg("data.french_dir")

# Chunking
CHUNK_SIZE            = cfg("ingestion.chunk_size")
CHUNK_OVERLAP         = cfg("ingestion.chunk_overlap")

# Retrieval
SIMILARITY_TOP_K      = cfg("retrieval.top_k")
RESPONSE_MODE         = cfg("retrieval.response_mode")

# Language
DEFAULT_LANGUAGE      = cfg("languages.default")

# Agreement
DEFAULT_AGREEMENT_YEAR_RANGE = cfg("agreement.default_year_range")

# Streamlit / UI
STREAMLIT_PAGE_TITLE  = cfg("ui.page_title")
STREAMLIT_PAGE_ICON   = cfg("ui.page_icon")
STREAMLIT_LAYOUT      = cfg("ui.layout")

# Feature Flags
ENABLE_SOURCE_DISPLAY = cfg("features.source_display")
ENABLE_LANGUAGE_STATS = cfg("features.language_stats")
ENABLE_CHAT_HISTORY   = cfg("features.chat_history")

# Copilot
COPilot_GITHUB_TOKEN  = cfg("copilot.github_token")
COPilot_GITHUB_ENDPOINT = cfg("copilot.github_models_endpoint")

# MODEL_API_AUTH
MODEL_API_AUTH = get_model_api_auth()
