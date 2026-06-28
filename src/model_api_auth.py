"""
MODEL_API_AUTH resolver — single source of truth for model API credentials.

Looks up (api_key, api_base) for a given model name in config.MODEL_API_AUTH.
If the model name is not listed, falls back to the "default" entry.

Usage:
    from model_api_auth import resolve_model_auth, get_openai_client, get_ollama_client

    auth = resolve_model_auth("claude-3-opus-20240229")
    # => {"api_key": "...", "api_base": "https://api.anthropic.com/v1"}

    client = get_openai_client("openrouter/owl-alpha")
    llm    = get_ollama_client("llama3.2:latest")
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

# Load MODEL_API_AUTH from config/config.yaml via config_loader
from config_loader import get_model_api_auth

MODEL_API_AUTH = get_model_api_auth()


def _env_key_for_model(model_name: str) -> str:
    """Convert a model name to an env-var-safe key, e.g. 'claude-3-opus' → 'CLAUDE_3_OPUS'."""
    return model_name.upper().replace("-", "_").replace(".", "_").replace("/", "_")


def resolve_model_auth(model_name: str) -> Dict[str, str]:
    """
    Resolve (api_key, api_base) for a model name.

    Lookup order:
      1. MODEL_API_AUTH[model_name]  — per-model config entry
      2. MODEL_API_AUTH["default"]   — fallback when model not listed

    Environment variable overlay (takes priority over config values):
      - MODEL_API_KEY_<SANITIZED_NAME>  — per-model api_key override
      - MODEL_API_BASE_<SANITIZED_NAME> — per-model api_base override
      - MODEL_API_KEY_DEFAULT           — default api_key override
      - MODEL_API_BASE_DEFAULT          — default api_base override
    """
    entry = MODEL_API_AUTH.get(model_name)
    if entry is None:
        entry = MODEL_API_AUTH.get("default", {})

    api_key  = entry.get("api_key", "")
    api_base = entry.get("api_base", "")

    # Env var overlay — per-model takes priority, then DEFAULT
    env_prefix = _env_key_for_model(model_name)
    api_key  = os.environ.get(f"MODEL_API_KEY_{env_prefix}",  api_key)
    api_base = os.environ.get(f"MODEL_API_BASE_{env_prefix}", api_base)

    # If still empty, try DEFAULT env vars
    if not api_key:
        api_key = os.environ.get("MODEL_API_KEY_DEFAULT", api_key)
    if not api_base or api_base == "":
        api_base = os.environ.get("MODEL_API_BASE_DEFAULT", api_base)

    return {"api_key": api_key, "api_base": api_base}


def get_openai_client(model_name: str, **kwargs):
    """
    Create an OpenAI-compatible client configured from MODEL_API_AUTH.

    Args:
        model_name:  Model identifier to look up in MODEL_API_AUTH.
        **kwargs:    Additional keyword arguments passed to OpenAI() constructor
                     (e.g. timeout, max_retries).

    Returns:
        openai.OpenAI client instance.
    """
    from openai import OpenAI

    auth = resolve_model_auth(model_name)
    return OpenAI(
        base_url=auth["api_base"],
        api_key=auth["api_key"] or "not-set",  # OpenAI SDK requires non-empty
        **kwargs,
    )


def get_ollama_client(model_name: str, is_embedding: bool = False, **kwargs):
    """
    Create an Ollama LLM or embedding client configured from MODEL_API_AUTH.

    The api_base is stripped of any trailing "/api" since Ollama clients
    expect the base URL without that suffix.

    Args:
        model_name:    Model identifier (also used as MODEL_API_AUTH lookup key).
        is_embedding:  If True, return OllamaEmbedding; otherwise Ollama LLM.
        **kwargs:      Additional keyword arguments for the Ollama constructor.

    Returns:
        Ollama or OllamaEmbedding instance.
    """
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.ollama import OllamaEmbedding

    auth = resolve_model_auth(model_name)
    # Strip trailing /api — Ollama client wants just the base
    base_url = auth["api_base"].rstrip("/")
    if base_url.endswith("/api"):
        base_url = base_url[:-4]

    if is_embedding:
        return OllamaEmbedding(
            model_name=model_name,
            base_url=base_url,
            **kwargs,
        )
    return Ollama(
        model=model_name,
        base_url=base_url,
        request_timeout=kwargs.pop("request_timeout", 120.0),
        **kwargs,
    )


def list_configured_models() -> Dict[str, Dict[str, str]]:
    """Return the full MODEL_API_AUTH mapping for inspection."""
    return dict(MODEL_API_AUTH)
