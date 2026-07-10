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
from dotenv import load_dotenv

# Automatically loads the .env file into the system environment
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

# Load MODEL_API_AUTH from config/config.yaml via config_loader
from config_loader import get_model_api_auth, _env_override

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

    api_base = entry.get("api_base", "")

    api_key  = entry.get("api_key", "")

    # Env var overlay — per-model takes priority, then DEFAULT
    env_prefix = _env_key_for_model(model_name)
    api_key  = os.environ.get(f"MODEL_API_KEY_{env_prefix}",  api_key)
    api_base = os.environ.get(f"MODEL_API_BASE_{env_prefix}", api_base)

    # If still empty, try DEFAULT env vars
    if not api_key:
        api_key = os.environ.get("MODEL_API_KEY_DEFAULT", api_key)
    if not api_base or api_base == "":
        api_base = os.environ.get("MODEL_API_BASE_DEFAULT", api_base)

    # OpenRouter models are usually named like 'tencent/hy3:free' (no 'openrouter'
    # in the name), so key off the api_base too: fall back to OPENROUTER_API_KEY.
    if not api_key and "openrouter" in (api_base or "").lower():
        api_key = os.environ.get("OPENROUTER_API_KEY", api_key)

    return {"api_key": api_key, "api_base": api_base}


# api_base markers that indicate an OpenAI-compatible (not Ollama) endpoint.
_OPENAI_COMPATIBLE_MARKERS = (
    "openrouter", "openai.com", "azure", "anthropic",
    "githubcopilot", "models.inference", "api.groq", "together",
)


def is_openai_compatible_base(api_base: str) -> bool:
    """
    True when an api_base should be driven by the OpenAI-compatible client rather
    than the Ollama client. Matches endpoints ending in '/v1' (the OpenAI-compatible
    convention) or containing a known cloud host marker (openrouter, etc.).
    """
    if not api_base:
        return False
    u = api_base.lower().rstrip("/")
    if u.endswith("/v1"):
        return True
    return any(m in u for m in _OPENAI_COMPATIBLE_MARKERS)


def get_llm_client(model_name: str, force_openai: bool = False, **kwargs):
    """
    Return an LLM client, auto-selecting the backend from the resolved api_base:
      - OpenAI-compatible endpoint (OpenRouter / '/v1' / cloud host)  → get_openai_llm
      - otherwise (local Ollama '/api')                               → get_ollama_client

    `force_openai=True` forces the OpenAI-compatible path regardless of api_base.
    `request_timeout` is mapped to the OpenAI client's `timeout` when routed there.
    """
    auth = resolve_model_auth(model_name)
    if force_openai or is_openai_compatible_base(auth.get("api_base", "")):
        rt = kwargs.pop("request_timeout", None)
        if rt is not None:
            kwargs.setdefault("timeout", rt)
        return get_openai_llm(model_name, **kwargs)
    return get_ollama_client(model_name, **kwargs)


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


_OPENAI_LIKE_CLS = None


def _openai_like_cls():
    """
    Build (once) a LlamaIndex OpenAI LLM subclass that tolerates arbitrary
    (non-OpenAI) model names served over an OpenAI-compatible endpoint.

    The stock `llama_index.llms.openai.OpenAI.metadata` raises ValueError for any
    model it doesn't recognise (e.g. 'claude-haiku-4-5'), which breaks the response
    synthesizer. We override `metadata` to report a chat model with a configurable
    context window (OPENAI_LLM_CONTEXT_WINDOW env, default 128000).
    """
    global _OPENAI_LIKE_CLS
    if _OPENAI_LIKE_CLS is None:
        from llama_index.llms.openai import OpenAI as _LIOpenAI
        from llama_index.core.llms import LLMMetadata

        _ctx = int(os.environ.get("OPENAI_LLM_CONTEXT_WINDOW", "128000"))

        # Reasoning-style models (GPT-5 / o-series / codex) reject the sampling
        # params `temperature`/`top_p` and require `max_completion_tokens` in place
        # of `max_tokens`. This llama_index build doesn't recognise newer names, so
        # we normalise the request kwargs ourselves.
        _REASONING_MARKERS = ("gpt-5", "o1", "o3", "o4", "codex")

        def _is_reasoning(model_name: str) -> bool:
            name = (model_name or "").lower()
            if os.environ.get("OPENAI_LLM_NO_SAMPLING", "").lower() in ("1", "true", "yes"):
                return True
            return any(t in name for t in _REASONING_MARKERS)

        class OpenAILikeLLM(_LIOpenAI):
            @property
            def metadata(self) -> LLMMetadata:
                return LLMMetadata(
                    context_window=_ctx,
                    num_output=self.max_tokens or 4096,
                    is_chat_model=True,
                    is_function_calling_model=False,
                    model_name=self.model,
                )

            def _get_model_kwargs(self, **kwargs):
                all_kwargs = super()._get_model_kwargs(**kwargs)
                if _is_reasoning(self.model):
                    all_kwargs.pop("temperature", None)
                    all_kwargs.pop("top_p", None)
                    if "max_tokens" in all_kwargs:
                        all_kwargs.setdefault(
                            "max_completion_tokens", all_kwargs.pop("max_tokens")
                        )
                return all_kwargs

        _OPENAI_LIKE_CLS = OpenAILikeLLM
    return _OPENAI_LIKE_CLS


def get_openai_llm(model_name: str, **kwargs):
    """
    Create a LlamaIndex-compatible LLM backed by an OpenAI-compatible endpoint
    (credentials from MODEL_API_AUTH). Unlike `get_openai_client` (which returns a
    raw `openai.OpenAI` SDK client with no `.metadata`/`.complete`), this returns an
    object usable as `engine.llm`, by the response synthesizer, and by the
    streaming helpers in llm_utils.
    """
    auth = resolve_model_auth(model_name)
    cls = _openai_like_cls()
    # temperature/max_tokens are supplied for all models; the OpenAILikeLLM wrapper
    # automatically strips temperature/top_p and switches max_tokens ->
    # max_completion_tokens for reasoning models (GPT-5 / o-series / codex).
    return cls(
        model=model_name,
        api_base=auth["api_base"] or None,
        api_key=auth["api_key"] or "not-set",
        temperature=kwargs.pop("temperature", 0.1),
        max_tokens=kwargs.pop("max_tokens", 4096),
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
