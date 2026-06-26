"""
GitHub Models / GitHub Copilot frontier model integration.
Uses the OpenAI-compatible GitHub Models API endpoint.

Setup:
  1. Create a GitHub Personal Access Token at https://github.com/settings/tokens
     (needs: models:read scope or Copilot subscription)
  2. Set environment variable:  export GITHUB_TOKEN=ghp_your_token_here
  3. Or add to config/config.yaml under copilot.github_token

Available models (GitHub Models API — March 2026):
  OpenAI  : gpt-4o, gpt-4.1, gpt-5
  Anthropic: claude-sonnet-4-5, claude-opus-4
  Google  : gemini-2.5-pro
  xAI     : grok-3
  Meta    : meta-llama-3.1-405b-instruct

GitHub Models endpoint: https://models.inference.ai.azure.com
"""

import os, sys, yaml
from pathlib import Path
from typing import Optional

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"

# Mapping of user-friendly aliases → actual GitHub Models API model IDs
MODEL_ALIASES = {
    # User's requested names → closest available GitHub Models IDs
    "claude-sonnet": "claude-sonnet-4-5",
    "claude": "claude-sonnet-4-5",
    "grok": "grok-3",
    "chatgpt": "gpt-4o",
    "gpt5": "gpt-5",
    "gpt-5": "gpt-5",
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt-4.1": "gpt-4.1",
    "gemini": "gemini-2.5-pro",
    "llama": "meta-llama-3.1-405b-instruct",
    "llama-405b": "meta-llama-3.1-405b-instruct",
    # Direct pass-through (already correct GitHub Models IDs)
    "claude-sonnet-4-5": "claude-sonnet-4-5",
    "claude-opus-4": "claude-opus-4",
    "grok-3": "grok-3",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "meta-llama-3.1-405b-instruct": "meta-llama-3.1-405b-instruct",
}

SYSTEM_PROMPTS = {
    "en": (
        "You are a helpful legal assistant specializing in university collective agreements. "
        "Answer ONLY using the provided context. Cite the source document name and page for every claim. "
        "If the context does not contain the answer, say so clearly. Respond in English."
    ),
    "fr": (
        "Tu es un assistant juridique spécialisé dans les conventions collectives universitaires. "
        "Réponds UNIQUEMENT à partir du contexte fourni. Cite le document source et la page pour chaque affirmation. "
        "Si le contexte ne contient pas la réponse, dis-le clairement. Réponds en français."
    ),
}


class CopilotEngine:
    """
    Wraps the GitHub Models API (OpenAI-compatible) to use frontier models
    for the generation step after local ChromaDB retrieval.
    """

    def __init__(
            self,
            model: str = "gpt-4o",
            github_token: Optional[str] = None,
            config_path: str = "config/config.yaml",
    ):
        self.model = MODEL_ALIASES.get(model, model)
        token = (
                github_token
                or os.environ.get("GITHUB_TOKEN")
                or self._load_token_from_config(config_path)
        )
        if not token:
            raise EnvironmentError(
                "GITHUB_TOKEN not found. Set it as an environment variable or in config.yaml "
                "under copilot.github_token. Get a token at https://github.com/settings/tokens"
            )

        self.client = OpenAI(
            base_url=GITHUB_MODELS_ENDPOINT,
            api_key=token,
        )
        print(f"[CopilotEngine] Initialized with model: {self.model}")

    @staticmethod
    def _load_token_from_config(path: str) -> Optional[str]:
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("copilot", {}).get("github_token")
        except FileNotFoundError:
            return None

    def generate(self, query: str, context: str, lang: str = "en") -> str:
        """
        Generate an answer from provided context using the frontier model.

        Args:
            query:   User's question
            context: Retrieved document chunks as a single string
            lang:    'en' or 'fr' — determines response language
        Returns:
            Generated answer string
        """
        system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
        prompt = (
            f"Context from collective agreement documents:\n\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # low temp for factual legal Q&A
                max_tokens=1024,
                timeout=120,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[CopilotEngine] Generation error: {e}")
            raise

    def generate_from_nodes(self, query: str, nodes: list, lang: str = "en") -> str:
        """
        Generate answer directly from LlamaIndex source nodes.
        Called after local ChromaDB retrieval.
        """
        context = "\n\n---\n\n".join(
            [f"[Source: {n.node.metadata.get('source_doc', 'unknown')} "
             f"p.{n.node.metadata.get('page', '?')}]\n{n.node.text}"
             for n in nodes]
        )
        return self.generate(query, context, lang)

    @classmethod
    def list_models(cls) -> dict:
        """Return the model alias → ID mapping for reference."""
        return MODEL_ALIASES