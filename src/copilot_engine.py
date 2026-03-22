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
    "claude-sonnet":    "claude-sonnet-4-5",
    "claude":           "claude-sonnet-4-5",
    "grok":             "grok-3",
    "chatgpt":          "gpt-4o",
    "gpt5":             "gpt-5",
    "gpt-5":            "gpt-5",
    "gpt4o":            "gpt-4o",
    "gpt-4o":           "gpt-4o",
    "gpt-4.1":          "gpt-4.1",
    "gemini":           "gemini-2.5-pro",
    "llama":            "meta-llama-3.1-405b-instruct",
    "llama-405b":       "meta-llama-3.1-405b-instruct",
    # Direct pass-through (already correct GitHub Models IDs)
    "claude-sonnet-4-5":           "claude-sonnet-4-5",
    "claude-opus-4":               "claude-opus-4",
    "grok-3":                      "grok-3",
    "gemini-2.5-pro":              "gemini-2.5-pro",
    "meta-llama-3.1-405b-instruct":"meta-llama-3.1-405b-instruct",
}

SYSTEM_PROMPTS = {
