"""
Unit tests for the reflect() function in src/reflector.py.
These tests use a stubbed LLM to avoid requiring a live Ollama instance.
Run with: python test_reflector.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Stub out llama_index before importing reflector so tests run without the
# full Ollama / llama-index stack installed.
sys.modules.setdefault('llama_index', MagicMock())
sys.modules.setdefault('llama_index.llms', MagicMock())
sys.modules.setdefault('llama_index.llms.ollama', MagicMock())

from reflector import reflect


def _make_llm(response_text):
    """Return a mock LLM whose complete() returns response_text."""
    llm = MagicMock()
    llm.complete.return_value = response_text
    return llm


class TestReflect(unittest.TestCase):

    def test_reflect_grounded_returns_true(self):
        """When the LLM responds with GROUNDED the function must return True."""
        llm = _make_llm("GROUNDED")
        self.assertIs(reflect("Some answer.", ["Chunk one.", "Chunk two."], llm), True)

    def test_reflect_grounded_case_insensitive(self):
        """GROUNDED detection should be case-insensitive (result is upper-cased internally)."""
        llm = _make_llm("grounded")
