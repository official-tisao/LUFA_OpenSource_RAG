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
        self.assertIs(reflect("Some answer.", ["Chunk one."], llm), True)

    def test_reflect_grounded_with_trailing_whitespace(self):
        """Leading/trailing whitespace around the token should be ignored."""
        llm = _make_llm("  GROUNDED  ")
        self.assertIs(reflect("Some answer.", ["Chunk one."], llm), True)

    def test_reflect_ungrounded_returns_false(self):
        """When the LLM responds with UNGROUNDED the function must return False."""
        llm = _make_llm("UNGROUNDED")
        self.assertIs(reflect("Some answer.", ["Chunk one."], llm), False)

    def test_reflect_ungrounded_case_insensitive(self):
        """UNGROUNDED detection should be case-insensitive."""
        llm = _make_llm("ungrounded")
        self.assertIs(reflect("Some answer.", ["Chunk one."], llm), False)

    def test_reflect_first_token_only(self):
        """Only the first token matters; extra text after GROUNDED should still return True."""
        llm = _make_llm("GROUNDED (some explanation)")
        self.assertIs(reflect("Some answer.", ["Chunk one."], llm), True)

    def test_reflect_first_token_ungrounded_with_extra_text(self):
        """Only the first token matters; extra text after UNGROUNDED should still return False."""
        llm = _make_llm("UNGROUNDED because the answer mentions facts not in the chunks.")
        self.assertIs(reflect("Some answer.", ["Chunk one."], llm), False)

    def test_reflect_exception_is_fail_closed(self):
        """On LLM error the function must return False (fail-closed) to match the implemented policy."""
        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("connection refused")
        self.assertIs(reflect("Some answer.", ["Chunk one."], llm), False)

    def test_reflect_empty_chunks_returns_false(self):
        """With no chunks there is nothing to ground against; must return False without calling LLM."""
        llm = MagicMock()
        self.assertIs(reflect("Some answer.", [], llm), False)
        llm.complete.assert_not_called()

    def test_reflect_uses_at_most_five_chunks(self):
        """Only the first five chunks should be sent to the LLM."""
        chunks = [f"Chunk {i}." for i in range(10)]
        llm = _make_llm("GROUNDED")
        reflect("Some answer.", chunks, llm)
        call_args = llm.complete.call_args[0][0]
        # Chunks 0-4 should appear; chunk 5 onwards should not
        for i in range(5):
            self.assertIn(f"Chunk {i}.", call_args)
        for i in range(5, 10):
            self.assertNotIn(f"Chunk {i}.", call_args)


if __name__ == "__main__":
    unittest.main()
