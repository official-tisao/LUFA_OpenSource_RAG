"""
Unit tests for the reflect() function in src/reflector.py.
These tests use a stubbed LLM to avoid requiring a live Ollama instance.
"""

import sys
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


def test_reflect_grounded_returns_true():
    """When the LLM responds with GROUNDED the function must return True."""
    llm = _make_llm("GROUNDED")
    assert reflect("Some answer.", ["Chunk one.", "Chunk two."], llm) is True


def test_reflect_grounded_case_insensitive():
    """GROUNDED detection should be case-insensitive (result is upper-cased internally)."""
    llm = _make_llm("grounded")
    assert reflect("Some answer.", ["Chunk one."], llm) is True


def test_reflect_grounded_with_trailing_whitespace():
    """Leading/trailing whitespace around the token should be ignored."""
    llm = _make_llm("  GROUNDED  ")
    assert reflect("Some answer.", ["Chunk one."], llm) is True


def test_reflect_ungrounded_returns_false():
    """When the LLM responds with UNGROUNDED the function must return False."""
    llm = _make_llm("UNGROUNDED")
    assert reflect("Some answer.", ["Chunk one."], llm) is False


def test_reflect_ungrounded_case_insensitive():
    """UNGROUNDED detection should be case-insensitive."""
    llm = _make_llm("ungrounded")
    assert reflect("Some answer.", ["Chunk one."], llm) is False


def test_reflect_first_token_only():
    """Only the first token matters; extra text after GROUNDED should still return True."""
    llm = _make_llm("GROUNDED (some explanation)")
    assert reflect("Some answer.", ["Chunk one."], llm) is True


def test_reflect_first_token_ungrounded_with_extra_text():
    """Only the first token matters; extra text after UNGROUNDED should still return False."""
    llm = _make_llm("UNGROUNDED because the answer mentions facts not in the chunks.")
    assert reflect("Some answer.", ["Chunk one."], llm) is False


def test_reflect_exception_is_fail_open():
    """On LLM error the function must return True (fail-open) to avoid infinite retry loops."""
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("connection refused")
    assert reflect("Some answer.", ["Chunk one."], llm) is True


def test_reflect_empty_chunks_returns_false():
    """With no chunks there is nothing to ground against; must return False without calling LLM."""
    llm = MagicMock()
    assert reflect("Some answer.", [], llm) is False
    llm.complete.assert_not_called()


def test_reflect_uses_at_most_five_chunks():
    """Only the first five chunks should be sent to the LLM."""
    chunks = [f"Chunk {i}." for i in range(10)]
    llm = _make_llm("GROUNDED")
    reflect("Some answer.", chunks, llm)
    call_args = llm.complete.call_args[0][0]
    # Chunks 0-4 should appear; chunk 5 onwards should not
    for i in range(5):
        assert f"Chunk {i}." in call_args
    for i in range(5, 10):
        assert f"Chunk {i}." not in call_args
