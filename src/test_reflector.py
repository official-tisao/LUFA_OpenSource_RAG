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
