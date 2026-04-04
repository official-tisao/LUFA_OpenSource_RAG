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
