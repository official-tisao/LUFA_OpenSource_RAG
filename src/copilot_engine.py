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
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,   # low temp for factual legal Q&A
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
            [f"[Source: {n.node.metadata.get('source_doc','unknown')} "
             f"p.{n.node.metadata.get('page','?')}]\n{n.node.text}"
             for n in nodes]
        )
        return self.generate(query, context, lang)

    @classmethod
    def list_models(cls) -> dict:
        """Return the model alias → ID mapping for reference."""
        return MODEL_ALIASES