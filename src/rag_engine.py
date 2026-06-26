        context     = "\n\n---\n\n".join([n.node.text for n in nodes])
        system      = self.query_handler.get_system_prompt(lang)
        instruction = "Réponds en français." if lang == "fr" else "Respond in English."

        prompt = f"""{system}
{instruction}
Answer ONLY using the context below.
Cite the source document name and page number for every claim.
Context:
{context}

Question: {original_query}
Answer:"""
        return str(self.llm.complete(prompt)).strip()

    def agentic_query(
        self,
        query_text:     str,
        return_sources: bool = False,
        max_retries:    int  = MAX_RETRIES
    ) -> dict:
        """Agentic RAG query loop with cross-lingual support."""
        original_lang      = detect_full_language(query_text)
        translation_applied = needs_translation(original_lang)
        translated_query   = None

        if translation_applied:
            lang_name = LANGUAGE_NAMES.get(original_lang, original_lang.upper())
            print(f"[AgenticRAG] Non-native language detected: {original_lang} ({lang_name})")
            print(f"[AgenticRAG] Translating query to English for processing...")
            translated_query = translate_to_english(query_text, original_lang, self.llm)
            processing_query = translated_query
            pipeline_lang    = "en"
        else:
            processing_query = query_text
            pipeline_lang    = original_lang

        print(f"[AgenticRAG] Pipeline language: {pipeline_lang}")

        current_query    = processing_query
        rewritten_query  = processing_query
        nodes            = []
        answer           = ""
        is_grounded      = False

        for attempt in range(1, max_retries + 1):
            print(f"[AgenticRAG] Attempt {attempt}/{max_retries}")
            rewritten_query = rewrite_query(current_query, pipeline_lang, self.llm)


            print(f"[AgenticRAG] Rewritten: {rewritten_query}")

            top_k = self.similarity_top_k + (attempt-1)
            nodes = self._retrieve_nodes(rewritten_query, top_k=top_k)
            print(f"[AgenticRAG] Retrieved {len(nodes)} chunks (top_k={top_k})")

            answer = self._generate_from_nodes(processing_query, nodes, pipeline_lang)

            chunk_texts = [n.node.text for n in nodes]
            is_grounded = reflect(answer, chunk_texts, self.llm)
            print(f"[AgenticRAG] Grounded: {is_grounded}")

            if is_grounded:
                break

            current_query = rewritten_query

        final_answer = answer
        if translation_applied:
            lang_name = LANGUAGE_NAMES.get(original_lang, original_lang.upper())
            print(f"[AgenticRAG] Translating answer back to {lang_name}...")
            final_answer = translate_to_target(answer, original_lang, self.llm)

        result = {
            'response':            final_answer,
            'english_response':    answer if translation_applied else None,
            'detected_language':   pipeline_lang,
            'original_language':   original_lang,
            'original_query':      query_text,
            'translated_query':    translated_query,
            'rewritten_query':     rewritten_query,
            'attempts':            attempt,
            'grounded':            is_grounded,
            'translation_applied': translation_applied,
        }

        if return_sources:
            sources = []
            for node in nodes:
                text    = node.node.text
                preview = text[:PREVIEW_LENGTH] + ('...' if len(text) > PREVIEW_LENGTH else '')
                sources.append({
                    'text':     preview,
                    'score':    node.score,
                    'metadata': node.node.metadata,
                    'node_id':  node.node.node_id
                })
            result['sources'] = sources

        return result


def create_rag_engine(
    db_path:         str = "db/chroma_db",
    llm_model:       str = "llama3.2:3b-instruct-q4_K_M",
    embedding_model: str = "nomic-embed-text-v2-moe"
) -> BilingualRAGEngine:
    return BilingualRAGEngine(
        db_path=db_path,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )