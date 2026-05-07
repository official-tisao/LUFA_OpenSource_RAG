            - fused via Reciprocal Rank Fusion(RRF)
            - Clause-based chunking
            """
        )

        with st.expander("Advanced Settings"):
            top_k = st.slider("Number of retrieved documents", 1, 10, 5)
            show_sources = st.checkbox("Show source documents", value=False)

        st.divider()

        st.subheader("📖 Instructions")
        if language == "English":
            st.markdown(
                """
                1. Make sure Ollama is running
                2. Ensure documents are ingested
                3. Type your question below
                4. Get your answer!

                **Example queries:**
                - "What are the office hours requirements?"
                - "What is the policy on academic freedom?"
                """
            )
        else:
            st.markdown(
                """
                1. Assurez-vous qu'Ollama fonctionne
                2. Assurez-vous que les documents sont ingérés
                3. Tapez votre question ci-dessous
                4. Obtenez votre réponse!

                **Exemples de questions:**
                - "Quelles sont les exigences concernant les heures de bureau?"
                - "Quelle est la politique sur la liberté académique?"
                """
            )

        if st.button("🔄 Reset Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Main content ─────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Chat Interface")

        rag_engine = load_rag_engine()
        if rag_engine is None:
            st.warning("⚠️ RAG engine not loaded. Please check the configuration.")
            return

        if hasattr(rag_engine, "set_similarity_top_k"):
            rag_engine.set_similarity_top_k(top_k)

        lang_names = _get_language_names()

        # ── Render chat history ───────────────────────────────────────────────
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["query"])

            with st.chat_message("assistant"):
                st.write(chat["response"])

                # Translation expander (history)
                if chat.get("translation_applied"):
                    _render_translation_expander(
                        orig_lang=chat.get("original_language", "unknown"),
                        lang_names=lang_names,
                        original_query=chat.get("original_query", chat["query"]),
                        translated_query=chat.get("translated_query", ""),
                        english_response=chat.get("english_response", ""),
                        final_response=chat["response"],
                    )

                # Language caption
                lang_emoji = (
                    "🇬🇧" if chat.get("language") == "en"
                    else "🇫🇷" if chat.get("language") == "fr"
                    else "🌐"
                )
                st.caption(f"{lang_emoji} Detected language: {chat.get('language', 'unknown')}")

                # Sources expander (history)
                if show_sources and chat.get("sources"):
                    _render_sources_expander(chat["sources"])

        # ── New query input ───────────────────────────────────────────────────
        placeholder = (
            "Ask a question / Posez une question"
            if language == "English"
            else "Posez une question / Ask a question"
        )
        query = st.chat_input(placeholder)

        if query:
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = rag_engine.query(query, return_sources=show_sources)

                        # ── Primary response ──────────────────────────────────
                        st.write(result["response"])

                        # ── Translation expander (new query) ──────────────────
