                        if result.get("translation_applied"):
                            _render_translation_expander(
                                orig_lang=result.get("original_language", "unknown"),
                                lang_names=lang_names,
                                original_query=result.get("original_query", query),
                                translated_query=result.get("translated_query", ""),
                                english_response=result.get("english_response", ""),
                                final_response=result["response"],
                            )

                        # ── Language caption ──────────────────────────────────
                        detected = result.get("detected_language", "unknown")
                        lang_emoji = (
                            "🇬🇧" if detected == "en"
                            else "🇫🇷" if detected == "fr"
                            else "🌐"
                        )
                        st.caption(f"{lang_emoji} Detected language: {detected}")

                        # ── Sources expander (new query) ──────────────────────
                        if show_sources and result.get("sources"):
                            _render_sources_expander(result["sources"])

                        # ── Persist to session state ──────────────────────────
                        st.session_state.chat_history.append({
                            "query": query,
                            "response": result["response"],
                            "language": detected,
                            "sources": result.get("sources", []),
                            "translation_applied": result.get("translation_applied", False),
                            "original_language": result.get("original_language"),
                            "original_query": result.get("original_query"),
                            "translated_query": result.get("translated_query"),
                            "english_response": result.get("english_response"),
                        })

                    except Exception as e:
                        st.error(f"Error processing query: {e}")
                        st.info(
                            "Make sure Ollama is running and the models are available."
                        )

    # ── Statistics panel ──────────────────────────────────────────────────────
    with col2:
        st.header("📊 Statistics")

        total = len(st.session_state.chat_history)
        st.metric("Total Questions Asked", total)

        if total:
            languages = [c.get("language", "unknown") for c in st.session_state.chat_history]
            en_count = languages.count("en")
            fr_count = languages.count("fr")
            other_count = total - en_count - fr_count

            st.subheader("Language Distribution")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("🇬🇧 EN", en_count)
            with col_b:
                st.metric("🇫🇷 FR", fr_count)
            with col_c:
                st.metric("🌐 Other", other_count)

            # Translation usage metric
            translated_count = sum(
                1 for c in st.session_state.chat_history if c.get("translation_applied")
            )
            if translated_count:
                st.metric("🌐 Queries Translated", translated_count)

            st.subheader("Recent Queries")
            for chat in reversed(st.session_state.chat_history[-5:]):
                lang = chat.get("language", "unknown")
                lang_emoji = "🇬🇧" if lang == "en" else "🇫🇷" if lang == "fr" else "🌐"
                preview = chat["query"][:50]
                ellipsis = "..." if len(chat["query"]) > 50 else ""
                with st.expander(f"{lang_emoji} {preview}{ellipsis}"):
                    st.write(f"**Query:** {chat['query']}")
                    if chat.get("translation_applied"):
                        orig_lang = chat.get("original_language", "?")
                        st.write(f"**Translated from:** {orig_lang.upper()}")
                    st.write(f"**Response:** {chat['response'][:200]}...")


if __name__ == "__main__":
    main()