    """
    Translate an English answer back to the user's original language.

    Args:
        text:        English answer text
        target_lang: ISO language code of the user's original language
        llm:         Shared Ollama LLM instance

    Returns:
        Translated answer (falls back to English on failure)
    """
    if target_lang in SUPPORTED_LANGUAGES or target_lang == "en":
        return text  # no translation needed

    lang_name = LANGUAGE_NAMES.get(target_lang, target_lang.upper())
