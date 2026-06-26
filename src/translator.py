    """
    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang.upper())
    prompt = TRANSLATE_TO_EN_PROMPT.format(
        source_language=lang_name,
        text=text
    )
    try:
        result = str(llm.complete(prompt)).strip()
        if result:
            print(f"[Translator] '{source_lang}' → 'en': {text[:60]}... → {result[:60]}...")
            return result
    except Exception as e:
        print(f"[Translator] translate_to_english failed: {e}")
    return text  # safe fallback


def translate_to_target(text: str, target_lang: str, llm: Ollama) -> str:
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
    prompt = TRANSLATE_TO_TARGET_PROMPT.format(
        target_language=lang_name,
        text=text
    )
    try:
        result = str(llm.complete(prompt)).strip()
        if result:
            print(f"[Translator] 'en' → '{target_lang}': {text[:60]}... → {result[:60]}...")
            return result
    except Exception as e:
        print(f"[Translator] translate_to_target failed: {e}")
    return text  # safe fallback — return English if translation fails