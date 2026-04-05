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
