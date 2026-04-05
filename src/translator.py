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