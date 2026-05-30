    "vi": "Vietnamese", "cy": "Welsh", "yo": "Yoruba", "zu": "Zulu",
}

TRANSLATE_TO_EN_PROMPT = """Translate the following text into English.
Output ONLY the translated text — no explanations, no labels, no extra content.

Text ({source_language}):
{text}

English translation:"""

TRANSLATE_TO_TARGET_PROMPT = """Translate the following text into {target_language}.
Output ONLY the translated text — no explanations, no labels, no extra content.

English text:
{text}

{target_language} translation:"""


def detect_full_language(text: str) -> str:
    """
    Detect the language of a text string.

    Returns:
        ISO 639-1 language code (e.g. 'en', 'fr', 'es', 'ar')
        Falls back to 'en' if detection fails.
    """
    try:
        code = detect(text)
        # Normalise Chinese variants → 'zh'
        if code.startswith("zh"):
            return "zh"
        return code
    except LangDetectException:
        print("[Translator] Language detection failed — defaulting to 'en'")
        return "en"


def needs_translation(lang_code: str) -> bool:
    """Return True if the language is not natively supported (i.e. not EN or FR)."""
    return lang_code not in SUPPORTED_LANGUAGES


def translate_to_english(text: str, source_lang: str, llm: Ollama) -> str:
    """
    Translate any non-EN/FR text to English before RAG processing.

    Args:
        text:        Text to translate
        source_lang: Detected ISO language code of the source text
        llm:         Shared Ollama LLM instance

    Returns:
        English-translated text (falls back to original on failure)
