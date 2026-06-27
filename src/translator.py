"""
Translation module for Agentic RAG multilingual support.
Converts non-EN/FR queries to English for processing,
then translates the final answer back to the original language.
"""

from llama_index.llms.ollama import Ollama
from langdetect import detect, LangDetectException

# Languages natively supported by the pipeline — no translation needed
SUPPORTED_LANGUAGES = {"en", "fr"}

# Human-readable language names for prompts
LANGUAGE_NAMES = {
    "af": "Afrikaans", "sq": "Albanian", "ar": "Arabic", "hy": "Armenian",
    "az": "Azerbaijani", "eu": "Basque", "be": "Belarusian", "bn": "Bengali",
    "bs": "Bosnian", "bg": "Bulgarian", "ca": "Catalan", "zh": "Chinese",
    "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)",
    "hr": "Croatian", "cs": "Czech", "da": "Danish", "nl": "Dutch",
    "en": "English", "et": "Estonian", "fi": "Finnish", "fr": "French",
    "gl": "Galician", "de": "German", "el": "Greek", "gu": "Gujarati",
    "ha": "Hausa", "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian",
    "id": "Indonesian", "ga": "Irish", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "kk": "Kazakh", "ko": "Korean", "lv": "Latvian",
    "lt": "Lithuanian", "mk": "Macedonian", "ms": "Malay", "ml": "Malayalam",
    "mt": "Maltese", "mr": "Marathi", "ne": "Nepali", "no": "Norwegian",
    "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "pa": "Punjabi",
    "ro": "Romanian", "ru": "Russian", "sr": "Serbian", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "es": "Spanish", "sw": "Swahili",
    "sv": "Swedish", "tl": "Filipino", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
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
