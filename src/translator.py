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
