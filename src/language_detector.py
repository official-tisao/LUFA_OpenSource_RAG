"""
Language detection module for bilingual RAG system.
Automatically detects whether documents are in English or French.
"""

from langdetect import detect, LangDetectException
from typing import Optional


def detect_language(text: str) -> str:
    """
    Detect the language of a text string.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Language code: 'en' for English, 'fr' for French
        
    Raises:
        LangDetectException: If language cannot be detected
    """
    try:
        lang = detect(text)
        if lang == 'fr':
            return 'fr'
        elif lang == 'en':
            return 'en'
        else:
            # Default to English for other languages
            return 'en'
    except LangDetectException:
        # Default to English if detection fails
        return 'en'


def detect_language_with_confidence(text: str) -> tuple[str, float]:
    """
    Detect language with confidence score.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (language_code, confidence_score)
    """
    from langdetect import detect_langs
    
    try:
        langs = detect_langs(text)
        if langs:
            top_lang = langs[0]
            lang_code = 'fr' if top_lang.lang == 'fr' else 'en'
            return (lang_code, top_lang.prob)
        else:
            return ('en', 0.0)
    except LangDetectException:
        return ('en', 0.0)


def is_french(text: str) -> bool:
    """
    Check if text is in French.
    
    Args:
        text: Input text to check
        
    Returns:
        True if text is French, False otherwise
    """
    return detect_language(text) == 'fr'


def is_english(text: str) -> bool:
    """
    Check if text is in English.
    
    Args:
        text: Input text to check
        
    Returns:
        True if text is English, False otherwise
    """
    return detect_language(text) == 'en'


def get_language_name(lang_code: str) -> str:
    """
    Get the full language name from language code.
    
    Args:
        lang_code: Language code ('en' or 'fr')
        
    Returns:
        Full language name
    """
    import pycountry
    
    try:
        language = pycountry.languages.get(alpha_2=lang_code)
        return language.name if language else lang_code.upper()
    except:
        return lang_code.upper()
