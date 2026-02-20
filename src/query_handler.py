"""
Query handler module for language-aware query processing.
Handles query language detection and routing to appropriate system prompts.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Optional
from language_detector import detect_language

# Add project root to path to allow importing config
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
if (_project_root / 'config.py').exists():
    from config import DEFAULT_AGREEMENT_YEAR_RANGE
else:
    from config_template import DEFAULT_AGREEMENT_YEAR_RANGE

# System prompts for bilingual support
SYSTEM_PROMPTS = {
    "en": "You are a helpful assistant answering questions about the Laurentian University Faculty Association collective agreement. Respond in English.",
    "fr": "Tu es un assistant utile qui répond aux questions sur la convention collective de l'Association des professeur(e)s de l'Université Laurentienne. Réponds en français."
}


class QueryHandler:
    """
    Handles query processing with language awareness.
    """
    
    def __init__(self):
        """Initialize the query handler."""
        self.system_prompts = SYSTEM_PROMPTS
    
    def detect_query_language(self, query: str) -> str:
        """
        Detect the language of a user query.
        
        Args:
            query: User query text
            
        Returns:
            Language code ('en' or 'fr')
        """
        return detect_language(query)
    
    def get_system_prompt(self, language: str) -> str:
        """
        Get the appropriate system prompt for the given language.
        
        Args:
            language: Language code ('en' or 'fr')
            
        Returns:
            System prompt string
        """
        return self.system_prompts.get(language, self.system_prompts['en'])
    
    def create_language_aware_query(self, query: str, language: Optional[str] = None) -> Dict[str, str]:
        """
        Create a language-aware query with appropriate system prompt.

        Args:
            query: User query text
            language: Optional language code. If not provided, will be auto-detected.

        Returns:
            Dictionary with 'query', 'language', and 'system_prompt'
        """
        if language is None:
            language = self.detect_query_language(query)

        query = self.augment_query_with_year(query, language)
        
        system_prompt = self.get_system_prompt(language)
        
        return {
            'query': query,
            'language': language,
            'system_prompt': system_prompt
        }

    def augment_query_with_year(self, user_input: str, language: Optional[str] = None) -> str:
        """
        Augment the user query with a year range if no year is present.

        If the query does not already contain a 4-digit year, appends
        'collective agreement 2020 - 2025' to help narrow retrieval.

        Args:
            user_input: Raw user query string
            language: Optional language code ('en' or 'fr'). If not provided, will be auto-detected.

        Returns:
            Augmented query string, or the original if a year was found
        """
        has_year = re.search(r'\b(19\d{2}|20\d{2})\b', user_input)

        if not has_year:
            if language is None:
                language = self.detect_query_language(user_input)

            if language == 'fr':
                return f"{user_input} convention collective {DEFAULT_AGREEMENT_YEAR_RANGE}"
            else:
                return f"{user_input} collective agreement {DEFAULT_AGREEMENT_YEAR_RANGE}"

        return user_input

    def format_response_instruction(self, language: str) -> str:
        """
        Create an instruction to ensure response is in the correct language.
        
        Args:
            language: Language code ('en' or 'fr')
            
        Returns:
            Response instruction string
        """
        if language == 'fr':
            return "Réponds en français. "
        else:
            return "Respond in English. "
