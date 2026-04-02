"""
Basic tests for the bilingual RAG system components.
These tests verify that the modules can be imported and basic functions work.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import ingestion
        print("✓ ingestion module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ingestion: {e}")
        return False
    
    try:
        import rag_engine
        print("✓ rag_engine module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import rag_engine: {e}")
        return False
    
    try:
        import app
        print("✓ app module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import app: {e}")
        return False
    
    return True


def test_augment_query_with_year():
    """Test that augment_query_with_year correctly appends year range when needed."""
    print("\nTesting augment_query_with_year...")
    try:
        from query_handler import QueryHandler
        handler = QueryHandler()

        # English query without a year should be augmented
        en_query = "What is the vacation policy?"
        en_result = handler.augment_query_with_year(en_query, language='en')
        assert en_result == "What is the vacation policy? collective agreement 2020 - 2025", \
            f"Unexpected English augmentation: '{en_result}'"
        print(f"✓ English query augmented: '{en_result}'")

        # French query without a year should be augmented
        fr_query = "Quelle est la politique de vacances?"
        fr_result = handler.augment_query_with_year(fr_query, language='fr')
        assert fr_result == "Quelle est la politique de vacances? convention collective 2020 - 2025", \
            f"Unexpected French augmentation: '{fr_result}'"
        print(f"✓ French query augmented: '{fr_result}'")

        # Query that already contains a year should not be modified
        year_query = "What changed in 2022 for sick leave?"
        year_result = handler.augment_query_with_year(year_query, language='en')
        assert year_result == year_query, \
            f"Query with year should not be modified, got: '{year_result}'"
        print(f"✓ Query with year unchanged: '{year_result}'")

        # 4-digit numbers outside the 19xx/20xx range should not block augmentation
        non_year_query = "Article 1234 about benefits"
        non_year_result = handler.augment_query_with_year(non_year_query, language='en')
        assert non_year_result == "Article 1234 about benefits collective agreement 2020 - 2025", \
            f"Non-year 4-digit number should not block augmentation, got: '{non_year_result}'"
        print(f"✓ Non-year 4-digit number triggers augmentation: '{non_year_result}'")

        non_year_query2 = "Clause 3000 override"
        non_year_result2 = handler.augment_query_with_year(non_year_query2, language='en')
        assert non_year_result2 == "Clause 3000 override collective agreement 2020 - 2025", \
            f"Non-year 4-digit number should not block augmentation, got: '{non_year_result2}'"
        print(f"✓ Non-year 4-digit number (3000) triggers augmentation: '{non_year_result2}'")

        return True
    except Exception as e:
        print(f"✗ augment_query_with_year failed: {e}")
        return False


def test_language_detection():
    """Test language detection functionality."""
    print("\nTesting language detection...")
    try:
        from ingestion import detect_language
        
        # Test English
        en_text = "This is an English text about artificial intelligence."
        en_result = detect_language(en_text)
        assert en_result == 'en', f"Expected 'en', got '{en_result}'"
        print(f"✓ English detection: '{en_result}'")
        
        # Test French
        fr_text = "Ceci est un texte en français sur l'intelligence artificielle."
        fr_result = detect_language(fr_text)
        assert fr_result == 'fr', f"Expected 'fr', got '{fr_result}'"
        print(f"✓ French detection: '{fr_result}'")
        
        return True
    except Exception as e:
        print(f"✗ Language detection failed: {e}")
        return False
