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


def test_directory_structure():
    """Test that all required directories exist."""
    print("\nTesting directory structure...")
    required_dirs = [
        'data/english',
        'data/french',
        'src',
        'db'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            all_exist = False
    
    return all_exist


def test_sample_documents():
    """Test that sample documents exist."""
    print("\nTesting sample documents...")
    sample_docs = [
        'data/english/sample_ai_document.txt',
        'data/french/sample_ai_document.txt'
    ]
    
    all_exist = True
    for doc_path in sample_docs:
        path = Path(doc_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✓ Document exists: {doc_path} ({size} bytes)")
        else:
            print(f"✗ Document missing: {doc_path}")
            all_exist = False
    
    return all_exist


def test_requirements():
    """Test that requirements.txt exists and has content."""
    print("\nTesting requirements.txt...")
    req_file = Path('requirements.txt')
    if not req_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    with open(req_file) as f:
        requirements = f.read().strip().split('\n')
    
    required_packages = [
        'llama-index',
        'llama-index-llms-ollama',
        'llama-index-embeddings-ollama',
        'llama-index-vector-stores-chroma',
        'chromadb',
        'streamlit',
        'pypdf',
        'langdetect'
    ]
    
    all_present = True
    for package in required_packages:
        # Check if any requirement line starts with the package name (handles version specifiers)
        if any(req.strip().startswith(package) for req in requirements):
            print(f"✓ Package listed: {package}")
        else:
            print(f"✗ Package missing: {package}")
            all_present = False
    
    return all_present


def test_reflect():
    """Test reflect() with stubbed LLM for all coverage paths."""
    print("\nTesting reflect()...")
    try:
        from unittest.mock import MagicMock
        from reflector import reflect

        llm = MagicMock()

        # GROUNDED response → True
        llm.complete.return_value = "GROUNDED"
        assert reflect("The answer.", ["chunk one"], llm) is True, "Expected True for GROUNDED"
        print("✓ GROUNDED response returns True")

        # UNGROUNDED response → False (must not match 'GROUNDED' substring)
        llm.complete.return_value = "UNGROUNDED"
        assert reflect("The answer.", ["chunk one"], llm) is False, "Expected False for UNGROUNDED"
        print("✓ UNGROUNDED response returns False")

        # Exception → False (fail-closed)
        llm.complete.side_effect = Exception("LLM unavailable")
        assert reflect("The answer.", ["chunk one"], llm) is False, "Expected False on exception"
        print("✓ Exception returns False (fail-closed)")

        # Empty chunks → False without calling LLM
        llm.complete.side_effect = None
        assert reflect("The answer.", [], llm) is False, "Expected False for empty chunks"
        print("✓ Empty chunks returns False")

        return True
    except Exception as e:
        print(f"✗ reflect() tests failed: {e}")
        return False


def test_rewrite_query():
    """Test rewrite_query() with stubbed LLM for all coverage paths."""
    print("\nTesting rewrite_query()...")
    try:
        from unittest.mock import MagicMock
        from query_rewriter import rewrite_query

        llm = MagicMock()

        # Normal rewrite – rewritten text returned
        rewritten = "What are the vacation entitlements under the collective agreement?"
        llm.complete.return_value = rewritten
        result = rewrite_query("vacation?", "en", llm)
        assert result == rewritten, f"Expected rewritten query, got '{result}'"
        print("✓ Valid rewrite is returned")

        # Empty response falls back to original query
        llm.complete.return_value = ""
        result = rewrite_query("vacation?", "en", llm)
        assert result == "vacation?", f"Expected original query on empty response, got '{result}'"
        print("✓ Empty response falls back to original")

        # Overly long response falls back to original query
        llm.complete.return_value = "x" * 500
        result = rewrite_query("vacation?", "en", llm)
        assert result == "vacation?", f"Expected original query for overly long response, got '{result}'"
        print("✓ Overly long response falls back to original")

        # Exception falls back to original query
        llm.complete.side_effect = Exception("LLM unavailable")
        result = rewrite_query("vacation?", "en", llm)
        assert result == "vacation?", f"Expected original query on exception, got '{result}'"
        print("✓ Exception falls back to original")

        # French language uses French prompt (verify no KeyError)
        llm.complete.side_effect = None
        llm.complete.return_value = "Quels sont les droits aux vacances?"
        result = rewrite_query("vacances?", "fr", llm)
        assert result == "Quels sont les droits aux vacances?", f"Unexpected French result: '{result}'"
        print("✓ French language prompt used correctly")

        return True
    except Exception as e:
        print(f"✗ rewrite_query() tests failed: {e}")
        return False


def test_bootstrap_script():
    """Test that bootstrap script exists and is executable."""
    print("\nTesting bootstrap script...")
    script = Path('bootstrap.sh')
    if not script.exists():
        print("✗ bootstrap.sh not found")
        return False
    
    print(f"✓ bootstrap.sh exists")
    
    # Check if executable
    import os
    if os.access(script, os.X_OK):
        print("✓ bootstrap.sh is executable")
        return True
    else:
        print("✗ bootstrap.sh is not executable")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Bilingual RAG System - Basic Tests")
    print("="*60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Sample Documents", test_sample_documents),
        ("Requirements", test_requirements),
        ("Bootstrap Script", test_bootstrap_script),
        ("Module Imports", test_imports),
        ("Language Detection", test_language_detection),
        ("Augment Query With Year", test_augment_query_with_year),
        ("Reflect", test_reflect),
        ("Rewrite Query", test_rewrite_query),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
