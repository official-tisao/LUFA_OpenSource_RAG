"""
Integration tests for the bilingual RAG system.
These tests require the full environment to be set up (run bootstrap.sh first).
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def check_ollama():
    """Check if Ollama is running."""
    print("Checking Ollama connection...")
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code == 200:
            print("✓ Ollama is running")
            return True
        else:
            print("✗ Ollama returned unexpected status")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Please ensure Ollama is running: ollama serve")
        return False


def test_language_detection():
    """Test language detection with real module."""
    print("\nTesting language detection...")
    try:
        from ingestion import detect_language

        # Test English
        en_text = "This is an English text about artificial intelligence and machine learning."
        en_result = detect_language(en_text)
        print(f"  English text detected as: {en_result}")
        assert en_result == 'en', f"Expected 'en', got '{en_result}'"

        # Test French
        fr_text = "Ceci est un texte en français sur l'intelligence artificielle et l'apprentissage automatique."
        fr_result = detect_language(fr_text)
        print(f"  French text detected as: {fr_result}")
        assert fr_result == 'fr', f"Expected 'fr', got '{fr_result}'"

        print("✓ Language detection works correctly")
        return True
    except Exception as e:
        print(f"✗ Language detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_loading():
    """Test document loading functionality."""
    print("\nTesting document loading...")
    try:
        from ingestion import load_documents_from_directory

        # Load English documents
        en_docs = load_documents_from_directory("data/english")
        print(f"  Loaded {len(en_docs)} English document(s)")

        # Load French documents
        fr_docs = load_documents_from_directory("data/french")
        print(f"  Loaded {len(fr_docs)} French document(s)")

        if len(en_docs) > 0 and len(fr_docs) > 0:
            print("✓ Document loading works correctly")
            return True
        else:
            print("✗ No documents loaded")
            return False
    except Exception as e:
        print(f"✗ Document loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_tagging():
    """Test document language tagging."""
    print("\nTesting document tagging...")
    try:
        from ingestion import load_documents_from_directory, tag_documents_with_language

        # Load and tag documents
        docs = load_documents_from_directory("data/english")
        if docs:
            tagged_docs = tag_documents_with_language(docs)

            if all('language' in doc.metadata for doc in tagged_docs):
                print(f"✓ All {len(tagged_docs)} documents tagged with language")
                return True
            else:
                print("✗ Some documents missing language tag")
                return False
        else:
            print("✗ No documents to tag")
            return False
    except Exception as e:
        print(f"✗ Document tagging failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ingestion(quick_mode=True):
    """Test document ingestion (creates index)."""
    print("\nTesting document ingestion...")
    if quick_mode:
        print("  (Skipped - would require Ollama models)")
        return True

    try:
        from ingestion import create_multilingual_index

        # This will attempt to create an index
        # It requires Ollama to be running with the right models
        index = create_multilingual_index(
            english_dir="data/english",
            french_dir="data/french",
            db_path="db/test_chroma_db"
        )

        print("✓ Document ingestion completed successfully")
        return True
    except Exception as e:
        print(f"✗ Document ingestion failed: {e}")
        print("  This is expected if Ollama models are not available")
        return False


def test_rag_engine_init(quick_mode=True):
    """Test RAG engine initialization."""
    print("\nTesting RAG engine initialization...")
    if quick_mode:
        print("  (Skipped - would require Ollama models)")
        return True

    try:
        from rag_engine import BilingualRAGEngine

        engine = BilingualRAGEngine(
            db_path="db/test_chroma_db",
            llm_model="llama3.2:latest",
            embedding_model="nomic-embed-text-v2-moe:latest"
        )

        print("✓ RAG engine initialized successfully")
        return True
    except Exception as e:
        print(f"✗ RAG engine initialization failed: {e}")
        print("  This is expected if Ollama models are not available")
        return False


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("Bilingual RAG System - Integration Tests")
    print("=" * 70)
    print("\nNote: Some tests require Ollama to be running with models pulled.")
    print("Run: ollama pull llama3.2:latest && ollama pull nomic-embed-text-v2-moe:latest")
    print()

    # Check if we're in quick mode (no Ollama required)
    quick_mode = not check_ollama()

    if quick_mode:
        print("\n⚠ Running in quick mode (Ollama not available)")
        print("  Some tests will be skipped\n")

    tests = [
        ("Language Detection", lambda: test_language_detection()),
        ("Document Loading", lambda: test_document_loading()),
        ("Document Tagging", lambda: test_document_tagging()),
        ("Document Ingestion", lambda: test_ingestion(quick_mode)),
        ("RAG Engine Init", lambda: test_rag_engine_init(quick_mode)),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nPassed: {passed}/{total}")

    if quick_mode:
        print("\n⚠ Note: Some tests were skipped due to Ollama not being available")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    # Check if dependencies are installed
    try:
        import langdetect
        import llama_index
    except ImportError as e:
        print("=" * 70)
        print("ERROR: Required dependencies not installed")
        print("=" * 70)
        print(f"\nMissing module: {e}")
        print("\nPlease run the bootstrap script first:")
        print("  ./bootstrap.sh")
        print("  conda activate LUFA_OpenSource_RAG")
        print("  python test_integration.py")
        exit(1)

    exit(main())
