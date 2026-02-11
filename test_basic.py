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
        if any(package in req for req in requirements):
            print(f"✓ Package listed: {package}")
        else:
            print(f"✗ Package missing: {package}")
            all_present = False
    
    return all_present


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
