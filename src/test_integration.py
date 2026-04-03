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
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
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
