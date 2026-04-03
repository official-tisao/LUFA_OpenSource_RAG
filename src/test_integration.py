        print(f"✗ RAG engine initialization failed: {e}")
        print("  This is expected if Ollama models are not available")
        return False


def main():
    """Run all integration tests."""
    print("="*70)
    print("Bilingual RAG System - Integration Tests")
    print("="*70)
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
    
    print("\n" + "="*70)
    print("Test Results Summary")
    print("="*70)
    
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
        print("="*70)
        print("ERROR: Required dependencies not installed")
        print("="*70)
        print(f"\nMissing module: {e}")
        print("\nPlease run the bootstrap script first:")
        print("  ./bootstrap.sh")
        print("  conda activate LUFA_OpenSource_RAG")
        print("  python test_integration.py")
        exit(1)
    
    exit(main())
