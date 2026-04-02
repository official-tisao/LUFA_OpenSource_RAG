        print(f"✓ Overly long response fallback: '{result}'")

        # French language uses French prompt (verify no crash)
        result = rewrite_query("vacances?", "fr", MockLLM())
        assert result == "What are the vacation entitlements in the 2020-2025 LUFA collective agreement?", \
            f"Unexpected French rewrite: '{result}'"
        print(f"✓ French language rewrite: '{result}'")

        return True
    except Exception as e:
        print(f"✗ Query rewriter test failed: {e}")
        return False


def test_reflector():
    """Test grounding verification with a stubbed LLM."""
    print("\nTesting reflector...")
    try:
        from reflector import reflect

        CHUNKS = ["The vacation policy states 20 days per year."]

        class MockLLM:
            def __init__(self, response):
                self._response = response

            def complete(self, prompt):
                return self._response

        # GROUNDED → returns True
        result = reflect("Employees get 20 days vacation.", CHUNKS, MockLLM("GROUNDED"))
        assert result is True, f"Expected True for GROUNDED, got {result}"
        print("✓ GROUNDED → True")

        # UNGROUNDED → returns False
        result = reflect("Employees get 50 days vacation.", CHUNKS, MockLLM("UNGROUNDED"))
        assert result is False, f"Expected False for UNGROUNDED, got {result}"
        print("✓ UNGROUNDED → False")

        # UNGROUNDED with punctuation must still not be treated as GROUNDED
        result = reflect("Employees get 50 days vacation.", CHUNKS, MockLLM("UNGROUNDED."))
        assert result is False, f"Expected False for 'UNGROUNDED.', got {result}"
        print("✓ 'UNGROUNDED.' punctuation check → False")

        # Tokens starting with GROUNDED but not exactly equal must not match
        result = reflect("Employees get 50 days vacation.", CHUNKS, MockLLM("GROUNDEDNESS"))
        assert result is False, f"Expected False for 'GROUNDEDNESS', got {result}"
        print("✓ 'GROUNDEDNESS' → False (exact token check)")

        # Empty chunks → always False (no source to ground against)
        result = reflect("Some answer.", [], MockLLM("GROUNDED"))
        assert result is False, f"Expected False for empty chunks, got {result}"
        print("✓ Empty chunks → False")

        # Exception → fail-closed (False)
        class FailingLLM:
            def complete(self, prompt):
                raise RuntimeError("LLM error")

        result = reflect("Some answer.", CHUNKS, FailingLLM())
        assert result is False, f"Expected False (fail-closed) on exception, got {result}"
        print("✓ Exception → False (fail-closed)")

        return True
    except Exception as e:
        print(f"✗ Reflector test failed: {e}")
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
        ("Query Rewriter", test_query_rewriter),
        ("Reflector", test_reflector),
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
