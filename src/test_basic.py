

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


def test_query_rewriter():
    """Test query rewriting with a stubbed LLM."""
    print("\nTesting query rewriter...")
    try:
        from query_rewriter import rewrite_query

        class MockLLM:
            def complete(self, prompt):
                return "What are the vacation entitlements in the 2020-2025 LUFA collective agreement?"

        # Normal rewrite returns the LLM output
        result = rewrite_query("vacation?", "en", MockLLM())
        assert result == "What are the vacation entitlements in the 2020-2025 LUFA collective agreement?", \
            f"Unexpected rewrite: '{result}'"
        print(f"✓ Normal rewrite: '{result}'")

        # Exception → falls back to original query
        class FailingLLM:
            def complete(self, prompt):
                raise RuntimeError("LLM error")

        result = rewrite_query("vacation?", "en", FailingLLM())
        assert result == "vacation?", f"Expected original query on failure, got: '{result}'"
        print(f"✓ Exception fallback: '{result}'")

        # Empty response → falls back to original query
        class EmptyLLM:
            def complete(self, prompt):
                return ""

        result = rewrite_query("vacation?", "en", EmptyLLM())
        assert result == "vacation?", f"Expected original query for empty response, got: '{result}'"
        print(f"✓ Empty response fallback: '{result}'")

        # Overly long response (>= 400 chars) → falls back to original query
        class LongLLM:
            def complete(self, prompt):
                return "x" * 401

        result = rewrite_query("vacation?", "en", LongLLM())
        assert result == "vacation?", f"Expected original query for long response, got: '{result}'"
