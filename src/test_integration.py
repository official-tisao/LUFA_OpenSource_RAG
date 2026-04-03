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
