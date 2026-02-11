"""
Document ingestion module for bilingual RAG system.
Handles document loading, language detection, chunking, and vector store indexing.
"""

import os
from pathlib import Path
from typing import List, Dict
from langdetect import detect, LangDetectException
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


def detect_language(text: str) -> str:
    """
    Detect the language of a text.
    
    Args:
        text: Input text to detect language
        
    Returns:
        Language code ('en' or 'fr'), defaults to 'en' if detection fails
    """
    try:
        lang = detect(text)
        # Map detected language to our supported languages
        if lang == 'fr':
            return 'fr'
        else:
            return 'en'  # Default to English for all other languages
    except LangDetectException:
        return 'en'  # Default to English if detection fails


def load_documents_from_directory(directory: str) -> List[Document]:
    """
    Load documents from a directory.
    
    Args:
        directory: Path to directory containing documents
        
    Returns:
        List of Document objects
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Skipping...")
        return []
    
    reader = SimpleDirectoryReader(directory, recursive=True)
    try:
        documents = reader.load_data()
        return documents
    except Exception as e:
        print(f"Error loading documents from {directory}: {e}")
        return []


def tag_documents_with_language(documents: List[Document]) -> List[Document]:
    """
    Tag documents with their detected language.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of Document objects with language metadata
    """
    for doc in documents:
        # Detect language from document text
        text_sample = doc.text[:1000]  # Use first 1000 chars for detection
        language = detect_language(text_sample)
        
        # Add language as metadata
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata['language'] = language
        
        print(f"Document tagged with language: {language}")
    
    return documents


def create_multilingual_index(
    english_dir: str = "data/english",
    french_dir: str = "data/french",
    db_path: str = "db/chroma_db",
    collection_name: str = "multilingual_docs",
    embedding_model: str = "bge-m3:latest"
) -> VectorStoreIndex:
    """
    Create a multilingual vector store index from English and French documents.
    
    Args:
        english_dir: Path to directory containing English documents
        french_dir: Path to directory containing French documents
        db_path: Path to ChromaDB database
        collection_name: Name of the ChromaDB collection
        embedding_model: Name of the embedding model to use
        
    Returns:
        VectorStoreIndex object
    """
    # Load documents from both directories
    print("Loading English documents...")
    english_docs = load_documents_from_directory(english_dir)
    
    print("Loading French documents...")
    french_docs = load_documents_from_directory(french_dir)
    
    # Combine documents
    all_documents = english_docs + french_docs
    
    if not all_documents:
        print("No documents found. Creating empty index...")
        all_documents = [Document(text="Empty placeholder document", metadata={"language": "en"})]
    
    # Tag documents with language
    print("Tagging documents with language...")
    all_documents = tag_documents_with_language(all_documents)
    
    # Initialize embedding model (bge-m3 is multilingual)
    print(f"Initializing embedding model: {embedding_model}")
    embed_model = OllamaEmbedding(
        model_name=embedding_model,
        base_url="http://localhost:11434",
    )
    
    # Initialize ChromaDB
    print(f"Initializing ChromaDB at {db_path}...")
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Create or get collection
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    
    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create text splitter for chunking
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    
    # Create index
    print("Creating vector store index...")
    index = VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[text_splitter],
        show_progress=True,
    )
    
    print(f"Index created successfully with {len(all_documents)} documents")
    return index


def ingest_documents(
    english_dir: str = "data/english",
    french_dir: str = "data/french",
    db_path: str = "db/chroma_db"
):
    """
    Main ingestion function to process and index all documents.
    
    Args:
        english_dir: Path to directory containing English documents
        french_dir: Path to directory containing French documents
        db_path: Path to ChromaDB database
    """
    print("Starting document ingestion...")
    index = create_multilingual_index(english_dir, french_dir, db_path)
    print("Document ingestion completed!")
    return index


if __name__ == "__main__":
    # Run ingestion when script is executed directly
    ingest_documents()
