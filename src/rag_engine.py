"""
RAG engine module for bilingual question-answering.
Handles query processing, cross-lingual retrieval, and response generation.
"""

from typing import List, Optional
from langdetect import detect, LangDetectException
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Constants
PREVIEW_LENGTH = 200  # Length of text preview for source documents


class BilingualRAGEngine:
    """
    Bilingual RAG engine that supports English and French queries.
    Features:
    - Auto-detect query language
    - Cross-lingual retrieval
    - Respond in the query language
    """
    
    def __init__(
        self,
        db_path: str = "db/chroma_db",
        collection_name: str = "multilingual_docs",
        llm_model: str = "llama3.2:latest",
        embedding_model: str = "bge-m3:latest",
        similarity_top_k: int = 3
    ):
        """
        Initialize the bilingual RAG engine.
        
        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the ChromaDB collection
            llm_model: Name of the LLM model to use
            embedding_model: Name of the embedding model to use
            similarity_top_k: Number of similar documents to retrieve
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k
        
        # Initialize LLM
        print(f"Initializing LLM: {llm_model}")
        self.llm = Ollama(
            model=llm_model,
            base_url="http://localhost:11434",
            request_timeout=120.0,
        )
        
        # Initialize embedding model
        print(f"Initializing embedding model: {embedding_model}")
        self.embed_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url="http://localhost:11434",
        )
        
        # Initialize index
        self.index = self._load_index()
        
        # Create query engine
        self.query_engine = self._create_query_engine()
    
    def _load_index(self) -> VectorStoreIndex:
        """
        Load the vector store index from ChromaDB.
        
        Returns:
            VectorStoreIndex object
        """
        print(f"Loading index from {self.db_path}...")
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # Get collection
        chroma_collection = chroma_client.get_or_create_collection(self.collection_name)
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create index from existing vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model,
        )
        
        print("Index loaded successfully")
        return index
    
    def set_similarity_top_k(self, top_k: int):
        """
        Update the number of similar documents to retrieve.
        
        Args:
            top_k: Number of documents to retrieve
        """
        self.similarity_top_k = top_k
        if hasattr(self, 'query_engine') and hasattr(self.query_engine, 'retriever'):
            # Note: Direct access to _similarity_top_k is necessary because LlamaIndex
            # VectorIndexRetriever doesn't provide a public setter method
            self.query_engine.retriever._similarity_top_k = top_k
    
    def _create_query_engine(self) -> RetrieverQueryEngine:
        """
        Create a query engine with custom retriever and response synthesizer.
        
        Returns:
            RetrieverQueryEngine object
        """
        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.similarity_top_k,
        )
        
        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode="compact",
        )
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        
        return query_engine
    
    def detect_query_language(self, query: str) -> str:
        """
        Detect the language of a query.
        
        Args:
            query: Input query text
            
        Returns:
            Language code ('en' or 'fr')
        """
        try:
            lang = detect(query)
            if lang == 'fr':
                return 'fr'
            else:
                return 'en'
        except LangDetectException:
            return 'en'  # Default to English
    
    def create_language_aware_prompt(self, query: str, language: str) -> str:
        """
        Create a language-aware prompt that instructs the model to respond in the query language.
        
        Args:
            query: Original user query
            language: Detected language code
            
        Returns:
            Enhanced query with language instruction
        """
        if language == 'fr':
            instruction = "Réponds en français. "
        else:
            instruction = "Respond in English. "
        
        return instruction + query
    
    def query(self, query_text: str, return_sources: bool = False) -> dict:
        """
        Process a query and return the response.
        
        Args:
            query_text: User query text
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary containing response and metadata
        """
        # Detect query language
        detected_language = self.detect_query_language(query_text)
        print(f"Detected query language: {detected_language}")
        
        # Create language-aware prompt
        enhanced_query = self.create_language_aware_prompt(query_text, detected_language)
        
        # Execute query
        response = self.query_engine.query(enhanced_query)
        
        # Prepare result
        result = {
            'response': str(response),
            'detected_language': detected_language,
        }
        
        # Add source information if requested
        if return_sources and hasattr(response, 'source_nodes'):
            sources = []
            for node in response.source_nodes:
                text = node.node.text
                preview = text[:PREVIEW_LENGTH] + ('...' if len(text) > PREVIEW_LENGTH else '')
                source_info = {
                    'text': preview,
                    'score': node.score,
                    'metadata': node.node.metadata,
                }
                sources.append(source_info)
            result['sources'] = sources
        
        return result
    
    def chat(self, query_text: str) -> str:
        """
        Simple chat interface that returns just the response text.
        
        Args:
            query_text: User query text
            
        Returns:
            Response text
        """
        result = self.query(query_text, return_sources=False)
        return result['response']


def create_rag_engine(
    db_path: str = "db/chroma_db",
    llm_model: str = "llama3.2:latest",
    embedding_model: str = "bge-m3:latest"
) -> BilingualRAGEngine:
    """
    Factory function to create a BilingualRAGEngine instance.
    
    Args:
        db_path: Path to ChromaDB database
        llm_model: Name of the LLM model to use
        embedding_model: Name of the embedding model to use
        
    Returns:
        BilingualRAGEngine instance
    """
    return BilingualRAGEngine(
        db_path=db_path,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )


if __name__ == "__main__":
    # Test the RAG engine
    print("Testing RAG engine...")
    engine = create_rag_engine()
    
    # Test with English query
    print("\nTesting English query:")
    result = engine.query("What is this document about?", return_sources=True)
    print(f"Response: {result['response']}")
    print(f"Language: {result['detected_language']}")
    
    # Test with French query
    print("\nTesting French query:")
    result = engine.query("De quoi parle ce document?", return_sources=True)
    print(f"Response: {result['response']}")
    print(f"Language: {result['detected_language']}")
