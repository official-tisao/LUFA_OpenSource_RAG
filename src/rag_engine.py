#!/usr/bin/env python3
"""
RAG engine module for bilingual question-answering.
Handles query processing, cross-lingual hybrid retrieval, and response generation.
"""
import re
from rank_bm25 import BM25Okapi
from llama_index.core.schema import NodeWithScore
from typing import List, Optional
from language_detector import detect_language
from query_handler import QueryHandler, SYSTEM_PROMPTS
from query_rewriter import rewrite_query
from reflector import reflect
from translator import (
    detect_full_language,
    needs_translation,
    translate_to_english,
    translate_to_target,
    LANGUAGE_NAMES,
)
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

PREVIEW_LENGTH = 200
MAX_RETRIES    = 3


class BilingualRAGEngine:
    """
    Bilingual RAG engine for Laurentian University Faculty Association collective agreement.
    Features cross-lingual retrieval, query rewriting, and hybrid RRF retrieval.
    """

    def __init__(
        self,
        db_path: str = "db/chroma_db",
        collection_name: str = "multilingual_docs",
        llm_model: str = "llama3.2:3b-instruct-q4_K_M",
        embedding_model: str = "nomic-embed-text-v2-moe",
        similarity_top_k: int = 5
    ):
        self.db_path          = db_path
        self.collection_name  = collection_name
        self.similarity_top_k = similarity_top_k
        self.query_handler    = QueryHandler()

        print(f"Initializing LLM: {llm_model}")
        self.llm = Ollama(
            model=llm_model,
            base_url="http://localhost:11434",
            request_timeout=120.0,
        )

        print(f"Initializing embedding model: {embedding_model}")
        self.embed_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url="http://localhost:11434",
        )

        self.index        = self._load_index()
        self.query_engine = self._create_query_engine()

    def _load_index(self) -> VectorStoreIndex:
        print(f"Loading index from {self.db_path}...")
        chroma_client     = chromadb.PersistentClient(path=self.db_path)
        chroma_collection = chroma_client.get_or_create_collection(self.collection_name)
        vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model,
        )
        print("Index loaded successfully")
        return index

    def set_similarity_top_k(self, top_k: int):
        self.similarity_top_k = top_k
        if hasattr(self, 'query_engine') and hasattr(self.query_engine, 'retriever'):
            self.query_engine.retriever._similarity_top_k = top_k

    def _create_query_engine(self) -> RetrieverQueryEngine:
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.similarity_top_k,
        )
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode="compact",
        )
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

    def detect_query_language(self, query: str) -> str:
        return self.query_handler.detect_query_language(query)

    def create_language_aware_prompt(self, query: str, language: str) -> str:
        system_prompt = self.query_handler.get_system_prompt(language)
        instruction   = "Réponds en français. " if language == 'fr' else "Respond in English. "
        return instruction + query

    def query(self, query_text: str, return_sources: bool = False) -> dict:
        """Standard single-pass RAG query."""
        detected_language = self.detect_query_language(query_text)
        print(f"Detected query language: {detected_language}")
        enhanced_query    = self.create_language_aware_prompt(query_text, detected_language)
        response          = self.query_engine.query(enhanced_query)
