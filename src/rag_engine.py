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
from query_handler import QueryHandler
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
from model_api_auth import get_ollama_client
from config_loader import cfg
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
        db_path: str = None,
        collection_name: str = None,
        llm_model: str = None,
        embedding_model: str = None,
        similarity_top_k: int = None
    ):
        # Defaults from config/config.yaml
        db_path          = db_path          or cfg("database.path")
        collection_name  = collection_name  or cfg("database.collection_name")
        llm_model        = llm_model        or cfg("models.llm.name")
        embedding_model  = embedding_model  or cfg("models.embedding.name")
        similarity_top_k = similarity_top_k or cfg("retrieval.top_k")
        self.db_path          = db_path
        self.collection_name  = collection_name
        self.similarity_top_k = similarity_top_k
        self.query_handler    = QueryHandler()

        print(f"Initializing LLM: {llm_model}")
        self.llm = get_ollama_client(llm_model, request_timeout=240.0)

        print(f"Initializing embedding model: {embedding_model}")
        self.embed_model = get_ollama_client(embedding_model, is_embedding=True)

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
        query = self.query_handler.augment_query_with_year(query, language)
        instruction = "Réponds en français. " if language == "fr" else "Respond in English. "
        return instruction + query

    def query(self, query_text: str, return_sources: bool = False) -> dict:
        """Standard single-pass RAG query."""
        detected_language = self.detect_query_language(query_text)
        print(f"Detected query language: {detected_language}")
        enhanced_query    = self.create_language_aware_prompt(query_text, detected_language)
        response          = self.query_engine.query(enhanced_query)

        result = {
            'response': str(response),
            'detected_language': detected_language,
        }

        if return_sources and hasattr(response, 'source_nodes'):
            sources = []
            for node in response.source_nodes:
                text = node.node.text
                preview = text[:PREVIEW_LENGTH] + ('...' if len(text) > PREVIEW_LENGTH else '')
                sources.append({
                    'text': preview,
                    'score': node.score,
                    'metadata': node.node.metadata,
                    'node_id': node.node.node_id
                })
            result['sources'] = sources

        return result

    def chat(self, query_text: str) -> str:
        return self.query(query_text, return_sources=False)['response']

    def _retrieve_nodes(self, query: str, top_k: int = 5) -> list:
        """
        Executes advanced hybrid retrieval. Sorts dense vector nodes by recency weights
        during similarity ties, then merges them with sparse BM25 tokens via RRF.
        Directly reads from ChromaDB to avoid ZeroDivisionError.
        """
        dense_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k * 2,
        )
        raw_dense_nodes = dense_retriever.retrieve(query)

        if raw_dense_nodes:
            raw_dense_nodes.sort(key=lambda x: x.score, reverse=True)
            dense_nodes = []
            i = 0
            tie_threshold = 0.02

            while i < len(raw_dense_nodes):
                bucket_score = raw_dense_nodes[i].score
                bucket = []
                j = i
                while j < len(raw_dense_nodes) and (bucket_score - raw_dense_nodes[j].score) < tie_threshold:
                    bucket.append(raw_dense_nodes[j])
                    j += 1

                bucket.sort(key=lambda x: float(x.node.metadata.get("recency_weight", 1.0)), reverse=True)
                dense_nodes.extend(bucket)
                i = j
        else:
            dense_nodes = []

        from llama_index.core.schema import TextNode
        chroma_client = chromadb.PersistentClient(path=self.db_path)
        chroma_collection = chroma_client.get_or_create_collection(self.collection_name)
        collection_data = chroma_collection.get(include=["documents", "metadatas"])

        all_nodes = []
        if collection_data and collection_data.get("ids"):
            for cid, doc, meta in zip(collection_data["ids"], collection_data["documents"],
                                      collection_data["metadatas"]):
                node = TextNode(id_=cid, text=doc, metadata=meta)
                all_nodes.append(node)

        if not all_nodes:
            return dense_nodes

        tokenized_corpus = [re.findall(r'\b\w+\b', node.text.lower()) for node in all_nodes]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = re.findall(r'\b\w+\b', query.lower())
        bm25_scores = bm25.get_scores(tokenized_query)

        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda idx: bm25_scores[idx], reverse=True)[:top_k * 2]
        sparse_nodes = [all_nodes[idx] for idx in top_bm25_indices]

        rrf_scores = {}
        node_mapping = {}

        for rank, node in enumerate(dense_nodes, start=1):
            node_id = node.node.node_id
            node_mapping[node_id] = node
            node.node.metadata["original_cosine_score"] = str(node.score)
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + (1.0 / (60 + rank))

        for rank, node in enumerate(sparse_nodes, start=1):
            node_id = node.node_id
            if node_id not in node_mapping:
                from llama_index.core.schema import NodeWithScore
                node_mapping[node_id] = NodeWithScore(node=node, score=0.0)
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + (1.0 / (60 + rank))

        sorted_node_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]

        final_nodes = []
        for nid in sorted_node_ids:
            wrapped_node = node_mapping[nid]
            wrapped_node.score = rrf_scores[nid]
            final_nodes.append(wrapped_node)

        return final_nodes

    def _generate_from_nodes(self, original_query: str, nodes, lang: str) -> str:

        context     = "\n\n---\n\n".join([n.node.text for n in nodes])
        system      = self.query_handler.get_system_prompt(lang)
        instruction = "Réponds en français." if lang == "fr" else "Respond in English."

        prompt = f"""{system}
{instruction}
Answer ONLY using the context below.
Cite the source document name and page number for every claim.
Context:
{context}

Question: {original_query}
Answer:"""
        return str(self.llm.complete(prompt)).strip()

    def agentic_query(
        self,
        query_text:     str,
        return_sources: bool = False,
        max_retries:    int  = MAX_RETRIES
    ) -> dict:
        """Agentic RAG query loop with cross-lingual support."""
        original_lang      = detect_full_language(query_text)
        translation_applied = needs_translation(original_lang)
        translated_query   = None

        if translation_applied:
            lang_name = LANGUAGE_NAMES.get(original_lang, original_lang.upper())
            print(f"[AgenticRAG] Non-native language detected: {original_lang} ({lang_name})")
            print(f"[AgenticRAG] Translating query to English for processing...")
            translated_query = translate_to_english(query_text, original_lang, self.llm)
            processing_query = translated_query
            pipeline_lang    = "en"
        else:
            processing_query = query_text
            pipeline_lang    = original_lang

        print(f"[AgenticRAG] Pipeline language: {pipeline_lang}")

        current_query    = processing_query
        rewritten_query  = processing_query
        nodes            = []
        answer           = ""
        is_grounded      = False

        for attempt in range(1, max_retries + 1):
            print(f"[AgenticRAG] Attempt {attempt}/{max_retries}")
            rewritten_query = rewrite_query(current_query, pipeline_lang, self.llm)


            print(f"[AgenticRAG] Rewritten: {rewritten_query}")

            top_k = self.similarity_top_k + (attempt-1)
            nodes = self._retrieve_nodes(rewritten_query, top_k=top_k)
            print(f"[AgenticRAG] Retrieved {len(nodes)} chunks (top_k={top_k})")

            answer = self._generate_from_nodes(processing_query, nodes, pipeline_lang)

            chunk_texts = [n.node.text for n in nodes]
            is_grounded = reflect(answer, chunk_texts, self.llm)
            print(f"[AgenticRAG] Grounded: {is_grounded}")

            if is_grounded:
                break

            current_query = rewritten_query

        final_answer = answer
        if translation_applied:
            lang_name = LANGUAGE_NAMES.get(original_lang, original_lang.upper())
            print(f"[AgenticRAG] Translating answer back to {lang_name}...")
            final_answer = translate_to_target(answer, original_lang, self.llm)

        result = {
            'response':            final_answer,
            'english_response':    answer if translation_applied else None,
            'detected_language':   pipeline_lang,
            'original_language':   original_lang,
            'original_query':      query_text,
            'translated_query':    translated_query,
            'rewritten_query':     rewritten_query,
            'attempts':            attempt,
            'grounded':            is_grounded,
            'translation_applied': translation_applied,
        }

        if return_sources:
            sources = []
            for node in nodes:
                text    = node.node.text
                preview = text[:PREVIEW_LENGTH] + ('...' if len(text) > PREVIEW_LENGTH else '')
                sources.append({
                    'text':     preview,
                    'score':    node.score,
                    'metadata': node.node.metadata,
                    'node_id':  node.node.node_id
                })
            result['sources'] = sources

        return result


    def naive_query(
            self,
            query_text:     str,
            return_sources: bool = False
    ) -> dict:
        """Naive RAG query with cross-lingual support."""
        max_retries = 1
        original_lang      = detect_full_language(query_text)
        translation_applied = needs_translation(original_lang)
        translated_query   = None

        if translation_applied:
            lang_name = LANGUAGE_NAMES.get(original_lang, original_lang.upper())
            print(f"[NaiveRAG] Non-native language detected: {original_lang} ({lang_name})")
            print(f"[NaiveRAG] Translating query to English for processing...")
            translated_query = translate_to_english(query_text, original_lang, self.llm)
            processing_query = translated_query
            pipeline_lang    = "en"
        else:
            processing_query = query_text
            pipeline_lang    = original_lang

        print(f"[NaiveRAG] Pipeline language: {pipeline_lang}")

        rewritten_query  = processing_query
        nodes            = []
        answer           = ""
        is_grounded      = False

        #for attempt in range(1, max_retries + 1):
        top_k = self.similarity_top_k
        nodes = self._retrieve_nodes(processing_query, top_k=top_k)
        print(f"[NaiveRAG] Retrieved {len(nodes)} chunks (top_k={top_k})")

        answer = self._generate_from_nodes(processing_query, nodes, pipeline_lang)

            # chunk_texts = [n.node.text for n in nodes]
            # is_grounded = reflect(answer, chunk_texts, self.llm)
            # print(f"[NaiveRAG] Grounded: {is_grounded}")

            # if is_grounded:
            #    break

            #current_query = rewritten_query

        final_answer = answer
        if translation_applied:
            lang_name = LANGUAGE_NAMES.get(original_lang, original_lang.upper())
            print(f"[NaiveRAG] Translating answer back to {lang_name}...")
            final_answer = translate_to_target(answer, original_lang, self.llm)

        result = {
            'response':            final_answer,
            'english_response':    answer if translation_applied else None,
            'detected_language':   pipeline_lang,
            'original_language':   original_lang,
            'original_query':      query_text,
            'translated_query':    translated_query,
            'rewritten_query':     rewritten_query,
            'attempts':            1,
            'grounded':            is_grounded,
            'translation_applied': translation_applied,
        }

        if return_sources:
            sources = []
            for node in nodes:
                text    = node.node.text
                preview = text[:PREVIEW_LENGTH] + ('...' if len(text) > PREVIEW_LENGTH else '')
                sources.append({
                    'text':     preview,
                    'score':    node.score,
                    'metadata': node.node.metadata,
                    'node_id':  node.node.node_id
                })
            result['sources'] = sources

        return result



def create_rag_engine(
    db_path:         str = None,
    llm_model:       str = None,
    embedding_model: str = None
) -> BilingualRAGEngine:
    return BilingualRAGEngine(
        db_path=db_path,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )