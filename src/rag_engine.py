"""
RAG engine module for bilingual question-answering.
Handles query processing, cross-lingual retrieval, and response generation.
"""
import re
from rank_bm25 import BM25Okapi
from llama_index.core.schema import NodeWithScore
from typing import List, Optional
from language_detector import detect_language
from query_handler import QueryHandler, SYSTEM_PROMPTS
from query_rewriter import rewrite_query
from reflector import reflect
from translator import (                          # ── TRANSLATION: new imports
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
    Features:
    - Auto-detect query language (all languages)
    - Translate non-EN/FR queries → English → RAG pipeline → translate answer back
    - Cross-lingual retrieval with nomic-embed-text-v2-moe
    - Agentic loop: query rewriting + reflection + adaptive re-retrieval
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

    # ── Unchanged internals ───────────────────────────────────────────────────

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

    # ── Original single-pass query — unchanged ────────────────────────────────

    def query(self, query_text: str, return_sources: bool = False) -> dict:
        """Standard single-pass RAG query (original behaviour, fully preserved)."""
        detected_language = self.detect_query_language(query_text)
        print(f"Detected query language: {detected_language}")
        enhanced_query    = self.create_language_aware_prompt(query_text, detected_language)
        response          = self.query_engine.query(enhanced_query)

        result = {
            'response':          str(response),
            'detected_language': detected_language,
        }

        if return_sources and hasattr(response, 'source_nodes'):
            sources = []
            for node in response.source_nodes:
                text    = node.node.text
                preview = text[:PREVIEW_LENGTH] + ('...' if len(text) > PREVIEW_LENGTH else '')
                sources.append({
                    'text':     preview,
                    'score':    node.score,
                    'metadata': node.node.metadata,
                })
            result['sources'] = sources

        return result

    def chat(self, query_text: str) -> str:
        return self.query(query_text, return_sources=False)['response']

    # ── Agentic helpers ───────────────────────────────────────────────────────

    # def _retrieve_nodes(self, query: str, top_k: int):
    #     retriever = VectorIndexRetriever(
    #         index=self.index,
    #         similarity_top_k=top_k,
    #     )
    #
    #     #def _retrieve_hybrid_nodes(self, query: str, top_k: int = 5) -> list:
    #     return retriever.retrieve(query)

    # Add this method to your BilingualRAGEngine class in src/rag_engine.py
    # Make sure to import: from rank_bm25 import BM25Okapi
    # and tokenize helper: import re

    def _retrieve_nodes(self, query: str, top_k: int = 5) -> list:
        """
        Executes hybrid retrieval by combining dense vector scores from ChromaDB
        and sparse token matching scores from rank-bm25, fused via RRF.
        """
        # 1. Fetch dense candidates
        dense_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k * 2,
        )
        dense_nodes = dense_retriever.retrieve(query)

        # 2. Extract text pool for BM25 from the index
        # For a fully dynamic production system, you can pull corpus nodes directly:
        all_nodes = list(self.index.docstore.docs.values())
        tokenized_corpus = [re.findall(r'\b\w+\b', node.text.lower()) for node in all_nodes]
        bm25 = BM25Okapi(tokenized_corpus)

        # 3. Fetch sparse candidates
        tokenized_query = re.findall(r'\b\w+\b', query.lower())
        bm25_scores = bm25.get_scores(tokenized_query)

        # Sort top nodes based on BM25 score
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
        sparse_nodes = [all_nodes[idx] for idx in top_bm25_indices]

        # 4. Apply Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        node_mapping = {}

        for rank, node in enumerate(dense_nodes, start=1):
            node_id = node.node.node_id
            node_mapping[node_id] = node
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + (1.0 / (60 + rank))

        for rank, node in enumerate(sparse_nodes, start=1):
            node_id = node.node_id
            if node_id not in node_mapping:
                # Wrap standard node in NodeWithScore for compatibility
                from llama_index.core.schema import NodeWithScore
                node_mapping[node_id] = NodeWithScore(node=node, score=0.0)
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + (1.0 / (60 + rank))

        # Sort by unified RRF ranking
        sorted_node_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]

        final_nodes = []
        for nid in sorted_node_ids:
            wrapped_node = node_mapping[nid]
            # Assign unified fusion calculation score
            wrapped_node.score = rrf_scores[nid]
            final_nodes.append(wrapped_node)

        return final_nodes

    def _generate_from_nodes(self, original_query: str, nodes, lang: str) -> str:
        context     = "\n\n---\n\n".join([n.node.text for n in nodes])
        system      = self.query_handler.get_system_prompt(lang)
        instruction = "Réponds en français." if lang == "fr" else "Respond in English."

        prompt = f"""{system}
{instruction}
Answer ONLY using the context below. Cite the source document name and page number 
for every claim.

Context:
{context}

Question: {original_query}
Answer:"""
        return str(self.llm.complete(prompt)).strip()

    # ── Agentic query — with translation layer ────────────────────────────────

    def agentic_query(
        self,
        query_text:     str,
        return_sources: bool = False,
        max_retries:    int  = MAX_RETRIES
    ) -> dict:
        """
        Agentic RAG query with full translation + 4-step loop:

          [PRE]  Detect original language
                 If not EN/FR → translate query to English
          [1]    Query Rewriting   (in EN or FR)
          [2]    Retrieval         (top-K, widens on retry)
          [3]    Generation        (grounded answer in EN or FR)
          [4]    Reflection        (GROUNDED / UNGROUNDED → loop)
          [POST] If original language was not EN/FR → translate answer back

        Args:
            query_text:    Raw user query in any language
            return_sources: Include source metadata in result
            max_retries:   Maximum agent loop iterations

        Returns:
            dict with keys:
              response, detected_language, original_language,
              translated_query (if applicable), rewritten_query,
              attempts, grounded, translation_applied, sources (optional)
        """

        # ── PRE-PROCESSING: language detection + translation ──────────────────
        original_lang      = detect_full_language(query_text)
        translation_applied = needs_translation(original_lang)
        translated_query   = None

        if translation_applied:
            lang_name = LANGUAGE_NAMES.get(original_lang, original_lang.upper())
            print(f"[AgenticRAG] Non-native language detected: {original_lang} ({lang_name})")
            print(f"[AgenticRAG] Translating query to English for processing...")
            translated_query = translate_to_english(query_text, original_lang, self.llm)
            processing_query = translated_query
            pipeline_lang    = "en"   # always process in English for non-native langs
        else:
            processing_query = query_text
            pipeline_lang    = original_lang   # "en" or "fr" — process natively

        print(f"[AgenticRAG] Pipeline language: {pipeline_lang}")

        # ── AGENT LOOP ────────────────────────────────────────────────────────
        current_query    = processing_query
        rewritten_query  = processing_query
        nodes            = []
        answer           = ""
        is_grounded      = False

        for attempt in range(1, max_retries + 1):
            print(f"[AgenticRAG] Attempt {attempt}/{max_retries}")

            # Step 1 — Query Rewriting
            if attempt == 1:
                rewritten_query = rewrite_query(current_query, pipeline_lang, self.llm)
            else:
                hint = (
                    f"{current_query} "
                    f"Focus on specific article numbers, salary grids, dates, or job ranks."
                )
                rewritten_query = rewrite_query(hint, pipeline_lang, self.llm)

            print(f"[AgenticRAG] Rewritten: {rewritten_query}")

            # Step 2 — Retrieval (broaden k on retries: 5 → 7 → 9)
            top_k = self.similarity_top_k + (attempt - 1) * 2
            nodes = self._retrieve_nodes(rewritten_query, top_k=top_k)
            print(f"[AgenticRAG] Retrieved {len(nodes)} chunks (top_k={top_k})")

            # Step 3 — Generation (always in EN or FR inside the pipeline)
            answer = self._generate_from_nodes(processing_query, nodes, pipeline_lang)

            # Step 4 — Reflection
            chunk_texts = [n.node.text for n in nodes]
            is_grounded = reflect(answer, chunk_texts, self.llm)
            print(f"[AgenticRAG] Grounded: {is_grounded}")

            if is_grounded:
                break

            current_query = rewritten_query   # refine from last rewrite on retry

        # ── POST-PROCESSING: translate answer back to original language ───────
        final_answer = answer
        if translation_applied:
            lang_name = LANGUAGE_NAMES.get(original_lang, original_lang.upper())
            print(f"[AgenticRAG] Translating answer back to {lang_name}...")
            final_answer = translate_to_target(answer, original_lang, self.llm)

        # ── Build result ──────────────────────────────────────────────────────
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
                })
            result['sources'] = sources

        return result


def create_rag_engine(
    db_path:         str = "db/chroma_db",
    llm_model:       str = "llama3.2:3b-instruct-q4_K_M",
    embedding_model: str = "nomic-embed-text-v2-moe"
) -> BilingualRAGEngine:
    return BilingualRAGEngine(
        db_path=db_path,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )


if __name__ == "__main__":
    engine = create_rag_engine()

    tests = [
        ("What is the salary grid for 2024?",   "English"),
        ("Quelles sont les heures de bureau?",  "French"),
        ("¿Cuáles son las horas de oficina?",   "Spanish → EN → answer → ES"),
        ("事務時間は何ですか？",                   "Japanese → EN → answer → JA"),
        ("Welche Gehaltsklasse gilt für 2024?", "German  → EN → answer → DE"),
    ]

    for query, label in tests:
        print(f"\n{'─'*60}")
        print(f"[TEST] {label}")
        print(f"Query: {query}")
        r = engine.agentic_query(query, return_sources=False)
        print(f"Original lang:       {r['original_language']}")
        print(f"Translation applied: {r['translation_applied']}")
        if r['translated_query']:
            print(f"Translated query:    {r['translated_query']}")
        print(f"Rewritten query:     {r['rewritten_query']}")
        print(f"Attempts:            {r['attempts']}")
        print(f"Grounded:            {r['grounded']}")
        print(f"Answer:              {r['response']}")