#!/usr/bin/env python3
"""
LUFA RAG System - Main Package Initialization

This package provides modular components for the LUFA RAG (Laurentian University Faculty Association
RAG) system. The system provides:

1. Metrics Module - Unified scoring for generation, retrieval, and judge metrics
2. CSV Utilities - CSV I/O with caching and error handling
3. Retrieval Module - Standalone retrieval functions for top-k chunks
4. Question Rewriter - Single and batch question rewriting with structured prompts
5. Ground Truth Module - Ground truth finding with multi-chunk expansion
6. Answer Generator - Naive and agentic answer generation with proper separation of concerns

All components are designed to be importable and usable independently or together
for building modular RAG pipelines.

Example Usage:
    # Import all components
    from src import metrics, csv_utils, retrieval, question_rewriter, ground_truth, answer_generator

    # Use metrics
    f1 = metrics.token_f1(prediction, reference)

    # Use retrieval
    results = retrieval.retrieve_from_dataframe(questions_df, engine)

    # Use question rewriter
    rewritten = question_rewriter.rewrite_questions_dataframe(questions_df, engine, llm)

    # Use ground truth
    ground_truth_df = ground_truth.find_ground_truth_for_dataframe(questions_df, db_path)

    # Use answer generator
    answer = answer_generator.naive_rag(engine, question_text)

All CSV utilities provide both path-based and DataFrame-returning variants for flexibility.
"""

# Version information
__version__ = "1.0.0"

# Package imports for backward compatibility and easy access
try:
    from . import metrics
    from . import csv_utils
    from . import retrieval
    from . import question_rewriter
    from . import ground_truth
    from . import answer_generator
    from .rag_engine import BilingualRAGEngine
    from .run_simulation import query_single_record

    # Convenience functions for common use cases
    def create_rag_engine(db_path=None, llm_model=None, embedding_model=None):
        """Create a bilingual RAG engine with default settings."""
        return BilingualRAGEngine(db_path=db_path, llm_model=llm_model, embedding_model=embedding_model)

    # Export key functions for easy access
    __all__ = [
        # Core components
        'metrics',
        'csv_utils',
        'retrieval',
        'question_rewriter',
        'ground_truth',
        'answer_generator',

        # Main classes
        'BilingualRAGEngine',

        # Helper functions
        'create_rag_engine',
        'query_single_record',

        # Import shortcuts for convenience
        'token_f1', 'compute_bleu', 'compute_rouge', 'compute_meteor',
        'recall_at_k', 'mrr', 'ndcg_at_k',
        'read_csv_cached', 'write_csv_row', 'get_completed_ids', 'ensure_columns',
        'retrieve_top_k', 'retrieve_from_dataframe',
        'rewrite_questions_batch', 'rewrite_single_question_batch',
        'find_ground_truth_for_questions', 'find_ground_truth_for_questions_path',
        'naive_rag', 'agentic_rag',
    ]

    # List of modules and functions available for import
    # (This helps users discover what's available)
    def discover_modules():
        """Return a list of available modules and functions."""
        return {
            'metrics': ['token_f1', 'compute_bleu', 'compute_rouge', 'compute_meteor', 'mrr', 'ndcg_at_k', 'recall_at_k'],
            'csv_utils': ['read_csv_cached', 'write_csv_row', 'get_completed_ids', 'ensure_columns', 'backup_csv', 'export_json'],
            'retrieval': ['retrieve_top_k', 'retrieve_from_dataframe', 'retrieve_from_question_row', 'retrieve_from_csv_path', 'extract_sources_from_nodes', 'sources_to_csv_row'],
            'question_rewriter': ['rewrite_single_question', 'rewrite_single_question_batch', 'rewrite_questions_batch', 'rewrite_questions_dataframe', 'rewrite_questions_from_csv', 'rewrite_batch_prompt', 'parse_batch_rewrite_response'],
            'ground_truth': ['find_ground_truth_for_questions', 'find_ground_truth_for_questions_path', 'find_ground_truth_for_dataframe', 'find_ground_truth_for_row'],
            'answer_generator': ['naive_rag', 'agentic_rag', 'generate_naive_answer', 'generate_agentic_answer'],
            'rag_engine': ['BilingualRAGEngine'],
        }

except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    __all__ = ['metrics', 'csv_utils', 'retrieval', 'question_rewriter', 'ground_truth', 'answer_generator']