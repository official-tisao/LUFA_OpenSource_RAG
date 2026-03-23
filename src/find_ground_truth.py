#!/usr/bin/env python3
"""
Find ground truth source IDs and exact clause texts for each question
by combining collection routing, metadata SQL WHERE filters, and text overlap matching.
Provides interactive step-by-step terminal output, processing counters, and
answer-based token validation safeguards.
"""

import sys
import argparse
from pathlib import Path
import re
import pandas as pd
import chromadb

sys.path.insert(0, str(Path(__file__).parent))
from rag_engine import create_rag_engine

DEFAULT_CSV = "tests/combined_test_data.csv"
OUTPUT_CSV = "tests/combined_test_data_and_ground_truth.csv"
DEFAULT_DB = "db/chroma_db"
DEFAULT_COLLECTION = "multilingual_docs"


def extract_article_number(text_to_search):
    """Detect explicit references to article numbers (e.g., Article 27)."""
    if not text_to_search or pd.isna(text_to_search):
        return None
    match = re.search(r'(?:article|clause)\s+(\d{1,3})', str(text_to_search), re.I)
    return match.group(1) if match else None


def calculate_token_overlap(text_a, text_b):
    """Calculate the word overlap ratio to find the most exact match."""
    words_a = set(re.findall(r'\b\w+\b', str(text_a).lower()))
    words_b = set(re.findall(r'\b\w+\b', str(text_b).lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a.intersection(words_b)
    return len(intersection) / len(words_b)


def calculate_answer_based_overlap(expected_answer, resolved_db_text):
    """
    Answer column-based validation. Calculates what percentage of distinct tokens
    present inside the expected answer column exist inside the longer resolved DB block.
    """
    if not expected_answer or not resolved_db_text or pd.isna(expected_answer) or pd.isna(resolved_db_text):
        return 0.0
    words_expected = set(re.findall(r'\b\w+\b', str(expected_answer).lower()))
    words_db = set(re.findall(r'\b\w+\b', str(resolved_db_text).lower()))
    if not words_expected:
        return 0.0
    intersection = words_expected.intersection(words_db)
    return len(intersection) / len(words_expected)


def find_exact_ground_truth(row, client, collection_name, embed_model):
    """
    Queries the unified ChromaDB collection using structured metadata logical
    WHERE clauses, then isolates the true node identifier via text correlation.
    """
    collection = client.get_collection(collection_name)
    question_text = str(row["question"])
    expected_text = str(row.get("expected_answer", ""))
    row_lang = str(row.get("language", "English")).lower()

