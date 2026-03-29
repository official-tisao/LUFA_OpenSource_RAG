#!/usr/bin/env python3
"""
Repair script for LUFA RAG system output data.
Reconstructs missing or mismatched source chunk IDs by matching text snippets
directly against persistent ChromaDB records and ground truth registries.
Provides interactive step-by-step terminal output and processing counters.
"""

import os
import sys
import re
import argparse
from pathlib import Path
import pandas as pd
import chromadb

DEFAULT_LUFA_CSV = "tests/lufa_out_data.csv"
DEFAULT_GROUND_TRUTH_CSV = "tests/combined_test_data_and_ground_truth.csv"
DEFAULT_DB = "db/chroma_db"
DEFAULT_COLLECTION = "multilingual_docs"


def calculate_token_overlap(text_a, text_b):
    """Calculate how much of text_a's distinct tokens are present inside text_b."""
    if not text_a or not text_b or pd.isna(text_a) or pd.isna(text_b):
        return 0.0
    words_a = set(re.findall(r'\b\w+\b', str(text_a).lower()))
    words_b = set(re.findall(r'\b\w+\b', str(text_b).lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a.intersection(words_b)
    return len(intersection) / len(words_a)


def repair_dataset(lufa_path, gt_path, db_path, collection_name):
    print("================================================================================")
    print("STAGE 1: Loading Datasets and Connecting to ChromaDB")
    print("================================================================================")

    # Verification checks for paths and fallbacks
    if not Path(lufa_path).exists():
        fallback = Path("data") / Path(lufa_path).name
        if fallback.exists():
            lufa_path = str(fallback)
        else:
            print(f"Error: Target RAG engine output file not found at {lufa_path}")
            return

    if not Path(gt_path).exists():
        fallback = Path("data") / Path(gt_path).name
        if fallback.exists():
            gt_path = str(fallback)

    print(f" -> Loading target RAG output log from: {lufa_path}")
    lufa_df = pd.read_csv(lufa_path)

    gt_lookup = {}
    if Path(gt_path).exists():
        print(f" -> Loading ground truth registry reference from: {gt_path}")
        gt_df = pd.read_csv(gt_path)
        for _, row in gt_df.iterrows():
            q_id = str(row.get("id", "")).strip()
            if q_id:
                gt_lookup[q_id] = row.to_dict()
    else:
        print(" -> Info: Ground truth reference file not found. Repair will rely solely on database matching.")

    print(f" -> Connecting to persistent ChromaDB instance at: {db_path}")
