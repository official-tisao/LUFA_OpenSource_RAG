#!/usr/bin/env python3
"""
Universal repair script. Fixes source IDs by checking text overlap against ChromaDB,
updates BOTH lufa_out_data.csv and evaluation_results.csv, recalculates retrieval
metrics (MRR, NDCG, Recall), and updates the HTML dashboard.
"""

import sys
import re
import math
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import generate_dashboard


def calculate_token_overlap(text_a, text_b):
    if not text_a or not text_b or pd.isna(text_a) or pd.isna(text_b):
        return 0.0
    words_a = set(re.findall(r'\b\w+\b', str(text_a).lower()))
    words_b = set(re.findall(r'\b\w+\b', str(text_b).lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a.intersection(words_b)
    return len(intersection) / max(len(words_a), 1)


def mrr(retrieved, ground_truth):
    gt_set = set(ground_truth)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gt_set:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved, ground_truth, k):
    if not ground_truth:
        return 0.0
    relevant_at_k = set(retrieved[:k]) & set(ground_truth)
    return len(relevant_at_k) / len(ground_truth)


def ndcg_at_k(retrieved, ground_truth, k=5):
    gt_set = set(ground_truth)
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, doc_id in enumerate(retrieved[:k], start=1) if doc_id in gt_set)
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def run_repair(lufa_csv, eval_csv, gt_csv, db_path, dash_out):
    print("Loading datasets for universal repair...")
    lufa_df = pd.read_csv(lufa_csv) if Path(lufa_csv).exists() else None
    eval_df = pd.read_csv(eval_csv) if Path(eval_csv).exists() else None
