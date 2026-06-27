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
from dashboard_generator import generate_dashboard


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
    gt_df = pd.read_csv(gt_csv) if Path(gt_csv).exists() else None

    if lufa_df is None or eval_df is None or gt_df is None:
        print("Missing required CSV files. Repair aborted.")
        return

    gt_lookup = {}
    for _, row in gt_df.iterrows():
        qid = str(row.get("id", "")).strip()
        gt_col = "ground_source_truth_id" if "ground_source_truth_id" in row else "ground_truth_source_ids"
        gt_lookup[qid] = [s.strip() for s in str(row.get(gt_col, "")).split("|") if s.strip()]

    import chromadb
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("multilingual_docs")
    chroma_data = collection.get(include=["documents"])
    db_ids = chroma_data.get("ids", [])
    db_docs = chroma_data.get("documents", [])

    updates = 0
    for idx, eval_row in eval_df.iterrows():
        qid = str(eval_row.get("question_id", "")).strip()

        repaired_ids = []
        for i in range(1, 6):
            text_val = eval_row.get(f"source{i}_text", "")
            if pd.isna(text_val) or str(text_val).strip() == "":
                continue

            source_clean = str(text_val).strip().lower()
            matched_id = eval_row.get(f"source{i}_id", "")

            if "_chunk" in str(matched_id):
                max_overlap = -1.0
                for cid, doc_text in zip(db_ids, db_docs):
                    if source_clean in str(doc_text).lower():
                        matched_id = cid
                        break
                    overlap = calculate_token_overlap(source_clean, str(doc_text).lower())
                    if overlap > max_overlap:
                        max_overlap = overlap
                        matched_id = cid

            if matched_id:
                repaired_ids.append(matched_id)
                eval_df.at[idx, f"source{i}_id"] = matched_id

                lufa_idx = lufa_df.index[lufa_df['question_id'] == qid].tolist()
                if lufa_idx:
                    lufa_df.at[lufa_idx[0], f"source{i}_id"] = matched_id

        gt_ids = gt_lookup.get(qid, [])
        if repaired_ids and gt_ids:
            new_mrr = round(mrr(repaired_ids, gt_ids), 4)
            new_ndcg = ndcg_at_k(repaired_ids, gt_ids, 5)

            if eval_row.get("mrr") != new_mrr or eval_row.get("ndcg_at_k") != new_ndcg:
                eval_df.at[idx, "mrr"] = new_mrr
                eval_df.at[idx, "ndcg_at_k"] = new_ndcg
                eval_df.at[idx, "recall_1"] = round(recall_at_k(repaired_ids, gt_ids, 1), 4)
                eval_df.at[idx, "recall_3"] = round(recall_at_k(repaired_ids, gt_ids, 3), 4)
                eval_df.at[idx, "recall_5"] = round(recall_at_k(repaired_ids, gt_ids, 5), 4)
                updates += 1
                print(f"Repaired Question {qid} -> MRR: {new_mrr} | NDCG: {new_ndcg}")

    if updates > 0:
        lufa_df.to_csv(lufa_csv, index=False)
        eval_df.to_csv(eval_csv, index=False)
        print(f"Saved {updates} row updates to CSVs.")

        generate_dashboard(eval_df, dash_out)
        print("Dashboard HTML successfully updated.")
    else:
        print("No repairs needed. Files are aligned.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lufa_csv", default="tests/lufa_out_data.csv")
    parser.add_argument("--eval_csv", default="tests/evaluation_results.csv")
    parser.add_argument("--gt_csv", default="tests/combined_test_data_and_ground_truth.csv")
    parser.add_argument("--db", default="db/chroma_db")
    parser.add_argument("--dash", default="dashboard/index.html")
    args = parser.parse_args()

    run_repair(args.lufa_csv, args.eval_csv, args.gt_csv, args.db, args.dash)