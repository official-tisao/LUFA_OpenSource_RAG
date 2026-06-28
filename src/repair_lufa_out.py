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

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import cfg

DEFAULT_LUFA_CSV = "tests/lufa_out_data.csv"
DEFAULT_GROUND_TRUTH_CSV = "tests/combined_test_data_and_ground_truth.csv"
DEFAULT_DB = cfg("database.path")
DEFAULT_COLLECTION = cfg("database.collection_name")


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
    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection(collection_name)
        db_results = collection.get(include=["documents"])
        print(f" -> Successfully loaded {len(db_results.get('ids', []))} total text chunks from database.")
    except Exception as e:
        print(f" 💥 Error connecting to collection '{collection_name}': {e}")
        return

    print("\n================================================================================")
    print("STAGE 2: Executing Source ID Repair Loop")
    print("================================================================================")

    total_repaired = 0
    total_fields = 0
    total_records = len(lufa_df)

    for idx, row in lufa_df.iterrows():
        current_counter = idx + 1
        q_id = str(row.get("question_id", "")).strip()
        print(f"\n[{current_counter}/{total_records}] Analyzing Question ID: {q_id}")

        gt_row = gt_lookup.get(q_id, {})

        for i in range(1, 6):
            text_col = f"source{i}_text"
            id_col = f"source{i}_id"

            source_text = row.get(text_col, "")
            if pd.isna(source_text) or str(source_text).strip() == "":
                continue

            total_fields += 1
            source_clean = str(source_text).strip().lower()
            repaired_id = ""
            method_used = ""

            # Strategy 1: Cross-verify Source 1 text with ground truth registry text directly
            if i == 1 and gt_row:
                gt_text = str(gt_row.get("ground_source_truth", "")).lower()
                gt_id_raw = str(gt_row.get("ground_source_truth_id", "")).strip()
                # Support pipe-separated ground_source_truth_id — use all IDs for matching
                gt_ids = [s.strip() for s in gt_id_raw.split("|") if s.strip()]
                gt_id = gt_ids[0] if gt_ids else ""  # primary ID for single-match scenarios
                if gt_text and gt_ids:
                    overlap = calculate_token_overlap(source_clean, gt_text)
                    if overlap > 0.85 or source_clean in gt_text:
                        repaired_id = gt_id
                        method_used = f"Ground Truth Registry Alignment (Score: {overlap:.2%})"

            # Strategy 2: Scan full ChromaDB document pool for matching substrings or overlap token sets
            if not repaired_id:
                best_match_id = ""
                max_overlap = -1.0
                exact_found = False

                for cid, doc_text in zip(db_results["ids"], db_results["documents"]):
                    doc_clean = str(doc_text).lower()
                    if source_clean in doc_clean:
                        best_match_id = cid
                        exact_found = True
                        break

                    overlap = calculate_token_overlap(source_clean, doc_clean)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_match_id = cid

                if exact_found:
                    repaired_id = best_match_id
                    method_used = "ChromaDB Substring Perfect Fit"
                elif max_overlap > 0.5:
                    repaired_id = best_match_id
                    method_used = f"ChromaDB Vector Token Overlap (Score: {max_overlap:.2%})"

            if repaired_id:
                old_id = row.get(id_col, "")
                if str(old_id) != str(repaired_id):
                    lufa_df.at[idx, id_col] = repaired_id
                    total_repaired += 1
                    print(f"   ✅ Repaired {id_col} -> {repaired_id} [{method_used}]")
                else:
                    print(f"   · Verified {id_col} is already correct -> {repaired_id}")
            else:
                print(f"   ❌ Warning: Could not locate matching chunk ID for {text_col}")

    print("\n================================================================================")
    print("STAGE 3: Saving Repaired Records and Summary")
    print("================================================================================")

    lufa_df.to_csv(lufa_path, index=False)
    print(f" -> Repaired file saved cleanly back to: {lufa_path}")
    print(f" -> Total source text fields evaluated: {total_fields}")
    print(f" -> Total source identifier columns updated: {total_repaired}")
    print("================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair missing or misplaced source chunk IDs in lufa output logs.")
    parser.add_argument("--lufa_csv", default=DEFAULT_LUFA_CSV, help="Path to lufa output log sheet")
    parser.add_argument("--gt_csv", default=DEFAULT_GROUND_TRUTH_CSV, help="Path to ground truth reference sheet")
    parser.add_argument("--db", default=DEFAULT_DB, help="ChromaDB persistent directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="ChromaDB collection name")
    args = parser.parse_args()

    repair_dataset(args.lufa_csv, args.gt_csv, args.db, args.collection)