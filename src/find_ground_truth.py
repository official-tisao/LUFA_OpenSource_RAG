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
    lang_code = "fr" if "fr" in row_lang or "fran" in row_lang else "en"
    art_no = extract_article_number(question_text) or extract_article_number(expected_text)

    # Pathway 1: Complex Metadata SQL WHERE Route
    if art_no:
        where_clause = {
            "$and": [
                {"article_number": str(art_no)},
                {"language": lang_code}
            ]
        }
        print(f"   [Dual Strategy] Executing SQL WHERE route -> Article: {art_no} | Lang: {lang_code}")
        db_results = collection.get(
            where=where_clause,
            include=["documents"]
        )
    # Pathway 2: Fallback Embedding Index Route
    else:
        where_clause = {"language": lang_code}
        print(
            f"   [Dual Strategy] No Article found. Executing fallback Neural Vector Index Query route -> Lang: {lang_code}")
        query_embedding = embed_model.get_text_embedding(question_text)
        query_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=20,
            where=where_clause,
            include=["documents"]
        )
        db_results = {
            "ids": query_results["ids"][0] if query_results["ids"] else [],
            "documents": query_results["documents"][0] if query_results["documents"] else []
        }

    ids = db_results.get("ids", [])
    docs = db_results.get("documents", [])

    if not ids or not docs:
        print("   ⚠️  Warning: No database records matched query filters.")
        return "", "", 0.0, 0.0

    best_id = ids[0]
    best_doc = docs[0]
    max_overlap = -1.0

    for cid, doc_text in zip(ids, docs):
        overlap = calculate_token_overlap(doc_text, expected_text)
        if overlap > max_overlap:
            max_overlap = overlap
            best_id = cid
            best_doc = doc_text

    # Cross-verify using answer column token footprint ratio
    answer_overlap_score = calculate_answer_based_overlap(expected_text, best_doc)
    return best_id, best_doc, max_overlap, answer_overlap_score


def run_pipeline(csv_path, db_path, collection_name, output_path):
    print(f"[Initialization] Loading base source text dataset from: {csv_path}")
    if not Path(csv_path).exists():
        print(f"Error: Source file {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    total_records = len(df)
    print(f"[Initialization] Total entries detected to process: {total_records}")

    df["ground_source_truth_id"] = ""
    df["ground_source_truth"] = ""
    df["answer_ground_truth_alignment"] = 0.0

    print(f"[Initialization] Establishing persistent connection to ChromaDB at: {db_path}")

    client = chromadb.PersistentClient(path=db_path)

    print("[Initialization] Instantiating neural layout embedding index models...")
    engine = create_rag_engine(db_path=db_path)
    embed_model = engine.embed_model

    print("\n" + "=" * 80)
    print("STARTING INTERACTIVE BILINGUAL PROCESSING LOOP")
    print("=" * 80)

    for idx, row in df.iterrows():
        current_counter = idx + 1
        question_id = row.get("id", f"row_{idx}")

        print(f"\n[{current_counter}/{total_records}] Processing Question ID: {question_id}")
        print(f"   -> Query Preview: \"{str(row['question'])[:65]}...\"")

        try:
            gt_id, gt_text, overlap_pct, answer_overlap_pct = find_exact_ground_truth(
                row=row,
                client=client,
                collection_name=collection_name,
                embed_model=embed_model
            )

            df.at[idx, "ground_source_truth_id"] = gt_id
            df.at[idx, "ground_source_truth"] = gt_text
            df.at[idx, "answer_ground_truth_alignment"] = answer_overlap_pct

            if gt_id:
                print(f"   ✅ Success: Chunk linked successfully.")
                print(f"      - Database Node UUID: {gt_id}")
                print(f"      - Text Alignment Score (Doc-to-Expected): {overlap_pct:.2%}")
                print(f"      - Answer Footprint Cross-Check (Expected-in-Doc): {answer_overlap_pct:.2%}")
                print(f"      - Answer-Ground-Truth Alignment: {answer_overlap_pct:.2%}")
            else:
                print(f"   ❌ Null Extraction: Found no underlying text blocks.")

        except Exception as e:
            print(f"   💥 Pipeline Exception on index row index [{idx}]: {e}")

    print("\n" + "=" * 80)
    print("PROCESSING CYCLE COMPLETED")
    print("=" * 80)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[Export] Integrated data structured table saved cleanly to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find true database ground truth mappings with runtime console feedback.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to evaluation test dataset")
    parser.add_argument("--db", default=DEFAULT_DB, help="ChromaDB persistent file location")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="ChromaDB target collection")
    parser.add_argument("--out", default=OUTPUT_CSV, help="Path for the output joined results")
    args = parser.parse_args()

    run_pipeline(args.csv, args.db, args.collection, args.out)