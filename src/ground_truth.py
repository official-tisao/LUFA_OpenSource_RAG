#!/usr/bin/env python3
"""
Ground truth finder module for LUFA RAG system.
Provides functions to find ground truth source IDs and texts for questions.
Refactored version of find_ground_truth.py with separate input variants.
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from collections import defaultdict

# Import from rag_engine for create_rag_engine
try:
    from rag_engine import create_rag_engine
except ImportError:
    create_rag_engine = None

# Import ChromaDB
import chromadb

# Constants
DEFAULT_CSV = "tests/combined_test_data.csv"
OUTPUT_CSV = "tests/combined_test_data_and_ground_truth.csv"
DEFAULT_DB = "db/chroma_db"
DEFAULT_COLLECTION = "multilingual_docs"
MULTI_CHUNK_THRESHOLD = 80
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
def passes_threshold(score, threshold=MULTI_CHUNK_THRESHOLD):
    """Returns True when round(score * 100) >= threshold."""
    return round(score * 100) >= threshold
class ChunkNeighborIndex:
    """
    Pre-caches every chunk in the ChromaDB collection and builds a lookup that
    maps each chunk ID to its immediate document-order neighbors.

    Adjacency is defined as: same ``language`` + same ``doc_source``, with
    ``chunk_index`` differing by exactly 1.  This ensures the neighbor is
    from the same physical document, not a random chunk that happened to be
    inserted nearby.
    """

    def __init__(self, client, collection_name):
        print("[NeighborIndex] Loading full collection for adjacency mapping...")
        collection = client.get_collection(collection_name)
        all_data = collection.get(include=["documents", "metadatas"])

        self._id_to_doc = {}
        self._id_to_meta = {}
        self._prev = {}
        self._next = {}

        ids = all_data.get("ids", [])
        docs = all_data.get("documents", [])
        metas = all_data.get("metadatas", [])

        for cid, doc, meta in zip(ids, docs, metas or [{}] * len(ids)):
            self._id_to_doc[cid] = doc or ""
            self._id_to_meta[cid] = meta or {}

        groups = defaultdict(list)
        for cid, meta in self._id_to_meta.items():
            lang = str(meta.get("language", "")).strip()
            src = str(meta.get("doc_source", "")).strip()
            try:
                idx_int = int(meta.get("chunk_index", -1))
            except (TypeError, ValueError):
                idx_int = -1
            groups[(lang, src)].append((idx_int, cid))

        for key in groups:
            ordered = sorted(groups[key], key=lambda t: t[0])
            for pos, (_, cid) in enumerate(ordered):
                self._prev[cid] = ordered[pos - 1][1] if pos > 0 else None
                self._next[cid] = ordered[pos + 1][1] if pos < len(ordered) - 1 else None

        print(f"[NeighborIndex] Indexed {len(ids)} chunks across {len(groups)} document groups.")

    def get_text(self, chunk_id):
        return self._id_to_doc.get(chunk_id, "")

    def get_prev_id(self, chunk_id):
        return self._prev.get(chunk_id)

    def get_next_id(self, chunk_id):
        return self._next.get(chunk_id)
def attempt_multi_chunk_expansion(best_id, expected_text, neighbor_index):
    """
    When the single-chunk overlap is below 100%, try expanding to adjacent
    chunks to capture answers that straddle a chunk boundary.

    Expansion passes (executed in order, short-circuits on 100%):
      1. Forward   :  chunk[N]   + chunk[N+1]
      2. Backward  :  chunk[N-1] + chunk[N]
      3. Combined  :  chunk[N-1] + chunk[N] + chunk[N+1]

    Returns the best-scoring expansion as a dict:
        { "ids": [str, ...], "text": str, "overlap": float, "label": str }
    or None if no neighbor exists at all.
    """
    prev_id = neighbor_index.get_prev_id(best_id)
    next_id = neighbor_index.get_next_id(best_id)

    base_text = neighbor_index.get_text(best_id)
    prev_text = neighbor_index.get_text(prev_id) if prev_id else ""
    next_text = neighbor_index.get_text(next_id) if next_id else ""

    passes = []

    # Pass 1 — Forward: N + N+1
    if next_id:
        combined = base_text + "\n\n" + next_text
        score = calculate_answer_based_overlap(expected_text, combined)
        passes.append({
            "ids": [best_id, next_id],
            "text": combined,
            "overlap": score,
            "label": "Forward  (N + N+1)",
        })
        if score >= 1.0:
            return passes[-1], passes

    # Pass 2 — Backward: N-1 + N
    if prev_id:
        combined = prev_text + "\n\n" + base_text
        score = calculate_answer_based_overlap(expected_text, combined)
        passes.append({
            "ids": [prev_id, best_id],
            "text": combined,
            "overlap": score,
            "label": "Backward (N-1 + N)",
        })
        if score >= 1.0:
            return passes[-1], passes

    # Pass 3 — Combined: N-1 + N + N+1
    if prev_id and next_id:
        combined = prev_text + "\n\n" + base_text + "\n\n" + next_text
        score = calculate_answer_based_overlap(expected_text, combined)
        passes.append({
            "ids": [prev_id, best_id, next_id],
            "text": combined,
            "overlap": score,
            "label": "Combined (N-1 + N + N+1)",
        })
        if score >= 1.0:
            return passes[-1], passes

    if not passes:
        return None, []

    # No pass reached 100% — pick the highest-scoring one.
    # Tiebreaker: when two passes share the same score, choose the one with
    # the shortest combined text length to save tokens downstream.
    best = max(passes, key=lambda p: (p["overlap"], -len(p["text"])))
    return best, passes
def _resolve_ids_from_text(text, chroma_data, min_overlap=0.3):
    """
    Resolve chunk IDs by matching ground_source_truth text against
    pre-cached ChromaDB documents using token overlap.

    Returns all chunk IDs whose overlap score >= min_overlap,
    pipe-joined in database order.
    """
    gt_words = set(re.findall(r'\b\w+\b', text.lower()))
    if not gt_words:
        return []

    db_ids = chroma_data.get("ids", [])
    db_docs = chroma_data.get("documents", [])

    matches = []
    for cid, doc in zip(db_ids, db_docs):
        if not doc:
            continue
        doc_words = set(re.findall(r'\b\w+\b', str(doc).lower()))
        if not doc_words:
            continue
        # Overlap: what fraction of the ground truth text's tokens appear in this chunk
        overlap = len(gt_words & doc_words) / len(gt_words)
        if overlap >= min_overlap:
            matches.append((cid, overlap))

    if not matches:
        # Lower threshold and retry with top-1 if nothing matched
        for cid, doc in zip(db_ids, db_docs):
            if not doc:
                continue
            doc_words = set(re.findall(r'\b\w+\b', str(doc).lower()))
            if not doc_words:
                continue
            overlap = len(gt_words & doc_words) / len(gt_words)
            if overlap > 0:
                matches.append((cid, overlap))
        if matches:
            # Just return the best match
            matches.sort(key=lambda x: x[1], reverse=True)
            return [matches[0][0]]
        return []

    # Return IDs sorted by overlap score (descending)
    matches.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in matches]
def find_ground_truth_for_text(text, chroma_data):
    """
    Find ground truth chunk ID(s) for given text by matching against ChromaDB.

    Args:
        text: Ground truth text to match
        chroma_data: ChromaDB data dict with keys: {"ids": [...], "documents": [...], "metadatas": [...]}

    Returns:
        Tuple: (chunk_id, text, overlap_percentage, answer_overlap_percentage)
    """
    if not text or not chroma_data:
        return "", "", 0.0, 0.0

    db_ids = chroma_data.get("ids", [])
    db_docs = chroma_data.get("documents", [])

    if not db_ids or not db_docs:
        return "", "", 0.0, 0.0

    best_id = db_ids[0]
    best_doc = db_docs[0]
    max_overlap = -1.0

    for cid, doc_text in zip(db_ids, db_docs):
        overlap = calculate_token_overlap(doc_text, text)
        if overlap > max_overlap:
            max_overlap = overlap
            best_id = cid
            best_doc = doc_text

    # Cross-verify using answer column token footprint ratio
    answer_overlap_score = calculate_answer_based_overlap(text, best_doc)

    return best_id, best_doc, max_overlap, answer_overlap_score
def find_ground_truth_for_row(row, chroma_data=None):
    """
    Find ground truth for a single DataFrame row.

    Args:
        row: DataFrame row (pandas Series)
        chroma_data: Pre-cached ChromaDB data dict

    Returns:
        Tuple: (chunk_ids, ground_truth_text, alignment_percentage)
    """
    # Extract ground truth text from row
    gt_text_col = "ground_source_truth" if "ground_source_truth" in row.index else None
    if not gt_text_col:
        return [], "", 0.0

    gt_text = str(row.get(gt_text_col, "")).strip()

    if not gt_text or gt_text == "nan" or len(gt_text) < 20:
        return [], "", 0.0

    # Find ground truth using chroma_data if available
    if chroma_data:
        gt_id, gt_doc, overlap_pct, answer_overlap_pct = find_ground_truth_for_text(gt_text, chroma_data)

        if not gt_id:
            return [], gt_text, 0.0

        alignment_int = round(answer_overlap_pct * 100)

        # Multi-chunk expansion if needed
        if answer_overlap_pct < 1.0:
            # Initialize neighbor index for multi-chunk expansion
            client = chromadb.PersistentClient(path="db/chroma_db")
            collection_name = "multilingual_docs"
            neighbor_index = ChunkNeighborIndex(client, collection_name)

            best_expansion, all_passes = attempt_multi_chunk_expansion(
                best_id=gt_id,
                expected_text=gt_text,
                neighbor_index=neighbor_index
            )

            if best_expansion is not None and passes_threshold(best_expansion["overlap"]):
                # Accept multi-chunk expansion
                gt_id = "|".join(best_expansion["ids"])
                gt_doc = best_expansion["text"]
                alignment_int = round(best_expansion["overlap"] * 100)

        return gt_id.split("|") if isinstance(gt_id, str) else [gt_id], gt_doc, alignment_int

    return [], gt_text, 0.0
def find_ground_truth_for_questions(questions_df: pd.DataFrame,
                                    chroma_data=None,
                                    db_path: str = None,
                                    collection_name: str = None) -> pd.DataFrame:
    """
    Find ground truth for all questions in a DataFrame.

    Args:
        questions_df: DataFrame with questions data (must have 'question', 'expected_answer', etc. columns)
        chroma_data: Pre-cached ChromaDB data dict
        db_path: Path to ChromaDB database
        collection_name: Name of ChromaDB collection

    Returns:
        pd.DataFrame: DataFrame with added ground truth columns
    """
    # Create a copy to avoid modifying the input
    result_df = questions_df.copy()

    # Initialize neighbor index if needed for multi-chunk expansion
    if chroma_data is None and (db_path is not None or collection_name is not None):
        try:
            client = chromadb.PersistentClient(path=db_path or "db/chroma_db")
            collection_name = collection_name or "multilingual_docs"
            chroma_client = chromadb.PersistentClient(path=db_path or "db/chroma_db")
            collection = chroma_client.get_collection(collection_name)
            chroma_data = collection.get(include=["documents", "metadatas"])
        except Exception as e:
            print(f"[GroundTruth] Could not load ChromaDB data: {e}")
            chroma_data = None

    # Process each row
    print(f"[GroundTruth] Processing {len(result_df)} questions...")

    for idx, row in result_df.iterrows():
        try:
            # Find ground truth for this row
            gt_ids, gt_text, alignment_pct = find_ground_truth_for_row(row, chroma_data)

            # Store results
            if gt_ids:
                # Convert list of IDs to pipe-separated string
                result_df.at[idx, "ground_source_truth_id"] = "|".join(gt_ids)
                result_df.at[idx, "ground_source_truth"] = gt_text
                result_df.at[idx, "answer_ground_truth_alignment"] = alignment_pct
            else:
                result_df.at[idx, "ground_source_truth_id"] = ""
                result_df.at[idx, "ground_source_truth"] = ""
                result_df.at[idx, "answer_ground_truth_alignment"] = 0

            print(f"      [GroundTruth] Processed question ID: {row.get('id', f'row_{idx}')}")

        except Exception as e:
            print(f"      [GroundTruth Error] Failed to process row {idx}: {e}")
            # Set empty values on error
            result_df.at[idx, "ground_source_truth_id"] = ""
            result_df.at[idx, "ground_source_truth"] = ""
            result_df.at[idx, "answer_ground_truth_alignment"] = 0

    print(f"[GroundTruth] Completed processing {len(result_df)} questions")

    return result_df
def find_ground_truth_for_questions_path(input_path: str,
                                           output_path: str,
                                           db_path: str = None,
                                           collection_name: str = None) -> None:
    """
    Find ground truth for questions from CSV path and save to output CSV.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        db_path: Path to ChromaDB database
        collection_name: Name of ChromaDB collection
    """
    print(f"[GroundTruth] Loading questions from: {input_path}")

    # Load input CSV
    df = pd.read_csv(input_path)

    # Initialize columns if they don't exist
    if "ground_source_truth_id" not in df.columns:
        df["ground_source_truth_id"] = ""

    if "ground_source_truth" not in df.columns:
        df["ground_source_truth"] = ""

    if "answer_ground_truth_alignment" not in df.columns:
        df["answer_ground_truth_alignment"] = 0

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Process questions
    result_df = find_ground_truth_for_questions(df, None, db_path, collection_name)

    # Save results
    result_df.to_csv(output_path, index=False)

    print(f"[GroundTruth] Saved ground truth results to: {output_path}")
def find_ground_truth_for_dataframe(questions_df: pd.DataFrame,
                                    chroma_data=None,
                                    db_path: str = None,
                                    collection_name: str = None) -> pd.DataFrame:
    """
    High-level function to find ground truth from DataFrame (alternative to CSV version).

    Args:
        questions_df: DataFrame with question data
        chroma_data: Pre-cached ChromaDB data dict
        db_path: Path to ChromaDB database
        collection_name: Name of ChromaDB collection

    Returns:
        pd.DataFrame: DataFrame with ground truth columns
    """
    return find_ground_truth_for_questions(questions_df, chroma_data, db_path, collection_name)
# Alias for backward compatibility (matching the original module)
run = find_ground_truth_for_questions_path
run_pipeline = find_ground_truth_for_questions