#!/usr/bin/env python3
"""
Find ground truth source IDs and exact clause texts for each question
by combining collection routing, metadata SQL WHERE filters, and text overlap matching.
Provides interactive step-by-step terminal output, processing counters, and
answer-based token validation safeguards.

Supports multi-chunk ground truth: when a single chunk scores below 100% overlap,
the script attempts adjacent-chunk expansion (forward, backward, combined) and
accepts multi-chunk IDs if the expanded overlap reaches the 80% threshold.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict
import re
import pandas as pd
import chromadb

sys.path.insert(0, str(Path(__file__).parent))
from rag_engine import create_rag_engine

DEFAULT_CSV = "tests/combined_test_data.csv"
OUTPUT_CSV = "tests/combined_test_data_and_ground_truth.csv"
DEFAULT_DB = "db/chroma_db"
DEFAULT_COLLECTION = "multilingual_docs"

# Multi-chunk expansion is accepted when round(score*100) >= this value.
MULTI_CHUNK_THRESHOLD = 80


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
#  ADJACENT CHUNK NEIGHBOR INDEX
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-CHUNK ADJACENT EXPANSION
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE-CHUNK GROUND TRUTH FINDER  (original logic, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(csv_path, db_path, collection_name, output_path):
    print(f"[Initialization] Loading base source text dataset from: {csv_path}")
    if not Path(csv_path).exists():
        print(f"Error: Source file {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    total_records = len(df)
    columns = [
    "id", "question", "expected_answer", "category", "difficulty", 
    "language", "ground_source_truth_id", "ground_source_truth", 
    "answer_ground_truth_alignment"
]

# Create an empty DataFrame with these columns
    new_df = pd.DataFrame(columns=columns)
    print(f"[Initialization] Total entries detected to process: {total_records}")

    df["ground_source_truth_id"] = ""
    df["ground_source_truth"] = ""
    df["answer_ground_truth_alignment"] = 0

    print(f"[Initialization] Establishing persistent connection to ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)

    print("[Initialization] Instantiating neural layout embedding index models...")
    engine = create_rag_engine(db_path=db_path)
    embed_model = engine.embed_model

    neighbor_index = ChunkNeighborIndex(client, collection_name)

    print("\n" + "=" * 80)
    print("STARTING INTERACTIVE BILINGUAL PROCESSING LOOP")
    print("=" * 80)

    multi_chunk_accepted = 0
    multi_chunk_attempted = 0

    for idx, row in df.iterrows():
        current_counter = idx + 1
        question_id = row.get("id", f"row_{idx}")

        print(f"\n[{current_counter}/{total_records}] Processing Question ID: {question_id}")
        print(f"   -> Query Preview: \"{str(row['question'])[:65]}...\"")

        try:
            # ── Step 1: Single-chunk ground truth (original logic) ────────────
            gt_id, gt_text, overlap_pct, answer_overlap_pct = find_exact_ground_truth(
                row=row,
                client=client,
                collection_name=collection_name,
                embed_model=embed_model
            )

            # Store as clean integer percentage
            alignment_int = round(answer_overlap_pct * 100)

            # Assign single-chunk result into the three columns.
            df.at[idx, "ground_source_truth_id"] = gt_id
            df.at[idx, "ground_source_truth"] = gt_text
            df.at[idx, "answer_ground_truth_alignment"] = alignment_int

            if gt_id:
                print(f"   ✅ Success: Chunk linked successfully.")
                print(f"      - Database Node UUID: {gt_id}")
                print(f"      - Text Alignment Score (Doc-to-Expected): {round(overlap_pct * 100)}%")
                print(f"      - Answer Footprint Cross-Check (Expected-in-Doc): {alignment_int}%")
                print(f"      - Answer-Ground-Truth Alignment: {alignment_int}%")
            else:
                print(f"   ❌ Null Extraction: Found no underlying text blocks.")
                continue

            # ── Step 2: Multi-chunk expansion (only when overlap < 100%) ──────
            if answer_overlap_pct >= 1.0:
                print(f"   ✅ Single-chunk overlap is 100% — no multi-chunk expansion needed.")
                new_df.loc[len(new_df)] = df.iloc[idx]
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                new_df.to_csv(output_path, index=False)
                continue

            multi_chunk_attempted += 1
            print(f"   🔄 Adjacent Chunk Expansion (single-chunk overlap {alignment_int}% < 100%):")

            best_expansion, all_passes = attempt_multi_chunk_expansion(
                best_id=gt_id,
                expected_text=str(row.get("expected_answer", "")),
                neighbor_index=neighbor_index
            )

            # Print every pass result to terminal.
            for i, p in enumerate(all_passes, start=1):
                ids_str = ", ".join(p["ids"])
                score_int = round(p["overlap"] * 100)
                marker = ""
                if p["overlap"] >= 1.0:
                    marker = "  ← PERFECT (100%)"
                elif passes_threshold(p["overlap"]):
                    marker = f"  ← ACCEPTED (≥{MULTI_CHUNK_THRESHOLD}%)"
                print(f"      - Pass {i} ({p['label']}): {score_int}%  [{ids_str}]{marker}")

            if best_expansion is None:
                print(f"      ⚠️  No adjacent neighbors available for expansion.")
                continue

            best_score = best_expansion["overlap"]
            best_score_int = round(best_score * 100)
            best_label = best_expansion["label"]
            best_ids = best_expansion["ids"]
            best_combined_text = best_expansion["text"]
            ids_pipe = "|".join(best_ids)

            print(f"      ── Best Expansion: {best_label} at {best_score_int}%")

            # Accept if round(score * 100) >= threshold.
            if passes_threshold(best_score):
                multi_chunk_accepted += 1
                print(f"      ✅ Multi-Chunk ACCEPTED ({best_score_int}% ≥ {MULTI_CHUNK_THRESHOLD}%)")
                print(f"         Multi-Chunk Ground Truth IDs: {ids_pipe}")

                # Overwrite the three columns with multi-chunk values.
                df.at[idx, "ground_source_truth_id"] = ids_pipe
                df.at[idx, "ground_source_truth"] = best_combined_text
                df.at[idx, "answer_ground_truth_alignment"] = best_score_int

                new_df.loc[len(new_df)] = df.iloc[idx]
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                new_df.to_csv(output_path, index=False)
            else:
                print(f"      ❌ Multi-Chunk REJECTED ({best_score_int}% < {MULTI_CHUNK_THRESHOLD}%)")
                print(f"         Keeping single-chunk assignment: {gt_id}")

        except Exception as e:
            print(f"   💥 Pipeline Exception on index row index [{idx}]: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PROCESSING CYCLE COMPLETED")
    print("=" * 80)
    print(f"\n   Total questions processed:        {total_records}")
    print(f"   Multi-chunk expansion attempted:  {multi_chunk_attempted}")
    print(f"   Multi-chunk expansions accepted:  {multi_chunk_accepted}")
    if multi_chunk_attempted > 0:
        print(f"   Acceptance rate:                  {multi_chunk_accepted / multi_chunk_attempted:.1%}")

    
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
