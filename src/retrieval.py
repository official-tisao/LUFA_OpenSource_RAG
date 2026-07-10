#!/usr/bin/env python3
"""
Standalone retrieval module for LUFA RAG system.
Provides retrieval functions for top-k chunk retrieval with data preservation.
Detects old single-column score schema and triggers full re-retrieval when found.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from copy import deepcopy

from csv_utils import read_csv_cached, get_completed_ids, write_csv_row

# Old single-column schema markers — presence triggers full re-retrieval
_OLD_SCHEMA_COLUMNS = {"original_cosine_score", "recency_adjusted_score", "RRF"}
_OLD_SOURCE_SCORE_PATTERN = "source{}_score"


def has_old_schema(df: pd.DataFrame) -> bool:
    """Check if a DataFrame uses the old single-column score schema."""
    cols = set(df.columns)
    if cols & _OLD_SCHEMA_COLUMNS:
        return True
    for i in range(1, 6):
        old_col = _OLD_SOURCE_SCORE_PATTERN.format(i)
        new_col = f"source{i}_cosine_score"
        if old_col in cols and new_col not in cols:
            return True
    return False


def extract_sources_from_nodes(nodes, max_sources: int = 5) -> List[Dict]:
    """Extract source information from RAG nodes into dictionary format."""
    sources = []
    for i, node in enumerate(nodes[:max_sources], start=1):
        cosine = float(node.node.metadata.get('original_cosine_score', node.score))
        recency_weight = float(node.node.metadata.get('recency_weight', 1.0))
        rrf = float(node.score)

        source = {
            'node_id': node.node.node_id,
            'score': rrf,
            'cosine_score': cosine,
            'recency_adjusted_cosine_score': round(cosine * recency_weight, 6),
            'rrf_score': rrf,
            'text': node.node.text[:500] if node.node.text else "",
            'metadata': node.node.metadata or {}
        }
        sources.append(source)

    return sources


def sources_to_csv_row(sources: List[Dict], max_sources: int = 5) -> Dict:
    """Convert sources list to CSV row format with per-chunk score columns."""
    row = {}
    for i in range(1, max_sources + 1):
        if i <= len(sources):
            s = sources[i - 1]
            row[f'source{i}_id'] = s['node_id']
            row[f'source{i}_cosine_score'] = round(float(s['cosine_score']), 4)
            row[f'source{i}_recency_adjusted_cosine_score'] = round(float(s['recency_adjusted_cosine_score']), 4)
            row[f'source{i}_rrf_score'] = round(float(s['rrf_score']), 4)
            row[f'source{i}_text'] = s['text']
        else:
            row[f'source{i}_id'] = ""
            row[f'source{i}_cosine_score'] = ""
            row[f'source{i}_recency_adjusted_cosine_score'] = ""
            row[f'source{i}_rrf_score'] = ""
            row[f'source{i}_text'] = ""

    return row


def preserve_existing_retrieval(row_dict: Dict, question_id: str,
                                 output_path: Union[str, Path],
                                 preserve_threshold: float = 0.01) -> Dict:
    """Preserve existing retrieval data if it appears valid (new schema only)."""
    path = Path(output_path)

    if not path.exists() or path.stat().st_size == 0:
        return row_dict

    try:
        existing_df = pd.read_csv(path)
        if "question_id" not in existing_df.columns:
            return row_dict

        if has_old_schema(existing_df):
            return row_dict

        existing_row = existing_df[existing_df["question_id"] == question_id]
        if existing_row.empty:
            return row_dict

        existing_row = existing_row.iloc[0]

        has_valid_retrieval = False
        for i in range(1, 6):
            for score_col in [f"source{i}_cosine_score", f"source{i}_rrf_score"]:
                if score_col in existing_row.index and not pd.isna(existing_row[score_col]):
                    try:
                        if float(existing_row[score_col]) > preserve_threshold:
                            has_valid_retrieval = True
                            break
                    except (ValueError, TypeError):
                        continue
            if has_valid_retrieval:
                break

        if has_valid_retrieval:
            print(f"      [Retrieval] Preserving existing retrieval for question {question_id}")
            preserved_data = existing_row.to_dict()
            merged_data = deepcopy(row_dict)
            for i in range(1, 6):
                source_cols = [
                    f"source{i}_id", f"source{i}_text",
                    f"source{i}_cosine_score",
                    f"source{i}_recency_adjusted_cosine_score",
                    f"source{i}_rrf_score",
                ]
                for col in source_cols:
                    if col in preserved_data and col not in merged_data:
                        merged_data[col] = preserved_data[col]
            return merged_data
    except Exception as e:
        print(f"      [Retrieval Warning] Could not preserve existing data: {e}")

    return row_dict


def retrieve_top_k(engine,
                    questions_df: pd.DataFrame,
                    top_k: int = 5,
                    output_path: Union[str, Path] = "tests/lufa_out_data.csv") -> pd.DataFrame:
    """Retrieve top-k chunks for questions in a DataFrame."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = set()
    if path.exists() and path.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(path)
            if "question_id" in existing_df.columns and not has_old_schema(existing_df):
                valid_ids = set()
                for i in range(1, 6):
                    score_col = f"source{i}_rrf_score"
                    if score_col in existing_df.columns:
                        mask = existing_df[score_col].notna()
                        valid_ids.update(
                            existing_df[mask]["question_id"]
                            .dropna().astype(str).str.strip().tolist()
                        )
                completed_ids = valid_ids
        except Exception as e:
            print(f"      [Retrieval Warning] Could not read existing output file: {e}")

    output_columns = ["question_id", "question"]
    for i in range(1, top_k + 1):
        output_columns.extend([
            f"source{i}_id",
            f"source{i}_cosine_score",
            f"source{i}_recency_adjusted_cosine_score",
            f"source{i}_rrf_score",
            f"source{i}_text",
        ])

    result_rows = []

    for idx, row in questions_df.iterrows():
        q_id = str(row.get("id")).strip()

        if q_id in completed_ids:
            print(f"      [Retrieval] Skipping question {q_id} (already completed)")
            if path.exists() and path.stat().st_size > 0:
                try:
                    existing_df = pd.read_csv(path)
                    existing_row = existing_df[existing_df["question_id"] == q_id]
                    if not existing_row.empty:
                        result_rows.append(existing_row.iloc[0].to_dict())
                        continue
                except Exception:
                    pass
            continue

        print(f"      [Retrieval] Processing question {q_id}")

        try:
            nodes = engine._retrieve_nodes(row["question"], top_k=top_k)
            sources = extract_sources_from_nodes(nodes, max_sources=top_k)
            source_row = sources_to_csv_row(sources, max_sources=top_k)

            result_row = {
                "question_id": q_id,
                "question": row["question"][:500] if "question" in row else "",
                **source_row
            }

            preserved_row = preserve_existing_retrieval(result_row, q_id, output_path)

            for col in output_columns:
                if col not in preserved_row:
                    preserved_row[col] = ""

            result_rows.append(preserved_row)

        except Exception as e:
            print(f"      [Retrieval Error] Failed to process question {q_id}: {e}")
            error_row = {
                "question_id": q_id,
                "question": row.get("question", "")[:500] if "question" in row else "",
            }
            for i in range(1, top_k + 1):
                error_row[f"source{i}_id"] = ""
                error_row[f"source{i}_cosine_score"] = ""
                error_row[f"source{i}_recency_adjusted_cosine_score"] = ""
                error_row[f"source{i}_rrf_score"] = ""
                error_row[f"source{i}_text"] = ""
            result_rows.append(error_row)

    result_df = pd.DataFrame(result_rows, columns=output_columns)

    try:
        file_exists = path.exists() and path.stat().st_size > 0
        mode = 'a' if file_exists else 'w'
        header = not file_exists

        result_df.to_csv(path, mode=mode, header=header,
                        index=False, encoding='utf-8')

        print(f"      [Retrieval] Saved {len(result_rows)} rows to {path}")
    except Exception as e:
        print(f"      [Retrieval Error] Failed to save results: {e}")

    return result_df


def retrieve_from_question_row(question_text: str,
                                engine,
                                top_k: int = 5) -> Tuple[List[Dict], Dict]:
    """Retrieve chunks for a single question."""
    nodes = engine._retrieve_nodes(question_text, top_k=top_k)
    sources = extract_sources_from_nodes(nodes, max_sources=top_k)
    return sources, {}


def retrieve_from_csv_path(input_path: Union[str, Path],
                           engine,
                           output_path: Union[str, Path] = "tests/lufa_out_data.csv",
                           top_k: int = 5) -> pd.DataFrame:
    """High-level function to retrieve from CSV path."""
    df = pd.read_csv(input_path)
    return retrieve_top_k(engine, df, top_k, output_path)


def retrieve_from_dataframe(questions_df: pd.DataFrame,
                           engine,
                           top_k: int = 5,
                           output_path: Union[str, Path] = "tests/lufa_out_data.csv") -> pd.DataFrame:
    """High-level function to retrieve from DataFrame."""
    return retrieve_top_k(engine, questions_df, top_k, output_path)


def _migrate_old_schema_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read a CSV with old schema, drop old single-column score columns,
    and rename source{n}_score → source{n}_cosine_score for migration.
    Returns a DataFrame with missing per-chunk columns added (empty).
    """
    df = pd.read_csv(csv_path)
    for col in _OLD_SCHEMA_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    for i in range(1, 6):
        old_col = f"source{i}_score"
        new_cosine = f"source{i}_cosine_score"
        new_recency = f"source{i}_recency_adjusted_cosine_score"
        new_rrf = f"source{i}_rrf_score"

        if old_col in df.columns and new_cosine not in df.columns:
            df = df.rename(columns={old_col: new_cosine})

        if new_recency not in df.columns:
            df[new_recency] = ""
        if new_rrf not in df.columns:
            df[new_rrf] = ""

    return df


if __name__ == "__main__":
    import argparse
    import sys
    import time
    import traceback
    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser(description="Run retrieval over LUFA test questions")
    parser.add_argument("--input", default="tests/combined_test_data_and_ground_truth.csv",
                        help="Input CSV with questions (must have 'id' and 'question' columns)")
    parser.add_argument("--output", default="tests/lufa_out_data.csv",
                        help="Output CSV path for retrieval results")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve per question")
    args = parser.parse_args()

    from rag_engine import create_rag_engine
    from csv_utils import upsert_row
    from run_simulation import OUTPUT_COLUMNS

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[Retrieval] Loading questions from {input_path}...")
    questions_df = pd.read_csv(input_path)
    print(f"[Retrieval] {len(questions_df)} questions loaded.")

    # Detect old schema → triggers full re-retrieval
    force_all = False
    completed_ids = set()
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(output_path)
            if has_old_schema(existing_df):
                print("[Retrieval] OLD SCHEMA DETECTED (single cosine_score/recency_adjusted_score/RRF).")
                print("[Retrieval] Migrating CSV structure and re-running retrieval for ALL rows...")
                migrated_df = _migrate_old_schema_csv(output_path)
                migrated_df.to_csv(output_path, index=False)
                force_all = True
            elif "question_id" in existing_df.columns:
                for i in range(1, 6):
                    score_col = f"source{i}_rrf_score"
                    if score_col in existing_df.columns:
                        completed_ids.update(
                            existing_df[existing_df[score_col].notna() 
                            & (existing_df[score_col].astype(str).str.strip() != "")
                            #& (existing_df[score_col].astype(float).round(5) > 0.0001)
                            ]
                            ["question_id"].dropna().astype(str).tolist()
                        )
                print(f"[Retrieval] Resuming — {len(completed_ids)} questions already processed.")
        except Exception as e:
            print(f"[Retrieval] Warning: could not read existing output: {e}")

    print("[Retrieval] Initializing RAG engine...")
    engine = create_rag_engine()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If force_all, we need to re-write the file from scratch with new retrieval
    if force_all:
        backup = output_path.with_suffix(".csv.bak")
        if output_path.exists():
            import shutil
            shutil.copy2(output_path, backup)
            print(f"[Retrieval] Backup saved to {backup}")
        # Read existing data to preserve non-retrieval columns (answer, etc.)
        existing_data = {}
        try:
            old_df = pd.read_csv(output_path)
            for _, r in old_df.iterrows():
                qid = str(r.get("question_id", "")).strip()
                if qid:
                    existing_data[qid] = r.to_dict()
        except Exception:
            pass
        # Clear file for fresh write
        output_path.unlink(missing_ok=True)

    processed = 0
    skipped = 0
    for idx, row in questions_df.iterrows():
        q_id = str(row.get("id", idx)).strip()
        q_text = str(row.get("question", "")).strip()
        counter = f"[{idx + 1}/{len(questions_df)}]"

        if not force_all and q_id in completed_ids:
            print(f"{counter} Skipping {q_id} (already done)")
            skipped += 1
            continue

        print(f"\n{counter} Retrieving for question {q_id}: \"{q_text[:60]}...\"")
        try:
            nodes = engine._retrieve_nodes(q_text, top_k=args.top_k)
            sources = extract_sources_from_nodes(nodes, max_sources=args.top_k)
            source_row = sources_to_csv_row(sources, max_sources=args.top_k)

            result_row = {
                "question_id": q_id,
                "question": q_text[:500],
                **source_row,
            }

            # If migrating, preserve non-retrieval columns from old data
            if force_all and q_id in existing_data:
                from csv_utils import resolve_language
                old_row = existing_data[q_id]
                for col in ["answer", "base_model_used", "language", "attempts", "grounded"]:
                    if col in old_row and col not in result_row:
                        val = old_row[col]
                        result_row[col] = "" if pd.isna(val) else val
                # Repair language from the question_id when old value is missing/corrupt
                result_row["language"] = resolve_language(old_row.get("language", ""), q_id)

            # Update the existing row in place (or append if new). result_row only
            # carries question_id/question/source{n}_* (plus carried fields in
            # migration mode), so a previously-written answer/grounded is preserved.
            upsert_row(result_row, output_path, OUTPUT_COLUMNS, key_cols=("question_id",))
            processed += 1
            print(f"   Retrieved {len(sources)} chunks — source columns updated in {output_path}")
            try:
                from dashboard_generator import refresh_dashboard
                refresh_dashboard(lufa_csv=str(output_path))
            except Exception as _de:
                print(f"   [Dashboard] refresh skipped: {_de}")
        except Exception as e:
            print(f"   Error on {q_id}: {e}")
            traceback.print_exc()
        time.sleep(0.1)

    print(f"\n[Retrieval] Done. Processed={processed}, Skipped={skipped}. Results in {output_path}")
