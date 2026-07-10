#!/usr/bin/env python3
"""
Cleanup and deduplication script for LUFA evaluation scorecards.
Applies quality filters to isolate and remove bad records, then deduplicates
the ledger by preserving the highest scoring row per question_id.
Automatically updates the real-time HTML dashboard with the clean outputs.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def _num_series(df, col):
    """Numeric column as a Series, or all-zeros when the column is absent."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _migrate_if_old(path, df, canonical):
    """If the frame uses the old single-score schema, migrate the file to the
    per-chunk canonical schema (backs up, renames source{n}_score, repairs
    language) and return the reloaded DataFrame."""
    from retrieval import has_old_schema
    from csv_utils import migrate_csv_schema
    if has_old_schema(df):
        print(f"[Clean] Old schema detected in {path} — migrating to per-chunk schema first...")
        migrate_csv_schema(str(path), list(canonical))
        return pd.read_csv(path, on_bad_lines="skip")
    return df


def clean_evaluation_file(file_path="tests/evaluation_results.csv", dashboard_path="dashboard/index.html"):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: Target evaluation file not found at {file_path}")
        return

    print(f"Loading ledger file for cleanup processing: {file_path}")
    df = pd.read_csv(path, on_bad_lines="skip")
    initial_count = len(df)
    print(f"Initial total record count: {initial_count}")

    if "question_id" not in df.columns:
        print("Error: Missing required column 'question_id' inside the file structure.")
        return

    # Migrate an old single-score schema to the per-chunk evaluation schema first.
    from evaluate import EVAL_COLUMNS
    df = _migrate_if_old(path, df, EVAL_COLUMNS)

    # Clear string formats to safe numeric types for sorting and matrix filtering
    df["attempts"] = _num_series(df, "attempts").fillna(1).astype(int)
    df["token_f1_score"] = _num_series(df, "token_f1_score")
    df["mrr"] = _num_series(df, "mrr")
    df["answer_relevance"] = _num_series(df, "answer_relevance")
    df["faithfulness"] = _num_series(df, "faithfulness")

    # Track true boolean array values for grounding check matching
    df["grounded_bool"] = df["grounded"].astype(str).str.strip().str.lower() == "true"

    # Identify bad rows based on advanced invalidation metrics criteria
    bad_rows_condition = (
            ((df["attempts"] > 2) & (~df["grounded_bool"])) |
            ((df["grounded_bool"]) & (df["faithfulness"] < 0.40)) |
            ((df["mrr"] > 0.0) & (df["token_f1_score"] == 0.0)) |
            ((df["attempts"] >= 2) & (df["answer_relevance"] < 0.40))
    )

    # Tag rows temporarily to isolate them during sorting
    df["is_corrupted_metric"] = bad_rows_condition

    # Sorting Strategy to bubble the absolute best records to the top:
    # 1. Valid metrics come first (is_corrupted_metric: False before True)
    # 2. Highest text matching score comes next (token_f1_score descending)
    # 3. Highest retrieval positioning comes next (mrr descending)
    # 4. Highest judge accuracy comes next (faithfulness descending)
    df_sorted = df.sort_values(
        by=["is_corrupted_metric", "token_f1_score", "mrr", "faithfulness"],
        ascending=[True, False, False, False]
    )

    # Drop duplicates on (question_id, rag_base_model) keeping first, which locks in the best entry per model
    cleaned_df = df_sorted.drop_duplicates(subset=["question_id", "rag_base_model"], keep="first")

    # Clean up working columns before writing back to disk
    cleaned_df = cleaned_df.drop(columns=["grounded_bool", "is_corrupted_metric"])

    # Re-sort chronologically by ID for visual presentation alignment
    #cleaned_df = cleaned_df.sort_values(by=["question_id"])

    cleaned_df.to_csv(path, index=False)
    final_count = len(cleaned_df)
    print("Cleanup pass completed successfully.")
    print(f"Cleanup record saved to {file_path} successfully.")
    print(f"Total duplicate or bad metric records eliminated: {initial_count - final_count}")
    print(f"Final clean unique record count saved to disk: {final_count}")

    # Automatically re-compile the user dashboard metrics view
    try:
        sys.path.insert(0, str(Path(file_path).parent.parent / "src"))
        from dashboard_generator import generate_dashboard
        generate_dashboard(cleaned_df, dashboard_path)
        print(f"Dashboard interface successfully updated with unique clean records at {dashboard_path}")
    except Exception as d_err:
        print(f"Note: Dashboard live UI compile step skipped: {d_err}")


def clean_lufa_file(file_path="tests/lufa_out_data.csv"):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: Target lufa_out file not found at {file_path}")
        return

    print(f"Loading lufa_out file for cleanup processing: {file_path}")
    df = pd.read_csv(path, on_bad_lines="skip")
    initial_count = len(df)
    print(f"Initial total record count: {initial_count}")

    if "question_id" not in df.columns:
        print("Error: Missing required column 'question_id' inside the file structure.")
        return

    # Migrate an old single-score schema to the per-chunk schema before sorting.
    from run_simulation import OUTPUT_COLUMNS as LUFA_COLUMNS
    df = _migrate_if_old(path, df, LUFA_COLUMNS)

    # Build TEMPORARY sort keys only — never mutate the real data columns.
    # (Coercing attempts/scores in place previously persisted junk like attempts=99.)
    sort_cols = []
    sort_asc = []

    for i in range(1, 6):
        key = f"_sort_rrf{i}"
        df[key] = _num_series(df, f"source{i}_rrf_score")
        sort_cols.append(key)
        sort_asc.append(False)
    for i in range(1, 6):
        key = f"_sort_rec{i}"
        df[key] = _num_series(df, f"source{i}_recency_adjusted_cosine_score")
        sort_cols.append(key)
        sort_asc.append(False)

    df["_sort_grounded"] = df.get("grounded", pd.Series("", index=df.index)).astype(str).str.strip().str.lower().map(
        lambda v: 0 if v in ("true", "1") else 1
    )
    df["_sort_has_answer"] = df.get("answer", pd.Series("", index=df.index)).map(
        lambda v: 0 if str(v).strip().lower() not in ("", "nan", "none", "error") else 1
    )
    # Lower attempts is better; blanks/garbage sort last WITHOUT overwriting the value.
    if "attempts" in df.columns:
        df["_sort_attempts"] = pd.to_numeric(df["attempts"], errors="coerce").fillna(9999)
    else:
        df["_sort_attempts"] = pd.Series(9999, index=df.index)

    sort_cols += ["_sort_grounded", "_sort_has_answer", "_sort_attempts"]
    sort_asc += [True, True, True]

    df_sorted = df.sort_values(by=sort_cols, ascending=sort_asc)

    cleaned_df = df_sorted.drop_duplicates(subset=["question_id", "base_model_used"], keep="first")

    cleaned_df = cleaned_df.drop(columns=[c for c in cleaned_df.columns if c.startswith("_sort_")])

    cleaned_df.to_csv(path, index=False)
    final_count = len(cleaned_df)
    print("Cleanup pass completed successfully.")
    print(f"Cleanup record saved to {file_path} successfully.")
    print(f"Total duplicate or bad records eliminated: {initial_count - final_count}")
    print(f"Final clean unique record count saved to disk: {final_count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean duplicates from evaluation or lufa_out CSVs")
    parser.add_argument("--target", choices=["eval", "lufa", "both"], default="both",
                        help="Which file(s) to clean (default: both)")
    parser.add_argument("--eval_csv", default="tests/evaluation_results.csv")
    parser.add_argument("--lufa_csv", default="tests/lufa_out_data.csv")
    parser.add_argument("--dashboard", default="dashboard/index.html")
    args = parser.parse_args()

    if args.target in ("eval", "both"):
        clean_evaluation_file(args.eval_csv, args.dashboard)
    if args.target in ("lufa", "both"):
        clean_lufa_file(args.lufa_csv)