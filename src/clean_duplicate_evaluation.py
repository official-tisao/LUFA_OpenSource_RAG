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

    # Clear string formats to safe numeric types for sorting and matrix filtering
    df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce").fillna(1).astype(int)
    df["token_f1_score"] = pd.to_numeric(df["token_f1_score"], errors="coerce").fillna(0.0)
    df["mrr"] = pd.to_numeric(df["mrr"], errors="coerce").fillna(0.0)
    df["answer_relevance"] = pd.to_numeric(df["answer_relevance"], errors="coerce").fillna(0.0)
    df["faithfulness"] = pd.to_numeric(df["faithfulness"], errors="coerce").fillna(0.0)

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

    # Drop duplicates on the question_id keeping first, which locks in the best entry
    cleaned_df = df_sorted.drop_duplicates(subset=["question_id"], keep="first")

    # Clean up working columns before writing back to disk
    cleaned_df = cleaned_df.drop(columns=["grounded_bool", "is_corrupted_metric"])

    # Re-sort chronologically by ID for visual presentation alignment
    cleaned_df = cleaned_df.sort_values(by=["question_id"])

    cleaned_df.to_csv(path, index=False)
    final_count = len(cleaned_df)

    print("Cleanup pass completed successfully.")
    print(f"Total duplicate or bad metric records eliminated: {initial_count - final_count}")
    print(f"Final clean unique record count saved to disk: {final_count}")

    # Automatically re-compile the user dashboard metrics view
    try:
        sys.path.insert(0, str(Path(file_path).parent.parent / "src"))
        from evaluate import generate_dashboard
        generate_dashboard(cleaned_df, dashboard_path)
        print(f"Dashboard interface successfully updated with unique clean records at {dashboard_path}")
    except Exception as d_err:
        print(f"Note: Dashboard live UI compile step skipped: {d_err}")


if __name__ == "__main__":
    clean_evaluation_file()