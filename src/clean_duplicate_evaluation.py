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
