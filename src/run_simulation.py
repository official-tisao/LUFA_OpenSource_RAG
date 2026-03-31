#!/usr/bin/env python3
"""
Simulation script — runs every question in combined_test_data.csv
through the Agentic RAG pipeline and stores results in lufa_out_data.csv.
Supports crash-resumption and saves outputs row-by-row incrementally.
Guarantees strict column alignment matching the evaluation dashboard schema.
Automatically identifies and re-runs error records, then continues with new ones.
"""

import sys
import argparse
import csv
import time
import traceback
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

INPUT_CSV = "tests/combined_test_data_and_ground_truth.csv"
OUTPUT_CSV = "tests/lufa_out_data.csv"
CONFIG = "config/config.yaml"

OUTPUT_COLUMNS = [
    "question_id", "question", "answer", "base_model_used", "language", "attempts", "grounded",
    "source1_id", "source1_score", "source1_text",
    "source2_id", "source2_score", "source2_text",
    "source3_id", "source3_score", "source3_text",
    "source4_id", "source4_score", "source4_text",
    "source5_id", "source5_score", "source5_text",
    "original_cosine_score", "recency_adjusted_score", "RRF"
]


def load_config(path=CONFIG):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def extract_sources(sources, max_sources=5):
    """Flatten up to 5 sources into flat dict keys for CSV, preserving original database UUIDs."""
    row = {}
    for i in range(1, max_sources + 1):
        if i <= len(sources):
            s = sources[i - 1]
            meta = s.get("metadata", {})

            row[f"source{i}_id"] = (s.get("node_id") or s.get("chunk_id") or meta.get("node_id") or meta.get("id") or
                                    str(meta.get("source_doc", "")) + f"_chunk{i}")

            if "original_cosine_score" in meta:
                row[f"source{i}_score"] = round(float(meta["original_cosine_score"]), 4)
            else:
                row[f"source{i}_score"] = round(float(s.get("score") or 0), 4)

            row[f"source{i}_text"] = str(s.get("text", ""))[:500]
        else:
            row[f"source{i}_id"] = ""
            row[f"source{i}_score"] = ""
            row[f"source{i}_text"] = ""
    return row


def query_single_record(record, mode, base_model, model_name, api_url, idx):
    """
    Core atomic method to prompt the backend RAG layout for a single row.
    Provides deep verbose terminal feedback for interactive visibility.
    """
    q_text = str(record["question"])
    q_id = str(record["id"])
    print(f"   [Simulation Engine] Initializing query transaction for ID: {q_id}")
    print(f"   [Simulation Engine] Query String: \"{q_text[:65]}...\"")
    print(f"   [Simulation Engine] Mode: '{mode}' | Target Model: '{model_name}'")

    try:
        if mode == "local":
            from rag_engine import create_rag_engine
            engine = create_rag_engine()
            result = engine.agentic_query(
                query_text=q_text,
                return_sources=True,
                max_retries=3,
            )

        elif mode == "api":
            import httpx
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(
                    f"{api_url}/agentic-query",
                    json={"query": q_text, "return_sources": True, "max_retries": 3},
                )
                resp.raise_for_status()
                result = resp.json()

        elif mode == "frontier":
            from rag_engine import create_rag_engine
            from copilot_engine import CopilotEngine
            rag_engine = create_rag_engine()
            copilot = CopilotEngine(model=model_name)
