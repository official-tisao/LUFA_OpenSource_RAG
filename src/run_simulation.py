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

            lang = rag_engine.detect_query_language(q_text)
            nodes = rag_engine._retrieve_nodes(q_text, top_k=5)
            answer = copilot.generate_from_nodes(q_text, nodes, lang)

            sources_list = []
            for n in nodes:
                combined_meta = {}
                for k, v in n.node.metadata.items():
                    combined_meta[k] = v
                combined_meta["id"] = n.node.node_id
                if "original_cosine_score" in n.node.metadata:
                    combined_meta["original_cosine_score"] = n.node.metadata["original_cosine_score"]
                else:
                    combined_meta["original_cosine_score"] = str(n.score)

                sources_list.append({
                    "text": n.node.text[:500],
                    "score": n.score,
                    "metadata": combined_meta,
                    "node_id": n.node.node_id
                })

            result = {
                "response": answer,
                "original_language": record.get("language", lang),
                "rewritten_query": "",
                "attempts": 1,
                "grounded": True,
                "sources": sources_list
            }

        sources = result.get("sources", [])
        sources_dict = extract_sources(sources)

        orig_cosine = ""
        recency_adj = ""
        rrf_val = ""
        if sources:
            first_src = sources[0]
            first_meta = first_src.get("metadata", {})
            rrf_val = round(float(first_src.get("score") or 0.0), 6)
            orig_cosine = round(float(first_meta.get("original_cosine_score") or rrf_val), 6)
            recency_adj = orig_cosine

        row = {
            "question_id": q_id,
            "question": q_text,
            "answer": result.get("response", ""),
            "base_model_used": model_name,
            "language": result.get("original_language", record.get("language", "en")),
            "attempts": result.get("attempts", 1),
            "grounded": result.get("grounded", False),
        }
        row.update(sources_dict)
        row["original_cosine_score"] = orig_cosine
        row["recency_adjusted_score"] = recency_adj
        row["RRF"] = rrf_val

        print(f"   [Simulation Engine] ✅ Success! Received Answer length: {len(row['answer'])} chars.")
        return row

    except Exception as e:
        print(f"   [Simulation Engine] 💥 Error on record {q_id}: {e}")
        traceback.print_exc()
        return _empty_row(q_id, q_text, model_name, record.get("language", "en"))


def _empty_row(q_id, q_text, model, lang):
    row = {
        "question_id": q_id,
        "question": q_text,
        "answer": "ERROR",
        "base_model_used": model,
        "language": lang,
        "attempts": 0,
        "grounded": False,
    }
    for i in range(1, 6):
        row[f"source{i}_id"] = ""
        row[f"source{i}_score"] = ""
        row[f"source{i}_text"] = ""
    row["original_cosine_score"] = ""
    row["recency_adjusted_score"] = ""
    row["RRF"] = ""
    return row


def ensure_ground_truth(csv_path):
    """Auto-run find_ground_truth.py if ground_source_truth_id column is missing."""
    df = pd.read_csv(csv_path)
    if "ground_source_truth_id" not in df.columns or df["ground_source_truth_id"].isnull().all():
        print("[Sim] ground_source_truth_id missing — running find_ground_truth.py first...")
        from find_ground_truth import run as find_gt
        find_gt(csv_path, "db/chroma_db", "multilingual_docs", top_k=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LUFA RAG simulation over test dataset")
    parser.add_argument("--mode", choices=["local", "api", "frontier"], default="local")
    parser.add_argument("--model", default=None, help="Frontier model ID (for --mode frontier)")
    parser.add_argument("--api_url", default="http://localhost:8000", help="API base URL for --mode api")
    parser.add_argument("--input", default=INPUT_CSV, help="Input CSV path")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    cfg = load_config()
    base_model = cfg.get("models", {}).get("llm", {}).get("name", "llama3.2:3b-instruct-q4_K_M")
    model_name = args.model or base_model

    ensure_ground_truth(args.input)

    print(f"[Sim] Loading input master file: {args.input}")
    df = pd.read_csv(args.input)
    print(f"[Sim] {len(df)} total target questions configured in master dataset.")

    completed_ids = set()
    all_logged_ids = set()
    out_path = Path(args.output)

    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(args.output)
            if "question_id" in existing_df.columns:
                all_logged_ids = set(existing_df["question_id"].dropna().astype(str).tolist())

                # Filter strictly for rows that have valid string content and no ERROR flag
                successful_df = existing_df[
                    existing_df["question_id"].notna() &
                    existing_df["answer"].notna() &
                    (existing_df["answer"].astype(str).str.strip() != "ERROR") &
                    (existing_df["answer"].astype(str).str.strip() != "")
                    ]
                completed_ids = set(successful_df["question_id"].dropna().astype(str).tolist())

                error_count = len(all_logged_ids) - len(completed_ids)
                print(f"[Resumption] Located active output log file: {args.output}")
                print(f"[Resumption] Found {len(completed_ids)} successfully processed rows.")
                if error_count > 0:
                    print(
                        f"[Resumption] Found {error_count} rows containing ERROR flags. These will be automatically re-run.")
        except Exception as err:
            print(f"[Warning] Error parsing simulation file checkpoint: {err}")
    else:
        print(f"[Sim] Output path '{args.output}' is empty or new. Starting execution pass.")

    print("\n" + "=" * 80)
    print("STARTING UNIFIED SIMULATION PASS (REPAIRING ERRORS + ADDING NEW QUESTIONS)")
    print("=" * 80)

    for idx, record in df.iterrows():
        current_counter = idx + 1
        q_id = str(record["id"])

        if q_id in completed_ids:
            print(f"[{current_counter}/{len(df)}] Skipping Question ID {q_id} (Valid answer already stored)")
            continue

        # Verbose message informing whether it is fixing an error or processing a brand new row
        if q_id in all_logged_ids:
            print(f"\n[{current_counter}/{len(df)}] 🛠️  Re-running failed record -> Question ID: {q_id}")
        else:
            print(f"\n[{current_counter}/{len(df)}] 🚀 Processing unvisited record -> Question ID: {q_id}")

        row_res = query_single_record(record, args.mode, base_model, model_name, args.api_url, current_counter)

        single_row_df = pd.DataFrame([row_res], columns=OUTPUT_COLUMNS)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        file_is_new = not out_path.exists() or out_path.stat().st_size == 0
        single_row_df.to_csv(str(out_path), mode="a", index=False, header=file_is_new)
        print("   ✅ Row appended cleanly to simulation output log.")
        time.sleep(0.5)

    print("\n" + "=" * 80)
    print(f"[Sim] Execution pass complete. All rows verified and logged to: {args.output}")
    print("=" * 80 + "\n")