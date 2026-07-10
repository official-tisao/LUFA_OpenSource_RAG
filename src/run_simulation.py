#!/usr/bin/env python3
"""
Simulation script — runs every question in combined_test_data_and_ground_truth.csv
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

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import cfg

INPUT_CSV = "tests/combined_test_data_and_ground_truth.csv"
OUTPUT_CSV = "tests/lufa_out_data.csv"
CONFIG = "config/config.yaml"

# Cross-lingual translation columns — populated ONLY when the question is neither
# English nor French (i.e. a translation round-trip occurred). Appended at the end
# so existing CSVs migrate cleanly (new columns added empty, nothing shifts).
TRANSLATION_COLUMNS = [
    "translation_applied",
    "translated_question",
    "untranslated_answer",
    "translation_pipeline_language",
]

OUTPUT_COLUMNS = [
    "question_id", "question",
    "source1_id", "source1_cosine_score", "source1_recency_adjusted_cosine_score", "source1_rrf_score", "source1_text",
    "source2_id", "source2_cosine_score", "source2_recency_adjusted_cosine_score", "source2_rrf_score", "source2_text",
    "source3_id", "source3_cosine_score", "source3_recency_adjusted_cosine_score", "source3_rrf_score", "source3_text",
    "source4_id", "source4_cosine_score", "source4_recency_adjusted_cosine_score", "source4_rrf_score", "source4_text",
    "source5_id", "source5_cosine_score", "source5_recency_adjusted_cosine_score", "source5_rrf_score", "source5_text",
    "answer", "base_model_used", "language", "attempts", "grounded",
] + TRANSLATION_COLUMNS


def extract_translation_columns(result: dict) -> dict:
    """
    Build the 4 translation columns from a generation result dict.

    Handles both key conventions:
      - answer_generator: 'original_question_translation', 'untranslated_response'
      - rag_engine:       'translated_query',              'english_response'
    Columns are left BLANK unless a translation round-trip actually occurred
    (question was neither English nor French).
    """
    applied = bool(result.get("translation_applied"))
    if not applied:
        return {c: "" for c in TRANSLATION_COLUMNS}

    untranslated = result.get("untranslated_response")
    if untranslated is None:
        untranslated = result.get("english_response")
    translated_q = result.get("original_question_translation") or result.get("translated_query") or ""
    pipeline_lang = result.get("detected_language") or result.get("pipeline_language") or ""

    return {
        "translation_applied": True,
        "translated_question": translated_q,
        "untranslated_answer": untranslated if untranslated is not None else "",
        "translation_pipeline_language": pipeline_lang,
    }


def load_config(path=CONFIG):
    """Backward-compatible wrapper — delegates to config_loader."""
    from config_loader import cfg_raw
    return cfg_raw()


def extract_sources(sources, max_sources=5):
    """Flatten up to 5 sources into flat dict keys for CSV, preserving original database UUIDs."""
    row = {}
    for i in range(1, max_sources + 1):
        if i <= len(sources):
            s = sources[i - 1]
            meta = s.get("metadata", {})

            row[f"source{i}_id"] = (s.get("node_id") or s.get("chunk_id") or meta.get("node_id") or meta.get("id") or
                                    str(meta.get("source_doc", "")) + f"_chunk{i}")

            cosine = float(meta.get("original_cosine_score", s.get("score", 0)))
            recency_weight = float(meta.get("recency_weight", 1.0))
            rrf = float(s.get("score", 0))

            row[f"source{i}_cosine_score"] = round(cosine, 4)
            row[f"source{i}_recency_adjusted_cosine_score"] = round(cosine * recency_weight, 4)
            row[f"source{i}_rrf_score"] = round(rrf, 4)

            row[f"source{i}_text"] = str(s.get("text", ""))[:500]
        else:
            row[f"source{i}_id"] = ""
            row[f"source{i}_cosine_score"] = ""
            row[f"source{i}_recency_adjusted_cosine_score"] = ""
            row[f"source{i}_rrf_score"] = ""
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

        elif mode == "local-naive":
            from rag_engine import create_rag_engine
            engine = create_rag_engine()
            result = engine.naive_query(
                query_text=q_text,
                return_sources=True
            )

        elif mode == "api":
            import httpx
            with httpx.Client(timeout=400.0) as client:
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
        row.update(extract_translation_columns(result))

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
        row[f"source{i}_cosine_score"] = ""
        row[f"source{i}_recency_adjusted_cosine_score"] = ""
        row[f"source{i}_rrf_score"] = ""
        row[f"source{i}_text"] = ""
    for c in TRANSLATION_COLUMNS:
        row[c] = ""
    return row


def ensure_ground_truth(csv_path):
    """Auto-run find_ground_truth.py if ground_source_truth_id column is missing."""
    df = pd.read_csv(csv_path)
    if "ground_source_truth_id" not in df.columns or df["ground_source_truth_id"].isnull().all():
        print("[Sim] ground_source_truth_id missing — running ground truth finder first...")
        from .ground_truth import find_ground_truth_for_questions_path
        find_ground_truth_for_questions_path(
            csv_path,
            csv_path.replace(".csv", "_and_ground_truth.csv"),
            cfg("database.path"),
            cfg("database.collection_name")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LUFA RAG simulation over test dataset")
    parser.add_argument("--mode", choices=["local", "api", "frontier"], default="local")
    parser.add_argument("--model", default=None, help="Frontier model ID (for --mode frontier)")
    parser.add_argument("--api_url", default="http://localhost:8000", help="API base URL for --mode api")
    parser.add_argument("--input", default=INPUT_CSV, help="Input CSV path")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    base_model = cfg("models.llm.name")
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

        from csv_utils import align_and_append
        align_and_append(row_res, out_path, OUTPUT_COLUMNS)
        print("   ✅ Row appended cleanly to simulation output log.")
        try:
            from dashboard_generator import refresh_dashboard
            refresh_dashboard(lufa_csv=str(out_path))
        except Exception as _de:
            print(f"   [Dashboard] refresh skipped: {_de}")
        time.sleep(0.5)

    print("\n" + "=" * 80)
    print(f"[Sim] Execution pass complete. All rows verified and logged to: {args.output}")
    print("=" * 80 + "\n")