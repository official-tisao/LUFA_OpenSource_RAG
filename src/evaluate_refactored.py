#!/usr/bin/env python3
"""
Enhanced evaluation script with granular persistence for immediate metric storage.
Ensures recovery from timeouts and systematic repair capabilities.
"""

import sys
import argparse
import time
import traceback
from pathlib import Path
from datetime import datetime
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import cfg

# Import modular components
sys.path.insert(0, str(Path(__file__).parent))
import csv_utils
import metrics
import retrieval
import ground_truth
import answer_generator

# Import original utilities for backward compatibility
from src.rag_engine import create_rag_engine
from src.run_simulation import query_single_record
from src.dashboard_generator import generate_dashboard

OUTPUT_COLUMNS = [
    "question_id", "question",
    "source1_id", "source1_cosine_score", "source1_recency_adjusted_cosine_score", "source1_rrf_score", "source1_text",
    "source2_id", "source2_cosine_score", "source2_recency_adjusted_cosine_score", "source2_rrf_score", "source2_text",
    "source3_id", "source3_cosine_score", "source3_recency_adjusted_cosine_score", "source3_rrf_score", "source3_text",
    "source4_id", "source4_cosine_score", "source4_recency_adjusted_cosine_score", "source4_rrf_score", "source4_text",
    "source5_id", "source5_cosine_score", "source5_recency_adjusted_cosine_score", "source5_rrf_score", "source5_text",
    "answer", "base_model_used", "language", "attempts", "grounded",
    "translation_applied", "translated_question", "untranslated_answer", "translation_pipeline_language",
]

LUFA_COLUMNS = list(OUTPUT_COLUMNS)

EVAL_COLUMNS = [
    "question_id", "id", "question", "answer", "base_model_used", "rag_base_model",
    "language", "judge_llm", "category", "difficulty", "attempts", "grounded",
    "source1_id", "source1_cosine_score", "source1_recency_adjusted_cosine_score", "source1_rrf_score", "source1_text",
    "source2_id", "source2_cosine_score", "source2_recency_adjusted_cosine_score", "source2_rrf_score", "source2_text",
    "source3_id", "source3_cosine_score", "source3_recency_adjusted_cosine_score", "source3_rrf_score", "source3_text",
    "source4_id", "source4_cosine_score", "source4_recency_adjusted_cosine_score", "source4_rrf_score", "source4_text",
    "source5_id", "source5_cosine_score", "source5_recency_adjusted_cosine_score", "source5_rrf_score", "source5_text",
    "token_f1_score", "sentence_bleu_score", "rouge1", "rouge2", "rougeL", "meteor",
    "mrr", "ndcg_at_k", "recall_1", "recall_3", "recall_5",
    "answer_relevance", "faithfulness", "context_precision",
    "translation_applied", "translated_question", "untranslated_answer", "translation_pipeline_language",
]
def safe_float(val, default_val=0.0):
    if pd.isna(val) or val == "":
        return default_val
    try:
        return float(val)
    except Exception:
        return default_val
def load_existing_data_safely(file_path, required_columns=None):
    """Load existing data with comprehensive error handling."""
    if not Path(file_path).exists() or Path(file_path).stat().st_size == 0:
        if required_columns:
            return pd.DataFrame(columns=required_columns)
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path, on_bad_lines="skip")
    except Exception as e:
        print(f"[Warning] Could not load {file_path}: {e}")
        if required_columns:
            return pd.DataFrame(columns=required_columns)
        return pd.DataFrame()
def save_dataframe_incrementally(df, file_path, file_format="csv"):
    """Save DataFrame with immediate persistence and backup."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Create backup before overwriting
        backup_path = file_path.with_suffix(f".backup_{int(time.time())}{file_path.suffix}")
        if file_path.exists():
            df.to_csv(backup_path, index=False)

        # Save main file
        df.to_csv(file_path, index=False)

        # Verify save was successful
        if Path(file_path).exists() and Path(file_path).stat().st_size > 0:
            print(f"      [✓] Successfully saved {len(df)} rows to {file_path.name}")
            return True
        else:
            print(f"      [✗] Failed to save data to {file_path.name}")
            return False

    except Exception as e:
        print(f"      [✗] Error saving to {file_path.name}: {e}")
        return False
def get_existing_metrics_completeness(eval_df, question_id):
    """Check which metrics are already present for a question."""
    mask = eval_df["question_id"] == question_id
    if not mask.any():
        return {"generation": False, "retrieval": False, "judge": False}

    row = eval_df[mask].iloc[0]

    # Check generation metrics (has meaningful values)
    gen_metrics = ["token_f1_score", "sentence_bleu_score", "rouge1", "rouge2", "rougeL", "meteor"]
    has_generation = any(row[col] not in [0, "", None] for col in gen_metrics)

    # Check retrieval metrics
    ret_metrics = ["mrr", "ndcg_at_k", "recall_1", "recall_3", "recall_5"]
    has_retrieval = any(row[col] not in [0, "", None] for col in ret_metrics)

    # Check judge metrics
    judge_metrics = ["answer_relevance", "faithfulness", "context_precision"]
    has_judge = any(row[col] not in [0, "", None] for col in judge_metrics)

    return {"generation": has_generation, "retrieval": has_retrieval, "judge": has_judge}
def calculate_generation_metrics(row, prediction, reference):
    """Calculate all generation metrics with error handling."""
    try:
        gen_metrics = {}

        gen_metrics["token_f1_score"] = round(metrics.token_f1(prediction, reference), 4)
        gen_metrics["sentence_bleu_score"] = round(metrics.compute_bleu(prediction, reference), 4)

        rouge_scores = metrics.compute_rouge(prediction, reference)
        gen_metrics.update({
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"]
        })

        gen_metrics["meteor"] = round(metrics.compute_meteor(prediction, reference), 4)

        return gen_metrics, None

    except Exception as e:
        print(f"      [Generation Error] Failed to calculate metrics: {e}")
        return {"token_f1_score": 0.0, "sentence_bleu_score": 0.0, "rouge1": 0.0,
                "rouge2": 0.0, "rougeL": 0.0, "meteor": 0.0}, e
def calculate_retrieval_metrics(retrieved_ids, ground_truth_ids):
    """Calculate retrieval metrics with error handling."""
    try:
        ret_metrics = {}

        ret_metrics["mrr"] = round(metrics.mrr(retrieved_ids, ground_truth_ids), 4)
        ret_metrics["ndcg_at_k"] = metrics.ndcg_at_k(retrieved_ids, ground_truth_ids, k=5)
        ret_metrics["recall_1"] = round(metrics.recall_at_k(retrieved_ids, ground_truth_ids, k=1), 4)
        ret_metrics["recall_3"] = round(metrics.recall_at_k(retrieved_ids, ground_truth_ids, k=3), 4)
        ret_metrics["recall_5"] = round(metrics.recall_at_k(retrieved_ids, ground_truth_ids, k=5), 4)

        return ret_metrics, None

    except Exception as e:
        print(f"      [Retrieval Error] Failed to calculate metrics: {e}")
        return {"mrr": 0.0, "ndcg_at_k": 0.0, "recall_1": 0.0, "recall_3": 0.0, "recall_5": 0.0}, e
def calculate_judge_metrics_with_timeout(question, answer, context, judge_llm_model, timeout_seconds, use_combined=False):
    """Calculate judge metrics with timeout protection."""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Judge evaluation timed out after {timeout_seconds} seconds")

    try:
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        if use_combined:
            # Use combined judge
            try:
                from metrics import combined_judge_llm_scores
                return combined_judge_llm_scores(question, answer, context, judge_llm_model)
            except ImportError:
                print(f"      [Judge Warning] Combined judge not available, falling back to separate calls")
                return calculate_separate_judge_metrics_safe(question, answer, context, judge_llm_model)
        else:
            # Use separate judges
            return calculate_separate_judge_metrics_safe(question, answer, context, judge_llm_model)

    except TimeoutError:
        print(f"      [Judge Timeout] Evaluation timed out after {timeout_seconds} seconds")
        return {"answer_relevance": 0.0, "faithfulness": 0.0, "context_precision": 0.0}
    except Exception as e:
        print(f"      [Judge Error] Evaluation failed: {e}")
        return {"answer_relevance": 0.0, "faithfulness": 0.0, "context_precision": 0.0}
    finally:
        signal.alarm(0)  # Cancel the alarm
def calculate_separate_judge_metrics_safe(question, answer, context, judge_llm_model):
    """Calculate separate judge metrics with error handling."""
    try:
        from metrics import JUDGE_LLM_PROMPTS, extract_score, _load_judge_model

        llm = _load_judge_model(judge_llm_model)
        scores = {}

        for metric, prompt_template in JUDGE_LLM_PROMPTS.items():
            try:
                prompt = prompt_template.format(
                    question=question[:500],
                    answer=answer[:800],
                    context=context[:3000],
                )
                from llm_utils import stream_complete
                raw = stream_complete(llm, prompt)
                scores[metric] = round(extract_score(raw), 4)
            except Exception as e:
                print(f"      [Judge Warning] {metric} calculation failed: {e}")
                scores[metric] = 0.0

        return scores

    except Exception as e:
        print(f"      [Judge Error] Separate judge calculation failed: {e}")
        return {"answer_relevance": 0.0, "faithfulness": 0.0, "context_precision": 0.0}
def format_metrics_for_storage(question_data, gen_metrics, ret_metrics, judge_metrics):
    """Format all metrics into a single row for storage."""
    row = {
        "question_id": question_data["id"],
        "id": question_data["id"],
        "question": question_data.get("question", ""),
        "answer": question_data.get("answer", ""),
        "base_model_used": question_data.get("base_model_used", ""),
        "language": question_data.get("language", ""),
        "attempts": int(question_data.get("attempts", 1)),
        "grounded": str(question_data.get("grounded", False)).lower(),

        # Source data preservation (per-chunk scores)
        **{f"source{i}_{f}": question_data.get(f"source{i}_{f}", "")
           for i in range(1, 6)
           for f in ["id", "cosine_score", "recency_adjusted_cosine_score", "rrf_score", "text"]},

        # Model information
        "rag_base_model": question_data.get("rag_base_model", ""),
        "judge_llm": question_data.get("judge_llm", ""),
        "category": question_data.get("category", ""),
        "difficulty": question_data.get("difficulty", ""),

        # All metrics
        **gen_metrics,
        **ret_metrics,
        **judge_metrics
    }

    return row
def consolidate_and_save_metrics(eval_df, out_csv, dashboard_out):
    """Consolidate all metrics and save final outputs."""
    try:
        # Remove duplicate rows based on question_id and rag_base_model
        clean_df = eval_df.drop_duplicates(subset=["question_id", "rag_base_model"], keep="last")

        # Save consolidated metrics
        if save_dataframe_incrementally(clean_df, out_csv):
            print(f"[✓] Consolidated metrics saved successfully")

            # Generate dashboard
            try:
                generate_dashboard(clean_df, dashboard_out)
                print(f"[✓] Dashboard generated successfully")
                return True
            except Exception as d_err:
                print(f"[Dashboard Warning] Could not generate dashboard: {d_err}")
                return False
        else:
            print(f"[✗] Failed to save consolidated metrics")
            return False

    except Exception as e:
        print(f"[✗] Error in consolidation process: {e}")
        return False
def run_enhanced_evaluation(
        lufa_csv="tests/lufa_out_data.csv",
        test_csv="tests/combined_test_data_and_ground_truth.csv",
        out_csv="tests/evaluation_results.csv",
        dashboard_out="dashboard/index.html",
        judge_llm_model=None,
        llm_model=None,
        sim_mode="local",
        api_url="http://localhost:8000",
        use_combined_judge=False,
        judge_timeout=240.0,
        force_regen=False
):
    """Enhanced evaluation with granular persistence and error recovery."""

    print("=" * 80)
    print("ENHANCED EVALUATION MODULE - GRANULAR METRIC PERSISTENCE")
    print("=" * 80)

    # Initialize configuration
    llm_model = llm_model or cfg("models.llm.name")
    judge_llm_model = judge_llm_model or cfg("models.judge_llm.name")

    # Load all existing data
    print("[1] Loading existing data...")
    test_df = pd.read_csv(test_csv)

    lufa_data = load_existing_data_safely(lufa_csv, LUFA_COLUMNS)
    evaluation_data = load_existing_data_safely(out_csv, EVAL_COLUMNS)

    # Create working dataframes
    lufa_records = {}
    for _, row in lufa_data.iterrows():
        q_id = str(row.get("question_id", "")).strip()
        if q_id and q_id != "nan":
            lufa_records[q_id] = row.to_dict()

    # Determine completion status
    completed_ids = determine_completed_ids(evaluation_data)

    # Display status
    total_questions = len(test_df)
    completed_count = len(completed_ids)
    pending_count = total_questions - completed_count

    print(f"\n[Status] Questions: {total_questions} total")
    print(f"[Status] Completed: {completed_count}")
    print(f"[Status] Pending: {pending_count}")

    if force_regen:
        print(f"\n[Force Regen] Selected - processing all questions (skip_completion_check=True)")

    # Pre-cache database data if available
    print("[2] Loading database resources...")
    chroma_cached_data = load_chroma_data_if_available()

    # Initialize evaluation dataframe with all columns
    eval_df = evaluation_data.copy() if not evaluation_data.empty else pd.DataFrame(columns=EVAL_COLUMNS)

    # Ensure all columns exist
    for col in EVAL_COLUMNS:
        if col not in eval_df.columns:
            eval_df[col] = ""

    # Start processing loop with granular persistence
    print("\n[3] Starting granular processing loop...")
    print("     Metrics will be saved immediately after each calculation")
    print("     " + "-" * 60)

    processing_start_time = time.time()

    for idx, row in test_df.iterrows():
        current_counter = idx + 1
        q_id = str(row.get("id")).strip()

        # Skip already completed unless force regeneration
        if q_id in completed_ids and not force_regen:
            completeness = get_existing_metrics_completeness(eval_df, q_id)
            all_complete = all(completeness.values())

            if all_complete:
                print(f"[{current_counter:4d}/{total_questions}] Question {q_id}: ✓ All metrics complete - skipping")
                continue
            else:
                print(f"[{current_counter:4d}/{total_questions}] Question {q_id}: ⚠ Partial metrics - attempting recovery")
                # Try to recover missing metrics

        print(f"\n[{current_counter:4d}/{total_questions}] Question {q_id}:")

        # Get or generate lufa data with immediate persistence
        try:
            active_rag_data = get_or_generate_lufa_data_improved(
                q_id, row, lufa_records, lufa_csv, sim_mode, cfg_base_model, llm_model, api_url, current_counter
            )
        except Exception as e:
            print(f"   ✗ Critical error getting/generating lufa data: {e}")
            continue

        # Calculate and immediately save generation metrics
        print("   📝 Calculating generation metrics...")
        try:
            prediction = "" if pd.isna(active_rag_data.get("answer")) else str(active_rag_data.get("answer"))
            reference = "" if pd.isna(row.get("expected_answer")) else str(row.get("expected_answer"))

            gen_metrics, gen_error = calculate_generation_metrics(row, prediction, reference)

            if gen_error:
                print(f"      ⚠ Generation metrics calculation had issues: {gen_error}")

            # Prepare storage row
            storage_row = format_metrics_for_storage(row, gen_metrics, {}, {})

            # Save immediately
            if save_dataframe_incrementally(pd.DataFrame([storage_row]), out_csv):
                print("      ✓ Generation metrics saved immediately")
            else:
                print("      ✗ Failed to save generation metrics")

        except Exception as e:
            print(f"   ✗ Error in generation metrics: {e}")
            continue

        # Calculate and immediately save retrieval metrics
        print("   🔍 Calculating retrieval metrics...")
        try:
            retrieved_ids = build_retrieved_ids_from_rag_data(active_rag_data)
            gt_ids = resolve_ground_truth_ids_enhanced(row, chroma_cached_data)

            ret_metrics, ret_error = calculate_retrieval_metrics(retrieved_ids, gt_ids)

            if ret_error:
                print(f"      ⚠ Retrieval metrics calculation had issues: {ret_error}")

            # Prepare storage row
            storage_row = format_metrics_for_storage(row, {}, ret_metrics, {})

            # Save immediately
            if save_dataframe_incrementally(pd.DataFrame([storage_row]), out_csv, mode='a'):
                print("      ✓ Retrieval metrics saved immediately")
            else:
                print("      ✗ Failed to save retrieval metrics")

        except Exception as e:
            print(f"   ✗ Error in retrieval metrics: {e}")
            continue

        # Calculate and immediately save judge metrics
        print("   ⚖️  Calculating judge metrics...")
        try:
            context = build_context_from_row_enhanced(active_rag_data)
            question = str(row.get("question", ""))
            answer = "" if pd.isna(active_rag_data.get("answer")) else str(active_rag_data.get("answer"))

            judge_metrics = calculate_judge_metrics_with_timeout(
                question, answer, context, judge_llm_model, judge_timeout, use_combined_judge
            )

            # Prepare storage row
            storage_row = format_metrics_for_storage(row, {}, {}, judge_metrics)

            # Save immediately
            if save_dataframe_incrementally(pd.DataFrame([storage_row]), out_csv, mode='a'):
                print("      ✓ Judge metrics saved immediately")
            else:
                print("      ✗ Failed to save judge metrics")

        except Exception as e:
            print(f"   ✗ Error in judge metrics: {e}")
            continue

        # Update evaluation DataFrame
        completeness = get_existing_metrics_completeness(eval_df, q_id)
        if all(completeness.values()):
            completed_ids.add(q_id)

        # Save intermediate results periodically
        if current_counter % 5 == 0 or current_counter == total_questions:
            print(f"   💾 Periodic save: Question {current_counter}/{total_questions} complete")
            consolidate_and_save_metrics(eval_df, out_csv, dashboard_out)

        # Progress display
        if current_counter % 20 == 0 or current_counter == total_questions:
            elapsed = time.time() - processing_start_time
            print(f"\n   📊 Progress: {current_counter}/{total_questions} questions ({elapsed:.1f}s elapsed)")

    # Final consolidation
    print(f"\n[4] Final consolidation and cleanup...")
    success = consolidate_and_save_metrics(eval_df, out_csv, dashboard_out)

    if success:
        elapsed_time = time.time() - processing_start_time
        print(f"\n[✓] Processing completed successfully!")
        print(f"[✓] Processed {len(eval_df)} questions in {elapsed_time:.1f} seconds")
        print(f"[✓] Results saved to: {out_csv}")
        print(f"[✓] Dashboard available at: {dashboard_out}")
    else:
        print(f"\n[✗] Processing completed with errors")
        print(f"[✗] Partial results may be available in: {out_csv}")

    return success
# Helper functions (continued from above)
def load_chroma_data_if_available():
    """Load ChromaDB data if available for ground truth resolution."""
    try:
        import chromadb
        from config_loader import cfg

        client = chromadb.PersistentClient(path=cfg("database.path"))
        collection = client.get_collection(cfg("database.collection_name"))
        return collection.get(include=["documents"])
    except Exception as e:
        print(f"[Warning] Could not preload ChromaDB data: {e}")
        return None
def build_retrieved_ids_from_rag_data(rag_data):
    """Extract retrieved IDs from RAG data."""
    ids = []
    for i in range(1, 6):
        sid = rag_data.get(f"source{i}_id")
        if sid and sid != "":
            ids.append(str(sid))
    return ids
def resolve_ground_truth_ids_enhanced(row, chroma_data):
    """Enhanced ground truth ID resolution."""
    try:
        from metrics import resolve_ground_truth_ids
        return resolve_ground_truth_ids(row, chroma_data)
    except ImportError:
        # Fallback implementation
        gt_col = "ground_source_truth_id" if "ground_source_truth_id" in row.index else "ground_truth_source_ids"
        if gt_col in row:
            raw_ids = row[gt_col]
            if raw_ids and not pd.isna(raw_ids):
                return [s.strip() for s in str(raw_ids).split("|") if s.strip()]
        return []
def build_context_from_row_enhanced(row):
    """Build context from RAG data with enhanced error handling."""
    parts = []
    for i in range(1, 6):
        text = row.get(f"source{i}_text", "")
        if text and text != "" and not pd.isna(text):
            parts.append(str(text))
    return "\n\n---\n\n".join(parts)
def get_or_generate_lufa_data_improved(q_id, row, lufa_records, lufa_csv, sim_mode,
                                         cfg_base_model, llm_model, api_url, current_counter):
    """Improved version of lufa data retrieval/generation."""
    if q_id not in lufa_records:
        print(f"   ⚠️  Question {q_id} not in lufa data - generating...")

        try:
            record_dict = row.to_dict()
            sim_output = query_single_record(
                record_dict, sim_mode, cfg_base_model, llm_model, api_url, current_counter
            )

            lufa_records[q_id] = sim_output

            # Save immediately (schema-safe — values always land under correct header)
            from csv_utils import align_and_append
            align_and_append(sim_output, lufa_csv, list(LUFA_COLUMNS))
            print(f"      ✓ Generated and saved lufa data for question {q_id}")

            return sim_output

        except Exception as e:
            print(f"      ✗ Failed to generate lufa data: {e}")
            raise
    else:
        print(f"      ✓ Using existing lufa data for question {q_id}")
        return lufa_records[q_id]
# Export all key functions for use
__all__ = [
    'run_enhanced_evaluation',
    'get_existing_metrics_completeness',
    'determine_completed_ids',
    'save_dataframe_incrementally',
    'calculate_generation_metrics',
    'calculate_retrieval_metrics',
    'calculate_judge_metrics_with_timeout',
    'format_metrics_for_storage',
    'consolidate_and_save_metrics',
    'load_existing_data_safely',
    'get_or_generate_lufa_data_improved',
    'build_retrieved_ids_from_rag_data',
    'resolve_ground_truth_ids_enhanced',
    'build_context_from_row_enhanced'
]
if __name__ == "__main__":
    # Command line interface
    parser = argparse.ArgumentParser(
        description="Enhanced Evaluation Module - Granular Metric Persistence"
    )
    parser.add_argument("--lufa_csv", default="tests/lufa_out_data.csv", help="Path to LUFA output CSV")
    parser.add_argument("--test_csv", default="tests/combined_test_data_and_ground_truth.csv", help="Path to test data CSV")
    parser.add_argument("--out_csv", default="tests/evaluation_results.csv", help="Path to output CSV")
    parser.add_argument("--dashboard", default="dashboard/index.html", help="Path to dashboard HTML")
    parser.add_argument("--judge_llm", default=None, help="Judge LLM model name")
    parser.add_argument("--llm_model", default=None, help="Main LLM model name")
    parser.add_argument("--sim_mode", choices=["local", "local-naive", "api", "frontier"], default="local", help="Simulation mode")
    parser.add_argument("--api_url", default="http://localhost:8000", help="API URL for remote mode")
    parser.add_argument("--use_combined_judge", action="store_true", help="Use combined judge prompt")
    parser.add_argument("--judge_timeout", type=float, default=240.0, help="Judge timeout in seconds")
    parser.add_argument("--force_regen", action="store_true", help="Force regeneration of all questions")

    args = parser.parse_args()

    # Run enhanced evaluation
    success = run_enhanced_evaluation(
        lufa_csv=args.lufa_csv,
        test_csv=args.test_csv,
        out_csv=args.out_csv,
        dashboard_out=args.dashboard,
        judge_llm_model=args.judge_llm,
        llm_model=args.llm_model,
        sim_mode=args.sim_mode,
        api_url=args.api_url,
        use_combined_judge=args.use_combined_judge,
        judge_timeout=args.judge_timeout,
        force_regen=args.force_regen
    )

    exit(0 if success else 1)