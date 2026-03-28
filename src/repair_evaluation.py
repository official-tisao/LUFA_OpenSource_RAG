#!/usr/bin/env python3
"""
Advanced self-healing validation script for LUFA Agentic RAG system.
Identifies catastrophic system failures, flushes corrupted rows,
and re-runs full simulation, grading, and dashboard updates in one pass.
Provides real-time terminal progress metrics.
"""

import sys
import json
import argparse
import math
import re
import warnings
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_simulation import query_single_record, load_config
from evaluate import (
    token_f1, compute_bleu, compute_rouge, compute_meteor,
    mrr, ndcg_at_k, recall_at_k, llm_judge_scores,
    parse_source_ids, build_retrieved_ids, build_context_from_row,
    safe_float, generate_dashboard, LUFA_COLUMNS, EVAL_COLUMNS
)

warnings.filterwarnings("ignore")


def check_row_invalidation(row):
    """
    Evaluates row-level metrics against failure thresholds.
    Returns True if the row is deeply broken and requires re-processing.
    """
    try:
        attempts = int(row.get("attempts", 1))
        grounded = str(row.get("grounded", "")).strip().lower() == "true"
        f1_val = float(row.get("token_f1_score", 0.0) if not pd.isna(row.get("token_f1_score")) else 0.0)
        mrr_val = float(row.get("mrr", 0.0) if not pd.isna(row.get("mrr")) else 0.0)
        relevance = float(row.get("answer_relevance", 1.0) if not pd.isna(row.get("answer_relevance")) else 1.0)
        faithful = float(row.get("faithfulness", 1.0) if not pd.isna(row.get("faithfulness")) else 1.0)

        # Rule 0: Looped to limit but failed grounding guardrail
        if attempts > 2 and not grounded:
            return True, "Baseline Failure (Attempts > 2 and ungrounded)"

        # Rule 1: False Positive Reflection Guardrail (Grounded but completely unfaithful)
        if grounded and faithful < 0.40:
            return True, f"False Positive Guardrail (Grounded=True, but Judge Faithfulness={faithful:.2f})"

        # Rule 2: Generation Choke (Perfect database retrieval but zero output match)
        if mrr_val > 0.0 and f1_val == 0.0:
            return True, f"Generation Choke (Correct Chunk Found, but Token F1={f1_val:.2f})"

        # Rule 3: Semantic Query Drift (Agent looped but lost original query intent)
        if attempts >= 2 and relevance < 0.40:
            return True, f"Semantic Query Drift (Loops={attempts}, but Judge Relevance={relevance:.2f})"

    except Exception:
        return True, "Structural Corruption / Parse Error"

    return False, ""


def process_healing_cycle(lufa_path, eval_path, gt_path, db_path, dash_path, llm_model, sim_mode, api_url):
    print("================================================================================")
    print("STAGE 1: Scanning For System Invalidation Gaps Across Scorecards")
    print("================================================================================")

    if not Path(eval_path).exists():
        print(f"Error: Target scorecard {eval_path} not found. Execution halted.")
        return

    eval_df = pd.read_csv(eval_path)
    gt_df = pd.read_csv(gt_path)
    lufa_df = pd.read_csv(lufa_path) if Path(lufa_path).exists() else pd.DataFrame(columns=LUFA_COLUMNS)

    print(f"Loaded evaluation ledger containing {len(eval_df)} calculated records.")

    invalidated_qids = {}
    for idx, row in eval_df.iterrows():
        qid = str(row.get("question_id", "")).strip()
        is_invalid, reason = check_row_invalidation(row)
        if is_invalid and qid:
            invalidated_qids[qid] = reason

    if not invalidated_qids:
        print(" -> System is fully aligned! No invalidated records or structural errors found.")
        return

    print(f" -> Flagged {len(invalidated_qids)} records for deep pipeline reconstruction.")
    for qid, reason in invalidated_qids.items():
        print(f"    * ID [{qid}] -> Reason: {reason}")

    # Flush out bad records from active frames
    eval_df = eval_df[~eval_df["question_id"].isin(invalidated_qids.keys())]
    lufa_df = lufa_df[~lufa_df["question_id"].isin(invalidated_qids.keys())]

    print("\n" + "=" * 80)
    print("STAGE 2: EXECUTING RE-GENERATION AND METRIC RESCORING LOOP")
    print("=" * 80)

    cfg = load_config()
    cfg_base_model = cfg.get("models", {}).get("llm", {}).get("name", "llama3.2:3b-instruct-q4_K_M")
