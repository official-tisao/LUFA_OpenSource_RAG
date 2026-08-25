#!/usr/bin/env python3
"""
Performance evaluation script for LUFA Agentic RAG system with row-by-row
checkpoint saving, crash-resumption support, verbose metric logging,
and live real-time HTML dashboard incremental updates.
Includes inline single-pass retrieval repair hooks to prevent zero metric tracking bugs.
"""

import sys
import json
import argparse
import math
import re
import warnings
from pathlib import Path
from datetime import datetime
from dashboard_generator import generate_dashboard

import pandas as pd

# NLP metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Dynamic simulation hooks
from run_simulation import (query_single_record, TRANSLATION_COLUMNS,
                            SYSTEM_METRIC_COLUMNS, EVAL_ONLY_METRIC_COLUMNS,
                            HUMAN_MANUAL_COLUMNS, CROSSLINGUAL_COLUMNS)
from citation_metrics import (citation_accuracy_regex, citation_accuracy_judge,
                              extract_gold_citation)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import cfg


# Check local cache availability before calling remote download endpoints to avoid WinError 10060
def ensure_nltk_packages():
    pkgs = {
        "punkt": "tokenizers/punkt",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "punkt_tab": "tokenizers/punkt_tab"
    }
    for pkg, path in pkgs.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


ensure_nltk_packages()

# ─────────────────────────────────────────────────────────────────────────────
#  STRICT MATRIX SCHEMA COLUMN DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
LUFA_COLUMNS = [
    "question_id", "question",
    "source1_id", "source1_cosine_score", "source1_recency_adjusted_cosine_score", "source1_rrf_score", "source1_text",
    "source2_id", "source2_cosine_score", "source2_recency_adjusted_cosine_score", "source2_rrf_score", "source2_text",
    "source3_id", "source3_cosine_score", "source3_recency_adjusted_cosine_score", "source3_rrf_score", "source3_text",
    "source4_id", "source4_cosine_score", "source4_recency_adjusted_cosine_score", "source4_rrf_score", "source4_text",
    "source5_id", "source5_cosine_score", "source5_recency_adjusted_cosine_score", "source5_rrf_score", "source5_text",
    "answer", "base_model_used", "language", "attempts", "grounded",
] + TRANSLATION_COLUMNS + SYSTEM_METRIC_COLUMNS + CROSSLINGUAL_COLUMNS

# Question ids whose ground_source_truth carries no article or clause number, so
# citation_accuracy_regex returns "" and the row leaves the citation mean altogether. Collected
# per run and reported at the end; see the note beside the citation print in the metric loop.
_CITATION_UNSCORABLE = []

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
] + TRANSLATION_COLUMNS + SYSTEM_METRIC_COLUMNS + CROSSLINGUAL_COLUMNS \
  + EVAL_ONLY_METRIC_COLUMNS + HUMAN_MANUAL_COLUMNS   # human cols stay LAST


# ─────────────────────────────────────────────────────────────────────────────
#  INLINE REPAIR ENGINE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def calculate_token_overlap(text_a, text_b):
    """Calculate the token intersection ratio between text segments."""
    if not text_a or not text_b or pd.isna(text_a) or pd.isna(text_b):
        return 0.0
    words_a = set(re.findall(r'\b\w+\b', str(text_a).lower()))
    words_b = set(re.findall(r'\b\w+\b', str(text_b).lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a.intersection(words_b)
    return len(intersection) / len(words_a)


def repair_single_row_sources(row_dict, chroma_data=None, db_path=None, collection_name=None):
    db_path = db_path or cfg("database.path")
    collection_name = collection_name or cfg("database.collection_name")
    """
    Examines a row's source text layers and extracts matching database keys
    by executing a local token scan over the persistent storage collection.
    """
    import chromadb
    repaired_ids = []

    if chroma_data is None:
        try:
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_collection(collection_name)
            chroma_data = collection.get(include=["documents"])
        except Exception as err:
            print(f"      [Chroma Connection Error] Live layer read skipped: {err}")
            return []

    db_ids = chroma_data.get("ids", [])
    db_docs = chroma_data.get("documents", [])

    if not db_ids or not db_docs:
        return []

    for i in range(1, 6):
        text_val = row_dict.get(f"source{i}_text", "")
        if pd.isna(text_val) or str(text_val).strip() == "":
            continue

        source_clean = str(text_val).strip().lower()
        matched_id = ""
        max_overlap = -1.0
        exact_found = False

        for cid, doc_text in zip(db_ids, db_docs):
            doc_clean = str(doc_text).lower()
            if source_clean in doc_clean:
                matched_id = cid
                exact_found = True
                break

            overlap = calculate_token_overlap(source_clean, doc_clean)
            if overlap > max_overlap:
                max_overlap = overlap
                matched_id = cid

        if matched_id and (exact_found or max_overlap > 0.5):
            repaired_ids.append(matched_id)
    return repaired_ids


# ─────────────────────────────────────────────────────────────────────────────
#  GENERATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())


def token_f1(prediction, reference):
    pred_tokens = set(tokenize(prediction))
    ref_tokens = set(tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_bleu(prediction, reference):
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens:
        return 0.0
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)


def compute_rouge(prediction, reference):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(str(reference), str(prediction))
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_meteor(prediction, reference):
    try:
        pred_tokens = tokenize(prediction)
        ref_tokens = tokenize(reference)
        if not pred_tokens or not ref_tokens:
            return 0.0
        return meteor_score([ref_tokens], pred_tokens)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  RETRIEVAL METRICS
# ─────────────────────────────────────────────────────────────────────────────
def parse_source_ids(raw):
    if not raw or pd.isna(raw):
        return []
    return [s.strip() for s in str(raw).split("|") if s.strip()]


def resolve_ground_truth_ids(row, chroma_data=None):
    """
    Resolve ground truth IDs for a row, supporting | -separated multi-chunk IDs
    and falling back to text-based resolution when IDs are missing.

    Resolution order:
      1. ground_source_truth_id column (pipe-separated) → split into list
      2. ground_truth_source_ids column (fallback column name) → split into list
      3. ground_source_truth column → match text against ChromaDB chunks
         to resolve chunk IDs (requires chroma_data pre-cache)

    Args:
        row:         DataFrame row dict (or pandas Series)
        chroma_data: Pre-cached ChromaDB data dict with keys:
                     {"ids": [...], "documents": [...], "metadatas": [...]}
                     Only needed for text-based fallback (step 3).

    Returns:
        List of ground truth ID strings.
    """
    # Step 1: Try pipe-separated ground_source_truth_id
    gt_col = "ground_source_truth_id" if "ground_source_truth_id" in row.index else (
             "ground_truth_source_ids" if "ground_truth_source_ids" in row.index else None)
    if gt_col:
        ids = parse_source_ids(row.get(gt_col, ""))
        if ids:
            return ids

    # Step 2: Try ground_truth_source_ids (alternate column name)
    alt_col = "ground_truth_source_ids" if "ground_truth_source_ids" in row.index else None
    if alt_col and alt_col != gt_col:
        ids = parse_source_ids(row.get(alt_col, ""))
        if ids:
            return ids

    # Step 3: Fallback — resolve from ground_source_truth text via ChromaDB matching
    gt_text_col = "ground_source_truth" if "ground_source_truth" in row.index else None
    if gt_text_col and chroma_data:
        gt_text = str(row.get(gt_text_col, "")).strip()
        if gt_text and gt_text != "nan" and len(gt_text) > 20:
            return _resolve_ids_from_text(gt_text, chroma_data)

    return []


def _resolve_ids_from_text(text, chroma_data, min_overlap=0.3):
    """
    Resolve chunk IDs by matching ground_source_truth text against
    pre-cached ChromaDB documents using token overlap.

    Returns all chunk IDs whose overlap score >= min_overlap,
    pipe-joined in database order.
    """
    gt_words = set(re.findall(r'\b\w+\b', text.lower()))
    if not gt_words:
        return []

    db_ids = chroma_data.get("ids", [])
    db_docs = chroma_data.get("documents", [])

    matches = []
    for cid, doc in zip(db_ids, db_docs):
        if not doc:
            continue
        doc_words = set(re.findall(r'\b\w+\b', str(doc).lower()))
        if not doc_words:
            continue
        # Overlap: what fraction of the ground truth text's tokens appear in this chunk
        overlap = len(gt_words & doc_words) / len(gt_words)
        if overlap >= min_overlap:
            matches.append((cid, overlap))

    if not matches:
        # Lower threshold and retry with top-1 if nothing matched
        for cid, doc in zip(db_ids, db_docs):
            if not doc:
                continue
            doc_words = set(re.findall(r'\b\w+\b', str(doc).lower()))
            if not doc_words:
                continue
            overlap = len(gt_words & doc_words) / len(gt_words)
            if overlap > 0:
                matches.append((cid, overlap))
        if matches:
            # Just return the best match
            matches.sort(key=lambda x: x[1], reverse=True)
            return [matches[0][0]]
        return []

    # Return IDs sorted by overlap score (descending)
    matches.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in matches]


def recall_at_k(retrieved, ground_truth, k):
    if not ground_truth:
        return 0.0
    relevant_at_k = set(retrieved[:k]) & set(ground_truth)
    return len(relevant_at_k) / len(ground_truth)


def precision_at_k(retrieved, ground_truth, k):
    """Precision@k (Ch4 §4.6.1): fraction of the top-k retrieved chunks that are
    relevant. P@3 is the primary retrieval metric. Denominator is the number of
    chunks actually present in the top-k (== k whenever ≥k were retrieved)."""
    topk = retrieved[:k]
    if not topk or not ground_truth:
        return 0.0
    relevant = set(topk) & set(ground_truth)
    return len(relevant) / len(topk)


def mrr(retrieved, ground_truth):
    gt_set = set(ground_truth)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gt_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved, ground_truth, k=5):
    gt_set = set(ground_truth)
    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in gt_set
    )
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  LLM-AS-JUDGE METRICS
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_LLM_PROMPTS = {
    "answer_relevance": """Score how relevant the answer is to the question.
Score 1-5: 1=completely irrelevant, 3=partially relevant, 5=fully relevant.
Reply with ONLY the integer score.

Question: {question}
Answer: {answer}
Score:""",

    "faithfulness": """Score how faithful the answer is to the provided context.
Score 1-5: 1=many unsupported claims, 3=mostly supported, 5=all claims verifiable in context.
Reply with ONLY the integer score.

Context: {context}
Answer: {answer}
Score:""",

    "context_precision": """Score how precisely the context chunks are relevant to the question.
Score 1-5: 1=context is mostly irrelevant, 3=some chunks relevant, 5=all chunks highly relevant.
Reply with ONLY the integer score.

Question: {question}
Context: {context}
Score:""",
}


def extract_score(text):
    match = re.search(r'\b([1-5])\b', str(text).strip())
    return float(match.group(1)) / 5.0 if match else 0.5


def get_judge_client(judge_llm_model=None, request_timeout=240.0):
    """Resolve and load the judge LLM client. The model name defaults to
    cfg('models.judge_llm.name') (prometheus2:8x7b) when not supplied, and routing
    (Ollama vs OpenAI-compatible) is auto-selected from the resolved api_base via
    get_llm_client — so a --judge_llm override pointed at a cloud/OpenRouter model
    works here too (fixes the previous Ollama-only hardcoding)."""
    judge_llm_model = judge_llm_model or cfg("models.judge_llm.name")
    from model_api_auth import get_llm_client
    return get_llm_client(judge_llm_model, request_timeout=request_timeout)


def judge_llm_scores(question, answer, context, judge_llm_model="None", judge_client=None):
    """Score the three RAGAS metrics with the judge LLM. Returns normalized 0–1
    scores. `judge_client` may be a pre-loaded client (reused across calls)."""
    llm = judge_client or get_judge_client(judge_llm_model)
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
            print(f"      [Judge Warning] {metric} calculation failure: {e}")
            scores[metric] = 0.0
    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EVALUATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def build_context_from_row(row):
    parts = []
    for i in range(1, 6):
        text = row.get(f"source{i}_text", "")
        if text and not pd.isna(text):
            parts.append(str(text))
    return "\n\n---\n\n".join(parts)


def build_retrieved_ids(row):
    ids = []
    for i in range(1, 6):
        sid = row.get(f"source{i}_id", "")
        if sid and not pd.isna(sid):
            ids.append(str(sid))
    return ids


def safe_float(val, default_val=0.0):
    if pd.isna(val) or val == "":
        return default_val
    try:
        return float(val)
    except Exception:
        return default_val
def write_metrics_row_to_csv(row_dict, output_path, mode="a", header=False):
    """
    Write a metrics row to CSV with immediate persistence.
    Used for granular storage of individual metric groups.

    Args:
        row_dict: Dictionary containing metrics to write
        output_path: Path to output CSV file
        mode: Write mode ("a" for append, "w" for write)
        header: Whether to write header

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from csv_utils import write_csv_row
        # Extract required columns for metrics CSV
        metrics_columns = list(EVAL_COLUMNS)

        # Filter row_dict to only include metrics columns
        filtered_row = {k: v for k, v in row_dict.items() if k in metrics_columns}

        return write_csv_row(filtered_row, output_path, mode=mode, header=header)
    except Exception as e:
        print(f"      [Metrics Write Error] Failed to write row to {output_path}: {e}")
        return False


def run_evaluation(
        lufa_csv="tests/lufa_out_data.csv",
        test_csv="tests/combined_test_data_and_ground_truth.csv",
        out_csv="tests/evaluation_results.csv",
        dashboard_out="dashboard/index.html",
        judge_llm_model=None,
        llm_model=None,
        sim_mode="local",
        api_url="http://localhost:8000",
        no_judge=False,
):
    _CITATION_UNSCORABLE.clear()
    llm_model = llm_model or cfg("models.llm.name")
    # Judge defaults to prometheus2:8x7b (cfg('models.judge_llm.name')) unless a
    # --judge_llm override is passed; --no_judge disables judge scoring entirely.
    if no_judge:
        judge_llm_model = None
    else:
        judge_llm_model = judge_llm_model or cfg("models.judge_llm.name")

    print(f"[Eval] Verifying ground truth data file from: {test_csv}")
    if not Path(test_csv).exists():
        raise FileNotFoundError(
            f"Critical System Error: Missing designated test data tracking registry file at: {test_csv}")
    test_df = pd.read_csv(test_csv)

    print(f"[Eval] Verifying RAG engine output log from: {lufa_csv}")

    if Path(lufa_csv).exists():
        try:
            out_df = pd.read_csv(lufa_csv, on_bad_lines="skip")
        except Exception as e:
            print(
                f"[Warning] Severe file corruption detected in {lufa_csv}. Resetting file layout schema framework: {e}")
            out_df = pd.DataFrame(columns=LUFA_COLUMNS)
            Path(lufa_csv).parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(lufa_csv, index=False)
    else:
        print(f"[Warning] Engine output file '{lufa_csv}' not detected. Constructing a clean file baseline.")
        out_df = pd.DataFrame(columns=LUFA_COLUMNS)
        Path(lufa_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(lufa_csv, index=False)

    lufa_records = {}
    for _, lufa_row in out_df.iterrows():
        s_qid = str(lufa_row.get("question_id", "")).strip()
        if s_qid and s_qid != "nan":
            lufa_records[s_qid] = lufa_row.to_dict()

    completed_ids = set()
    existing_eval_rows = {}   # q_id -> prior eval row (used to preserve manual human columns)
    out_path = Path(out_csv)

    chroma_cached_data = None
    try:
        import chromadb
        client = chromadb.PersistentClient(path=cfg("database.path"))
        collection = client.get_collection(cfg("database.collection_name"))
        chroma_cached_data = collection.get(include=["documents"])
    except Exception as e:
        print(f"[Warning] Could not pre-cache database tables: {e}")

    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_csv, on_bad_lines="skip")
            if "question_id" in existing_df.columns:
                mrr_s = pd.to_numeric(existing_df.get("mrr", 0), errors='coerce').fillna(0.0)
                ndcg_s = pd.to_numeric(existing_df.get("ndcg_at_k", 0), errors='coerce').fillna(0.0)

                valid_completions = existing_df[
                    existing_df["question_id"].notna() &
                    ((mrr_s > 0.0) | (ndcg_s > 0.0))
                    ]
                completed_ids = set(valid_completions["question_id"].dropna().astype(str).str.strip().tolist())

                # Snapshot prior rows so any hand-entered human/IAA columns survive
                # a re-run (align_and_append + dedup-keep-last would otherwise drop
                # them when an incomplete row is re-appended).
                for _, erow in existing_df.iterrows():
                    eid = str(erow.get("question_id", "")).strip()
                    if eid and eid != "nan":
                        existing_eval_rows[eid] = erow.to_dict()

                total_rows = len(existing_df["question_id"].dropna().unique())
                zero_count = total_rows - len(completed_ids)
                print(f"[Resumption] Detected existing output scorecard. Found {len(completed_ids)} completed entries.")
                if zero_count > 0:
                    print(
                        f"[Resumption] Flagged {zero_count} rows containing zero metrics. Bypassing cache to execute inline repair pass.")
        except Exception as err:
            print(f"[Warning] Error parsing current results file structure, resetting fresh compilation: {err}")
    else:
        print(
            f"[Eval] Output tracking database path '{out_csv}' not present. Will be generated dynamically row-by-row.")

    cfg_base_model = cfg("models.llm.name")

    print(f"[Eval] Starting verification loop on {len(test_df)} ground truth questions...")
    print("\n" + "=" * 80)
    print("BEGINNING PIPELINE PROCESSING LOOP (WITH INLINE SIMULATION & REPAIR)")
    print("=" * 80)

    for idx, row in test_df.iterrows():
        current_counter = idx + 1
        q_id = str(row.get("id")).strip()

        if q_id in completed_ids:
            print(
                f"[{current_counter}/{len(test_df)}] Skipping Question ID {q_id} (Already verified in checkpoint file)")
            continue

        print(f"\n[{current_counter}/{len(test_df)}] Active Evaluation Context -> Question ID: {q_id}")

        if q_id not in lufa_records:
            print(f"   ⚠️  Notice: Question ID '{q_id}' not found in RAG output log records.")
            print(f"   -> Executing real-time dynamic backend simulation pass...")

            record_dict = row.to_dict()
            sim_row_output = query_single_record(record_dict, sim_mode, cfg_base_model, llm_model, api_url,
                                                 current_counter)

            lufa_records[q_id] = sim_row_output
            from csv_utils import align_and_append
            align_and_append(sim_row_output, lufa_csv, LUFA_COLUMNS)
            print(f"   -> Content generated successfully and appended permanently to {lufa_csv}")

        active_rag_data = lufa_records[q_id]

        prediction = "" if pd.isna(active_rag_data.get("answer")) else str(active_rag_data.get("answer"))
        reference = "" if pd.isna(row.get("expected_answer")) else str(row.get("expected_answer"))
        retrieved_ids = build_retrieved_ids(active_rag_data)

        gt_col = "ground_source_truth_id" if "ground_source_truth_id" in test_df.columns else "ground_truth_source_ids"
        ground_truth_ids = resolve_ground_truth_ids(row, chroma_data=chroma_cached_data)

        context = build_context_from_row(active_rag_data)
        question = "" if pd.isna(row.get("question")) else str(row.get("question"))
        language_val = "" if pd.isna(row.get("language")) else str(row.get("language"))

        print("   -> Calculating Generation metrics...")
        f1_val = round(token_f1(prediction, reference), 4)
        print(f"      * Token F1 Score: {f1_val}")

        bleu_val = round(compute_bleu(prediction, reference), 4)
        print(f"      * Sentence BLEU Score: {bleu_val}")

        rouge_scores = compute_rouge(prediction, reference)
        print(
            f"      * ROUGE-1: {rouge_scores['rouge1']} | ROUGE-2: {rouge_scores['rouge2']} | ROUGE-L: {rouge_scores['rougeL']}")

        meteor_val = round(compute_meteor(prediction, reference), 4)
        print(f"      * METEOR Score: {meteor_val}")

        print("   -> Calculating Retrieval position metrics...")
        mrr_val = round(mrr(retrieved_ids, ground_truth_ids), 4)
        ndcg_val = ndcg_at_k(retrieved_ids, ground_truth_ids, k=5)
        rec1 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=1), 4)
        rec3 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=3), 4)
        rec5 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=5), 4)

        if mrr_val == 0.0 and ndcg_val == 0.0:
            print("   ⚠️  Retrieval metrics returned 0.0. Activating embedded local token repair pass...")
            try:
                fixed_ids = repair_single_row_sources(active_rag_data, chroma_cached_data,
                                                      cfg("database.path"),
                                                      cfg("database.collection_name"))

                if fixed_ids:
                    retrieved_ids = fixed_ids
                    mrr_val = round(mrr(retrieved_ids, ground_truth_ids), 4)
                    ndcg_val = ndcg_at_k(retrieved_ids, ground_truth_ids, k=5)
                    rec1 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=1), 4)
                    rec3 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=3), 4)
                    rec5 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=5), 4)
                    print(f"      * Repaired Retrieval Scores -> MRR: {mrr_val} | NDCG@5: {ndcg_val}")

                    for i, cid in enumerate(retrieved_ids, start=1):
                        active_rag_data[f"source{i}_id"] = cid

                    try:
                        current_lufa_df = pd.read_csv(lufa_csv, on_bad_lines="skip")
                        idx_matches = current_lufa_df[current_lufa_df["question_id"] == q_id].index
                        if len(idx_matches) > 0:
                            for i, cid in enumerate(retrieved_ids, start=1):
                                current_lufa_df.loc[idx_matches, f"source{i}_id"] = cid
                            current_lufa_df.to_csv(lufa_csv, index=False)
                    except Exception as lufa_save_err:
                        print(f"      [Warning] Could not sync repaired source IDs back to base log: {lufa_save_err}")
            except Exception as repair_err:
                print(f"      [Live Repair Error] Single row recovery pass failed: {repair_err}")

        # Precision@k (Ch4 §4.6.1; P@3 primary) — uses the final (possibly repaired) IDs.
        prec1 = round(precision_at_k(retrieved_ids, ground_truth_ids, k=1), 4)
        prec3 = round(precision_at_k(retrieved_ids, ground_truth_ids, k=3), 4)
        prec5 = round(precision_at_k(retrieved_ids, ground_truth_ids, k=5), 4)

        # Citation accuracy — deterministic regex now; judge score below if enabled.
        gold_text = "" if pd.isna(row.get("ground_source_truth")) else str(row.get("ground_source_truth"))
        cit_regex = citation_accuracy_regex(prediction, gold_text)

        print(f"      * Mean Reciprocal Rank (MRR): {mrr_val}")
        print(f"      * NDCG@5 Index: {ndcg_val}")
        print(f"      * Recall Distribution -> Recall@1: {rec1} | Recall@3: {rec3} | Recall@5: {rec5}")
        print(f"      * Precision Distribution -> P@1: {prec1} | P@3: {prec3} | P@5: {prec5}")
        # An unscorable row is NOT a zero: it drops out of the citation mean entirely, so a
        # corpus whose gold text has lost its article numbers silently shrinks the denominator
        # rather than lowering the score. Count and report it instead of letting it pass.
        if cit_regex == "":
            _CITATION_UNSCORABLE.append(str(q_id))
            print(f"      * Citation Accuracy (regex): NOT SCORABLE — no article or clause "
                  f"number in ground_source_truth, so this row is excluded from the citation "
                  f"mean rather than counted as 0")
        else:
            print(f"      * Citation Accuracy (regex): {cit_regex}")

        for si in range(1, 6):
            cos_v = safe_float(active_rag_data.get(f"source{si}_cosine_score", 0.0))
            rec_v = safe_float(active_rag_data.get(f"source{si}_recency_adjusted_cosine_score", 0.0))
            rrf_v = safe_float(active_rag_data.get(f"source{si}_rrf_score", 0.0))
            if cos_v > 0 or rrf_v > 0:
                print(f"   -> Source{si} Scores -> Cosine: {cos_v} | Recency-Adj: {rec_v} | RRF: {rrf_v}")

        rec = {}
        for col in LUFA_COLUMNS:
            val = active_rag_data.get(col)
            rec[col] = "" if pd.isna(val) else val

        rec["id"] = q_id
        rec["question_id"] = q_id
        rec["question"] = question
        rec["language"] = language_val
        rec["rag_base_model"] = "" if pd.isna(active_rag_data.get("base_model_used")) else str(
            active_rag_data.get("base_model_used"))
        rec["judge_llm"] = judge_llm_model
        rec["category"] = "" if pd.isna(row.get("category")) else str(row.get("category"))
        rec["difficulty"] = "" if pd.isna(row.get("difficulty")) else str(row.get("difficulty"))

        rec["token_f1_score"] = f1_val
        rec["sentence_bleu_score"] = bleu_val
        rec["rouge1"] = rouge_scores["rouge1"]
        rec["rouge2"] = rouge_scores["rouge2"]
        rec["rougeL"] = rouge_scores["rougeL"]
        rec["meteor"] = meteor_val

        rec["mrr"] = mrr_val
        rec["ndcg_at_k"] = ndcg_val
        rec["recall_1"] = rec1
        rec["recall_3"] = rec3
        rec["recall_5"] = rec5

        # Precision@k + citation (deterministic regex); judge citation filled below.
        rec["precision_1"] = prec1
        rec["precision_3"] = prec3
        rec["precision_5"] = prec5
        rec["citation_accuracy_regex"] = cit_regex
        rec["citation_accuracy_judge"] = ""

        # Human/IAA manual columns: carry forward any hand-entered values from a
        # prior run for this question; otherwise leave blank for the researcher.
        _prev_eval = existing_eval_rows.get(q_id, {})
        for _hc in HUMAN_MANUAL_COLUMNS:
            _pv = _prev_eval.get(_hc, "")
            rec[_hc] = "" if (pd.isna(_pv) if not isinstance(_pv, (list, dict)) else False) else _pv

        rec["answer_relevance"] = 0.0
        rec["faithfulness"] = 0.0
        rec["context_precision"] = 0.0

        if judge_llm_model and prediction and prediction != "" and prediction != "ERROR":
            print(f"   -> Dispatching prompt topologies to Judge Model ({judge_llm_model})...")
            try:
                judge_client = get_judge_client(judge_llm_model)
                judge = judge_llm_scores(question, prediction, context, judge_llm_model,
                                         judge_client=judge_client)
                rec["answer_relevance"] = judge.get("answer_relevance", 0.0)
                rec["faithfulness"] = judge.get("faithfulness", 0.0)
                rec["context_precision"] = judge.get("context_precision", 0.0)
                print(
                    f"      * Judge Feedback -> Relevance: {rec['answer_relevance']} | Faithfulness: {rec['faithfulness']} | Precision: {rec['context_precision']}")
                # Citation accuracy via the SAME judge model (own dedicated prompt).
                cit_judge = citation_accuracy_judge(judge_client, prediction, gold_text)
                rec["citation_accuracy_judge"] = cit_judge
                print(f"      * Citation Accuracy (judge): {cit_judge}")
            except Exception as e:
                print(f"      [Judge Model Connection Error] Skipping scoring pass on entry {q_id}: {e}")

        from csv_utils import align_and_append
        align_and_append(rec, out_path, EVAL_COLUMNS)
        print("   ✅ Row recorded safely to checkpoint score file.")

        print("   -> Re-compiling performance HTML dashboard layer with current progress data...")
        try:
            current_progress_df = pd.read_csv(str(out_path))
            current_progress_df = current_progress_df.drop_duplicates(subset=["question_id", "rag_base_model"], keep="last")
            Path(dashboard_out).parent.mkdir(parents=True, exist_ok=True)
            generate_dashboard(current_progress_df, dashboard_out)
            print(
                f"      * Live view updated at {dashboard_out} (Current processed set size: {len(current_progress_df)})")
        except Exception as d_err:
            print(f"      [Dashboard Warning] Skipping live UI compilation step: {d_err}")

    if out_path.exists() and out_path.stat().st_size > 0:
        final_results_df = pd.read_csv(str(out_path))
        final_results_df = final_results_df.drop_duplicates(subset=["question_id", "rag_base_model"], keep="last")
    else:
        final_results_df = pd.DataFrame()

    print("\n" + "=" * 80)
    print("PROCESSING CYCLE COMPLETED")
    print("=" * 80)
    print(f"\n[Export] Full results available at: {out_csv}")
    print(f"[Export] Interactive dashboard finalized completely at: {dashboard_out}")

    if len(final_results_df) > 0:
        print("\n" + "─" * 60)
        print("FINAL EVALUATION METRIC OVERVIEW SUMMARY")
        print("─" * 60)
        summary_cols = ["token_f1_score", "sentence_bleu_score", "rougeL", "meteor", "mrr", "ndcg_at_k",
                        "recall_5", "answer_relevance", "faithfulness", "context_precision"]
        for col in summary_cols:
            if col in final_results_df.columns:
                num_series = pd.to_numeric(final_results_df[col], errors='coerce')
                print(f"  {col:<22}: {num_series.mean():.4f}")

        if "citation_accuracy_regex" in final_results_df.columns:
            cit = pd.to_numeric(final_results_df["citation_accuracy_regex"], errors='coerce')
            print(f"  {'citation_accuracy_regex':<22}: {cit.mean():.4f}"
                  f"   (over {cit.notna().sum()} of {len(final_results_df)} rows)")

    # Unscorable rows are excluded from the citation mean, not counted as zero, so the mean
    # above can look healthy while resting on a shrunken denominator. Say so explicitly.
    if _CITATION_UNSCORABLE:
        n = len(_CITATION_UNSCORABLE)
        print("\n" + "!" * 60)
        print(f"CITATION ACCURACY: {n} row(s) could not be scored at all, because their "
              f"ground_source_truth contains no article or clause number for "
              f"extract_gold_citation to match against.")
        print("These rows are EXCLUDED from the citation mean rather than counted as 0, so the "
              "mean is computed over a smaller sample than the row count suggests.")
        print("First affected question ids: " + ", ".join(_CITATION_UNSCORABLE[:10])
              + (" ..." if n > 10 else ""))
        print("!" * 60)

    return final_results_df


# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD HTML GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def df_to_js_data(df):
    """Prepare aggregated data structures for Chart.js dashboard views."""
    cleaned_df = df.copy().drop_duplicates(subset=["question_id", "rag_base_model"], keep="last")
    models = cleaned_df["rag_base_model"].dropna().unique().tolist()

    def avg_by(group_col, metric):
        numeric_series = pd.to_numeric(cleaned_df[metric], errors='coerce')
        temp_df = cleaned_df.copy()
        temp_df[metric] = numeric_series
        return {
            str(k): round(float(v), 4)
            for k, v in temp_df.groupby(group_col)[metric].mean().items()
        }

    gen_metrics = ["token_f1_score", "sentence_bleu_score", "rouge1", "rouge2", "rougeL", "meteor"]
    ret_metrics = ["mrr", "ndcg_at_k", "recall_1", "recall_3", "recall_5"]
    judge_metrics = ["answer_relevance", "faithfulness", "context_precision"]

    for m in gen_metrics + ret_metrics + judge_metrics:
        if m in cleaned_df.columns:
            cleaned_df[m] = pd.to_numeric(cleaned_df[m], errors='coerce').fillna(0.0)

    overall_metrics = {}
    for m in gen_metrics + ret_metrics + judge_metrics:
        if m in cleaned_df.columns:
            overall_metrics[m] = round(float(cleaned_df[m].mean()), 4)

    data = {
        "models": models,
        "overall": overall_metrics,
        "by_model": {
            metric: avg_by("rag_base_model", metric)
            for metric in gen_metrics + ret_metrics + judge_metrics
            if metric in cleaned_df.columns
        },
        "by_language": {m: avg_by("language", m) for m in gen_metrics + judge_metrics if m in cleaned_df.columns},
        "by_difficulty": {m: avg_by("difficulty", m) for m in gen_metrics + ret_metrics if m in cleaned_df.columns},
        "by_category": {m: avg_by("category", m) for m in gen_metrics + judge_metrics if m in cleaned_df.columns},
        "grounded_rate": round(
            float(cleaned_df["grounded"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).mean()),
            4) if "grounded" in cleaned_df.columns else 0,
        "avg_attempts": round(float(pd.to_numeric(cleaned_df["attempts"], errors='coerce').fillna(1).mean()),
                              2) if "attempts" in cleaned_df.columns else 1,
        "total_questions": len(cleaned_df),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "rows": cleaned_df[["question_id", "question", "rag_base_model", "language",
                            "category", "difficulty", "token_f1_score", "sentence_bleu_score", "rougeL", "meteor",
                            "mrr", "ndcg_at_k", "recall_5", "answer_relevance",
                            "faithfulness", "context_precision", "grounded", "attempts"]
        ].fillna("").to_dict(orient="records"),
    }
    return data

#
# def generate_dashboard(df, output_path):
#     data = df_to_js_data(df)
#     data_json = json.dumps(data, ensure_ascii=False, default=str)
#     html = DASHBOARD_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LUFA RAG system performance")
    parser.add_argument("--lufa_csv", default="tests/lufa_out_data.csv")
    parser.add_argument("--test_csv", default="tests/combined_test_data_and_ground_truth.csv")
    parser.add_argument("--out_csv", default="tests/evaluation_results.csv")
    parser.add_argument("--dashboard", default="dashboard/index.html")
    parser.add_argument("--judge_llm", default=None,
                        help="Judge model override. Defaults to models.judge_llm.name "
                             "(prometheus2:8x7b) when omitted.")
    parser.add_argument("--no_judge", action="store_true",
                        help="Disable LLM-judge scoring (RAGAS + citation judge) entirely.")
    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--sim_mode", choices=["local", "local-naive", "api", "frontier"], default="local")
    parser.add_argument("--api_url", default="http://localhost:8000")
    args = parser.parse_args()

    run_evaluation(
        lufa_csv=args.lufa_csv,
        test_csv=args.test_csv,
        out_csv=args.out_csv,
        dashboard_out=args.dashboard,
        judge_llm_model=args.judge_llm,
        llm_model=args.llm_model,
        sim_mode=args.sim_mode,
        api_url=args.api_url,
        no_judge=args.no_judge,
    )