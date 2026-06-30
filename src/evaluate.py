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
from run_simulation import query_single_record

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
    "question_id", "question", "answer", "base_model_used", "language", "attempts", "grounded",
    "source1_id", "source1_score", "source1_text",
    "source2_id", "source2_score", "source2_text",
    "source3_id", "source3_score", "source3_text",
    "source4_id", "source4_score", "source4_text",
    "source5_id", "source5_score", "source5_text",
    "original_cosine_score", "recency_adjusted_score", "RRF"
]

EVAL_COLUMNS = [
    "question_id", "question", "answer", "base_model_used", "language", "attempts", "grounded",
    "source1_id", "source1_score", "source1_text",
    "source2_id", "source2_score", "source2_text",
    "source3_id", "source3_score", "source3_text",
    "source4_id", "source4_score", "source4_text",
    "source5_id", "source5_score", "source5_text",
    "original_cosine_score", "recency_adjusted_score", "RRF",
    "id", "rag_base_model", "judge_llm", "category", "difficulty",
    "token_f1_score", "sentence_bleu_score", "rouge1", "rouge3", "rougeL", "meteor",
    "mrr", "ndcg_at_k", "recall_1", "recall_3", "recall_5",
    "answer_relevance", "faithfulness", "context_precision"
]


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
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge", "rougeL"], use_stemmer=True)
    scores = scorer.score(str(reference), str(prediction))
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge3": round(scores["rouge3"].fmeasure, 4),
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


def judge_llm_scores(question, answer, context, judge_llm_model="None"):
    judge_llm_model = judge_llm_model or cfg("models.judge_llm.name")
    """Use local Ollama model as judge. Returns normalized 0–1 scores."""
    from llama_index.llms.ollama import Ollama
    from model_api_auth import get_ollama_client
    llm = get_ollama_client(judge_llm_model, request_timeout=240.0)
    scores = {}
    for metric, prompt_template in JUDGE_LLM_PROMPTS.items():
        try:
            prompt = prompt_template.format(
                question=question[:500],
                answer=answer[:800],
                context=context[:3000],
            )
            raw = str(llm.complete(prompt)).strip()
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


def run_evaluation(
        lufa_csv="tests/lufa_out_data.csv",
        test_csv="tests/combined_test_data_and_ground_truth.csv",
        out_csv="tests/evaluation_results.csv",
        dashboard_out="dashboard/index.html",
        judge_llm_model=None,
        llm_model=None,
        sim_mode="local",
        api_url="http://localhost:8000"
):
    llm_model = llm_model or cfg("models.llm.name")
    #judge_llm_model = judge_llm_model or cfg("models.judge_llm.name")
    
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
            sim_df = pd.DataFrame([sim_row_output], columns=LUFA_COLUMNS)
            sim_df.to_csv(lufa_csv, mode="a", index=False, header=False)
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
            f"      * ROUGE-1: {rouge_scores['rouge1']} | ROUGE-3: {rouge_scores['rouge3']} | ROUGE-L: {rouge_scores['rougeL']}")

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

        print(f"      * Mean Reciprocal Rank (MRR): {mrr_val}")
        print(f"      * NDCG@5 Index: {ndcg_val}")
        print(f"      * Recall Distribution -> Recall@1: {rec1} | Recall@3: {rec3} | Recall@5: {rec5}")

        primary_logged_score = safe_float(active_rag_data.get("source1_score", 0.0))
        orig_cos = safe_float(active_rag_data.get("original_cosine_score", primary_logged_score))
        rec_adj = safe_float(active_rag_data.get("recency_adjusted_score", primary_logged_score))
        rrf_val = safe_float(active_rag_data.get("RRF", primary_logged_score))
        print(
            f"   -> Tracked Hybrid Vectors -> Cosine: {orig_cos} | Recency-Adjusted: {rec_adj} | Fused RRF: {rrf_val}")

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
        rec["rouge3"] = rouge_scores["rouge3"]
        rec["rougeL"] = rouge_scores["rougeL"]
        rec["meteor"] = meteor_val

        rec["mrr"] = mrr_val
        rec["ndcg_at_k"] = ndcg_val
        rec["recall_1"] = rec1
        rec["recall_3"] = rec3
        rec["recall_5"] = rec5

        rec["original_cosine_score"] = orig_cos
        rec["recency_adjusted_score"] = rec_adj
        rec["RRF"] = rrf_val

        rec["answer_relevance"] = 0.0
        rec["faithfulness"] = 0.0
        rec["context_precision"] = 0.0

        if judge_llm_model and prediction and prediction != "" and prediction != "ERROR":
            print(f"   -> Dispatching prompt topologies to Judge Model ({judge_llm_model})...")
            try:
                judge = judge_llm_scores(question, prediction, context, judge_llm_model)
                rec["answer_relevance"] = judge.get("answer_relevance", 0.0)
                rec["faithfulness"] = judge.get("faithfulness", 0.0)
                rec["context_precision"] = judge.get("context_precision", 0.0)
                print(
                    f"      * Judge Feedback -> Relevance: {rec['answer_relevance']} | Faithfulness: {rec['faithfulness']} | Precision: {rec['context_precision']}")
            except Exception as e:
                print(f"      [Judge Model Connection Error] Skipping scoring pass on entry {q_id}: {e}")

        single_row_df = pd.DataFrame([rec], columns=EVAL_COLUMNS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        file_is_new = not out_path.exists() or out_path.stat().st_size == 0
        single_row_df.to_csv(str(out_path), mode="a", index=False, header=file_is_new)
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

    gen_metrics = ["token_f1_score", "sentence_bleu_score", "rouge1", "rouge3", "rougeL", "meteor"]
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
    parser.add_argument("--judge_llm", default=None)
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
        api_url=args.api_url
    )