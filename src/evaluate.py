#!/usr/bin/env python3
"""
Performance evaluation script for LUFA Agentic RAG system with row-by-row
checkpoint saving, crash-resumption support, verbose metric logging,
and live real-time HTML dashboard incremental updates.
Preserves all original columns from the agentic RAG system output,
explicitly including true question text and language metadata with strict schema alignment.
"""

import sys
import json
import argparse
import math
import re
import warnings
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# NLP metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Dynamic simulation hooks
from run_simulation import query_single_record, load_config

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

# Download NLTK data silently
for pkg in ["punkt", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

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
    "token_f1_score", "sentence_bleu_score", "rouge1", "rouge2", "rougeL", "meteor",
    "mrr", "ndcg_at_k", "recall_1", "recall_3", "recall_5",
    "answer_relevance", "faithfulness", "context_precision"
]


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
LLM_JUDGE_PROMPTS = {
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


def llm_judge_scores(question, answer, context, llm_model="llama3.2:3b-instruct-q4_K_M"):
    """Use local Ollama model as judge. Returns normalized 0–1 scores."""
    from llama_index.llms.ollama import Ollama
    llm = Ollama(model=llm_model, base_url="http://localhost:11434", request_timeout=60.0)
    scores = {}
    for metric, prompt_template in LLM_JUDGE_PROMPTS.items():
        try:
            prompt = prompt_template.format(
                question=question[:500],
                answer=answer[:800],
                context=context[:1000],
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
        use_llm_judge=True,
        llm_model="llama3.2:3b-instruct-q4_K_M",
        sim_mode="local",
        api_url="http://localhost:8000"
):
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

    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_csv, on_bad_lines="skip")
            if "question_id" in existing_df.columns:
                completed_ids = set(existing_df["question_id"].dropna().astype(str).str.strip().tolist())
                print(
                    f"[Resumption] Detected existing output scorecard. Found {len(completed_ids)} pre-calculated records.")
        except Exception as err:
            print(f"[Warning] Error parsing current results file structure, resetting fresh compilation: {err}")
    else:
        print(
            f"[Eval] Output tracking database path '{out_csv}' not present. Will be generated dynamically row-by-row.")

    cfg = load_config()
    cfg_base_model = cfg.get("models", {}).get("llm", {}).get("name", "llama3.2:3b-instruct-q4_K_M")

    print(f"[Eval] Starting verification loop on {len(test_df)} ground truth questions...")
    print("\n" + "=" * 80)
    print("BEGINNING PIPELINE PROCESSING LOOP (WITH INLINE SIMULATION)")
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
        ground_truth_ids = parse_source_ids(row.get(gt_col, ""))

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
        print(f"      * Mean Reciprocal Rank (MRR): {mrr_val}")

        ndcg_val = ndcg_at_k(retrieved_ids, ground_truth_ids, k=5)
        print(f"      * NDCG@5 Index: {ndcg_val}")

        rec1 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=1), 4)
        rec3 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=3), 4)
        rec5 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=5), 4)
        print(f"      * Recall Distribution -> Recall@1: {rec1} | Recall@3: {rec3} | Recall@5: {rec5}")

        primary_logged_score = safe_float(active_rag_data.get("source1_score", 0.0))
        orig_cos = safe_float(active_rag_data.get("original_cosine_score", primary_logged_score))
        rec_adj = safe_float(active_rag_data.get("recency_adjusted_score", primary_logged_score))
        rrf_val = safe_float(active_rag_data.get("RRF", primary_logged_score))
        print(
            f"   -> Tracked Hybrid Vectors -> Cosine: {orig_cos} | Recency-Adjusted: {rec_adj} | Fused RRF: {rrf_val}")

        # Seed record dictionary using authoritative tracking variables
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
        rec["judge_llm"] = llm_model
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

        rec["original_cosine_score"] = orig_cos
        rec["recency_adjusted_score"] = rec_adj
        rec["RRF"] = rrf_val

        rec["answer_relevance"] = 0.0
        rec["faithfulness"] = 0.0
        rec["context_precision"] = 0.0

        if use_llm_judge and prediction and prediction != "" and prediction != "ERROR":
            print(f"   -> Dispatching prompt topologies to Judge Model ({llm_model})...")
            try:
                judge = llm_judge_scores(question, prediction, context, llm_model)
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
            Path(dashboard_out).parent.mkdir(parents=True, exist_ok=True)
            generate_dashboard(current_progress_df, dashboard_out)
            print(
                f"      * Live view updated at {dashboard_out} (Current processed set size: {len(current_progress_df)})")
        except Exception as d_err:
            print(f"      [Dashboard Warning] Skipping live UI compilation step: {d_err}")

    if out_path.exists() and out_path.stat().st_size > 0:
        final_results_df = pd.read_csv(str(out_path))
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
    models = df["rag_base_model"].dropna().unique().tolist()

    def avg_by(group_col, metric):
        numeric_series = pd.to_numeric(df[metric], errors='coerce')
        temp_df = df.copy()
        temp_df[metric] = numeric_series
        return {
            str(k): round(float(v), 4)
            for k, v in temp_df.groupby(group_col)[metric].mean().items()
        }

    gen_metrics = ["token_f1_score", "sentence_bleu_score", "rouge1", "rouge2", "rougeL", "meteor"]
    ret_metrics = ["mrr", "ndcg_at_k", "recall_1", "recall_3", "recall_5"]
    judge_metrics = ["answer_relevance", "faithfulness", "context_precision"]

    cleaned_df = df.copy()
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
            if metric in df.columns
        },
        "by_language": {m: avg_by("language", m) for m in gen_metrics + judge_metrics if m in df.columns},
        "by_difficulty": {m: avg_by("difficulty", m) for m in gen_metrics + ret_metrics if m in df.columns},
        "by_category": {m: avg_by("category", m) for m in gen_metrics + judge_metrics if m in df.columns},
        "grounded_rate": round(float(pd.to_numeric(df["grounded"], errors='coerce').fillna(0).astype(bool).mean()),
                               4) if "grounded" in df.columns else 0,
        "avg_attempts": round(float(pd.to_numeric(df["attempts"], errors='coerce').fillna(1).mean()),
                              2) if "attempts" in df.columns else 1,
        "total_questions": len(df),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "rows": cleaned_df[["question_id", "question", "rag_base_model", "language",
                            "category", "difficulty", "token_f1_score", "sentence_bleu_score", "rougeL", "meteor",
                            "mrr", "ndcg_at_k", "recall_5", "answer_relevance",
                            "faithfulness", "context_precision", "grounded", "attempts"]
        ].fillna("").to_dict(orient="records"),
    }
    return data


def generate_dashboard(df, output_path):
    data = df_to_js_data(df)
    data_json = json.dumps(data, ensure_ascii=False, default=str)
    html = DASHBOARD_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>LUFA RAG Evaluation Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body { background:#0f172a; color:#e2e8f0; font-family:'Segoe UI',sans-serif; }
  .card { background:#1e293b; border-radius:12px; padding:20px; border:1px solid #334155; }
  .metric-card { background:linear-gradient(135deg,#1e3a5f,#1e293b); border-radius:10px; padding:16px; border:1px solid #2563eb44; }
  .section-title { font-size:1.1rem; font-weight:700; color:#93c5fd; margin-bottom:12px; text-transform:uppercase; letter-spacing:.05em; }
  .badge { display:inline-block; padding:2px 8px; border-radius:9999px; font-size:.7rem; font-weight:600; }
  .badge-en { background:#1d4ed8; color:#bfdbfe; }
  .badge-fr { background:#7c3aed; color:#ddd6fe; }
  .badge-other { background:#374151; color:#d1d5db; }
  table { width:100%; border-collapse:collapse; font-size:.78rem; }
  th { background:#0f172a; color:#94a3b8; padding:8px 10px; text-align:left; position:sticky; top:0; }
  td { padding:7px 10px; border-bottom:1px solid #1e293b; }
  tr:hover td { background:#1e3a5f22; }
  .score-high { color:#4ade80; }
  .score-mid  { color:#facc15; }
  .score-low  { color:#f87171; }
  canvas { max-height:280px; }
  ::-webkit-scrollbar { width:6px; height:6px; }
  ::-webkit-scrollbar-track { background:#0f172a; }
  ::-webkit-scrollbar-thumb { background:#334155; border-radius:3px; }
</style>
</head>
<body class="p-6">

<script>const D = __DATA_PLACEHOLDER__;</script>

<div class="mb-8">
  <h1 class="text-3xl font-bold text-white mb-1">🎓 LUFA RAG Evaluation Dashboard</h1>
  <p class="text-slate-400 text-sm">Agentic RAG for Cross-Lingual Retrieval of University Collective Agreements &nbsp;·&nbsp;
     <span id="gen-at" class="text-slate-500"></span></p>
</div>

<div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4 mb-8" id="kpi-row"></div>

<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
  <div class="card">
    <div class="section-title">Generation Metrics by Model</div>
    <canvas id="genChart"></canvas>
  </div>
  <div class="card">
    <div class="section-title">Retrieval Metrics by Model</div>
    <canvas id="retChart"></canvas>
  </div>
</div>

<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
  <div class="card">
    <div class="section-title">LLM-as-Judge Metrics (Radar)</div>
    <canvas id="radarChart"></canvas>
  </div>
  <div class="card">
    <div class="section-title">Performance by Language</div>
    <canvas id="langChart"></canvas>
  </div>
</div>

<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
  <div class="card">
    <div class="section-title">F1 Score by Difficulty</div>
    <canvas id="diffChart"></canvas>
  </div>
  <div class="card">
    <div class="section-title">ROUGE-L by Category</div>
    <canvas id="catChart"></canvas>
  </div>
</div>

<div class="card mb-8">
  <div class="section-title">Detailed Results</div>
  <div style="max-height:400px;overflow:auto;">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Question</th><th>Model</th><th>Lang</th>
          <th>F1</th><th>BLEU</th><th>ROUGE-L</th><th>METEOR</th>
          <th>MRR</th><th>Recall@5</th>
          <th>Relevance</th><th>Faithful</th><th>Precision</th>
          <th>Grounded</th><th>Attempts</th>
        </tr>
      </thead>
      <tbody id="results-tbody"></tbody>
    </table>
  </div>
</div>

<p class="text-center text-slate-600 text-xs pb-4">
  LUFA Agentic RAG Thesis &nbsp;·&nbsp; Laurentian University &nbsp;·&nbsp;
  Computational Sciences &nbsp;·&nbsp; Generated <span id="footer-date"></span>
</p>

<script>
const COLORS = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4'];
const LIGHT  = ['#93c5fd','#6ee7b7','#fcd34d','#fca5a5','#c4b5fd','#f9a8d4','#67e8f9'];

function scoreClass(v){ return v>=0.7?'score-high':v>=0.4?'score-mid':'score-low'; }
function fmt(v){ return (typeof v==='number')?v.toFixed(3):v||''; }

document.getElementById('gen-at').textContent = D.generated_at;
document.getElementById('footer-date').textContent = D.generated_at;

// ── KPI Cards ─────────────────────────────────────────────────────────────────
const kpis = [
  {label:'Questions', value: D.total_questions, unit:'', icon:'📊'},
  {label:'Avg F1', value: (D.overall.token_f1_score||0).toFixed(3), unit:'', icon:'🎯'},
  {label:'Avg ROUGE-L',value: (D.overall.rougeL||0).toFixed(3),unit:'',icon:'📝'},
  {label:'Avg MRR', value: (D.overall.mrr||0).toFixed(3), unit:'', icon:'🔍'},
  {label:'Recall@5', value: (D.overall.recall_5||0).toFixed(3),unit:'',icon:'📈'},
  {label:'Faithfulness',value:(D.overall.faithfulness||0).toFixed(3),unit:'',icon:'✅'},
  {label:'Grounded', value: (D.grounded_rate*100||0).toFixed(1),unit:'%',icon:'🔒'},
  {label:'Avg Attempts',value: D.avg_attempts, unit:'', icon:'🔄'},
];
const kpiRow = document.getElementById('kpi-row');
kpis.forEach(k=>{
  kpiRow.innerHTML += `<div class="metric-card text-center">
    <div class="text-2xl mb-1">${k.icon}</div>
    <div class="text-2xl font-bold text-blue-300">${k.value}${k.unit}</div>
    <div class="text-xs text-slate-400 mt-1">${k.label}</div>
  </div>`;
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function barChart(id, labels, datasets, opts={}){
  new Chart(document.getElementById(id), {
    type:'bar',
    data:{labels, datasets},
    options:{
      responsive:true, maintainAspectRatio:true,
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:11}}}},
      scales:{
        x:{ticks:{color:'#64748b'},grid:{color:'#1e293b'}},
        y:{ticks:{color:'#64748b'},grid:{color:'#334155'}, beginAtZero:true, max:1,...(opts.yMax?{max:opts.yMax}:{})},
      },
    }
  });
}

const models = D.models.length ? D.models : ['default'];

// ── Generation Chart ──────────────────────────────────────────────────────────
{
  const metrics = ['token_f1_score','sentence_bleu_score','rouge1','rouge2','rougeL','meteor'];
  const labels = ['F1','BLEU','ROUGE-1','ROUGE-2','ROUGE-L','METEOR'];
  const datasets = models.map((m,i)=>({
    label: m,
    data: metrics.map(met=>(D.by_model[met]||{})[m]||0),
    backgroundColor: COLORS[i%COLORS.length]+'99',
    borderColor: COLORS[i%COLORS.length],
    borderWidth:1,
  }));
  barChart('genChart', labels, datasets);
}

// ── Retrieval Chart ───────────────────────────────────────────────────────────
{
  const metrics = ['mrr','ndcg_at_k','recall_1','recall_3','recall_5'];
  const labels = ['MRR','NDCG','Recall@1','Recall@3','Recall@5'];
  const datasets = models.map((m,i)=>({
    label: m,
    data: metrics.map(met=>(D.by_model[met]||{})[m]||0),
    backgroundColor: LIGHT[i%LIGHT.length]+'88',
    borderColor: LIGHT[i%LIGHT.length],
    borderWidth:1,
  }));
  barChart('retChart', labels, datasets);
}

// ── Radar Chart (LLM-Judge) ───────────────────────────────────────────────────
{
  const judgeLabels = ['Answer Relevance','Faithfulness','Context Precision'];
  const judgeKeys = ['answer_relevance','faithfulness','context_precision'];
  new Chart(document.getElementById('radarChart'),{
    type:'radar',
    data:{
      labels: judgeLabels,
      datasets: models.map((m,i)=>({
        label: m,
        data: judgeKeys.map(k=>(D.by_model[k]||{})[m]||0),
        borderColor: COLORS[i%COLORS.length],
        backgroundColor: COLORS[i%COLORS.length]+'33',
        pointBackgroundColor: COLORS[i%COLORS.length],
        borderWidth:2,
      })),
    },
    options:{
      responsive:true, maintainAspectRatio:true,
      scales:{r:{min:0,max:1,ticks:{stepSize:.2,color:'#475569',backdropColor:'transparent'},
        grid:{color:'#334155'},pointLabels:{color:'#94a3b8',font:{size:11}}}},
      plugins:{legend:{labels:{color:'#94a3b8'}}},
    }
  });
}

// ── Language Chart ────────────────────────────────────────────────────────────
{
  const langs = Object.keys(D.by_language.token_f1_score||{});
  const datasets = [
    {label:'F1', data:langs.map(l=>(D.by_language.token_f1_score||{})[l]||0), backgroundColor:COLORS[0]+'99',borderColor:COLORS[0],borderWidth:1},
    {label:'ROUGE-L',data:langs.map(l=>(D.by_language.rougeL||{})[l]||0), backgroundColor:COLORS[1]+'99',borderColor:COLORS[1],borderWidth:1},
    {label:'METEOR', data:langs.map(l=>(D.by_language.meteor||{})[l]||0), backgroundColor:COLORS[2]+'99',borderColor:COLORS[2],borderWidth:1},
  ];
  barChart('langChart', langs, datasets);
}

// ── Difficulty Chart ──────────────────────────────────────────────────────────
{
  const diffs = Object.keys(D.by_difficulty.token_f1_score||{});
  barChart('diffChart', diffs, [{
    label:'F1 by Difficulty',
    data: diffs.map(d=>(D.by_difficulty.token_f1_score||{})[d]||0),
    backgroundColor: diffs.map((_,i)=>COLORS[i%COLORS.length]+'bb'),
    borderColor: diffs.map((_,i)=>COLORS[i%COLORS.length]),
    borderWidth:1,
  }]);
}

// ── Category Chart ────────────────────────────────────────────────────────────
{
  const cats = Object.keys(D.by_category.rougeL||{});
  barChart('catChart', cats.map(c=>c.length>18?c.slice(0,16)+'…':c), [{
    label:'ROUGE-L by Category',
    data: cats.map(c=>(D.by_category.rougeL||{})[c]||0),
    backgroundColor: cats.map((_,i)=>LIGHT[i%LIGHT.length]+'99'),
    borderColor: cats.map((_,i)=>LIGHT[i%LIGHT.length]),
    borderWidth:1,
  }]);
}

// ── Table ─────────────────────────────────────────────────────────────────────
const tbody = document.getElementById('results-tbody');
(D.rows||[]).forEach((r,i)=>{
  const langBadge = r.language==='en'?'badge-en':r.language==='fr'?'badge-fr':'badge-other';
  tbody.innerHTML += `<tr>
    <td class="text-slate-500">${i+1}</td>
    <td class="max-w-xs truncate" title="${(r.question||'').replace(/"/g,"'").replace(/</g,"&lt;").replace(/>/g,"&gt;")}">
      ${(r.question||'').slice(0,60)}${(r.question||'').length>60?'…':''}</td>
    <td class="text-blue-300 text-xs">${r.rag_base_model||''}</td>
    <td><span class="badge ${langBadge}">${r.language||''}</span></td>
    <td class="${scoreClass(r.token_f1_score)}">${fmt(r.token_f1_score)}</td>
    <td class="${scoreClass(r.sentence_bleu_score)}">${fmt(r.sentence_bleu_score)}</td>
    <td class="${scoreClass(r.rougeL)}">${fmt(r.rougeL)}</td>
    <td class="${scoreClass(r.meteor)}">${fmt(r.meteor)}</td>
    <td class="${scoreClass(r.mrr)}">${fmt(r.mrr)}</td>
    <td class="${scoreClass(r.recall_5)}">${fmt(r.recall_5)}</td>
    <td class="${scoreClass(r.answer_relevance)}">${fmt(r.answer_relevance)}</td>
    <td class="${scoreClass(r.faithfulness)}">${fmt(r.faithfulness)}</td>
    <td class="${scoreClass(r.context_precision)}">${fmt(r.context_precision)}</td>
    <td>${r.grounded?'<span class="text-green-400">✓</span>':'<span class="text-red-400">✗</span>'}</td>
    <td class="text-center text-slate-400">${r.attempts||1}</td>
  </tr>`;
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LUFA RAG system performance")
    parser.add_argument("--lufa_csv", default="tests/lufa_out_data.csv")
    parser.add_argument("--test_csv", default="tests/combined_test_data_and_ground_truth.csv")
    parser.add_argument("--out_csv", default="tests/evaluation_results.csv")
    parser.add_argument("--dashboard", default="dashboard/index.html")
    parser.add_argument("--no_llm_judge", action="store_true", help="Skip LLM-as-judge step")
    parser.add_argument("--llm_model", default="llama3.2:3b-instruct-q4_K_M")
    parser.add_argument("--sim_mode", choices=["local", "api", "frontier"], default="local")
    parser.add_argument("--api_url", default="http://localhost:8000")
    args = parser.parse_args()

    run_evaluation(
        lufa_csv=args.lufa_csv,
        test_csv=args.test_csv,
        out_csv=args.out_csv,
        dashboard_out=args.dashboard,
        use_llm_judge=not args.no_llm_judge,
        llm_model=args.llm_model,
        sim_mode=args.sim_mode,
        api_url=args.api_url
    )