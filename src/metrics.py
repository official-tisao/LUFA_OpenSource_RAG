#!/usr/bin/env python3
"""
Unified metrics module for LUFA RAG system.
Contains all metric calculations used in evaluation and repair.
"""

import re
import warnings
import pandas as pd
import time
from typing import Dict, List, Tuple

from llm_utils import stream_complete

warnings.filterwarnings("ignore")

# NLP imports (lazy-loaded to avoid startup overhead)
_nltk_initialized = False
def _init_nltk():
    global _nltk_initialized
    if not _nltk_initialized:
        import nltk
        global sentence_bleu, SmoothingFunction, meteor_score, rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        from rouge_score import rouge_scorer
        _nltk_initialized = True
def tokenize(text):
    """Tokenize text into words for metric calculations."""
    return re.findall(r'\b\w+\b', str(text).lower())
def token_f1(prediction, reference):
    """Calculate Token F1 score between prediction and reference."""
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
    """Calculate BLEU score."""
    try:
        pred_tokens = tokenize(prediction)
        ref_tokens = tokenize(reference)
        if not pred_tokens:
            return 0.0

        _init_nltk()
        smoothie = SmoothingFunction().method4
        from nltk.translate.bleu_score import sentence_bleu
        return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
    except Exception:
        return 0.0
def compute_rouge(prediction, reference):
    """Calculate ROUGE scores (R1, R2, RL)."""
    try:
        _init_nltk()
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(str(reference), str(prediction))
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
def compute_meteor(prediction, reference):
    """Calculate METEOR score."""
    try:
        _init_nltk()
        pred_tokens = tokenize(prediction)
        ref_tokens = tokenize(reference)
        if not pred_tokens or not ref_tokens:
            return 0.0

        from nltk.translate.meteor_score import meteor_score
        return meteor_score([ref_tokens], pred_tokens)
    except Exception:
        return 0.0
def parse_source_ids(raw):
    """Parse pipe-separated source IDs string."""
    if not raw or pd.isna(raw):
        return []
    return [s.strip() for s in str(raw).split("|") if s.strip()]
def recall_at_k(retrieved, ground_truth, k):
    """Calculate Recall at K."""
    if not ground_truth:
        return 0.0
    relevant_at_k = set(retrieved[:k]) & set(ground_truth)
    return len(relevant_at_k) / len(ground_truth)
def mrr(retrieved, ground_truth):
    """Calculate Mean Reciprocal Rank."""
    gt_set = set(ground_truth)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gt_set:
            return 1.0 / rank
    return 0.0
def ndcg_at_k(retrieved, ground_truth, k=5):
    """Calculate NDCG at K."""
    import math
    gt_set = set(ground_truth)
    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in gt_set
    )
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0
def extract_score(text):
    """Extract score from judge LLM response."""
    match = re.search(r'\b([1-5])\b', str(text).strip())
    return float(match.group(1)) / 5.0 if match else 0.5
def _load_judge_model(judge_llm_model, request_timeout=240.0):
    """Load the judge LLM, auto-routing to the OpenAI-compatible client for
    cloud/OpenRouter endpoints or Ollama for local models (based on api_base)."""
    from model_api_auth import get_llm_client
    return get_llm_client(judge_llm_model, request_timeout=request_timeout)
def judge_llm_scores(question, answer, context, judge_llm_model="None"):
    """Use local Ollama model as judge. Returns normalized 0-1 scores."""
    judge_llm_model = judge_llm_model or "None"
    from llama_index.llms.ollama import Ollama

    llm = _load_judge_model(judge_llm_model)
    scores = {}
    for metric, prompt_template in JUDGE_LLM_PROMPTS.items():
        try:
            prompt = prompt_template.format(
                question=question[:500],
                answer=answer[:800],
                context=context[:3000],
            )
            raw = stream_complete(llm, prompt)
            scores[metric] = round(extract_score(raw), 4)
        except Exception as e:
            print(f"      [Judge Warning] {metric} calculation failure: {e}")
            scores[metric] = 0.0
    return scores
# Combined judge prompt (replace/keep original separate prompts)
COMBINED_JUDGE_PROMPT = """
You are evaluating a RAG system's output. Score the following three metrics on a 1-5 scale.

1. answer_relevance: How relevant is the answer to the question?
   1=completely irrelevant, 3=partially relevant, 5=fully relevant.

2. faithfulness: How faithful is the answer to the provided context (no unsupported claims)?
   1=many unsupported claims, 3=mostly supported, 5=all claims verifiable in context.

3. context_precision: How precisely is the retrieved context relevant to the question?
   1=context is mostly irrelevant, 3=some chunks relevant, 5=all chunks highly relevant.

Question: {question}
Context: {context}
Answer: {answer}

For each metric, first write one sentence of reasoning, then give the score.
Reply in EXACTLY this format, nothing else:

answer_relevance_reasoning: <one sentence>
answer_relevance_score: <int>
faithfulness_reasoning: <one sentence>
faithfulness_score: <int>
context_precision_reasoning: <one sentence>
context_precision_score: <int>
"""
def combined_judge_llm_scores(question, answer, context, judge_llm_model="None"):
    """Combined judge prompt evaluating all three metrics in one LLM call."""
    judge_llm_model = judge_llm_model or "None"
    from llama_index.llms.ollama import Ollama

    llm = _load_judge_model(judge_llm_model)
    try:
        prompt = COMBINED_JUDGE_PROMPT.format(
            question=question[:500],
            answer=answer[:800],
            context=context[:3000],
        )
        raw = stream_complete(llm, prompt)
        return parse_combined_judge_response(raw)
    except Exception as e:
        print(f"      [Judge Warning] Combined judge calculation failure: {e}")
        return {
            "answer_relevance": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0
        }
def parse_combined_judge_response(response):
    """Parse combined judge response extracting reasoning and scores."""
    scores = {
        "answer_relevance": 0.0,
        "faithfulness": 0.0,
        "context_precision": 0.0
    }

    lines = response.split('\n')
    current_metric = None

    for line in lines:
        line = line.strip()
        if line.startswith('answer_relevance_score:'):
            try:
                scores["answer_relevance"] = round(float(line.split(':')[1].strip()) / 5.0, 4)
            except (ValueError, IndexError):
                scores["answer_relevance"] = 0.5
        elif line.startswith('faithfulness_score:'):
            try:
                scores["faithfulness"] = round(float(line.split(':')[1].strip()) / 5.0, 4)
            except (ValueError, IndexError):
                scores["faithfulness"] = 0.5
        elif line.startswith('context_precision_score:'):
            try:
                scores["context_precision"] = round(float(line.split(':')[1].strip()) / 5.0, 4)
            except (ValueError, IndexError):
                scores["context_precision"] = 0.5

    return scores
# Original separate judge prompts (kept for backward compatibility)
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


if __name__ == "__main__":
    import argparse
    import sys
    import math
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser(description="Compute evaluation metrics from LUFA RAG output")
    parser.add_argument("--lufa_csv", default="tests/lufa_out_data.csv",
                        help="RAG output CSV (answers + retrieved sources)")
    parser.add_argument("--test_csv", default="tests/combined_test_data_and_ground_truth.csv",
                        help="Ground truth CSV (expected_answer, ground_source_truth_id)")
    parser.add_argument("--out_csv", default="tests/evaluation_results.csv",
                        help="Output CSV for computed metrics")
    parser.add_argument("--judge_llm", default=None,
                        help="Ollama model for LLM-as-judge scoring (skipped if not set)")
    args = parser.parse_args()

    from retrieval import has_old_schema
    from csv_utils import migrate_csv_schema, resolve_language
    from evaluate import EVAL_COLUMNS
    from run_simulation import TRANSLATION_COLUMNS

    lufa_path = Path(args.lufa_csv)
    test_path = Path(args.test_csv)
    out_path = Path(args.out_csv)

    if not lufa_path.exists():
        print(f"[Metrics] ERROR: {lufa_path} not found. Run retrieval.py or answer_generator.py first.")
        sys.exit(1)
    if not test_path.exists():
        print(f"[Metrics] ERROR: {test_path} not found.")
        sys.exit(1)

    lufa_df = pd.read_csv(lufa_path)
    test_df = pd.read_csv(test_path)

    # Metric groups: deterministic metrics are cheap and always recomputable;
    # judge metrics need an Ollama judge model (--judge_llm) to (re)compute.
    DETERMINISTIC_METRICS = [
        "token_f1_score", "sentence_bleu_score", "rouge1", "rouge2", "rougeL", "meteor",
        "mrr", "ndcg_at_k", "recall_1", "recall_3", "recall_5",
    ]
    JUDGE_METRICS = ["answer_relevance", "faithfulness", "context_precision"]

    def _is_zero_or_blank(v):
        """True when a metric cell is missing, empty, or numerically zero."""
        if v is None:
            return True
        try:
            if pd.isna(v):
                return True
        except (TypeError, ValueError):
            pass
        s = str(v).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return True
        try:
            return float(s) == 0.0
        except (TypeError, ValueError):
            return False

    # Load existing evaluation rows (migrating an old single-score schema first).
    # These are the baseline we update in place — any zero/blank metric cell is
    # re-attempted; existing non-zero (esp. judge) scores are preserved.
    existing_rows = {}
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            existing_eval_df = pd.read_csv(out_path)
            if has_old_schema(existing_eval_df):
                print("[Metrics] OLD SCHEMA DETECTED in evaluation_results.csv — migrating to per-chunk schema first...")
                migrate_csv_schema(out_path, EVAL_COLUMNS)
                existing_eval_df = pd.read_csv(out_path)
            if "question_id" in existing_eval_df.columns:
                for _, erow in existing_eval_df.iterrows():
                    eid = str(erow.get("question_id", "")).strip()
                    if eid:
                        existing_rows[eid] = erow.to_dict()
                print(f"[Metrics] Loaded {len(existing_rows)} existing rows — zero/blank metrics will be recomputed.")
        except Exception as e:
            print(f"[Metrics] Warning: could not read existing output: {e}")

    # Index lufa rows by question_id
    lufa_records = {}
    if "question_id" in lufa_df.columns:
        for _, r in lufa_df.iterrows():
            qid = str(r.get("question_id", "")).strip()
            if qid:
                lufa_records[qid] = r.to_dict()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _safe(val):
        return "" if pd.isna(val) else val

    def _safe_float(v):
        try:
            return round(float(v), 6)
        except (TypeError, ValueError):
            return ""

    def _get_retrieved_ids(rag_row):
        ids = []
        for i in range(1, 6):
            sid = str(rag_row.get(f"source{i}_id", "")).strip()
            if sid:
                ids.append(sid)
        return ids

    def _get_ground_truth_ids(test_row):
        for col in ("ground_source_truth_id", "ground_truth_source_ids"):
            raw = test_row.get(col, "")
            if raw and not pd.isna(raw):
                ids = [s.strip() for s in str(raw).split("|") if s.strip()]
                if ids:
                    return ids
        return []

    def _build_context(rag_row):
        parts = []
        for i in range(1, 6):
            txt = str(rag_row.get(f"source{i}_text", "")).strip()
            if txt:
                parts.append(txt)
        return " ".join(parts)[:3000]

    # results = final ledger. Start from existing rows so anything we don't touch
    # (e.g. rows without answers, or already-complete rows) is preserved verbatim.
    results = dict(existing_rows)

    # One-time backup of the previous ledger before we begin rewriting it per-row.
    if out_path.exists() and out_path.stat().st_size > 0:
        bak = out_path.with_suffix(out_path.suffix + ".metricsbak")
        pd.read_csv(out_path, on_bad_lines="skip").to_csv(bak, index=False)
        print(f"[Metrics] Backed up previous evaluation -> {bak}")

    def _persist_and_refresh():
        """Write the full ledger to disk and regenerate the dashboard (per-row safe)."""
        df_out = pd.DataFrame([results[k] for k in sorted(results)], columns=EVAL_COLUMNS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        try:
            from dashboard_generator import refresh_dashboard
            refresh_dashboard(eval_csv=str(out_path), lufa_csv=str(lufa_path))
        except Exception as _de:
            print(f"      [Dashboard] refresh skipped: {_de}")
        return df_out

    total = len(test_df)
    skipped_no_answer = []
    recomputed_det = 0
    judged = 0
    unchanged = 0

    for idx, row in test_df.iterrows():
        # if int(idx) > 0:
        #     time.sleep(2)  # Pauses execution for exactly 3.0 seconds
        #     print("2 seconds have passed!")

        q_id = str(row.get("id", "")).strip()
        counter = f"[{idx + 1}/{total}]"

        if q_id not in lufa_records:
            skipped_no_answer.append(q_id)
            continue

        rag = lufa_records[q_id]
        prediction = "" if pd.isna(rag.get("answer")) else str(rag.get("answer", ""))
        if not prediction or prediction.strip() == "" or prediction.strip().upper() == "ERROR":
            skipped_no_answer.append(q_id)
            continue

        prev = existing_rows.get(q_id, {})
        is_new = q_id not in existing_rows

        # Recompute deterministic metrics when the row is new OR any deterministic
        # cell is zero/blank. (These are cheap, so we refresh the whole group.)
        need_det = is_new or any(_is_zero_or_blank(prev.get(m)) for m in DETERMINISTIC_METRICS)
        # Recompute judge cells only when a judge model is available AND some judge
        # cell is zero/blank; non-zero judge scores are always preserved.
        judge_zero = any(_is_zero_or_blank(prev.get(m)) for m in JUDGE_METRICS)
        #no_naive_judge = args.mode == "local-naive"
        need_judge = bool(args.judge_llm) and (is_new or judge_zero)

        if not need_det and not need_judge:
            unchanged += 1
            continue

        reference = "" if pd.isna(row.get("expected_answer")) else str(row.get("expected_answer", ""))
        question = "" if pd.isna(row.get("question")) else str(row.get("question", ""))
        context = _build_context(rag)
        retrieved_ids = _get_retrieved_ids(rag)
        ground_truth_ids = _get_ground_truth_ids(row)

        if str(row.get("translation_applied")).strip().lower() == "true":
            prediction = str(row.get("untranslated_answer", prediction))
            question = str(row.get("translated_question", question))

        print(f"\n{counter} {q_id}: \"{question[:55]}...\" (deterministic={need_det}, judge={need_judge})")

        # ---- deterministic metrics ----
        if need_det:
            f1_val = round(token_f1(prediction, reference), 4)
            bleu_val = round(compute_bleu(prediction, reference), 4)
            rouge_scores = compute_rouge(prediction, reference)
            meteor_val = round(compute_meteor(prediction, reference), 4)
            mrr_val = round(mrr(retrieved_ids, ground_truth_ids), 4)
            ndcg_val = ndcg_at_k(retrieved_ids, ground_truth_ids, k=5)
            rec1 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=1), 4)
            rec3 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=3), 4)
            rec5 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=5), 4)
            print(f"   F1={f1_val} BLEU={bleu_val} R1={rouge_scores['rouge1']} METEOR={meteor_val} "
                  f"| MRR={mrr_val} NDCG@5={ndcg_val} R@1/3/5={rec1}/{rec3}/{rec5}")
            recomputed_det += 1
        else:
            f1_val = _safe_float(prev.get("token_f1_score")) or 0.0
            bleu_val = _safe_float(prev.get("sentence_bleu_score")) or 0.0
            rouge_scores = {
                "rouge1": _safe_float(prev.get("rouge1")) or 0.0,
                "rouge2": _safe_float(prev.get("rouge2")) or 0.0,
                "rougeL": _safe_float(prev.get("rougeL")) or 0.0,
            }
            meteor_val = _safe_float(prev.get("meteor")) or 0.0
            mrr_val = _safe_float(prev.get("mrr")) or 0.0
            ndcg_val = _safe_float(prev.get("ndcg_at_k")) or 0.0
            rec1 = _safe_float(prev.get("recall_1")) or 0.0
            rec3 = _safe_float(prev.get("recall_3")) or 0.0
            rec5 = _safe_float(prev.get("recall_5")) or 0.0

        # ---- judge metrics: preserve non-zero, fill zero cells when judge model set ----
        ar = _safe_float(prev.get("answer_relevance")) or 0.0
        faith = _safe_float(prev.get("faithfulness")) or 0.0
        cp = _safe_float(prev.get("context_precision")) or 0.0
        if need_judge:
            print(f"   Running LLM judge ({args.judge_llm}) on zero cells...")
            try:
                j = combined_judge_llm_scores(question, prediction, context, args.judge_llm)
                if _is_zero_or_blank(prev.get("answer_relevance")):
                    ar = j.get("answer_relevance", ar)
                if _is_zero_or_blank(prev.get("faithfulness")):
                    faith = j.get("faithfulness", faith)
                if _is_zero_or_blank(prev.get("context_precision")):
                    cp = j.get("context_precision", cp)
                judged += 1
                print(f"   relevance={ar} faithfulness={faith} context_precision={cp}")
            except Exception as je:
                print(f"   [Judge Warning] {je}")

        # is_grounded = False
        # check_grounded = False
        new_grounded ="false"
        if str(rag.get("grounded")).lower() == "false":
            new_grounded = "true" if float(cp) > 0.4 else "false"

        #     from reflector import reflect
        #     chunk_texts = [n.node.text for n in nodes]
        #     is_grounded = reflect(final_answer, chunk_texts, engine.llm)
        # ---- assemble the full row (carry forward any prior fields) ----
        out_row = dict(prev)
        out_row.update({
            "question_id": q_id,
            "id": q_id,
            "question": question,
            "answer": _safe(rag.get("answer")),
            "base_model_used": _safe(rag.get("base_model_used")),
            "rag_base_model": _safe(rag.get("base_model_used")) or _safe(prev.get("rag_base_model")),
            "language": resolve_language(row.get("language", prev.get("language", "")), q_id),
            "judge_llm": args.judge_llm or _safe(prev.get("judge_llm")),
            "category": _safe(row.get("category")) or _safe(prev.get("category")),
            "difficulty": _safe(row.get("difficulty")) or _safe(prev.get("difficulty")),
            "attempts": _safe(rag.get("attempts")),
            "grounded": _safe(new_grounded),
        })
        for i in range(1, 6):
            out_row[f"source{i}_id"] = _safe(rag.get(f"source{i}_id"))
            out_row[f"source{i}_cosine_score"] = _safe_float(rag.get(f"source{i}_cosine_score"))
            out_row[f"source{i}_recency_adjusted_cosine_score"] = _safe_float(rag.get(f"source{i}_recency_adjusted_cosine_score"))
            out_row[f"source{i}_rrf_score"] = _safe_float(rag.get(f"source{i}_rrf_score"))
            out_row[f"source{i}_text"] = _safe(rag.get(f"source{i}_text"))
        out_row.update({
            "token_f1_score": f1_val,
            "sentence_bleu_score": bleu_val,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "meteor": meteor_val,
            "mrr": mrr_val,
            "ndcg_at_k": ndcg_val,
            "recall_1": rec1,
            "recall_3": rec3,
            "recall_5": rec5,
            "answer_relevance": ar,
            "faithfulness": faith,
            "context_precision": cp,
        })
        # Carry cross-lingual translation columns straight from the lufa row.
        for _tc in TRANSLATION_COLUMNS:
            out_row[_tc] = _safe(rag.get(_tc))
        results[q_id] = out_row
        # Persist this row immediately (crash-safe) and refresh the dashboard.
        _persist_and_refresh()

    # Final consolidated write (also covers the no-rows-processed case).
    out_df = _persist_and_refresh()

    print(f"\n[Metrics] Done. Deterministic recomputed: {recomputed_det} | judged: {judged} "
          f"| already-complete: {unchanged} | total rows: {len(out_df)}.")
    print(f"[Metrics] Evaluation results in {out_path}")
    if skipped_no_answer:
        print(f"[Metrics] {len(skipped_no_answer)} questions skipped (missing/invalid answer).")
        print(f"[Metrics] Run `python src/retrieval.py` then `python src/answer_generator.py` to generate them.")