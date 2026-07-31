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

# ── Chapter-4 instrumentation column groups (single source of truth) ──────────
# SYSTEM_METRIC_COLUMNS are captured at SIMULATION time (they can only be
# measured while the query actually runs). They live in lufa_out_data.csv and are
# carried into evaluation_results.csv (exactly like TRANSLATION_COLUMNS). Every
# value is BLANK when unavailable — old rows that predate instrumentation, and
# GPU VRAM / RAM on cloud/api runs (Ch4 §4.6.4: hardware metrics apply only to
# the local systems A/B/C). Appended at the very end so existing CSVs migrate
# cleanly (new columns added empty, nothing shifts).
#   retrieval_latency_s  — wall-clock of the first-pass hybrid retrieval (§4.6.4)
#   ttft_s               — time-to-first-token of first-pass generation (§4.6.4)
#   end_to_end_latency_s — total query wall-clock, submission→answer (§4.6.4)
#   gpu_vram_mb          — PEAK GPU memory during the query (nvidia-smi), local only
#   system_ram_mb        — peak process RSS (psutil), local only (§4.6.4)
#   cpu_percent          — mean system CPU utilisation during the query, local only
#   gpu_util_percent     — mean GPU utilisation during the query, local only
#   warmup_applied       — 1 on the one warm-up (cold-start-discarded) question
# cpu_percent / gpu_util_percent exist because Ollama silently splits a model across
# CPU and GPU when it does not fit in VRAM; recording both makes the actual execution
# device visible in the data rather than silently distorting the latency figures.
#   gpu_vram_dedicated_mb / gpu_vram_shared_mb — Windows adapter-memory split. When a
#     model exceeds the 6 GB card, WDDM spills into shared system memory (24 GB is
#     dedicated to this here). nvidia-smi cannot see that spill, so it is recorded
#     separately and its performance cost stays visible instead of hidden.
#   context_window_used / prompt_tokens_est / predicted_output_tokens — the per-query
#     context budget chosen at request time by src/llm_utils.compute_context_window.
SYSTEM_METRIC_COLUMNS = [
    "retrieval_latency_s",
    "ttft_s",
    "end_to_end_latency_s",
    "gpu_vram_mb",
    "gpu_vram_dedicated_mb",
    "gpu_vram_shared_mb",
    "system_ram_mb",
    "cpu_percent",
    "gpu_util_percent",
    "context_window_used",
    "prompt_tokens_est",
    "predicted_output_tokens",
    "warmup_applied",
]

# EVAL_ONLY_METRIC_COLUMNS are computed at EVALUATION time (they need the gold
# labels), so they live only in evaluation_results.csv.
#   precision_1/3/5           — P@k, P@3 is the primary retrieval metric (§4.6.1)
#   citation_accuracy_regex   — deterministic article/clause match vs gold (§4.8.4)
#   citation_accuracy_judge   — LLM-judge citation score, own prompt (§4.6.3)
EVAL_ONLY_METRIC_COLUMNS = [
    "precision_1",
    "precision_3",
    "precision_5",
    "citation_accuracy_regex",
    "citation_accuracy_judge",
]

# HUMAN_MANUAL_COLUMNS are filled in by the researcher (Ch4 §4.4.6 IAA + §4.8.3
# human adjudication on the 30-output sample). They default BLANK so an unfilled
# cell is never mistaken for a real score, and are always the LAST columns.
#   in_human_sample         — mark the row as part of the stratified 30-output set
#   human_annot{1,2}_citation  — each annotator's citation-accuracy score (1/0.5/0)
#   human_annot{1,2}_relevance — each annotator's binary relevance judgment (for Kappa)
#   human_citation_accuracy — adjudicated citation-accuracy score
#   human_appropriateness   — adjudicated binary "fit for real use" score (§4.8.3)
HUMAN_MANUAL_COLUMNS = [
    "in_human_sample",
    "human_annot1_citation",
    "human_annot2_citation",
    "human_annot1_relevance",
    "human_annot2_relevance",
    "human_citation_accuracy",
    "human_appropriateness",
]

# Cross-lingual "no-translation" columns. In the no-translation pipeline the query is
# NOT translated: retrieval runs on the raw foreign-language query (the real test of
# cross-lingual retrieval) and the answer is generated in the QUESTION's language.
# The benchmark's expected_answer / ground_source_truth are in English, though, so a
# post-hoc translation of the answer is kept purely so the lexical metrics (Token-F1,
# BLEU, ROUGE, METEOR) compare like with like. It is NOT part of the RAG pipeline and
# is never shown to the judge — the judge sees the native-language answer.
#   answer_metrics_translation — answer rendered in the ground-truth language
#   metrics_language           — language of that rendering (e.g. 'en')
#   pipeline_translation_mode  — 'none' (no-translation run) or 'bridge' (translate to EN)
CROSSLINGUAL_COLUMNS = [
    "answer_metrics_translation",
    "metrics_language",
    "pipeline_translation_mode",
]

OUTPUT_COLUMNS = [
    "question_id", "question",
    "source1_id", "source1_cosine_score", "source1_recency_adjusted_cosine_score", "source1_rrf_score", "source1_text",
    "source2_id", "source2_cosine_score", "source2_recency_adjusted_cosine_score", "source2_rrf_score", "source2_text",
    "source3_id", "source3_cosine_score", "source3_recency_adjusted_cosine_score", "source3_rrf_score", "source3_text",
    "source4_id", "source4_cosine_score", "source4_recency_adjusted_cosine_score", "source4_rrf_score", "source4_text",
    "source5_id", "source5_cosine_score", "source5_recency_adjusted_cosine_score", "source5_rrf_score", "source5_text",
    "answer", "base_model_used", "language", "attempts", "grounded",
] + TRANSLATION_COLUMNS + SYSTEM_METRIC_COLUMNS + CROSSLINGUAL_COLUMNS


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


def extract_system_metric_columns(result: dict, mode: str,
                                  e2e_latency="", warmup_applied="", sysm: dict = None) -> dict:
    """
    Assemble the 6 sim-time performance columns for a lufa_out row.

    - retrieval_latency_s / ttft_s come from the engine result dict (blank when the
      backend did not report them, e.g. `api` mode where retrieval runs server-side).
    - end_to_end_latency_s is the caller-measured wall-clock for the whole query.
    - gpu_vram_mb / system_ram_mb come from `sysm` (blank for cloud/api — see
      system_metrics.sample_system_metrics, which blanks non-local modes).
    - warmup_applied is 1 only on the single warm-up (cold-start-discarded) question.
    """
    sysm = sysm or {}
    return {
        "retrieval_latency_s": result.get("retrieval_latency_s", "") if result else "",
        "ttft_s": result.get("ttft_s", "") if result else "",
        "end_to_end_latency_s": e2e_latency,
        "gpu_vram_mb": sysm.get("gpu_vram_mb", ""),
        "gpu_vram_dedicated_mb": sysm.get("gpu_vram_dedicated_mb", ""),
        "gpu_vram_shared_mb": sysm.get("gpu_vram_shared_mb", ""),
        "system_ram_mb": sysm.get("system_ram_mb", ""),
        "cpu_percent": sysm.get("cpu_percent", ""),
        "gpu_util_percent": sysm.get("gpu_util_percent", ""),
        "context_window_used": (result.get("context_window_used", "") if result else ""),
        "prompt_tokens_est": (result.get("prompt_tokens_est", "") if result else ""),
        "predicted_output_tokens": (result.get("predicted_output_tokens", "") if result else ""),
        "warmup_applied": warmup_applied,
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


def query_single_record(record, mode, base_model, model_name, api_url, idx, do_warmup=False):
    """
    Core atomic method to prompt the backend RAG layout for a single row.
    Provides deep verbose terminal feedback for interactive visibility.

    When `do_warmup` is True (the very first question of a local batch), retrieval
    is run once as a discarded warm-up before the timed pass, so cold-start cost
    (index load, BM25 corpus build, first embedding) does not distort the recorded
    retrieval latency (Ch4 §4.6.4 warm-up protocol).
    """
    import time as _time
    from system_metrics import sample_system_metrics

    q_text = str(record["question"])
    q_id = str(record["id"])
    print(f"   [Simulation Engine] Initializing query transaction for ID: {q_id}")
    print(f"   [Simulation Engine] Query String: \"{q_text[:65]}...\"")
    print(f"   [Simulation Engine] Mode: '{mode}' | Target Model: '{model_name}'")

    warmup_flag = ""
    e2e_latency = ""
    sysm = {"gpu_vram_mb": "", "system_ram_mb": ""}

    try:
        if mode == "local":
            from rag_engine import create_rag_engine
            engine = create_rag_engine()
            if do_warmup:
                print("   [Warmup] Running first-question retrieval warm-up (timing discarded)...")
                engine.warmup_retrieve(q_text)
                warmup_flag = 1
            _t0 = _time.perf_counter()
            result = engine.agentic_query(
                query_text=q_text,
                return_sources=True,
                max_retries=3,
            )
            e2e_latency = round(_time.perf_counter() - _t0, 4)

        elif mode == "local-naive":
            from rag_engine import create_rag_engine
            engine = create_rag_engine()
            if do_warmup:
                print("   [Warmup] Running first-question retrieval warm-up (timing discarded)...")
                engine.warmup_retrieve(q_text)
                warmup_flag = 1
            _t0 = _time.perf_counter()
            result = engine.naive_query(
                query_text=q_text,
                return_sources=True
            )
            e2e_latency = round(_time.perf_counter() - _t0, 4)

        elif mode == "api":
            import httpx
            _t0 = _time.perf_counter()
            with httpx.Client(timeout=400.0) as client:
                resp = client.post(
                    f"{api_url}/agentic-query",
                    json={"query": q_text, "return_sources": True, "max_retries": 3},
                )
                resp.raise_for_status()
                result = resp.json()
            e2e_latency = round(_time.perf_counter() - _t0, 4)

        elif mode == "frontier":
            from rag_engine import create_rag_engine
            from copilot_engine import CopilotEngine
            rag_engine = create_rag_engine()
            copilot = CopilotEngine(model=model_name)

            if do_warmup:
                print("   [Warmup] Running first-question retrieval warm-up (timing discarded)...")
                rag_engine.warmup_retrieve(q_text)
                warmup_flag = 1

            _t0 = _time.perf_counter()
            lang = rag_engine.detect_query_language(q_text)
            nodes = rag_engine._retrieve_nodes(q_text, top_k=5)
            frontier_retrieval_s = getattr(rag_engine, "_last_retrieval_seconds", "")
            answer = copilot.generate_from_nodes(q_text, nodes, lang)
            e2e_latency = round(_time.perf_counter() - _t0, 4)

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
                "sources": sources_list,
                "retrieval_latency_s": frontier_retrieval_s,
                "ttft_s": "",
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
        sysm = sample_system_metrics(mode)
        row.update(extract_system_metric_columns(result, mode, e2e_latency, warmup_flag, sysm))

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
    for c in SYSTEM_METRIC_COLUMNS:
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

    # Warm-up protocol (Ch4 §4.6.4): the FIRST question actually processed in this
    # batch runs retrieval twice — once discarded (cold start), then the recorded
    # pass. Every subsequent question is recorded on its single first run.
    warmed_up = False

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

        do_warmup = not warmed_up
        row_res = query_single_record(record, args.mode, base_model, model_name, args.api_url,
                                      current_counter, do_warmup=do_warmup)
        warmed_up = True

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