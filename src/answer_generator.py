#!/usr/bin/env python3
"""
Answer generator module for LUFA RAG system.
Provides answer generation functions (naive and agentic) with proper separation of concerns.
Uses per-chunk score columns: source{n}_cosine_score, source{n}_recency_adjusted_cosine_score, source{n}_rrf_score.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from copy import deepcopy

from openai import max_retries

from csv_utils import read_csv_cached, write_csv_row, get_completed_ids
from metrics import token_f1
from dashboard_generator import refresh_dashboard


def check_language(input_string):
    if "_en" in input_string:
        return "en"
    elif "_fr" in input_string:
        return "fr"
    elif "_de" in input_string:
        return "de"
    elif "_es" in input_string:
        return "es"

    return input_string[5:7]


def generate_naive_answer(engine,
                          query_text: str,
                          top_k: int = 5,
                          check_grounded: bool = False,
                          output_path: Union[str, Path] = "tests/lufa_out_data.csv", query_id: str = None,
                          cached_nodes=None,
                          no_translate: bool = False,
                          metrics_language: str = "en") -> Dict:
    """Generate answer using naive RAG approach (single retrieval + generation)."""

    return generate_agentic_answer(engine, query_text, 1, check_grounded, output_path, query_id,
                                   cached_nodes, no_translate=no_translate,
                                   metrics_language=metrics_language)


def generate_agentic_answer(engine,
                            query_text: str,
                            max_retries: int = 3,
                            check_grounded: bool = True,
                            output_path: Union[str, Path] = "tests/lufa_out_data.csv", query_id: str = None,
                            cached_nodes=None,
                            no_translate: bool = False,
                            metrics_language: str = "en") -> Dict:
    """Generate answer using agentic RAG approach (retrieval + retry loop with groundedness checking).

    no_translate=True runs the TRUE cross-lingual pipeline: the query is NOT translated,
    retrieval runs on the raw foreign-language query against the multilingual index, and
    the answer is generated in the question's own language. A post-hoc translation of the
    answer into `metrics_language` (the benchmark's ground-truth language) is produced
    separately, purely so lexical metrics compare like with like — it never enters the
    pipeline and is never shown to the judge.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # from language_detector import detect_full_language
    from translator import needs_translation, translate_to_english, translate_to_target, LANGUAGE_NAMES
    from query_rewriter import rewrite_query, build_headers
    from reflector import reflect

    # from language_detector import detect_full_language
    original_lang = (check_language(query_id) if query_id else check_language(query_text)) or "en"
    # In no-translate mode the pipeline never bridges through English: the raw query is
    # retrieved with and answered in its own language.
    translation_applied = (not no_translate) and needs_translation(original_lang)
    translated_query = query_text
    if no_translate and needs_translation(original_lang):
        print(f"[AnswerGenerator] NO-TRANSLATION cross-lingual mode: retrieving and answering "
              f"directly in '{original_lang}' (no English bridge).")
    if translation_applied:
        lang_name = LANGUAGE_NAMES.get(original_lang, original_lang.upper())
        print(f"[AnswerGenerator] Non-native language detected: {original_lang} ({lang_name})")
        print(f"[AnswerGenerator] Translating query to English for processing...")
        translated_query = translate_to_english(query_text, original_lang, engine.llm)
        processing_query = translated_query
        pipeline_lang = "en"
    else:
        processing_query = query_text
        pipeline_lang = original_lang

    print(f"[AnswerGenerator] Pipeline language: {pipeline_lang}")

    import time as _time
    from system_metrics import ResourceSampler

    rewritten_query = processing_query
    nodes = []
    answer = ""
    is_grounded = False
    real_attempt = 0;
    # Carried into the NEXT attempt's rewrite. Empty on attempt 1, so attempt 1 never rewrites.
    prev_headers = []
    prev_answer = ""
    # Ch4 §4.6.4 timing: first-pass retrieval latency + TTFT, and end-to-end.
    first_retrieval_s = ""
    first_ttft_s = ""
    ctx_used = ptok_used = otok_pred = ""
    # Sample CPU / GPU utilisation for the WHOLE query (all attempts + any
    # translation), not just an instant after it finished.
    _sampler = ResourceSampler(interval=2.0, enabled=True).start()
    _e2e_start = _time.perf_counter()
    for attempt in range(1, max_retries + 1):
        print(f"[AnswerGenerator] Attempt {attempt}/{max_retries}")

        if attempt > 1:
            # Rewrite from `processing_query`, the original question, never from the previous
            # rewrite: feeding the rewrite back made attempt 3 a rewrite of a rewrite and
            # drifted further from what was asked on every pass. The rewriter is given the
            # provision titles the previous pass actually found (headers only, never the noisy
            # chunk bodies) plus the answer the reflector rejected.
            rewritten_query = rewrite_query(
                processing_query, pipeline_lang, engine.llm,
                headers=prev_headers, prev_answer=prev_answer, attempt=attempt,
            )
            print(f"[AnswerGenerator] Rewritten: {rewritten_query}")

        # First pass reuses the cached top-K from lufa_out (no re-retrieval). Since attempt 1
        # performs no rewrite, that cached retrieval is the naive pipeline's retrieval, which
        # is what makes the agentic first pass identical to the naive arm by construction.
        if cached_nodes and attempt == 1:
            nodes = cached_nodes
            print(
                f"[AnswerGenerator] Reusing {len(nodes)} cached top-K chunks from lufa_out (first pass, no re-retrieval).")
        else:
            # Fixed top_k on every attempt. Widening it by (attempt-1) admitted progressively
            # weaker chunks, giving a retry a second way to end up worse than its own first
            # pass and confounding the comparison against the naive arm.
            top_k = engine.similarity_top_k
            nodes = engine._retrieve_nodes(rewritten_query, top_k=top_k)
            if attempt == 1:
                first_retrieval_s = getattr(engine, "_last_retrieval_seconds", "")
            print(f"[AnswerGenerator] Retrieved {len(nodes)} chunks (top_k={top_k})")

        answer = engine._generate_from_nodes(processing_query, nodes, pipeline_lang)
        if attempt == 1:
            first_ttft_s = getattr(engine, "_last_ttft_seconds", "")
        # Context budget chosen for the LAST generation call. top_k is now fixed at 5 on every
        # attempt, so this no longer grows with retries.
        ctx_used = getattr(engine.llm, "_last_context_window", "")
        ptok_used = getattr(engine.llm, "_last_prompt_tokens", "")
        otok_pred = getattr(engine.llm, "_last_predicted_output_tokens", "")

        # Record the attempt count on EVERY iteration, not only when the loop breaks
        # on a grounded answer — otherwise a run that exhausts all retries reports
        # attempts=0 and understates the true cost of the agentic loop.
        real_attempt = attempt

        if check_grounded:
            chunk_texts = [n.node.text for n in nodes]
            is_grounded = reflect(answer, chunk_texts, engine.llm)
            print(f"[AnswerGenerator] Grounded: {is_grounded}")

            if is_grounded:
                break

            # Rejected. Hand the next rewrite what was actually found and what was wrongly
            # said, so it can target a real provision instead of inventing one.
            prev_headers = build_headers(nodes)
            prev_answer = answer
        else:
            is_grounded = False
            real_attempt = attempt
            break

    final_answer = answer
    if translation_applied:
        lang_name = LANGUAGE_NAMES.get(original_lang, original_lang.upper())
        print(f"[AnswerGenerator] Translating answer back to {lang_name}...")
        final_answer = translate_to_target(answer, original_lang, engine.llm)

    # ── Post-hoc rendering of the answer in the ground-truth language ───────────
    # Only for the no-translation cross-lingual run, and only when the answer is not
    # already in the benchmark's language. Used SOLELY by the lexical metrics; the
    # judge and every pipeline stage keep using the native-language answer.
    answer_metrics_translation = ""
    if no_translate and original_lang != metrics_language and final_answer.strip():
        tgt = LANGUAGE_NAMES.get(metrics_language, metrics_language.upper())
        print(f"[AnswerGenerator] Rendering answer in {tgt} for lexical metrics only "
              f"(not used by the judge)...")
        try:
            from translator import TRANSLATE_TO_EN_PROMPT, TRANSLATE_TO_TARGET_PROMPT
            from llm_utils import stream_complete
            src_name = LANGUAGE_NAMES.get(original_lang, original_lang.upper())
            if metrics_language == "en":
                prompt = TRANSLATE_TO_EN_PROMPT.format(source_language=src_name, text=final_answer)
            else:
                prompt = TRANSLATE_TO_TARGET_PROMPT.format(target_language=tgt, text=final_answer)
            answer_metrics_translation = stream_complete(engine.llm, prompt) or ""
        except Exception as _te:
            print(f"[AnswerGenerator] metrics translation failed: {_te}")
            answer_metrics_translation = ""

    result = {
        'response': final_answer,
        'sources': [],
        'detected_language': pipeline_lang,
        'original_language': original_lang,
        'translation_applied': translation_applied,
        'rewritten_query': rewritten_query,
        'attempts': real_attempt,
        'grounded': is_grounded,
        'original_query_id': query_id,
        'original_question': query_text,
        'original_question_translation': translated_query,
        'untranslated_response': answer,
        'retrieval_latency_s': first_retrieval_s,
        'ttft_s': first_ttft_s,
        'end_to_end_latency_s': round(_time.perf_counter() - _e2e_start, 4),
        'context_window_used': ctx_used,
        'prompt_tokens_est': ptok_used,
        'predicted_output_tokens': otok_pred,
        'answer_metrics_translation': answer_metrics_translation,
        'metrics_language': metrics_language if answer_metrics_translation else "",
        'pipeline_translation_mode': "none" if no_translate else "bridge",
    }
    # Hardware utilisation across the whole query (§4.6.4). Blank if unavailable.
    _sysm = _sampler.stop()
    for _k in ("gpu_vram_mb", "gpu_vram_dedicated_mb", "gpu_vram_shared_mb",
               "system_ram_mb", "cpu_percent", "gpu_util_percent"):
        result[_k] = _sysm.get(_k, "")

    if nodes:
        result['sources'] = _extract_sources_from_nodes(nodes)
        result.update(_sources_to_csv_format(result['sources'], engine.similarity_top_k))

    return result


def _sources_to_csv_format(sources: List[Dict], max_sources: int = 5) -> Dict:
    """Convert sources list to CSV format with per-chunk score columns."""
    row = {}
    for i in range(1, max_sources + 1):
        if i <= len(sources):
            s = sources[i - 1]
            row[f'source{i}_id'] = s['node_id']
            row[f'source{i}_cosine_score'] = round(float(s['cosine_score']), 4)
            row[f'source{i}_recency_adjusted_cosine_score'] = round(float(s['recency_adjusted_cosine_score']), 4)
            row[f'source{i}_rrf_score'] = round(float(s['rrf_score']), 4)
            row[f'source{i}_text'] = s['text']
        else:
            row[f'source{i}_id'] = ""
            row[f'source{i}_cosine_score'] = ""
            row[f'source{i}_recency_adjusted_cosine_score'] = ""
            row[f'source{i}_rrf_score'] = ""
            row[f'source{i}_text'] = ""

    return row


def _extract_sources_from_nodes(nodes) -> List[Dict]:
    """Extract sources from NodeWithScore objects with per-chunk scores."""
    sources = []
    for node in nodes:
        cosine = float(node.node.metadata.get('original_cosine_score', node.score))
        recency_weight = float(node.node.metadata.get('recency_weight', 1.0))
        rrf = float(node.score)

        source = {
            'node_id': node.node.node_id,
            'score': rrf,
            'cosine_score': cosine,
            'recency_adjusted_cosine_score': round(cosine * recency_weight, 6),
            'rrf_score': rrf,
            'text': node.node.text[:500] if node.node.text else "",
            'metadata': node.node.metadata or {}
        }
        sources.append(source)

    return sources


def build_cached_nodes(source_row: Dict, max_sources: int = 5):
    """
    Reconstruct NodeWithScore objects from the top-K chunks already stored in a
    lufa_out row, so generation can REUSE them instead of re-retrieving.
    Returns [] when the row has no usable cached sources.
    """
    from llama_index.core.schema import TextNode, NodeWithScore

    def _f(key):
        try:
            return float(source_row.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    nodes = []
    for i in range(1, max_sources + 1):
        sid = str(source_row.get(f"source{i}_id", "") or "").strip()
        text = source_row.get(f"source{i}_text", "")
        text = "" if (text is None or (isinstance(text, float) and pd.isna(text))) else str(text)
        if not sid or sid.lower() in ("nan", "none") or not text.strip():
            continue
        cosine = _f(f"source{i}_cosine_score")
        recency_adj = _f(f"source{i}_recency_adjusted_cosine_score")
        rrf = _f(f"source{i}_rrf_score")
        recency_weight = (recency_adj / cosine) if cosine else 1.0
        node = TextNode(id_=sid, text=text, metadata={
            "original_cosine_score": str(cosine),
            "recency_weight": recency_weight,
        })
        nodes.append(NodeWithScore(node=node, score=rrf))
    return nodes


def generate_answer_record(engine, q_id, q_text, base_model, mode="local",
                           max_retries=1, cached_nodes=None) -> Dict:
    """
    Produce a run_simulation-style lufa_out row (answer + per-chunk sources).
    When `cached_nodes` are supplied they are reused for the first-pass generation
    (no re-retrieval); agentic retries still regenerate the top-K.
    """
    if mode == "local-naive":
        result = generate_naive_answer(engine, q_text, check_grounded=True,
                                       query_id=q_id, cached_nodes=cached_nodes)
    else:
        result = generate_agentic_answer(engine, q_text, max_retries=max_retries,
                                         check_grounded=True, query_id=q_id, cached_nodes=cached_nodes)

    row = {
        "question_id": q_id,
        "question": q_text,
        "answer": result.get("response", ""),
        "base_model_used": base_model,
        "language": result.get("original_language", "en"),
        "attempts": result.get("attempts", 1),
        "grounded": result.get("grounded", False),
    }
    for i in range(1, 6):
        for suf in ["id", "cosine_score", "recency_adjusted_cosine_score", "rrf_score", "text"]:
            row[f"source{i}_{suf}"] = result.get(f"source{i}_{suf}", "")
    from run_simulation import extract_translation_columns, extract_system_metric_columns
    row.update(extract_translation_columns(result))
    row.update(extract_system_metric_columns(
        result, "local",
        e2e_latency=result.get("end_to_end_latency_s", ""),
        warmup_applied="",
        sysm={k: result.get(k, "") for k in
              ("gpu_vram_mb", "gpu_vram_dedicated_mb", "gpu_vram_shared_mb",
               "system_ram_mb", "cpu_percent", "gpu_util_percent")}))
    return row


def naive_rag(engine, query_text: str, max_sources: int = 5,
              check_grounded: bool = True,
              output_path: Union[str, Path] = "tests/lufa_out_data.csv", query_id: str = None,
              cached_nodes=None) -> Dict:
    """High-level function for naive RAG processing."""
    return generate_naive_answer(engine, query_text, max_sources, check_grounded, output_path,
                                 query_id, cached_nodes=cached_nodes)


def agentic_rag(engine, query_text: str, max_retries: int = 3,
                check_grounded: bool = True,
                output_path: Union[str, Path] = "tests/lufa_out_data.csv", query_id: str = None,
                cached_nodes=None) -> Dict:
    """High-level function for agentic RAG processing."""
    return generate_agentic_answer(engine, query_text, max_retries, check_grounded, output_path,
                                   query_id, cached_nodes=cached_nodes)


if __name__ == "__main__":
    import argparse
    import sys
    import time
    import traceback
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    parser = argparse.ArgumentParser(description="Generate RAG answers for LUFA test questions")
    parser.add_argument("--input", default="tests/combined_test_data_and_ground_truth.csv",
                        help="Input CSV with questions (must have 'id' and 'question' columns)")
    parser.add_argument("--output", default="tests/lufa_out_data.csv",
                        help="Output CSV path (same schema as run_simulation.py)")
    parser.add_argument("--mode", choices=["local", "local-naive"], default="local",
                        help="local = agentic query, local-naive = single-pass query")
    parser.add_argument("--max_retries", type=int, default=1,
                        help="Max retry attempts for agentic mode")
    parser.add_argument("--llm_model", default=None,
                        help="Supply one of many models to use")
    parser.add_argument("--resume_regen", action="store_true",
                        help="Resume an interrupted --force regeneration: rows that already carry "
                             "generation telemetry (ttft_s) are skipped, everything else is "
                             "regenerated. Use this instead of --force after a crash.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate answers for EVERY question, including ones that already "
                             "have an answer (needed to capture ttft/latency/VRAM on a complete CSV).")
    parser.add_argument("--no_dashboard", action="store_true",
                        help="Skip the per-row dashboard refresh (much faster on long runs).")
    parser.add_argument("--no_translate", action="store_true",
                        help="TRUE cross-lingual mode: never bridge through English. Retrieve with "
                             "the raw query and answer in the question's own language. A post-hoc "
                             "translation into --metrics_language is stored for lexical metrics only.")
    parser.add_argument("--metrics_language", default="en",
                        help="Language of the benchmark's expected_answer / ground_source_truth, "
                             "used only to render the answer for lexical metrics (default: en).")


    def _str2bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


    parser.add_argument("--openai_client", type=_str2bool, nargs="?", const=True, default=False,
                        help="Use an OpenAI-compatible endpoint for the LLM instead of Ollama "
                             "(e.g. --openai_client True for cloud models like claude-haiku-4-5)")

    args = parser.parse_args()

    from rag_engine import create_rag_engine
    from config_loader import cfg
    from run_simulation import (OUTPUT_COLUMNS, extract_translation_columns,
                                extract_system_metric_columns, SYSTEM_METRIC_COLUMNS,
                                CROSSLINGUAL_COLUMNS)
    from csv_utils import upsert_row

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[AnswerGenerator] Loading questions from {input_path}...")
    questions_df = pd.read_csv(input_path)
    print(f"[AnswerGenerator] {len(questions_df)} questions loaded.")

    completed_ids = set()
    cached_map = {}
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(output_path)
            if "question_id" in existing_df.columns:
                # Cache the top-K already retrieved for each question (e.g. written
                # by retrieval.py) so generation can reuse it instead of re-retrieving.
                for _, r in existing_df.iterrows():
                    qid = str(r.get("question_id", "")).strip()
                    if qid:
                        cached_map[qid] = r.to_dict()

                if "answer" in existing_df.columns:
                    clean_ans = existing_df["answer"].astype(str).str.strip()
                    clean_ground = existing_df["grounded"].astype(str).str.strip().str.lower()

                    # Apply conditions cleanly
                    successful = existing_df[(
                            existing_df["answer"].notna() & (clean_ans != "") & (clean_ans != "ERROR")
                         & ((args.mode != "local") | (int(args.max_retries) < 2) | (clean_ground == "true"))
                    )]

                    completed_ids = set(successful["question_id"].dropna().astype(str).tolist())
            if args.resume_regen:
                # Correct resume semantics for an interrupted --force run. After a crash,
                # every row still holds an answer (the pre-existing one for rows not yet
                # reached), so plain resume would skip everything while --force would
                # restart all 426. A row counts as done only when THIS run measured it —
                # i.e. it has generation telemetry (ttft_s) and a usable answer.
                tel = (existing_df["ttft_s"].astype(str).str.strip()
                       if "ttft_s" in existing_df.columns else pd.Series("", index=existing_df.index))
                measured = existing_df[
                    tel.ne("") & ~tel.str.lower().isin(["nan", "none"])
                    & existing_df["answer"].notna()
                    & existing_df["answer"].astype(str).str.strip().ne("")
                    & existing_df["answer"].astype(str).str.strip().ne("ERROR")
                ]
                completed_ids = set(measured["question_id"].dropna().astype(str).tolist())
                print(f"[AnswerGenerator] --resume_regen: {len(completed_ids)} rows already "
                      f"regenerated (have ttft_s); the rest WILL be regenerated.")
            elif args.force:
                print(f"[AnswerGenerator] --force set: ignoring {len(completed_ids)} existing answers "
                      f"— ALL questions will be regenerated (answers WILL be overwritten).")
                completed_ids = set()
            print(f"[AnswerGenerator] Resuming — {len(completed_ids)} answered, "
                  f"{len(cached_map)} questions with cached top-K available.")
        except Exception as e:
            print(f"[AnswerGenerator] Warning: could not read existing output: {e}")

    if args.llm_model is None:
        base_model = cfg("models.llm.name")
    else:
        base_model = args.llm_model
    print(f"[AnswerGenerator] Mode: {args.mode} | Model: {base_model}")
    print("[AnswerGenerator] Initializing RAG engine...")
    engine = create_rag_engine(None, base_model, None, None, openai_client=args.openai_client)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Warm-up protocol (Ch4 §4.6.4): warm the retrieval path once on the first
    # question actually processed; that row is tagged warmup_applied=1.
    warmed_up = False

    for idx, row in questions_df.iterrows():
        q_id = str(row.get("id", idx)).strip()
        q_text = str(row.get("question", "")).strip()
        counter = f"[{idx + 1}/{len(questions_df)}]"

        if q_id in completed_ids:
            print(f"{counter} Skipping {q_id} (already answered)")
            continue

        warmup_flag = ""
        if not warmed_up:
            print(f"{counter} [Warmup] Priming retrieval on first question (timing discarded)...")
            try:
                engine.warmup_retrieve(q_text)
                warmup_flag = 1
            except Exception as _we:
                print(f"   [Warmup] skipped: {_we}")
            warmed_up = True

        print(f"\n{counter} Processing question {q_id}: \"{q_text[:60]}...\"")
        try:
            cached_nodes = build_cached_nodes(cached_map.get(q_id, {}))
            if cached_nodes:
                print(
                    f"   Found {len(cached_nodes)} cached top-K chunks for {q_id} — will reuse for first-pass generation.")
            if args.mode == "local-naive":
                result = generate_naive_answer(engine, q_text, top_k=5,
                                               check_grounded=False, output_path=args.output,
                                               query_id=q_id, cached_nodes=cached_nodes,
                                               no_translate=args.no_translate,
                                               metrics_language=args.metrics_language)
            else:
                result = generate_agentic_answer(engine, q_text, max_retries=args.max_retries,
                                                 check_grounded=True, output_path=args.output,
                                                 query_id=q_id, cached_nodes=cached_nodes,
                                                 no_translate=args.no_translate,
                                                 metrics_language=args.metrics_language)

            sources = result.get("sources", [])
            source_cols = {}
            is_valid_response = result.get("response", "").strip() not in ("", "ERROR")
            # Fallback source MUST be the previously-stored lufa row (written by
            # retrieval.py), NOT the input questions row — the input CSV has no
            # source* columns, so using it would blank out good retrieval data.
            prev_lufa = cached_map.get(q_id, {})
            for i in range(1, 6):
                if i <= len(sources) and is_valid_response:
                    s = sources[i - 1]
                    source_cols[f"source{i}_id"] = s.get("node_id", "")
                    source_cols[f"source{i}_cosine_score"] = round(float(s.get("cosine_score", 0)), 4)
                    source_cols[f"source{i}_recency_adjusted_cosine_score"] = round(
                        float(s.get("recency_adjusted_cosine_score", 0)), 4)
                    source_cols[f"source{i}_rrf_score"] = round(float(s.get("rrf_score", 0)), 4)
                    source_cols[f"source{i}_text"] = str(s.get("text", ""))
                else:
                    for suf in ("id", "cosine_score", "recency_adjusted_cosine_score",
                                "rrf_score", "text"):
                        v = prev_lufa.get(f"source{i}_{suf}", "")
                        source_cols[f"source{i}_{suf}"] = "" if (
                            v is None or (not isinstance(v, (list, dict)) and pd.isna(v))) else v

            out_row = {
                "question_id": q_id,
                "question": q_text,
                "answer": result.get("response", ""),
                "base_model_used": base_model,
                "language": result.get("original_language", row.get("language", "en")),
                "attempts": result.get("attempts", 1),
                "grounded": result.get("grounded", False),
                **source_cols,
                **extract_translation_columns(result),
                **extract_system_metric_columns(
                    result, "local",
                    e2e_latency=result.get("end_to_end_latency_s", ""),
                    warmup_applied=warmup_flag,
                    sysm={k: result.get(k, "") for k in
                          ("gpu_vram_mb", "gpu_vram_dedicated_mb", "gpu_vram_shared_mb",
                           "system_ram_mb", "cpu_percent", "gpu_util_percent")}),
                **{c: result.get(c, "") for c in CROSSLINGUAL_COLUMNS},
            }

            # Never clobber a telemetry value measured by an earlier batch with a blank.
            # The first pass reuses retrieval.py's cached top-K, so no retrieval happens
            # here and retrieval_latency_s/warmup_applied come back empty — dropping the
            # blank keys makes upsert_row preserve what batch 1 recorded.
            for _tk in SYSTEM_METRIC_COLUMNS:
                if str(out_row.get(_tk, "")).strip() == "":
                    out_row.pop(_tk, None)

            # Update the existing row in place (or append if new) so grounded,
            # attempts, answer and all source{n}_* columns are refreshed per row.
            upsert_row(out_row, output_path, OUTPUT_COLUMNS, key_cols=("question_id",))
            print(f"   Answer length: {len(out_row['answer'])} chars | Grounded: {out_row['grounded']} — saved.")
            if not args.no_dashboard:
                try:
                    refresh_dashboard(lufa_csv=str(output_path))
                except Exception as _de:
                    print(f"   [Dashboard] refresh skipped: {_de}")
        except Exception as e:
            print(f"   Error on {q_id}: {e}")
            traceback.print_exc()
            # Record the failure WITHOUT destroying retrieval or telemetry already on
            # this row: only the generation-owned fields are supplied to upsert_row.
            empty = {"question_id": q_id, "question": q_text, "answer": "ERROR",
                     "base_model_used": base_model, "language": row.get("language", "en"),
                     "attempts": 0, "grounded": False}
            upsert_row(empty, output_path, OUTPUT_COLUMNS, key_cols=("question_id",))
        time.sleep(0.5)
    try:
        refresh_dashboard(lufa_csv=str(output_path))
        print(f"   [Dashboard] refreshed.")
    except Exception as _de:
        print(f"   [Dashboard] final refresh skipped: {_de}")
    print(f"\n[AnswerGenerator] Done. Results in {output_path}")
