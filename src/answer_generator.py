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

from csv_utils import read_csv_cached, write_csv_row, get_completed_ids
from metrics import token_f1

def check_language(input_string):
    if "_en" in input_string:
        return "en"
    elif "_fr" in input_string:
        return "fr"
    return None
def generate_naive_answer(engine,
                         query_text: str,
                         top_k: int = 5,
                         check_grounded: bool = False,
                         output_path: Union[str, Path] = "tests/lufa_out_data.csv", query_id: str = None,
                         cached_nodes=None) -> Dict:
    """Generate answer using naive RAG approach (single retrieval + generation)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if cached_nodes:
        nodes = cached_nodes
        print(f"[AnswerGenerator] Reusing {len(nodes)} cached top-K chunks from lufa_out (no re-retrieval).")
    else:
        nodes = engine._retrieve_nodes(query_text, top_k=top_k)

    sources = _extract_sources_from_nodes(nodes)

    #from language_detector import detect_full_language
    #original_lang = detect_full_language(query_text)
    original_lang = check_language(query_id) if query_id else "en"
    from translator import needs_translation, translate_to_english, translate_to_target, LANGUAGE_NAMES

    translation_applied = needs_translation(original_lang)
    if translation_applied:
        print(f"[AnswerGenerator] Translating query to English for processing...")
        query_for_processing = translate_to_english(query_text, original_lang, engine.llm)
        pipeline_lang = "en"
    else:
        query_for_processing = query_text
        pipeline_lang = original_lang

    answer = engine._generate_from_nodes(query_for_processing, nodes, pipeline_lang)

    final_answer = answer
    if translation_applied:
        lang_name = LANGUAGE_NAMES.get(original_lang, original_lang.upper())
        print(f"[AnswerGenerator] Translating answer back to {lang_name}...")
        final_answer = translate_to_target(answer, original_lang, engine.llm)

    is_grounded = False
    if check_grounded:
        from reflector import reflect
        chunk_texts = [n.node.text for n in nodes]
        is_grounded = reflect(final_answer, chunk_texts, engine.llm)

    result = {
        'response': final_answer,
        'sources': sources,
        'detected_language': pipeline_lang,
        'original_language': original_lang,
        'translation_applied': translation_applied,
        'grounded': is_grounded,
        'attempts': 1,
    }

    result.update(_sources_to_csv_format(sources, top_k))

    return result


def generate_agentic_answer(engine,
                               query_text: str,
                               max_retries: int = 3,
                               check_grounded: bool = True,
                               output_path: Union[str, Path] = "tests/lufa_out_data.csv", query_id: str = None,
                               cached_nodes=None) -> Dict:
    """Generate answer using agentic RAG approach (retrieval + retry loop with groundedness checking)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    #from language_detector import detect_full_language
    from translator import needs_translation, translate_to_english, translate_to_target, LANGUAGE_NAMES
    from query_rewriter import rewrite_query
    from reflector import reflect

    #from language_detector import detect_full_language
    original_lang = (check_language(query_id) if query_id else check_language(query_text)) or "en"
    translation_applied = needs_translation(original_lang)

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

    current_query = processing_query
    rewritten_query = processing_query
    nodes = []
    answer = ""
    is_grounded = False

    for attempt in range(1, max_retries + 1):
        print(f"[AnswerGenerator] Attempt {attempt}/{max_retries}")

        if attempt > 1:
            rewritten_query = rewrite_query(current_query, pipeline_lang, engine.llm)
            print(f"[AnswerGenerator] Rewritten: {rewritten_query}")

        # First pass reuses the cached top-K from lufa_out (no re-retrieval).
        # Retries (attempt > 1) regenerate the top-K by re-retrieving with a wider k.
        if cached_nodes and attempt == 1:
            nodes = cached_nodes
            print(f"[AnswerGenerator] Reusing {len(nodes)} cached top-K chunks from lufa_out (first pass, no re-retrieval).")
        else:
            top_k = engine.similarity_top_k + (attempt - 1)
            nodes = engine._retrieve_nodes(rewritten_query, top_k=top_k)
            print(f"[AnswerGenerator] Retrieved {len(nodes)} chunks (top_k={top_k})")

        answer = engine._generate_from_nodes(processing_query, nodes, pipeline_lang)

        if check_grounded:
            chunk_texts = [n.node.text for n in nodes]
            is_grounded = reflect(answer, chunk_texts, engine.llm)
            print(f"[AnswerGenerator] Grounded: {is_grounded}")

            if is_grounded:
                break

            current_query = rewritten_query
        else:
            is_grounded = False
            break

    final_answer = answer
    if translation_applied:
        lang_name = LANGUAGE_NAMES.get(original_lang, original_lang.upper())
        print(f"[AnswerGenerator] Translating answer back to {lang_name}...")
        final_answer = translate_to_target(answer, original_lang, engine.llm)

    result = {
        'response': final_answer,
        'sources': [],
        'detected_language': pipeline_lang,
        'original_language': original_lang,
        'translation_applied': translation_applied,
        'rewritten_query': rewritten_query,
        'attempts': attempt,
        'grounded': is_grounded,
    }

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
                           max_retries=3, cached_nodes=None) -> Dict:
    """
    Produce a run_simulation-style lufa_out row (answer + per-chunk sources).
    When `cached_nodes` are supplied they are reused for the first-pass generation
    (no re-retrieval); agentic retries still regenerate the top-K.
    """
    if mode == "local-naive":
        result = generate_naive_answer(engine, q_text, check_grounded=False,
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
    return row


def naive_rag(engine, query_text: str, max_sources: int = 5,
              check_grounded: bool = False,
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
    args = parser.parse_args()

    from rag_engine import create_rag_engine
    from config_loader import cfg
    from run_simulation import OUTPUT_COLUMNS
    from csv_utils import align_and_append

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
                    successful = existing_df[
                        existing_df["answer"].notna() &
                        (existing_df["answer"].astype(str).str.strip() != "") &
                        (existing_df["answer"].astype(str).str.strip() != "ERROR")
                    ]
                    completed_ids = set(successful["question_id"].dropna().astype(str).tolist())
            print(f"[AnswerGenerator] Resuming — {len(completed_ids)} answered, "
                  f"{len(cached_map)} questions with cached top-K available.")
        except Exception as e:
            print(f"[AnswerGenerator] Warning: could not read existing output: {e}")

    base_model = cfg("models.llm.name")
    print(f"[AnswerGenerator] Mode: {args.mode} | Model: {base_model}")
    print("[AnswerGenerator] Initializing RAG engine...")
    engine = create_rag_engine()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, row in questions_df.iterrows():
        q_id = str(row.get("id", idx)).strip()
        q_text = str(row.get("question", "")).strip()
        counter = f"[{idx + 1}/{len(questions_df)}]"

        if q_id in completed_ids:
            print(f"{counter} Skipping {q_id} (already answered)")
            continue

        print(f"\n{counter} Processing question {q_id}: \"{q_text[:60]}...\"")
        try:
            cached_nodes = build_cached_nodes(cached_map.get(q_id, {}))
            if cached_nodes:
                print(f"   Found {len(cached_nodes)} cached top-K chunks for {q_id} — will reuse for first-pass generation.")
            if args.mode == "local-naive":
                result = generate_naive_answer(engine, q_text, top_k=5,
                                               check_grounded=False, output_path=args.output,
                                               query_id=q_id, cached_nodes=cached_nodes)
            else:
                result = generate_agentic_answer(engine, q_text, max_retries=args.max_retries,
                                                 check_grounded=True, output_path=args.output,
                                                 query_id=q_id, cached_nodes=cached_nodes)

            sources = result.get("sources", [])
            source_cols = {}
            for i in range(1, 6):
                if i <= len(sources):
                    s = sources[i - 1]
                    source_cols[f"source{i}_id"] = s.get("node_id", "")
                    source_cols[f"source{i}_cosine_score"] = round(float(s.get("cosine_score", 0)), 4)
                    source_cols[f"source{i}_recency_adjusted_cosine_score"] = round(float(s.get("recency_adjusted_cosine_score", 0)), 4)
                    source_cols[f"source{i}_rrf_score"] = round(float(s.get("rrf_score", 0)), 4)
                    source_cols[f"source{i}_text"] = str(s.get("text", ""))[:500]
                else:
                    source_cols[f"source{i}_id"] = ""
                    source_cols[f"source{i}_cosine_score"] = ""
                    source_cols[f"source{i}_recency_adjusted_cosine_score"] = ""
                    source_cols[f"source{i}_rrf_score"] = ""
                    source_cols[f"source{i}_text"] = ""

            out_row = {
                "question_id": q_id,
                "question": q_text,
                "answer": result.get("response", ""),
                "base_model_used": base_model,
                "language": result.get("original_language", row.get("language", "en")),
                "attempts": result.get("attempts", 1),
                "grounded": result.get("grounded", False),
                **source_cols,
            }

            align_and_append(out_row, output_path, OUTPUT_COLUMNS)
            print(f"   Answer length: {len(out_row['answer'])} chars | Grounded: {out_row['grounded']} — appended.")
            try:
                from dashboard_generator import refresh_dashboard
                refresh_dashboard(lufa_csv=str(output_path))
            except Exception as _de:
                print(f"   [Dashboard] refresh skipped: {_de}")
        except Exception as e:
            print(f"   Error on {q_id}: {e}")
            traceback.print_exc()
            empty = {"question_id": q_id, "question": q_text, "answer": "ERROR",
                     "base_model_used": base_model, "language": row.get("language", "en"),
                     "attempts": 0, "grounded": False}
            for i in range(1, 6):
                empty[f"source{i}_id"] = ""
                empty[f"source{i}_cosine_score"] = ""
                empty[f"source{i}_recency_adjusted_cosine_score"] = ""
                empty[f"source{i}_rrf_score"] = ""
                empty[f"source{i}_text"] = ""
            align_and_append(empty, output_path, OUTPUT_COLUMNS)
        time.sleep(0.5)

    print(f"\n[AnswerGenerator] Done. Results in {output_path}")
