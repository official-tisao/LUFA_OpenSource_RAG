#!/usr/bin/env python3
"""
Question rewriter module for LUFA RAG system.
Handles single and batch question rewriting with both small and large batch modes.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
from copy import deepcopy

from src.csv_utils import read_csv_cached, write_csv_row, get_completed_ids
from src.query_rewriter import rewrite_single_question
def rewrite_single_question_standalone(query: str, lang: str, llm) -> str:
    """Standalone version of rewrite_single_question from query_rewriter module."""
    return rewrite_single_question(query, lang, llm)
def rewrite_batch_prompt(questions_list: List[str], lang: str) -> str:
    """
    Create a batch rewrite prompt for multiple questions.

    Args:
        questions_list: List of questions to rewrite
        lang: Language code ('en' or 'fr')

    Returns:
        Formatted batch prompt string
    """
    if len(questions_list) == 0:
        return ""

    prompt = f"""Rewrite the following {len(questions_list)} questions to be more specific and precise so they retrieve the most relevant clauses or articles from the agreement. Output ONLY the rewritten queries — do not answer them or explain your work.

"""

    for i, question in enumerate(questions_list, 1):
        prompt += f"{i}. {question}\n"

    prompt += f"\nFor each, output in format:\n"

    for i in range(1, len(questions_list) + 1):
        prompt += f"{i}. {{{{rewritten_{i}}}}}\n"

    return prompt
def parse_batch_rewrite_response(response: str, expected_count: int) -> List[str]:
    """
    Parse LLM response for batch rewrite.

    Args:
        response: LLM response text
        expected_count: Expected number of rewritten questions

    Returns:
        List of rewritten questions
    """
    rewritten = []
    lines = response.strip().split('\n')

    # Look for numbered lines with pattern like "1. <rewritten question>"
    for i, line in enumerate(lines, 1):
        line = line.strip()
        # Match "1. <text>" pattern
        if line.startswith(f"{i}."):
            rewritten_text = line.split('.', 1)[1].strip()
            rewritten.append(rewritten_text)

    # If parsing failed (e.g., unexpected format), fallback to line-by-line approach
    if len(rewritten) == 0 and expected_count > 0:
        # Try to extract any line that looks like a rewritten question
        # This is a simple fallback - in production you'd want more sophisticated parsing
        for line in lines:
            line = line.strip()
            if line and not line.startswith('{') and not line.startswith('['):
                # Avoid action words that indicate incomplete responses
                if not any(action in line.lower() for action in ['rewrite', 'rewrite query', 'original query']):
                    rewritten.append(line)

    # If still no results and expected_count > 0, return empty list with warning
    if len(rewritten) == 0 and expected_count > 0:
        print(f"      [QuestionRewriter] Warning: Could not parse batch rewrite response. Expected {expected_count} questions, got {len(rewritten)}")

    return rewritten[:expected_count]  # Return only expected number
def rewrite_single_question_standalone(query: str, lang: str, llm) -> str:
    """Standalone version of rewrite_single_question from query_rewriter module."""
    from src.query_rewriter import rewrite_query
    return rewrite_query(query, lang, llm)
def rewrite_single_question_standalone(query: str, lang: str, llm) -> str:
    """Standalone version of rewrite_single_question from query_rewriter module."""
    return query  # Simplified fallback for now

def rewrite_single_question_batch(question_df: pd.DataFrame,
                                 engine,
                                 llm,
                                 output_path: Union[str, Path] = "tests/lufa_out_data.csv") -> pd.DataFrame:
    """
    Rewrite questions one at a time (small batch mode) and save to CSV.

    Args:
        question_df: DataFrame with question data
        engine: RAG engine instance
        llm: LLM instance
        output_path: Path to output CSV file

    Returns:
        pd.DataFrame: DataFrame with rewritten questions
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if file exists
    existing_df = read_csv_cached(path) if path.exists() else pd.DataFrame()

    # Get completed question IDs
    completed_ids = get_completed_ids(existing_df) if not existing_df.empty else set()

    # Prepare result DataFrame
    result_rows = []
    processed_count = 0

    for idx, row in question_df.iterrows():
        q_id = str(row.get("id")).strip()

        if q_id in completed_ids:
            print(f"      [QuestionRewriter] Skipping question {q_id} (already processed)")
            # Try to get existing rewritten question
            if not existing_df.empty:
                existing_row = existing_df[existing_df["question_id"] == q_id]
                if not existing_row.empty:
                    result_rows.append(existing_row.iloc[0].to_dict())
            continue

        question_text = row.get("question", "")
        lang = row.get("language", "en")

        print(f"      [QuestionRewriter] Processing question {q_id}: {question_text[:65]}...")

        try:
            # Rewrite using single question function (from query_rewriter.py)
            rewritten_question = rewrite_single_question_standalone(question_text, lang, llm)

            print(f"      [QuestionRewriter] Rewritten: {rewritten_question[:65]}...")

            # Create result row
            result_row = {
                "question_id": q_id,
                "question": question_text[:500] if len(question_text) > 500 else question_text,
                "question_rewrite": rewritten_question
            }

            # Add other columns from original row
            for col in question_df.columns:
                if col not in ["id", "question"] and col in row:
                    result_row[col] = row[col]

            result_rows.append(result_row)
            processed_count += 1

        except Exception as e:
            print(f"      [QuestionRewriter Error] Failed to process question {q_id}: {e}")
            # Create error row
            error_row = {
                "question_id": q_id,
                "question": question_text[:500] if "question" in row else "",
                "question_rewrite": ""
            }
            result_rows.append(error_row)

    # Create result DataFrame
    result_df = pd.DataFrame(result_rows)

    # Save to CSV (append if file exists)
    try:
        file_exists = path.exists() and path.stat().st_size > 0
        mode = 'a' if file_exists else 'w'
        header = not file_exists

        result_df.to_csv(path, mode=mode, header=header,
                        index=False, encoding='utf-8')

        print(f"      [QuestionRewriter] Saved {processed_count} rewritten questions to {path}")
    except Exception as e:
        print(f"      [QuestionRewriter Error] Failed to save results: {e}")

    return result_df
def rewrite_questions_batch(questions_df: pd.DataFrame,
                             engine,
                             llm,
                             batch_size: int = 10,
                             use_combined_judge: bool = False,
                             output_path: Union[str, Path] = "tests/lufa_out_data.csv") -> pd.DataFrame:
    """
    Rewrite questions in large batches (N questions in single LLM call).

    Args:
        questions_df: DataFrame with question data
        engine: RAG engine instance (for potential future use)
        llm: LLM instance
        batch_size: Number of questions to process in each batch
        use_combined_judge: Whether to use combined judge prompt format
        output_path: Path to output CSV file

    Returns:
        pd.DataFrame: DataFrame with rewritten questions
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if file exists
    existing_df = read_csv_cached(path) if path.exists() else pd.DataFrame()

    # Get completed question IDs
    completed_ids = get_completed_ids(existing_df) if not existing_df.empty else set()

    # Prepare result DataFrame
    result_rows = []
    processed_count = 0

    # Process in batches
    for batch_start in range(0, len(questions_df), batch_size):
        batch_df = questions_df.iloc[batch_start:batch_start + batch_size]

        print(f"      [QuestionRewriter] Processing batch {batch_start//batch_size + 1} with {len(batch_df)} questions")

        # Collect questions for this batch
        batch_questions = []
        batch_question_ids = []
        batch_langs = []

        for idx, row in batch_df.iterrows():
            q_id = str(row.get("id")).strip()

            if q_id in completed_ids:
                print(f"      [QuestionRewriter] Skipping question {q_id} (already processed)")
                continue

            question_text = row.get("question", "")
            lang = row.get("language", "en")

            batch_questions.append(question_text)
            batch_question_ids.append(q_id)
            batch_langs.append(lang)

        if not batch_questions:
            continue

        # Create batch prompt
        batch_prompt = rewrite_batch_prompt(batch_questions, batch_langs[0] if batch_langs else "en")

        print(f"      [QuestionRewriter] Sending {len(batch_questions)} questions in single prompt")

        try:
            # Call LLM to rewrite all questions
            response = llm.complete(batch_prompt)

            # Parse the response
            rewritten_questions = parse_batch_rewrite_response(
                str(response).strip(), len(batch_questions)
            )

            # If parsing failed, try one-by-one as fallback
            if len(rewritten_questions) == 0 and len(batch_questions) > 0:
                print(f"      [QuestionRewriter] Batch parsing failed, falling back to one-by-one")
                rewritten_questions = []
                for i, question in enumerate(batch_questions):
                    lang = batch_langs[i] if i < len(batch_langs) else "en"
                    # Use the simplified standalone version
                    rewritten = rewrite_single_question_standalone(question, lang, llm)
                    rewritten_questions.append(rewritten)

            # Create result rows
            for i, q_id in enumerate(batch_question_ids):
                rewritten = rewritten_questions[i] if i < len(rewritten_questions) else ""

                result_row = {
                    "question_id": q_id,
                    "question": batch_questions[i][:500] if len(batch_questions[i]) > 500 else batch_questions[i],
                    "question_rewrite": rewritten
                }

                # Add other columns from original row
                original_row = batch_df.iloc[i] if i < len(batch_df) else None
                if original_row is not None:
                    for col in batch_df.columns:
                        if col not in ["id", "question"] and col in original_row:
                            result_row[col] = original_row[col]

                result_rows.append(result_row)
                processed_count += 1

        except Exception as e:
            print(f"      [QuestionRewriter Error] Failed to process batch: {e}")

            # Create error rows for this batch
            for i, q_id in enumerate(batch_question_ids):
                error_row = {
                    "question_id": q_id,
                    "question": batch_questions[i][:500] if i < len(batch_questions) else "",
                    "question_rewrite": ""
                }
                result_rows.append(error_row)

        print(f"      [QuestionRewriter] Completed batch, processed {processed_count} total questions")

    # Create result DataFrame
    result_df = pd.DataFrame(result_rows)

    # Save to CSV (append if file exists)
    try:
        file_exists = path.exists() and path.stat().st_size > 0
        mode = 'a' if file_exists else 'w'
        header = not file_exists

        result_df.to_csv(path, mode=mode, header=header,
                        index=False, encoding='utf-8')

        print(f"      [QuestionRewriter] Saved {processed_count} rewritten questions to {path}")
    except Exception as e:
        print(f"      [QuestionRewriter Error] Failed to save results: {e}")

    return result_df
def rewrite_questions_dataframe(questions_df: pd.DataFrame,
                                 engine,
                                 llm,
                                 use_combined_judge: bool = False,
                                 output_path: Union[str, Path] = "tests/lufa_out_data.csv") -> pd.DataFrame:
    """
    High-level function to rewrite questions from DataFrame (alternative to CSV version).

    Args:
        questions_df: DataFrame with question data
        engine: RAG engine instance (for potential future use)
        llm: LLM instance
        use_combined_judge: Whether to use combined judge prompt format
        output_path: Path to output CSV file

    Returns:
        pd.DataFrame: DataFrame with rewritten questions
    """
    # Determine batch size based on questions count
    batch_size = min(10, len(questions_df))

    return rewrite_questions_batch(questions_df, engine, llm,
                                  batch_size=batch_size,
                                  use_combined_judge=use_combined_judge,
                                  output_path=output_path)

def rewrite_questions_from_csv(input_path: Union[str, Path],
                              engine,
                              llm,
                              batch_size: int = 10,
                              use_combined_judge: bool = False,
                              output_path: Union[str, Path] = "tests/lufa_out_data.csv") -> pd.DataFrame:
    """
    High-level function to rewrite questions from CSV (wrapper for backward compatibility).

    Args:
        input_path: Path to input CSV with questions
        engine: RAG engine instance
        llm: LLM instance
        batch_size: Number of questions to process in each batch
        use_combined_judge: Whether to use combined judge prompt format
        output_path: Path to output CSV file

    Returns:
        pd.DataFrame: DataFrame with rewritten questions
    """
    df = pd.read_csv(input_path)
    return rewrite_questions_dataframe(df, engine, llm,
                                     use_combined_judge=use_combined_judge,
                                     output_path=output_path)