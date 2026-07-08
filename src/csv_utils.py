#!/usr/bin/env python3
"""
CSV utility functions for LUFA RAG system.
Provides CSV I/O with caching, error handling, and both
path-based and DataFrame-returning variants.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

# Simple cache for file metadata
_file_cache = {}
def read_csv_cached(path: Union[str, Path],
                     encoding: str = "utf-8",
                     on_bad_lines: str = "skip") -> pd.DataFrame:
    """
    Read CSV file with caching and error handling.

    Args:
        path: Path to CSV file
        encoding: File encoding
        on_bad_lines: How to handle bad lines ('skip', 'error')

    Returns:
        pandas.DataFrame
    """
    path = str(path)

    # Check cache first
    cache_key = (path, encoding, on_bad_lines)
    if cache_key in _file_cache:
        cached_data, timestamp = _file_cache[cache_key]
        # Cache for 5 minutes to avoid stale reads
        if datetime.now().timestamp() - timestamp < 300:
            return cached_data

    try:
        df = pd.read_csv(path, encoding=encoding, on_bad_lines=on_bad_lines)

        # Update cache
        _file_cache[cache_key] = (df, datetime.now().timestamp())

        return df
    except Exception as e:
        print(f"      [CSV Read Error] Failed to read {path}: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame()
def write_csv_row(row_dict: Dict,
                   path: Union[str, Path],
                   columns: Optional[List[str]] = None,
                   mode: str = "a",
                   encoding: str = "utf-8") -> bool:
    """
    Write a single row to CSV file with header management.

    Args:
        row_dict: Dictionary containing row data
        path: Path to CSV file
        columns: Column order (if None, use row_dict keys)
        mode: Write mode ('a' for append, 'w' for overwrite)
        encoding: File encoding

    Returns:
        bool: True if successful, False otherwise
    """
    path = Path(path)
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # When a canonical schema is supplied and we're appending, delegate to the
        # schema-safe appender so values can never land under the wrong header.
        if columns and mode == 'a':
            return align_and_append(row_dict, path, list(columns), encoding=encoding)

        # Determine if file exists and has content
        file_exists = path.exists() and path.stat().st_size > 0

        # Convert to DataFrame
        df = pd.DataFrame([row_dict])

        # Reorder columns if specified
        if columns:
            # Keep existing columns plus new ones
            existing_cols = df.columns.tolist()
            final_columns = list(columns) + [c for c in existing_cols if c not in columns]
            df = df[final_columns]

        # Write to CSV
        write_mode = 'a' if (mode == 'a' and file_exists) else 'w'
        header = not file_exists  # Write header only if file doesn't exist

        df.to_csv(path, mode=write_mode, header=header,
                 index=False, encoding=encoding)

        return True
    except Exception as e:
        print(f"      [CSV Write Error] Failed to write to {path}: {e}")
        return False
def ensure_columns(df: pd.DataFrame,
                    required_columns: List[str]) -> pd.DataFrame:
    """
    Ensure DataFrame has all required columns, adding missing ones with empty values.

    Args:
        df: Input DataFrame
        required_columns: List of column names that must exist

    Returns:
        pd.DataFrame: DataFrame with all required columns
    """
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    return df
def get_completed_ids(df: pd.DataFrame,
                      id_column: str = "question_id") -> set:
    """
    Extract set of completed question IDs from DataFrame.
    Filters for non-null and non-empty IDs.

    Args:
        df: Input DataFrame
        id_column: Name of ID column to check

    Returns:
        set: Set of completed question IDs
    """
    if id_column not in df.columns:
        return set()

    # Filter out null and empty string IDs
    ids = df[id_column].dropna().astype(str).str.strip()
    return set(ids[ids != ""].tolist())
def read_csv_df(path: Union[str, Path],
               encoding: str = "utf-8",
               on_bad_lines: str = "skip") -> pd.DataFrame:
    """
    Read CSV file and return DataFrame (alternative to read_csv_cached).

    Args:
        path: Path to CSV file
        encoding: File encoding
        on_bad_lines: How to handle bad lines

    Returns:
        pandas.DataFrame
    """
    return read_csv_cached(path, encoding, on_bad_lines)
def write_csv_df(df: pd.DataFrame,
                path: Union[str, Path],
                encoding: str = "utf-8",
                mode: str = "a") -> bool:
    """
    Write DataFrame to CSV (alternative to write_csv_row).

    Args:
        df: DataFrame to write
        path: Path to CSV file
        encoding: File encoding
        mode: Write mode ('a' for append, 'w' for overwrite)

    Returns:
        bool: True if successful, False otherwise
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = path.exists() and path.stat().st_size > 0
        write_mode = 'a' if (mode == 'a' and file_exists) else 'w'
        header = not file_exists

        df.to_csv(path, mode=write_mode, header=header,
                 index=False, encoding=encoding)
        return True
    except Exception as e:
        print(f"      [CSV Write Error] Failed to write to {path}: {e}")
        return False
def _read_header(path: Union[str, Path], encoding: str = "utf-8") -> List[str]:
    """Return the column header of an existing CSV (empty list if unreadable)."""
    try:
        return pd.read_csv(path, nrows=0, encoding=encoding).columns.tolist()
    except Exception:
        return []


# Legacy single-score aggregate columns that the per-chunk schema replaces.
_LEGACY_AGGREGATE_COLS = ["original_cosine_score", "recency_adjusted_score", "RRF"]


def infer_language_from_qid(qid) -> str:
    """
    Infer 'en' / 'fr' from a question_id such as 'test_en_004' or 'test_fr_216'.
    Returns "" when the id carries no language marker.
    """
    s = str(qid).lower()
    if "_fr" in s or s.startswith("fr_"):
        return "fr"
    if "_en" in s or s.startswith("en_"):
        return "en"
    return ""


def resolve_language(lang, qid) -> str:
    """
    Normalise a language value to 'en' / 'fr'. When the stored value is empty or
    obviously corrupted (e.g. chunk text landed here via a misaligned append),
    fall back to inferring the language from the question_id.
    """
    s = str(lang).strip().lower()
    if s in ("en", "english", "eng"):
        return "en"
    if s in ("fr", "french", "français", "francais", "fra"):
        return "fr"
    inferred = infer_language_from_qid(qid)
    if inferred:
        return inferred
    return "" if s in ("", "nan", "none") else str(lang)


def normalize_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bring an old-schema DataFrame in line with the per-chunk schema *by name*,
    before a schema rebuild carries columns over:

      - rename source{n}_score        -> source{n}_cosine_score
      - repair the `language` column   -> 'en' / 'fr' (inferred from question_id
                                          when the stored value is missing/corrupt)

    Legacy aggregate columns (original_cosine_score / recency_adjusted_score / RRF)
    are intentionally left untouched here — they simply drop out during the
    carry-by-name rebuild because they are not part of any canonical schema.
    The per-chunk recency/rrf columns cannot be recovered from old data, so they
    are left empty for regeneration by retrieval.py.
    """
    df = df.copy()
    rename_map = {f"source{i}_score": f"source{i}_cosine_score"
                  for i in range(1, 6)
                  if f"source{i}_score" in df.columns and f"source{i}_cosine_score" not in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    if "question_id" in df.columns:
        lang_series = df["language"] if "language" in df.columns else [""] * len(df)
        df["language"] = [resolve_language(l, q) for l, q in zip(lang_series, df["question_id"])]

    return df


def migrate_csv_schema(path: Union[str, Path],
                       columns: List[str],
                       encoding: str = "utf-8",
                       make_backup: bool = True) -> List[str]:
    """
    Rebuild `path` to the canonical `columns` schema.

    - Backs up the old file to `<name>.csv.bak`.
    - Carries existing columns over BY NAME (positionally safe — never shifts values).
    - Leaves genuinely-new columns empty (to be regenerated by the caller).
    - Drops stale columns not present in `columns`.

    Returns the list of newly-added (empty) columns so the caller can regenerate them.
    """
    path = Path(path)
    old = pd.read_csv(path, encoding=encoding, on_bad_lines="skip")

    if make_backup:
        bpath = path.with_suffix(path.suffix + ".bak")
        old.to_csv(bpath, index=False, encoding=encoding)
        print(f"      [CSV Migrate] Backed up old structure -> {bpath}")

    # Reconcile legacy naming (source{n}_score -> source{n}_cosine_score) and
    # repair the language column BEFORE carrying columns over by name.
    old = normalize_legacy_columns(old)

    rebuilt = pd.DataFrame(
        {c: (old[c] if c in old.columns else "") for c in columns},
        columns=columns,
    )
    rebuilt.to_csv(path, mode="w", header=True, index=False, encoding=encoding)

    dropped = [c for c in old.columns if c not in columns]
    added = [c for c in columns if c not in old.columns]
    if dropped:
        print(f"      [CSV Migrate] Dropped stale columns: {dropped}")
    if added:
        print(f"      [CSV Migrate] Added new columns (need regeneration): {added}")
    return added


def upsert_row(row_dict: Dict,
               path: Union[str, Path],
               columns: List[str],
               key_cols=("question_id",),
               owned_cols: Optional[List[str]] = None,
               encoding: str = "utf-8") -> bool:
    """
    Update-or-insert a single row by key, rewriting the whole file so progress is
    visible immediately and no duplicate rows accumulate.

    - If a row whose `key_cols` match already exists, OVERWRITE only `owned_cols`
      (default: every column the caller actually supplied in `row_dict`), leaving
      all other existing columns intact. This lets retrieval update source columns
      while preserving a previously-written answer/grounded, and lets the answer
      generator update answer/grounded/attempts/sources without wiping metadata.
    - If no matching row exists, append a new one.
    - An old single-score schema is migrated to `columns` first (backup + rename).

    Values always land under the correct header regardless of dict order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(columns)

    if path.exists() and path.stat().st_size > 0:
        existing_header = _read_header(path, encoding)
        if set(existing_header) != set(columns):
            migrate_csv_schema(path, columns, encoding=encoding)
        df = pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
        for c in columns:
            if c not in df.columns:
                df[c] = ""
        df = df[columns].astype(object)
        df = df.where(pd.notna(df), "")
    else:
        df = pd.DataFrame(columns=columns).astype(object)

    owned = list(owned_cols) if owned_cols is not None else [c for c in columns if c in row_dict]

    # Locate existing row(s) matching the key.
    if len(df):
        mask = pd.Series(True, index=df.index)
        for k in key_cols:
            kv = str(row_dict.get(k, "")).strip()
            col = df[k].astype(str).str.strip() if k in df.columns else pd.Series("", index=df.index)
            mask = mask & (col == kv)
    else:
        mask = pd.Series([], dtype=bool)

    if len(df) and mask.any():
        for idx in df.index[mask]:
            for c in owned:
                if c in columns:
                    df.at[idx, c] = row_dict.get(c, df.at[idx, c])
    else:
        new_row = {c: row_dict.get(c, "") for c in columns}
        df = pd.concat([df, pd.DataFrame([new_row], columns=columns)], ignore_index=True)

    df.to_csv(path, index=False, encoding=encoding)
    return True


def align_and_append(rows: Union[Dict, List[Dict]],
                     path: Union[str, Path],
                     columns: List[str],
                     encoding: str = "utf-8") -> bool:
    """
    Schema-safe row append. GUARANTEES every value lands under the correct
    header regardless of the order in which the `rows` dicts were built.

    pandas' plain `to_csv(mode="a", header=False)` writes columns in the
    DataFrame's own order, ignoring the existing file's header — which silently
    corrupts data when different producers build rows in different orders. This
    helper prevents that:

      - New / empty file       -> write canonical `columns` header, then rows.
      - Same column SET        -> reindex rows to the file's existing header
                                  order, then append (no header line).
      - Different structure     -> back up + migrate the file to canonical
                                  `columns` (old data carried over by name),
                                  then append.

    `rows` may be a single dict or a list of dicts. Extra keys not in `columns`
    are dropped (prevents phantom trailing Column1/Column2 fields).
    """
    if isinstance(rows, dict):
        rows = [rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build the new-row frame strictly in canonical column order.
    df_new = pd.DataFrame(rows)
    for c in columns:
        if c not in df_new.columns:
            df_new[c] = ""
    df_new = df_new[columns]

    file_exists = path.exists() and path.stat().st_size > 0
    if not file_exists:
        df_new.to_csv(path, mode="w", header=True, index=False, encoding=encoding)
        return True

    existing = _read_header(path, encoding=encoding)

    if existing == columns:
        # identical header -> straight append
        df_new.to_csv(path, mode="a", header=False, index=False, encoding=encoding)
    elif set(existing) == set(columns):
        # same columns, different order -> match the file so values stay aligned
        df_new[existing].to_csv(path, mode="a", header=False, index=False, encoding=encoding)
    else:
        # structural mismatch -> migrate the file, then append cleanly
        migrate_csv_schema(path, columns, encoding=encoding)
        df_new.to_csv(path, mode="a", header=False, index=False, encoding=encoding)
    return True


def backup_csv(path: Union[str, Path],
               suffix: str = "_backup") -> Optional[str]:
    """
    Create a backup copy of CSV file.

    Args:
        path: Path to original CSV file
        suffix: Suffix to add to backup filename

    Returns:
        str: Path to backup file if successful, None otherwise
    """
    try:
        original_path = Path(path)
        backup_path = original_path.parent / f"{original_path.stem}{suffix}{original_path.suffix}"

        # Copy file using pandas
        df = pd.read_csv(original_path)
        df.to_csv(backup_path, index=False)

        return str(backup_path)
    except Exception as e:
        print(f"      [CSV Backup Error] Failed to backup {path}: {e}")
        return None
def export_json(data, path: Union[str, Path]) -> bool:
    """
    Export data to JSON file.

    Args:
        data: Data to export
        path: Path to JSON file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"      [JSON Export Error] Failed to export {path}: {e}")
        return False