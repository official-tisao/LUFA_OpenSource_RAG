#!/usr/bin/env python3
"""
reuse_retrieval.py — copy retrieval results from one lufa_out CSV to others.

The retriever is shared by every generator system: the same ChromaDB collection, the
same Nomic embedding model, the same BM25 corpus and the same RRF fusion. Given the
same question, `_retrieve_nodes` is deterministic, so re-running batch 1 separately for
llama-3.1-8b, llama-3.2-3b and mistral-7b produces byte-identical source columns and
merely re-measures the same retriever. Copying one measured run into the others is both
faster and MORE internally consistent — it also matches the thesis argument (Ch6 §6.5)
that all systems inherit one shared retrieval ceiling.

Copied columns (nothing else is touched — `upsert_row` only overwrites what it is given):
    source{1..5}_{id,cosine_score,recency_adjusted_cosine_score,rrf_score,text}
    retrieval_latency_s, warmup_applied

NOT valid for the cross-lingual German set: it asks different questions (test_de_*) in a
different language, so its retrieval must be measured on its own.

Usage:
  python src/reuse_retrieval.py \
      --source tests/llama-3.2-3b/Judge-Prometheus-8x7b-v2.0/lufa_out_data.csv \
      --target tests/llama-3.1-8b/Judge-Prometheus-8x7b-v2.0/lufa_out_data.csv \
      --target tests/mistral-7b/Judge-Prometheus-8x7b-v2.0/lufa_out_data.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from csv_utils import migrate_csv_schema, _read_header   # noqa: E402
from run_simulation import OUTPUT_COLUMNS                # noqa: E402

RETRIEVAL_COLUMNS = [
    f"source{i}_{suf}"
    for i in range(1, 6)
    for suf in ("id", "cosine_score", "recency_adjusted_cosine_score", "rrf_score", "text")
] + ["retrieval_latency_s", "warmup_applied"]


def _blank(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return str(v).strip() == "" or str(v).strip().lower() in ("nan", "none")


def reuse(source_csv, target_csv, dry_run=False):
    """
    Bulk-copy the retrieval columns in a SINGLE rewrite of the target.

    An earlier version called `upsert_row` once per question, which rewrote the whole
    1.6 MB CSV 426 times (O(n^2) I/O) and intermittently tripped a Windows file-handle
    error mid-run. Everything is now done as one vectorised merge: read both files once,
    assign the columns, write once. Only rows already present in the target are touched
    and only non-blank source values overwrite, so nothing else in the file is disturbed.
    """
    src = pd.read_csv(source_csv)
    if "question_id" not in src.columns:
        print(f"[reuse] ERROR: {source_csv} has no question_id column")
        return 1

    missing = [c for c in RETRIEVAL_COLUMNS if c not in src.columns]
    if missing:
        print(f"[reuse] WARNING: source is missing {missing} — those will be skipped.")
    cols = [c for c in RETRIEVAL_COLUMNS if c in src.columns]

    tgt_path = Path(target_csv)
    if not tgt_path.exists():
        print(f"[reuse] ERROR: target {tgt_path} does not exist")
        return 1

    # Bring the target to the canonical schema first (backs up to .bak, carries by name).
    header = _read_header(tgt_path)
    if set(header) != set(OUTPUT_COLUMNS):
        print("[reuse] target schema differs from canonical — migrating first.")
        if not dry_run:
            migrate_csv_schema(tgt_path, OUTPUT_COLUMNS)

    tgt = pd.read_csv(tgt_path)
    for c in OUTPUT_COLUMNS:
        if c not in tgt.columns:
            tgt[c] = ""
    tgt = tgt[OUTPUT_COLUMNS].astype(object)

    src_key = src["question_id"].astype(str).str.strip()
    tgt_key = tgt["question_id"].astype(str).str.strip()

    copied_cells = 0
    for c in cols:
        # Map qid -> value, dropping blanks so a blank source never clears a target cell.
        s = src[c]
        keep = ~s.map(_blank)
        mapping = dict(zip(src_key[keep], s[keep]))
        if not mapping:
            continue
        new_vals = tgt_key.map(mapping)
        mask = new_vals.notna()
        tgt.loc[mask, c] = new_vals[mask]
        copied_cells += int(mask.sum())

    matched = int(tgt_key.isin(set(src_key)).sum())
    unmatched = len(tgt) - matched
    if not dry_run:
        tgt.to_csv(tgt_path, index=False)

    print(f"[reuse] {source_csv}\n     -> {target_csv}: rows matched={matched} "
          f"unmatched={unmatched} cells copied={copied_cells} across {len(cols)} columns"
          f"{' (DRY RUN)' if dry_run else ''}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Copy shared retrieval results between lufa_out CSVs.")
    ap.add_argument("--source", required=True, help="lufa_out_data.csv with the measured retrieval")
    ap.add_argument("--target", action="append", required=True,
                    help="lufa_out_data.csv to receive it (repeatable)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    for t in args.target:
        if Path(t).resolve() == Path(args.source).resolve():
            print(f"[reuse] skipping target identical to source: {t}")
            continue
        if reuse(args.source, t, args.dry_run):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
