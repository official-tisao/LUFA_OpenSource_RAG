#!/usr/bin/env python3
"""
Standalone interactive-dashboard builder for the LUFA RAG system.

Takes the raw evaluation artifacts and produces a single self-contained HTML
dashboard (static-readable + JS-enhanced, with per-column filtering and a
Normalize toggle — see dashboard_generator for the full feature list).

Inputs
------
  --eval_csv   tests/evaluation_results.csv     (primary; has all metric columns)
  --lufa_csv   tests/lufa_out_data.csv          (optional; RAG engine output log,
                                                 used to enrich answer / source1_id
                                                 so the Normalize toggle is accurate)
  --gt_csv     tests/combined_test_data_and_ground_truth.csv
                                                 (optional; fills category / difficulty
                                                 / question / language when missing)
  --out        dashboard/index.html

Typical use
-----------
  python src/build_dashboard.py \
      --eval_csv tests/evaluation_results.csv \
      --lufa_csv tests/lufa_out_data.csv \
      --gt_csv   tests/combined_test_data_and_ground_truth.csv \
      --out      dashboard/index.html

Only --eval_csv is strictly required; the others are merged in when present.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dashboard_generator import generate_dashboard  # noqa: E402


def _read(path):
    if path and Path(path).exists():
        try:
            return pd.read_csv(path, on_bad_lines="skip")
        except Exception as e:
            print(f"[Warn] Could not read {path}: {e}")
    return None


def build(eval_csv, lufa_csv, gt_csv, out_path):
    eval_df = _read(eval_csv)
    if eval_df is None:
        raise FileNotFoundError(
            f"Required evaluation results file not found / unreadable: {eval_csv}")

    if "question_id" in eval_df.columns:
        eval_df = eval_df.drop_duplicates(subset=["question_id", "rag_base_model"], keep="last")

    # ── Enrich with answer + source1_id from the lufa engine log (for Normalize)
    lufa_df = _read(lufa_csv)
    if lufa_df is not None and "question_id" in lufa_df.columns:
        keep = [c for c in ["question_id", "answer", "source1_id"] if c in lufa_df.columns]
        lufa_small = lufa_df[keep].drop_duplicates(subset=["question_id", "base_model_used"], keep="last")
        merged = eval_df.merge(lufa_small, on="question_id", how="left", suffixes=("", "_lufa"))
        for col in ("answer", "source1_id"):
            lufa_col = col + "_lufa"
            if lufa_col in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[col].where(
                        merged[col].notna() & (merged[col].astype(str).str.strip() != ""),
                        merged[lufa_col])
                else:
                    merged[col] = merged[lufa_col]
                merged = merged.drop(columns=[lufa_col])
        eval_df = merged

    # ── Fill metadata gaps from ground-truth file
    gt_df = _read(gt_csv)
    if gt_df is not None and "id" in gt_df.columns:
        gt_small = gt_df.rename(columns={"id": "question_id"})
        meta_cols = [c for c in ["question_id", "category", "difficulty", "language", "question"]
                     if c in gt_small.columns]
        gt_small = gt_small[meta_cols].drop_duplicates(subset=["question_id", "rag_base_model"], keep="last")
        merged = eval_df.merge(gt_small, on="question_id", how="left", suffixes=("", "_gt"))
        for col in ("category", "difficulty", "language", "question"):
            gt_col = col + "_gt"
            if gt_col in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[col].where(
                        merged[col].notna() & (merged[col].astype(str).str.strip() != ""),
                        merged[gt_col])
                else:
                    merged[col] = merged[gt_col]
                merged = merged.drop(columns=[gt_col])
        eval_df = merged

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    generate_dashboard(eval_df, out_path)
    print(f"[Done] Interactive dashboard written to: {out_path}")
    print(f"       Rows: {len(eval_df)}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the LUFA interactive evaluation dashboard.")
    p.add_argument("--eval_csv", default="tests/evaluation_results.csv")
    p.add_argument("--lufa_csv", default="tests/lufa_out_data.csv")
    p.add_argument("--gt_csv", default="tests/combined_test_data_and_ground_truth.csv")
    p.add_argument("--out", default="dashboard/index.html")
    args = p.parse_args()
    build(args.eval_csv, args.lufa_csv, args.gt_csv, args.out)
