#!/usr/bin/env python3
"""
build_naive_from_agentic.py: seed the naive-RAG dataset from the agentic results.

In `answer_generator.generate_agentic_answer` the query rewriter only fires when
`attempt > 1`, so attempt 1 retrieves with the RAW question and generates once, which is
exactly the naive pipeline. Any agentic row that finished in a single attempt is therefore
already a valid naive observation and is copied wholesale (answer, per-chunk scores,
telemetry, deterministic metrics AND judge scores) instead of being regenerated.

A judge score is a function of (question, answer, context). All three are identical for a
copied row, so the score transfers, but this script VERIFIES the answer text matches
between the lufa and evaluation ledgers before trusting it, and refuses to copy the judge
cells for any row where it does not.

Rows with attempts > 1 are not copied. Their ids are written to
logs/naive_missing_<key>.txt plus a matching input CSV, so the generation step can be
scoped to exactly those questions (never run the generator against the full question set:
its resume gate also treats ungrounded rows as incomplete).

All writes are single bulk writes. Never per-row upsert_row here: 426 full rewrites of a
1.7 MB CSV is O(n^2) and intermittently trips a Windows file-handle error.

Usage:
  python src/build_naive_from_agentic.py
  python src/build_naive_from_agentic.py --dry_run
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from run_simulation import OUTPUT_COLUMNS          # noqa: E402
from evaluate import EVAL_COLUMNS                  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SUB = "Judge-Prometheus-8x7b-v2.0"

# key -> (agentic source dir, naive target dir)
PAIRS = [
    ("llama-3.2-3b", f"tests/llama-3.2-3b/{SUB}",  f"tests/naive-rag/llama-3.2-3b/{SUB}"),
    ("llama-3.1-8b", f"tests/llama-3.1-8b/{SUB}",  f"tests/naive-rag/llama-3.1-8b/{SUB}"),
    ("mistral-7b",   f"tests/mistral-7b/{SUB}",    f"tests/naive-rag/mistral-7b/{SUB}"),
]

JUDGE_COLS = ["answer_relevance", "faithfulness", "context_precision",
              "citation_accuracy_judge"]


def _norm(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(v).split())


def build(key, src_dir, tgt_dir, dry_run=False):
    src, tgt = REPO / src_dir, REPO / tgt_dir
    s_lufa, s_eval = src / "lufa_out_data.csv", src / "evaluation_results.csv"
    s_test = src / "combined_test_data_and_ground_truth.csv"
    for p in (s_lufa, s_eval, s_test):
        if not p.exists():
            print(f"[{key}] ERROR: missing {p}")
            return 1

    lufa = pd.read_csv(s_lufa)
    ev = pd.read_csv(s_eval)
    test = pd.read_csv(s_test)

    attempts = pd.to_numeric(lufa["attempts"], errors="coerce")
    ans = lufa["answer"].astype(str).str.strip()
    one_pass = (attempts == 1) & ans.ne("") & ans.ne("ERROR") & ans.ne("nan")

    keep_ids = set(lufa.loc[one_pass, "question_id"].astype(str))
    all_ids = set(lufa["question_id"].astype(str))
    missing_ids = sorted(all_ids - keep_ids)

    # Answer-text agreement check before any judge score is inherited.
    la = dict(zip(lufa["question_id"].astype(str), lufa["answer"].map(_norm)))
    ea = dict(zip(ev["question_id"].astype(str), ev["answer"].map(_norm)))
    mismatched = {q for q in keep_ids if q in ea and la.get(q, "") != ea.get(q, "")}
    if mismatched:
        print(f"[{key}] WARNING: {len(mismatched)} rows have differing answer text between "
              f"the lufa and evaluation ledgers; their judge scores will be BLANKED "
              f"rather than inherited.")

    lufa_keep = lufa[lufa["question_id"].astype(str).isin(keep_ids)].copy()
    ev_keep = ev[ev["question_id"].astype(str).isin(keep_ids)].copy()
    for c in OUTPUT_COLUMNS:
        if c not in lufa_keep.columns:
            lufa_keep[c] = ""
    for c in EVAL_COLUMNS:
        if c not in ev_keep.columns:
            ev_keep[c] = ""
    lufa_keep = lufa_keep[OUTPUT_COLUMNS]
    ev_keep = ev_keep[EVAL_COLUMNS]

    if mismatched:
        m = ev_keep["question_id"].astype(str).isin(mismatched)
        for c in JUDGE_COLS:
            ev_keep.loc[m, c] = ""

    print(f"[{key}] copied={len(lufa_keep)}  to_generate={len(missing_ids)}  "
          f"judge_inherited={len(ev_keep) - len(mismatched)}")

    if dry_run:
        return 0

    tgt.mkdir(parents=True, exist_ok=True)
    lufa_keep.to_csv(tgt / "lufa_out_data.csv", index=False)
    ev_keep.to_csv(tgt / "evaluation_results.csv", index=False)
    shutil.copyfile(s_test, tgt / "combined_test_data_and_ground_truth.csv")

    # Scoped input for the generation step: exactly the rows still needed.
    logs = REPO / "logs"
    logs.mkdir(exist_ok=True)
    (logs / f"naive_missing_{key}.txt").write_text("\n".join(missing_ids), encoding="utf-8")
    sub = test[test["id"].astype(str).isin(missing_ids)]
    sub.to_csv(logs / f"naive_input_{key}.csv", index=False)
    print(f"[{key}] wrote {tgt.relative_to(REPO)} and logs/naive_input_{key}.csv "
          f"({len(sub)} questions)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Seed naive-RAG data from one-attempt agentic rows.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    rc = 0
    for key, s, t in PAIRS:
        rc |= build(key, s, t, args.dry_run)
    print("\n[naive] done." if rc == 0 else "\n[naive] completed with errors.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
