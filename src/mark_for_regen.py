#!/usr/bin/env python3
"""
mark_for_regen.py: select the rows a pipeline change actually affects, so a regeneration
touches those and nothing else.

Why a subset. The corrective loop was changed in three ways: the rewriter no longer fires on
attempt 1, top_k no longer widens with each attempt, and from attempt 2 the rewriter receives
the rejected answer plus the retrieved provision titles. None of that touches a query that was
grounded on its first attempt: attempt 1 performs no rewrite and retrieves top_k=5, exactly as
before. Only rows where the loop actually fired took the changed path.

That is 450 rows across the four local directories rather than 1,298, and at roughly a minute
and a half per retry query it is the difference between half a day and a day and a half.

How the selection is communicated. `answer_generator.py --resume_regen` treats a row as already
done when it carries generation telemetry (`ttft_s`) and a usable answer. Blanking `ttft_s` on
the chosen rows is therefore all it takes to queue them, with no change to the generator.

The source columns are blanked too, and that part is not optional. `answer_generator` reuses the
stored top-K as attempt 1's retrieval, but for a retry row those columns hold the LAST attempt's
retrieval, not the first: batch 2 overwrites them on every pass. Seeding the new attempt 1 with
the old widened, rewritten retrieval would silently reintroduce the behaviour being removed.
With the columns blank the generator re-retrieves, and since attempt 1 no longer rewrites and no
longer widens, that retrieval is the naive single pass by construction.

Run:
  python src/mark_for_regen.py --dry-run
  python src/mark_for_regen.py
  python src/mark_for_regen.py --dir tests/mistral-7b/Judge-Prometheus-8x7b-v2.0
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# The four directories whose answers come from the local agentic loop. The naive and cloud arms
# never enter the corrective loop, so nothing in them changed and they are deliberately absent.
DEFAULT_DIRS = [
    "tests/llama-3.2-3b/Judge-Prometheus-8x7b-v2.0",
    "tests/llama-3.1-8b/Judge-Prometheus-8x7b-v2.0",
    "tests/mistral-7b/Judge-Prometheus-8x7b-v2.0",
    "tests/cross-lingual-german/llama-3.1-8b/Judge-Prometheus-8x7b-v2.0",
]
LUFA = "lufa_out_data.csv"

# Blanking ttft_s is what --resume_regen reads. The rest is stale measurement that would
# otherwise describe the previous pipeline until the row happened to be rewritten.
TELEMETRY = ["ttft_s", "retrieval_latency_s", "end_to_end_latency_s",
             "gpu_vram_mb", "gpu_vram_dedicated_mb", "gpu_vram_shared_mb",
             "system_ram_mb", "cpu_percent", "gpu_util_percent",
             "context_window_used", "prompt_tokens_est", "predicted_output_tokens",
             "warmup_applied", "attempts", "grounded"]
SOURCES = [f"source{i}_{s}" for i in range(1, 6)
           for s in ("id", "cosine_score", "recency_adjusted_cosine_score", "rrf_score", "text")]


def select(df):
    """Rows whose corrective loop fired. attempts > 1 is the definition of 'took the changed path'."""
    a = pd.to_numeric(df.get("attempts", pd.Series(dtype=str)), errors="coerce")
    return a.fillna(0) > 1


def mark(path, dry_run=False, backup=True):
    p = Path(path) / LUFA
    if not p.exists():
        print(f"   skip {p}: not found")
        return 0
    df = pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)
    if "attempts" not in df.columns:
        print(f"   skip {p}: no attempts column")
        return 0

    mask = select(df)
    n = int(mask.sum())
    if not n:
        print(f"   {p}: nothing to regenerate")
        return 0
    if dry_run:
        print(f"   {p}: would queue {n} of {len(df)} rows")
        return n

    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = p.with_suffix(f".csv.pre_regen_{stamp}")
        shutil.copy2(p, bak)
        print(f"   backup -> {bak.name}")

    for c in TELEMETRY + SOURCES:
        if c in df.columns:
            df.loc[mask, c] = ""
    df.to_csv(p, index=False, encoding="utf-8")
    print(f"   {p}: queued {n} of {len(df)} rows "
          f"({len(df) - n} grounded on attempt 1, left untouched)")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", dest="dirs",
                    help="directory holding a lufa_out_data.csv; repeatable. "
                         "Defaults to the four local agentic directories.")
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    dirs = args.dirs or DEFAULT_DIRS
    print(f"[mark] queueing rows whose corrective loop fired (attempts > 1)"
          + (" — DRY RUN" if args.dry_run else ""))
    total = sum(mark(d, args.dry_run, not args.no_backup) for d in dirs)
    print(f"\n[mark] {total} row(s) queued for regeneration.")
    if not args.dry_run and total:
        print("[mark] Now run generation with --resume_regen; rows still carrying ttft_s are skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
