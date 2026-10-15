#!/usr/bin/env python3
"""
filter_results_to_subset.py: drop the questions a human judged implausible out of every
result ledger, without touching the ledgers themselves.

reports/review_questions.html adds one column, `plausible`, to
tests/combined_test_data_and_ground_truth.csv. A row marked false is a question that does not
make sense as something a member would ask, so no system should be credited or penalised for
answering it. Those rows stay IN the repaired ground-truth file, flagged rather than deleted,
because deleting them would remove results with no record that a human had looked.

This script is where the flag is acted on. For every evaluation_results.csv (and its
lufa_out_data.csv) under --tests, it writes a filtered copy into a mirrored directory tree so
the originals survive untouched as the evidence already reported in the thesis.

Run:
  python src/filter_results_to_subset.py
  python src/filter_results_to_subset.py --subset tests/combined_test_data_and_ground_truth.csv
  python src/filter_results_to_subset.py --out-suffix _plausible --dry-run
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_SUBSET = "tests/combined_test_data_and_ground_truth.csv"
DEFAULT_TESTS = "tests"
DEFAULT_SUFFIX = "_plausible"
PLAUSIBLE = "plausible"
# Both names appear across the pipeline: the ground-truth file uses `id`, the ledgers use
# `question_id`.
ID_COLS = ("id", "question_id")
TARGETS = ("evaluation_results.csv", "lufa_out_data.csv")

TRUE = {"1", "true", "yes", "y"}
FALSE = {"0", "false", "no", "n"}


def _id_col(df):
    for c in ID_COLS:
        if c in df.columns:
            return c
    return None


def load_subset(path):
    """
    Return (keep_ids, drop_ids, undecided_ids) from the repaired ground-truth file.

    An undecided row (blank `plausible`) is KEPT. Nobody judged it, and silently discarding
    questions no human ruled on would be the same failure this whole exercise exists to fix.
    """
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    idc = _id_col(d)
    if idc is None:
        raise SystemExit(f"[filter] {path} has no id or question_id column.")
    if PLAUSIBLE not in d.columns:
        raise SystemExit(
            f"[filter] {path} has no '{PLAUSIBLE}' column, so there is nothing to filter on.\n"
            f"         Export it from reports/review_questions.html first."
        )
    keep, drop, undecided = set(), set(), set()
    for _, r in d.iterrows():
        qid = str(r[idc]).strip()
        if not qid:
            continue
        v = str(r[PLAUSIBLE]).strip().lower()
        if v in FALSE:
            drop.add(qid)
        elif v in TRUE:
            keep.add(qid)
        else:
            undecided.add(qid)
    return keep, drop, undecided


def filter_file(src, drop_ids, out_path, dry_run=False):
    """Write src minus the dropped question ids. Returns (kept, removed, absent)."""
    try:
        d = pd.read_csv(src, dtype=str, keep_default_na=False, low_memory=False)
    except Exception as e:                      # a locked or truncated file must not abort
        print(f"   skip {src}: {e}")            # the whole sweep
        return None
    idc = _id_col(d)
    if idc is None:
        print(f"   skip {src}: no id column")
        return None
    ids = d[idc].astype(str).str.strip()
    mask = ~ids.isin(drop_ids)
    kept, removed = int(mask.sum()), int((~mask).sum())
    # Ids present in the ledger but in neither list: a question that is not in the repaired
    # ground truth at all. Reported rather than dropped, since it usually means the ledger and
    # the ground-truth file have drifted apart.
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        d[mask].to_csv(out_path, index=False, encoding="utf-8")
    return kept, removed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default=DEFAULT_SUBSET,
                    help="repaired ground-truth file carrying the 'plausible' column")
    ap.add_argument("--tests", default=DEFAULT_TESTS, help="root to sweep for result ledgers")
    ap.add_argument("--out-suffix", default=DEFAULT_SUFFIX,
                    help="appended to each ledger's parent directory name for the filtered copy")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    keep, drop, undecided = load_subset(args.subset)
    total = len(keep) + len(drop) + len(undecided)
    print(f"[filter] {args.subset}: {total} questions")
    print(f"   plausible      {len(keep)}")
    print(f"   NOT plausible  {len(drop)}   <- removed from every ledger")
    print(f"   undecided      {len(undecided)}   <- kept, since nobody judged them")
    if not drop:
        print("[filter] nothing marked false, so there is nothing to remove. "
              "Have you exported from reports/review_questions.html yet?")
        return 0

    root = Path(args.tests)
    files = sorted(p for name in TARGETS for p in root.rglob(name)
                   if args.out_suffix not in str(p))
    if not files:
        print(f"[filter] no {' or '.join(TARGETS)} found under {root}")
        return 1

    print(f"\n[filter] {len(files)} ledger file(s)"
          + (" (dry run, nothing written)" if args.dry_run else ""))
    changed = 0
    for p in files:
        # Mirror the tree, renaming only the leaf directory, so the originals stay in place as
        # the evidence already reported.
        out = p.parent.with_name(p.parent.name + args.out_suffix) / p.name
        res = filter_file(p, drop, out, args.dry_run)
        if res is None:
            continue
        kept, removed = res
        changed += 1
        rel = p.relative_to(root)
        if removed:
            print(f"   {rel}: {kept} kept, {removed} removed -> {out}")
        else:
            print(f"   {rel}: {kept} kept, none of the dropped ids present")

    print(f"\n[filter] {changed} file(s) processed.")
    if not args.dry_run:
        print("[filter] Re-run evaluate.py against the filtered directories, or aggregate from "
              "them directly. The original ledgers are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
