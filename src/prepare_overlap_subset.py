#!/usr/bin/env python3
"""
prepare_overlap_subset.py — draw the second-annotator overlap subset.

The full human sample is 75 items (src/prepare_human_sample.py). A second rater is
available for part of it, so agreement is measured on a designed overlap rather than
on whatever happened to get scored twice. This script writes that overlap as its own
blind sheet.

Design, 30 items (40% of the sample):
  * all 15 of Sample B, the purposive citation-band draw. This is where disagreement
    is most likely and most informative, because it spans citation scores 0.0/0.5/1.0
    by construction.
  * 15 from Samples A and A-N, balanced across language and retry status and split
    between the two arms, so the overlap mirrors the structure of the rest.

Agreement is then computed only where BOTH raters recorded a value, which
compute_iaa.py already does, so the 45 single-rater items need no special handling.

Run:
  python src/prepare_overlap_subset.py
  python src/prepare_overlap_subset.py --english_only   # if the second rater reads no French
  python src/prepare_overlap_subset.py --n 40
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

HUMAN_EVAL_DIR = Path("tests/human_eval")
SEED = 42

# Cells to fill from Samples A and A-N, and how many from each. Sums to 15.
STRATA_QUOTA = {
    ("English", "en/retried"): 4,
    ("English", "en/single"): 4,
    ("French", "fr/retried"): 4,
    ("French", "fr/single"): 3,
}
ENGLISH_ONLY_QUOTA = {
    ("English", "en/retried"): 8,
    ("English", "en/single"): 7,
}

ANNOTATION_COLUMNS = ["citation_score", "context_relevant", "answer_faithful",
                      "answer_appropriate", "question_realistic", "notes"]


def pick(key, quota, rng):
    """Stratified draw from Samples A and A-N, alternating the two arms within each cell."""
    chosen = []
    pool = key[key["sample"].isin(["A", "A-N"])]
    for (lang, strata), n in quota.items():
        cell = pool[(pool["language"] == lang) & (pool["strata"] == strata)]
        # Alternate arms so the overlap covers the agentic and single-pass sides evenly.
        arms = [cell[cell["sample"] == "A"], cell[cell["sample"] == "A-N"]]
        buckets = [sorted(a.index.tolist()) for a in arms]
        for b in buckets:
            rng.shuffle(b)
        taken, i = [], 0
        while len(taken) < n and any(buckets):
            b = buckets[i % 2]
            if b:
                taken.append(b.pop())
            elif buckets[(i + 1) % 2]:
                taken.append(buckets[(i + 1) % 2].pop())
            i += 1
        chosen.extend(taken)
    return chosen


def main():
    ap = argparse.ArgumentParser(description="Draw the second-annotator overlap subset.")
    ap.add_argument("--sheet_dir", default=str(HUMAN_EVAL_DIR))
    ap.add_argument("--english_only", action="store_true",
                    help="second rater reads no French: draw the non-Sample-B half from "
                         "English items only, and drop French Sample B items")
    ap.add_argument("--n", type=int, default=30, help="target overlap size (default 30)")
    args = ap.parse_args()

    d = Path(args.sheet_dir)
    key = pd.read_csv(d / "sample_key.csv")
    sheet = pd.read_csv(d / "annotation_sheet_annot2.csv")

    rng = random.Random(SEED)

    sample_b = key[key["sample"] == "B"]
    if args.english_only:
        sample_b = sample_b[sample_b["language"] == "English"]
    b_idx = sorted(sample_b.index.tolist())

    quota = ENGLISH_ONLY_QUOTA if args.english_only else STRATA_QUOTA
    remaining = max(0, args.n - len(b_idx))
    scaled = {}
    total = sum(quota.values())
    for k, v in quota.items():
        scaled[k] = round(v * remaining / total)
    # Rounding can drift by one; correct against the largest cell.
    drift = remaining - sum(scaled.values())
    if drift:
        biggest = max(scaled, key=lambda c: scaled[c])
        scaled[biggest] += drift

    a_idx = pick(key, scaled, rng)
    idx = sorted(set(b_idx) | set(a_idx))

    ids = key.loc[idx, "item_id"].tolist()
    out = sheet[sheet["item_id"].isin(ids)].copy()
    for c in ANNOTATION_COLUMNS:
        out[c] = ""

    out_path = d / "annotation_sheet_annot2_subset.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # A record for you, NOT for the second rater: it names the strata and would break blinding.
    rec = key[key["item_id"].isin(ids)][["item_id", "sample", "system", "language", "strata"]]
    rec.to_csv(d / "overlap_subset_key.csv", index=False, encoding="utf-8-sig")

    print(f"[overlap] {len(out)} items -> {out_path}")
    print(f"[overlap] composition:")
    print(rec.groupby(["sample", "language"]).size().to_string())
    print(f"[overlap] key (keep private) -> {d / 'overlap_subset_key.csv'}")


if __name__ == "__main__":
    main()
