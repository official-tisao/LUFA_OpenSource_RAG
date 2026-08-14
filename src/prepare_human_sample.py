#!/usr/bin/env python3
"""
prepare_human_sample.py — draw the Chapter 4 §4.8.3 human-evaluation sample and write
blind annotation sheets.

Chapter 4 §4.8.3 specifies "a stratified random sample of 30 outputs produced by the
proposed System, including 15 queries from each of the two language query categories",
scored by two annotators who are blind to the automated scores. This script produces
exactly that, plus two extensions that cost almost nothing and answer questions the
30-row sample cannot:

  SAMPLE A  n=30  System A (Llama 3.2 3B, agentic), the proposed system.
                  Stratified language x retry-status: per language, 8 queries the
                  agentic loop retried and 7 it answered in one pass. Retry status is
                  the stratifier because RQ1 lives on the retry subset, so a sample
                  blind to it could miss the effect entirely.

  SAMPLE A-N n=30 The NAIVE arm's answers to the SAME 30 question ids. Same question,
                  same gold, second answer. This is what lets the RQ1 finding be
                  corroborated by a human rather than resting only on automated metrics.

  SAMPLE B  n=15  Purposive: 5 rows from each deterministic citation band (0.0 / 0.5 /
                  1.0), drawn from System A outside Sample A. Used ONLY to check the
                  citation regex against human reading across its whole range. It is
                  kept separate from Sample A because forcing band balance would bias
                  any population estimate; the natural distribution would put only ~2
                  full-credit rows in 30, too few to say anything about that band.

Blinding: all items are pooled, stripped of system label and of every automated score,
and shuffled before opaque item ids are assigned. The annotator therefore cannot tell an
agentic answer from a naive one, which is what makes the paired A / A-N comparison
usable. The mapping lives in sample_key.csv, which annotators must not open.

Outputs (into tests/human_eval/):
  annotation_sheet_annot1.csv   identical blank sheets, one per annotator
  annotation_sheet_annot2.csv
  sample_key.csv                item_id -> question_id, system, strata. DO NOT SHARE.

Deterministic: seeded, so re-running reproduces the same sample.

Run:  python src/prepare_human_sample.py
"""

import argparse
import random
import sys
import textwrap
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from citation_metrics import extract_gold_citation  # noqa: E402

SEED = 42

AGENTIC_DIR = "tests/llama-3.2-3b/Judge-Prometheus-8x7b-v2.0"
NAIVE_DIR = "tests/naive-rag/llama-3.2-3b/Judge-Prometheus-8x7b-v2.0"
GROUND_TRUTH = "reports/combined_test_data_and_ground_truth.csv"
OUT_DIR = Path("tests/human_eval")

PER_LANG_RETRY = 8      # retried queries per language in Sample A
PER_LANG_SINGLE = 7     # single-attempt queries per language in Sample A
PER_BAND = 5            # rows per citation band in Sample B

# Columns the annotator fills. Names match what compute_iaa.py and the
# evaluation_results.csv schema already expect, so nothing new is invented.
ANNOTATION_COLUMNS = [
    "citation_score",       # 1.0 / 0.5 / 0.0   -> human_annot{N}_citation
    "context_relevant",     # 1 / 0             -> human_annot{N}_relevance
    "answer_faithful",      # 1 / 0
    "answer_appropriate",   # 1 / 0
    "question_realistic",   # 1 / 0
    "notes",
]


def _wrap(text, width=110, limit=2600):
    """Keep cells readable in a spreadsheet without truncating meaning."""
    s = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    s = " ".join(s.split())
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0] + " […truncated for the sheet…]"
    return "\n".join(textwrap.wrap(s, width)) if s else ""


def _context(row):
    """Concatenate the retrieved chunks the generator actually saw, labelled."""
    parts = []
    for i in range(1, 6):
        t = row.get(f"source{i}_text")
        if isinstance(t, str) and t.strip():
            parts.append(f"[chunk {i}]\n{_wrap(t, limit=1200)}")
    return "\n\n".join(parts)


def load(agentic_dir, naive_dir, gt_csv):
    ev = pd.read_csv(Path(agentic_dir) / "evaluation_results.csv", low_memory=False)
    lu = pd.read_csv(Path(agentic_dir) / "lufa_out_data.csv", low_memory=False)
    nev = pd.read_csv(Path(naive_dir) / "evaluation_results.csv", low_memory=False)
    gt = pd.read_csv(gt_csv, low_memory=False)
    return ev, lu, nev, gt


def build_sample(ev, lu, nev, gt, seed=SEED):
    rng = random.Random(seed)

    attempts = dict(zip(lu["question_id"], lu["attempts"]))
    gold_text = dict(zip(gt["id"], gt["ground_source_truth"]))
    question = dict(zip(gt["id"], gt["question"]))
    difficulty = dict(zip(gt["id"], gt.get("difficulty", pd.Series(dtype=str))))

    ev = ev.copy()
    ev["_lang"] = ev["question_id"].astype(str).str[5:7]
    ev["_retry"] = ev["question_id"].map(lambda q: float(attempts.get(q, 1)) > 1)

    # ---- Sample A: stratified language x retry -----------------------------------
    sample_a = []
    for lang in ("en", "fr"):
        for retry, n in ((True, PER_LANG_RETRY), (False, PER_LANG_SINGLE)):
            pool = sorted(ev.loc[(ev["_lang"] == lang) & (ev["_retry"] == retry),
                                 "question_id"].astype(str).tolist())
            if len(pool) < n:
                raise SystemExit(
                    f"stratum {lang}/retry={retry} has {len(pool)} rows, need {n}")
            sample_a += [(q, lang, "retried" if retry else "single") for q in rng.sample(pool, n)]

    a_ids = {q for q, _, _ in sample_a}

    # ---- Sample B: purposive, by citation band, disjoint from Sample A -----------
    sample_b = []
    for band in (0.0, 0.5, 1.0):
        pool = sorted(ev.loc[(ev["citation_accuracy_regex"] == band)
                             & (~ev["question_id"].astype(str).isin(a_ids)),
                             "question_id"].astype(str).tolist())
        take = min(PER_BAND, len(pool))
        if take < PER_BAND:
            print(f"   note: citation band {band} has only {len(pool)} rows outside Sample A")
        sample_b += [(q, band) for q in rng.sample(pool, take)]

    # ---- Assemble items ----------------------------------------------------------
    ev_by_id = ev.set_index("question_id")
    nev_by_id = nev.set_index("question_id")

    items = []

    def add(qid, system, frame, sample_name, strata):
        if qid not in frame.index:
            return
        row = frame.loc[qid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        g = gold_text.get(qid, "")
        art, cl = extract_gold_citation(str(g))
        items.append({
            "question_id": qid,
            "system": system,
            "sample": sample_name,
            "strata": strata,
            "language": "English" if str(qid).startswith("test_en_") else "French",
            "difficulty": difficulty.get(qid, ""),
            "question": _wrap(question.get(qid, row.get("question", ""))),
            "answer": _wrap(row.get("answer", "")),
            "retrieved_context": _context(row),
            "gold_provision": _wrap(g, limit=1800),
            "gold_article": art,
            "gold_clause": cl,
        })

    for qid, lang, retry in sample_a:
        add(qid, "A", ev_by_id, "A", f"{lang}/{retry}")
        add(qid, "A-N", nev_by_id, "A-N", f"{lang}/{retry}")
    for qid, band in sample_b:
        add(qid, "A", ev_by_id, "B", f"citation={band}")

    rng.shuffle(items)
    for i, it in enumerate(items, start=1):
        it["item_id"] = f"H{i:03d}"
    return items


def write_sheets(items, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The blind sheet carries no system, no sample name, no strata, no automated score.
    sheet_cols = ["item_id", "language", "question", "answer",
                  "retrieved_context", "gold_provision", "gold_article", "gold_clause"]
    sheet = pd.DataFrame(items)[sheet_cols].copy()
    for c in ANNOTATION_COLUMNS:
        sheet[c] = ""

    paths = []
    for n in (1, 2):
        p = out_dir / f"annotation_sheet_annot{n}.csv"
        sheet.to_csv(p, index=False, encoding="utf-8-sig")  # BOM so Excel reads UTF-8
        paths.append(p)

    key = pd.DataFrame(items)[["item_id", "question_id", "system", "sample",
                               "strata", "language", "difficulty"]]
    kp = out_dir / "sample_key.csv"
    key.to_csv(kp, index=False, encoding="utf-8-sig")
    return paths, kp, sheet, key


def main():
    ap = argparse.ArgumentParser(description="Draw the Ch4 §4.8.3 human-evaluation sample.")
    ap.add_argument("--agentic_dir", default=AGENTIC_DIR)
    ap.add_argument("--naive_dir", default=NAIVE_DIR)
    ap.add_argument("--ground_truth", default=GROUND_TRUTH)
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    print("[sample] loading...")
    ev, lu, nev, gt = load(args.agentic_dir, args.naive_dir, args.ground_truth)
    items = build_sample(ev, lu, nev, gt, seed=args.seed)
    paths, kp, sheet, key = write_sheets(items, args.out_dir)

    print(f"[sample] {len(items)} items drawn (seed={args.seed})")
    print(key.groupby(["sample", "system"]).size().to_string())
    print("\n[sample] language split:")
    print(key["language"].value_counts().to_string())
    for p in paths:
        print(f"[sample] wrote {p}")
    print(f"[sample] wrote {kp}   <- DO NOT open this while annotating")
    print("\nNext: follow thesis/HUMAN_VALIDATION_PROTOCOL.md")


if __name__ == "__main__":
    main()
