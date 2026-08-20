#!/usr/bin/env python3
"""
export_chunk_store.py: build the offline data file that reports/review_questions.html needs
to repair gold labels.

Writes reports/chunk_store.json containing:

  chunks    every chunk in the index: id, text, article_number, clause_id, language,
            end_year, chunk_index, section_title. Full text, no truncation, because the
            reviewer has to read a clause to judge it.
  gold      per question_id, the chunk ids currently recorded as ground truth.
  retrieved per question_id, the union of the top-5 retrieved chunk ids across EVERY system
            that has an evaluation_results.csv.

Why the union rather than one system's retrieval: if the candidate list came from System A
alone, a reviewer picking a candidate would be handing System A gold labels drawn from its
own top-5, and System A's Recall@5 and MRR would rise for reasons that have nothing to do
with retrieval quality. Pooling across all ten result files removes the per-system tilt. It
does not remove the structural point that any finite candidate list favours chunks that some
system could reach, which is why the HTML also offers a search over the whole corpus and
records where each accepted chunk came from.

Run:
  python src/export_chunk_store.py
  python src/export_chunk_store.py --out reports/chunk_store.json --db db/chroma_db
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_DB = "db/chroma_db"
DEFAULT_COLLECTION = "multilingual_docs"
DEFAULT_OUT = "reports/chunk_store.json"
GT_CSV = "tests/combined_test_data_and_ground_truth.csv"
TESTS_DIR = "tests"

# The gold id column packs multiple chunks with a pipe. Keep that convention on the way out
# so the repaired column drops straight back into the same pipeline.
SEP = "|"


def _split_ids(raw):
    return [p.strip() for p in str(raw).split(SEP) if p.strip()]


def load_chunks(db, collection):
    import chromadb
    col = chromadb.PersistentClient(path=db).get_collection(collection)
    g = col.get(include=["metadatas", "documents"])
    out = []
    for cid, doc, m in zip(g["ids"], g["documents"], g["metadatas"]):
        out.append({
            "id": cid,
            "text": doc,
            "article": str(m.get("article_number", "")),
            "clause": str(m.get("clause_id", "")),
            "lang": str(m.get("language", "")),
            "year": str(m.get("end_year", "")),
            "idx": int(m.get("chunk_index", 0) or 0),
            "section": str(m.get("section_title", ""))[:120],
        })
    return out


def load_gold(gt_csv):
    d = pd.read_csv(gt_csv, dtype=str, keep_default_na=False)
    idc = "id" if "id" in d.columns else "question_id"
    gold = {}
    for _, r in d.iterrows():
        qid = str(r[idc]).strip()
        if qid:
            gold[qid] = _split_ids(r.get("ground_source_truth_id", ""))
    return gold


def load_retrieved(tests_dir):
    """Union of top-5 retrieved ids per question across every evaluation_results.csv."""
    files = sorted(Path(tests_dir).rglob("evaluation_results.csv"))
    retrieved, seen_files = {}, []
    for p in files:
        try:
            d = pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)
        except Exception as e:                       # a truncated or locked file must not
            print(f"   skip {p}: {e}")               # abort the whole export
            continue
        if "question_id" not in d.columns:
            continue
        cols = [c for c in (f"source{i}_id" for i in range(1, 6)) if c in d.columns]
        if not cols:
            continue
        seen_files.append(str(p))
        for _, r in d.iterrows():
            qid = str(r["question_id"]).strip()
            if not qid:
                continue
            bucket = retrieved.setdefault(qid, [])
            for c in cols:
                v = str(r[c]).strip()
                # Preserve first-seen order so the list is deterministic across runs, which
                # keeps the exported file diffable.
                if v and v.lower() not in ("nan", "none") and v not in bucket:
                    bucket.append(v)
    return retrieved, seen_files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--gt", default=GT_CSV)
    ap.add_argument("--tests", default=TESTS_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    print("[export] reading the index")
    chunks = load_chunks(args.db, args.collection)
    known = {c["id"] for c in chunks}
    print(f"   {len(chunks)} chunks")

    print("[export] reading current gold labels")
    gold = load_gold(args.gt)
    miss = sorted({i for ids in gold.values() for i in ids if i not in known})
    print(f"   {len(gold)} questions, {sum(len(v) for v in gold.values())} gold chunk refs"
          + (f", {len(miss)} refs not present in the index" if miss else ""))

    print("[export] pooling retrieved chunks across every system")
    retrieved, files = load_retrieved(args.tests)
    # Drop ids the index no longer contains, or the picker would offer a blank candidate.
    dropped = 0
    for qid, ids in retrieved.items():
        keep = [i for i in ids if i in known]
        dropped += len(ids) - len(keep)
        retrieved[qid] = keep
    print(f"   {len(files)} result files, {len(retrieved)} questions covered"
          + (f", {dropped} stale ids dropped" if dropped else ""))
    if retrieved:
        n = sorted(len(v) for v in retrieved.values())
        print(f"   pooled candidates per question: min {n[0]} median {n[len(n)//2]} max {n[-1]}")

    payload = {
        "generated_from": {"db": args.db, "collection": args.collection,
                           "ground_truth": args.gt, "result_files": files},
        "chunks": chunks, "gold": gold, "retrieved": retrieved,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"[export] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print("[export] open reports/review_questions.html and load this file when asked.")


if __name__ == "__main__":
    main()
