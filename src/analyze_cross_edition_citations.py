#!/usr/bin/env python3
"""
analyze_cross_edition_citations.py — how much of the zero-citation band is an
edition mismatch rather than a genuine failure to ground?

The corpus is not one document. It holds the Laurentian agreement plus the federated
agreements (Huntington, Thorneloe, University of Sudbury) across editions from 2002 to
2025, so the same provision carries different clause numbers in different documents.
The deterministic scorer compares the answer's identifiers against ONE gold passage, so
an answer that cites the right provision in the wrong edition scores 0 exactly like an
answer that cites nothing relevant at all.

This script separates those two cases. For every row scoring 0 it asks:

  1. does the answer cite a dotted clause identifier at all?
  2. does that identifier exist somewhere in the indexed corpus?
  3. does it exist in a DIFFERENT source document from the gold passage?
  4. is the chunk carrying it topically related to the question?

A row answering yes to all four is an edition mismatch: the system found a real
provision on the right subject and was scored zero for citing the wrong document's
numbering.

Run:
  python src/analyze_cross_edition_citations.py
  python src/analyze_cross_edition_citations.py --system A
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from citation_metrics import extract_citations, extract_gold_citation  # noqa: E402

SYSTEMS = {
    "A":   ("tests/llama-3.2-3b/Judge-Prometheus-8x7b-v2.0", "Llama 3.2 3B (agentic)"),
    "B":   ("tests/llama-3.1-8b/Judge-Prometheus-8x7b-v2.0", "Llama 3.1 8B (agentic)"),
    "C":   ("tests/mistral-7b/Judge-Prometheus-8x7b-v2.0", "Mistral 7B (agentic)"),
    "A-N": ("tests/naive-rag/llama-3.2-3b/Judge-Prometheus-8x7b-v2.0", "Llama 3.2 3B (naive)"),
    "D":   ("tests/cloud/chatgpt-5-4-Mini", "GPT-5.4 Mini (cloud)"),
}
GT_CSV = "tests/combined_test_data_and_ground_truth.csv"

STOP = set("""what how is are the a an of to in for and or on according collective agreement does
provide regarding apply conditions shall be with that this from at by member members university
any all may not no if when which who whom whose it its their there here also such other than
quelles quel quels quelle que est sont le la les des du de au aux dans pour et ou selon convention
collective indique sujet applique sur une un ses leur leurs cette ces avec par plus comme tout
tous toute toutes doit peut sera etre être""".split())


def toks(t):
    return {w for w in re.findall(r"[a-zA-ZÀ-ÿ]{4,}", str(t).lower()) if w not in STOP}


def load_corpus():
    """clause id -> {doc_source: [chunk_text, ...]}, and chunk id -> doc_source."""
    import chromadb
    client = chromadb.PersistentClient(path="db/chroma_db")
    col = client.get_collection("multilingual_docs")
    got = col.get(include=["documents", "metadatas"])
    clause_docs = defaultdict(lambda: defaultdict(list))
    id_to_source = {}
    for cid, text, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        # doc_source is a constant across the whole index and doc_id/ref_doc_id are None,
        # so end_year is the only field that distinguishes one edition from another.
        m = meta or {}
        src = f"{m.get('end_year', '?')}/{m.get('language', '?')}"
        id_to_source[cid] = src
        _, clauses = extract_citations(text)
        for c in clauses:
            clause_docs[c][src].append(text)
    return clause_docs, id_to_source, len(got["ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default=None, help="one of A B C A-N D; default all")
    ap.add_argument("--overlap", type=float, default=2,
                    help="min shared content words for 'topically related' (default 2)")
    args = ap.parse_args()

    clause_docs, id_to_source, n_chunks = load_corpus()
    print(f"[corpus] {n_chunks} chunks, {len(clause_docs)} distinct clause identifiers")
    sources = sorted({s for d in clause_docs.values() for s in d})
    print(f"[corpus] {len(sources)} source documents\n")

    gt = pd.read_csv(GT_CSV).set_index("id")

    targets = [args.system] if args.system else list(SYSTEMS)
    rows = []
    for key in targets:
        d, label = SYSTEMS[key]
        path = Path(d) / "evaluation_results.csv"
        if not path.exists():
            print(f"[skip] {label}: {path} not found")
            continue
        df = pd.read_csv(path, low_memory=False)

        n_zero = n_no_id = n_unknown = n_same = n_cross = n_cross_topical = 0
        examples = []
        for _, r in df.iterrows():
            score = pd.to_numeric(r.get("citation_accuracy_regex"), errors="coerce")
            if score != 0:
                continue
            n_zero += 1
            qid = r.get("question_id")
            _, cited = extract_citations(r.get("generated_answer") or r.get("answer") or "")
            cited = {c for c in cited if "." in c}
            if not cited:
                n_no_id += 1
                continue

            gold_srcs = set()
            if qid in gt.index:
                for gid in str(gt.loc[qid, "ground_source_truth_id"]).split("|"):
                    if gid.strip() in id_to_source:
                        gold_srcs.add(id_to_source[gid.strip()])
            q = toks(gt.loc[qid, "question"]) if qid in gt.index else set()

            # Mutually exclusive, and "present in the gold's own edition" wins. If the
            # cited clause exists in the edition the question was scored against, the
            # model could have been right within that edition, so the miss is a genuine
            # wrong-clause error and NOT an artefact of the corpus holding many editions.
            found_cross = found_same = in_corpus = False
            best = None
            for c in cited:
                if c not in clause_docs:
                    continue
                in_corpus = True
                srcs = set(clause_docs[c])
                if gold_srcs & srcs:
                    found_same = True
                other = srcs - gold_srcs
                if other and not (gold_srcs & srcs):
                    found_cross = True
                    for src in other:
                        for t in clause_docs[c][src]:
                            ov = q & toks(t)
                            if len(ov) >= args.overlap:
                                best = (c, src, sorted(ov)[:5])
            if not in_corpus:
                n_unknown += 1
            elif found_same:
                n_same += 1
            elif found_cross:
                n_cross += 1
                if best:
                    n_cross_topical += 1
                    if len(examples) < 4:
                        examples.append((qid, str(gt.loc[qid, "question"])[:58], best))

        pct = (lambda x: f"{x/n_zero:.1%}" if n_zero else "-")
        rows.append({
            "system": label, "zero_rows": n_zero,
            "no identifier at all": f"{n_no_id} ({pct(n_no_id)})",
            "clause not in corpus": f"{n_unknown} ({pct(n_unknown)})",
            "same doc as gold": f"{n_same} ({pct(n_same)})",
            "DIFFERENT doc": f"{n_cross} ({pct(n_cross)})",
            "different doc + on topic": f"{n_cross_topical} ({pct(n_cross_topical)})",
        })
        print(f"=== {label}: {n_zero} rows scored 0 ===")
        for k, v in rows[-1].items():
            if k not in ("system", "zero_rows"):
                print(f"    {k:26s} {v}")
        for qid, q, (c, src, ov) in examples:
            print(f"    e.g. {qid}: \"{q}\"\n         cites {c}, present in {src}, shared: {ov}")
        print()

    if rows:
        out = Path("reports/cross_edition_citation_analysis.csv")
        out.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[out] {out}")


if __name__ == "__main__":
    main()
