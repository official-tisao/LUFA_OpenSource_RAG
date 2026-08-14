#!/usr/bin/env python3
"""
compute_iaa.py — inter-annotator agreement for the LUFA human-evaluation sample
(Ch4 §4.4.6). IAA is a STUDY-LEVEL statistic, not a per-row score, so it is not
stored in evaluation_results.csv; instead this helper reads the two annotators'
manual columns and writes a small summary to tests/human_eval/iaa_summary.csv.

Computes:
  * Cohen's Kappa on the BINARY relevance judgment  (human_annot1_relevance vs
    human_annot2_relevance). Ch4 threshold: kappa >= 0.75.
  * Exact-agreement on the citation-accuracy judgment (human_annot1_citation vs
    human_annot2_citation) — the fraction of sampled queries where both annotators
    gave the same citation score.
  * Cohen's Kappa on the citation judgment (reported for completeness).

Only rows flagged in_human_sample (truthy) with BOTH annotator values present are
used. Blank/unfilled cells are skipped, so this runs safely before the sample is
fully annotated (it just reports the count it found).

Usage:
  python src/compute_iaa.py
  python src/compute_iaa.py --eval_csv tests/evaluation_results.csv \
                            --out_csv tests/human_eval/iaa_summary.csv
"""

import sys
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "1.0", "true", "yes", "y")


def _present(v) -> bool:
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    return str(v).strip() != "" and str(v).strip().lower() not in ("nan", "none")


def cohens_kappa(a, b):
    """Cohen's kappa for two equal-length label sequences. None if undefined."""
    pairs = [(x, y) for x, y in zip(a, b)]
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b), key=str)
    po = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe >= 1.0:
        return 1.0  # perfect/degenerate agreement
    return (po - pe) / (1.0 - pe)


def exact_agreement(a, b):
    """Fraction of positions where the two sequences match. None if empty."""
    n = len(a)
    if n == 0:
        return None
    return sum(1 for x, y in zip(a, b) if x == y) / n


def _paired(df, col1, col2):
    """Return two aligned label lists from rows where BOTH cells are present and
    the row is in the human sample."""
    a, b = [], []
    in_sample = df["in_human_sample"] if "in_human_sample" in df.columns else pd.Series([""] * len(df))
    for (_, row), flag in zip(df.iterrows(), in_sample):
        if not _truthy(flag):
            continue
        v1, v2 = row.get(col1), row.get(col2)
        if _present(v1) and _present(v2):
            a.append(str(v1).strip())
            b.append(str(v2).strip())
    return a, b


def compute_iaa(eval_csv="tests/evaluation_results.csv",
                out_csv="tests/human_eval/iaa_summary.csv"):
    path = Path(eval_csv)
    if not path.exists():
        print(f"[IAA] ERROR: {path} not found.")
        return None
    df = pd.read_csv(path)

    rel_a, rel_b = _paired(df, "human_annot1_relevance", "human_annot2_relevance")
    cit_a, cit_b = _paired(df, "human_annot1_citation", "human_annot2_citation")

    kappa_rel = cohens_kappa(rel_a, rel_b)
    exact_cit = exact_agreement(cit_a, cit_b)
    kappa_cit = cohens_kappa(cit_a, cit_b)

    rows = [
        {"metric": "relevance", "coefficient": "cohens_kappa",
         "value": "" if kappa_rel is None else round(kappa_rel, 4),
         "n_pairs": len(rel_a), "threshold": 0.75,
         "meets_threshold": "" if kappa_rel is None else (kappa_rel >= 0.75),
         "notes": "Ch4 §4.4.6 binary relevance IAA; required kappa >= 0.75"},
        {"metric": "citation", "coefficient": "exact_agreement",
         "value": "" if exact_cit is None else round(exact_cit, 4),
         "n_pairs": len(cit_a), "threshold": "",
         "meets_threshold": "",
         "notes": "Ch4 §4.4.6 exact-agreement on citation judgment"},
        {"metric": "citation", "coefficient": "cohens_kappa",
         "value": "" if kappa_cit is None else round(kappa_cit, 4),
         "n_pairs": len(cit_a), "threshold": "",
         "meets_threshold": "",
         "notes": "Kappa on citation judgment (reported for completeness)"},
    ]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["metric", "coefficient", "value", "n_pairs",
                                "threshold", "meets_threshold", "notes"]).to_csv(out_path, index=False)

    print(f"[IAA] relevance Cohen's kappa = {rows[0]['value']} (n={len(rel_a)})")
    print(f"[IAA] citation exact-agreement = {rows[1]['value']} (n={len(cit_a)})")
    print(f"[IAA] citation Cohen's kappa   = {rows[2]['value']} (n={len(cit_a)})")
    if not rel_a and not cit_a:
        print("[IAA] Note: no annotated pairs found. Fill in the human_annot* columns "
              "and set in_human_sample=1 on the sampled rows, then re-run.")
    print(f"[IAA] Summary written -> {out_path}")
    return rows


def _num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _adjudicated(df, col):
    """(question_id, value) pairs for in-sample rows with an adjudicated value."""
    out = {}
    if col not in df.columns:
        return out
    for _, r in df.iterrows():
        if not _truthy(r.get("in_human_sample")):
            continue
        v = _num(r.get(col))
        if v is not None:
            out[r.get("question_id")] = v
    return out


def compute_validation(eval_csv, naive_csv=None,
                       out_csv="tests/human_eval/validation_summary.csv"):
    """
    Compare the automated metrics against the adjudicated human judgments, and report
    the ecological-validity rate. This is what Chapter 5 §5.3.7 is written from.

    Run only AFTER adjudication: it reads the consensus columns, not the per-annotator
    ones, because a metric should be validated against the agreed human view.
    """
    df = pd.read_csv(eval_csv, low_memory=False)
    rows = []

    # 1. Deterministic citation regex against the human citation judgment.
    human_cit = _adjudicated(df, "human_citation_accuracy")
    pairs = [(_num(r.get("citation_accuracy_regex")), human_cit[r.get("question_id")])
             for _, r in df.iterrows()
             if r.get("question_id") in human_cit and _num(r.get("citation_accuracy_regex")) is not None]
    if pairs:
        exact = sum(1 for a, b in pairs if a == b) / len(pairs)
        harsher = sum(1 for a, b in pairs if a < b) / len(pairs)
        rows.append({"comparison": "citation regex vs human", "n": len(pairs),
                     "agreement": round(exact, 4),
                     "detail": f"regex scored LOWER than the human on {harsher:.1%} of items",
                     "notes": "regex is a string match; a correct but uncited answer scores 0"})

    # 2. LLM judge faithfulness against the human faithfulness judgment.
    human_f = _adjudicated(df, "human_faithfulness")
    fp = [(_num(r.get("faithfulness")), human_f[r.get("question_id")])
          for _, r in df.iterrows()
          if r.get("question_id") in human_f and _num(r.get("faithfulness")) is not None]
    if fp:
        # The judge emits a graded score; the human emits 0/1. Threshold at 0.5.
        agree = sum(1 for j, h in fp if (j >= 0.5) == (h >= 0.5)) / len(fp)
        rows.append({"comparison": "judge faithfulness vs human", "n": len(fp),
                     "agreement": round(agree, 4),
                     "detail": f"judge mean {sum(j for j, _ in fp)/len(fp):.3f} "
                               f"vs human mean {sum(h for _, h in fp)/len(fp):.3f}",
                     "notes": "judge score thresholded at 0.5 to compare with a binary human call"})

    # 3. Ecological validity: the realism rate.
    real = _adjudicated(df, "human_question_realistic")
    if real:
        rate = sum(real.values()) / len(real)
        rows.append({"comparison": "question realism (ecological validity)", "n": len(real),
                     "agreement": round(rate, 4),
                     "detail": f"{int(sum(real.values()))}/{len(real)} judged plausible",
                     "notes": "Ch4 §4.9 ecological-validity parameter"})

    # 4. Appropriateness, agentic against naive on the same questions.
    app_a = _adjudicated(df, "human_appropriateness")
    if naive_csv and Path(naive_csv).exists() and app_a:
        ndf = pd.read_csv(naive_csv, low_memory=False)
        app_n = _adjudicated(ndf, "human_appropriateness")
        both = sorted(set(app_a) & set(app_n))
        if both:
            ma = sum(app_a[q] for q in both) / len(both)
            mn = sum(app_n[q] for q in both) / len(both)
            rows.append({"comparison": "appropriateness agentic vs naive", "n": len(both),
                         "agreement": round(ma - mn, 4),
                         "detail": f"agentic {ma:.3f} vs naive {mn:.3f} (value is the difference)",
                         "notes": "human corroboration of the RQ1 finding; negative favours naive"})

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["comparison", "n", "agreement", "detail",
                                "notes"]).to_csv(out_path, index=False)
    for r in rows:
        print(f"[validate] {r['comparison']:42s} n={r['n']:<4} {r['agreement']}  {r['detail']}")
    if not rows:
        print("[validate] no adjudicated values found. Run merge_human_annotations.py "
              "--adjudicated first.")
    print(f"[validate] Summary written -> {out_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute inter-annotator agreement from human eval columns.")
    parser.add_argument("--eval_csv", default="tests/evaluation_results.csv")
    parser.add_argument("--out_csv", default="tests/human_eval/iaa_summary.csv")
    parser.add_argument("--validation", action="store_true",
                        help="after adjudication: compare the automated metrics against "
                             "the agreed human judgments")
    parser.add_argument("--naive_csv",
                        default="tests/naive-rag/llama-3.2-3b/Judge-Prometheus-8x7b-v2.0/evaluation_results.csv",
                        help="naive arm, for the paired appropriateness comparison")
    args = parser.parse_args()
    if args.validation:
        compute_validation(args.eval_csv, args.naive_csv)
    else:
        compute_iaa(args.eval_csv, args.out_csv)
