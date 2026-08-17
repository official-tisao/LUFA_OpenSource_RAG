#!/usr/bin/env python3
"""
merge_human_annotations.py — fold the filled annotation sheets back into the evaluation
ledgers, and drive the adjudication step.

The blind sheets carry an opaque item_id; sample_key.csv maps that to (question_id,
system). This script joins the two, routes each judgment to the evaluation_results.csv of
the system it came from, and sets in_human_sample=1 on those rows. Values land in the
HUMAN_MANUAL_COLUMNS schema (src/run_simulation.py), which metrics.py carries forward on
re-runs, so annotation survives a full metrics rebuild.

Two passes, matching Chapter 4 §4.4.6's requirement that agreement is measured BEFORE
adjudication:

  pass 1 (default)     merge both annotators' raw judgments, then write
                       tests/human_eval/adjudicated.csv listing ONLY the disagreements,
                       pre-filled with both scores for you to resolve.
                       -> now run compute_iaa.py, before you discuss anything.

  pass 2 (--adjudicated) read the resolved file back and write the consensus columns
                       (human_citation_accuracy, human_appropriateness,
                       human_faithfulness, human_question_realistic). Items the two
                       annotators already agreed on are consensus by definition and are
                       filled automatically; only genuine disagreements need your input.

Run:
  python src/merge_human_annotations.py
  python src/merge_human_annotations.py --adjudicated tests/human_eval/adjudicated.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from csv_utils import migrate_csv_schema  # noqa: E402
from run_simulation import HUMAN_MANUAL_COLUMNS  # noqa: E402

HUMAN_EVAL_DIR = Path("tests/human_eval")
SYSTEM_DIRS = {
    "A":   "tests/llama-3.2-3b/Judge-Prometheus-8x7b-v2.0",
    "A-N": "tests/naive-rag/llama-3.2-3b/Judge-Prometheus-8x7b-v2.0",
}

# sheet column -> (per-annotator eval column stem, adjudicated eval column)
JUDGMENTS = {
    "citation_score":     ("citation",    "human_citation_accuracy"),
    "context_relevant":   ("relevance",   None),
    "answer_faithful":    ("faithful",    "human_faithfulness"),
    "answer_appropriate": ("appropriate", "human_appropriateness"),
    "question_realistic": ("realistic",   "human_question_realistic"),
}


def _norm(v):
    """Normalise a hand-entered cell. Blank stays blank so unfilled rows are skipped."""
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return ""
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f == int(f) else str(f)


def load_sheets(sheet_dir):
    key_path = Path(sheet_dir) / "sample_key.csv"
    if not key_path.exists():
        sys.exit(f"[merge] ERROR: {key_path} not found. Run prepare_human_sample.py first.")
    key = pd.read_csv(key_path).set_index("item_id")

    sheets = {}
    for n in (1, 2):
        p = Path(sheet_dir) / f"annotation_sheet_annot{n}.csv"
        if not p.exists():
            sys.exit(f"[merge] ERROR: {p} not found.")
        df = pd.read_csv(p).set_index("item_id")
        for c in JUDGMENTS:
            if c not in df.columns:
                sys.exit(f"[merge] ERROR: {p} is missing the '{c}' column.")
            df[c] = df[c].map(_norm)
        sheets[n] = df
    return key, sheets


def merge_raw(key, sheets, dry_run=False):
    """Write both annotators' raw judgments into each system's evaluation ledger."""
    written = {}
    for system, d in SYSTEM_DIRS.items():
        ids = key.index[key["system"] == system]
        if len(ids) == 0:
            continue
        path = Path(d) / "evaluation_results.csv"
        if not path.exists():
            print(f"[merge] skip {system}: {path} not found")
            continue

        # Add any missing human columns without disturbing existing data. This writes to
        # disk, so it must be skipped under --dry_run: a dry run that migrates the schema
        # of the real evaluation ledger is not dry.
        from evaluate import EVAL_COLUMNS
        if not dry_run:
            migrate_csv_schema(path, EVAL_COLUMNS)

        df = pd.read_csv(path, low_memory=False)
        # An all-empty human column loads as float64, and pandas refuses to write the
        # string "0.5" into it. Force object dtype before any assignment.
        for c in HUMAN_MANUAL_COLUMNS:
            df[c] = df[c].astype(object) if c in df.columns else ""
        df = df.set_index("question_id", drop=False)

        n_rows = 0
        for item_id in ids:
            qid = key.loc[item_id, "question_id"]
            if qid not in df.index:
                print(f"[merge] warn: {qid} absent from {path.name}")
                continue
            touched = False
            for sheet_col, (stem, _) in JUDGMENTS.items():
                for a in (1, 2):
                    val = sheets[a].loc[item_id, sheet_col] if item_id in sheets[a].index else ""
                    if val != "":
                        df.loc[qid, f"human_annot{a}_{stem}"] = val
                        touched = True
            if touched:
                df.loc[qid, "in_human_sample"] = 1
                n_rows += 1

        if not dry_run:
            df.reset_index(drop=True).to_csv(path, index=False)
        written[system] = (n_rows, path)
        print(f"[merge] {system}: {n_rows} rows -> {path}")
    return written


def build_adjudication_file(key, sheets, out_path):
    """List only the items where the two annotators differ on some judgment."""
    rows = []
    for item_id in key.index:
        if item_id not in sheets[1].index or item_id not in sheets[2].index:
            continue
        rec, disputed = {"item_id": item_id}, False
        for sheet_col in JUDGMENTS:
            v1 = sheets[1].loc[item_id, sheet_col]
            v2 = sheets[2].loc[item_id, sheet_col]
            rec[f"{sheet_col}_annot1"] = v1
            rec[f"{sheet_col}_annot2"] = v2
            # Only ask for a decision where they actually disagree and both answered.
            rec[f"{sheet_col}_final"] = "" if (v1 != v2 and v1 != "" and v2 != "") else v1
            if v1 != v2 and v1 != "" and v2 != "":
                disputed = True
        if disputed:
            rows.append(rec)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["item_id"] + [f"{c}_{s}" for c in JUDGMENTS for s in ("annot1", "annot2", "final")]
    pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[merge] {len(rows)} item(s) need adjudication -> {out_path}")
    if rows:
        print("[merge] fill every blank *_final cell, then re-run with --adjudicated")
    return rows


def apply_adjudicated(key, sheets, adj_path):
    """Write consensus values. Agreed items are consensus already; only conflicts come
    from the adjudication file."""
    adj = pd.read_csv(adj_path).set_index("item_id") if Path(adj_path).exists() else pd.DataFrame()

    for system, d in SYSTEM_DIRS.items():
        ids = key.index[key["system"] == system]
        path = Path(d) / "evaluation_results.csv"
        if len(ids) == 0 or not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        for c in HUMAN_MANUAL_COLUMNS:
            df[c] = df[c].astype(object) if c in df.columns else ""
        df = df.set_index("question_id", drop=False)

        n = 0
        for item_id in ids:
            qid = key.loc[item_id, "question_id"]
            if qid not in df.index:
                continue
            touched = False
            for sheet_col, (_, final_col) in JUDGMENTS.items():
                if final_col is None:
                    continue
                v1 = sheets[1].loc[item_id, sheet_col] if item_id in sheets[1].index else ""
                v2 = sheets[2].loc[item_id, sheet_col] if item_id in sheets[2].index else ""
                if v1 != "" and v1 == v2:
                    value = v1                      # agreed, no adjudication needed
                elif item_id in adj.index:
                    value = _norm(adj.loc[item_id].get(f"{sheet_col}_final", ""))
                else:
                    value = ""
                if value != "":
                    df.loc[qid, final_col] = value
                    touched = True
            n += 1 if touched else 0

        df.reset_index(drop=True).to_csv(path, index=False)
        print(f"[merge] {system}: consensus written for {n} rows -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Merge human annotation sheets into the eval ledgers.")
    ap.add_argument("--sheet_dir", default=str(HUMAN_EVAL_DIR))
    ap.add_argument("--adjudicated", default=None,
                    help="path to the resolved adjudication file (pass 2)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    key, sheets = load_sheets(args.sheet_dir)

    filled = sum((sheets[a][list(JUDGMENTS)] != "").any(axis=1).sum() for a in (1, 2))
    print(f"[merge] sheets loaded: {len(key)} items, {filled} annotator-item cells filled")
    if filled == 0:
        sys.exit("[merge] nothing annotated yet. Fill the sheets first.")

    merge_raw(key, sheets, dry_run=args.dry_run)

    if args.adjudicated:
        apply_adjudicated(key, sheets, args.adjudicated)
        print("\n[merge] done. Now run:\n"
              "  python src/compute_iaa.py --eval_csv "
              "tests/llama-3.2-3b/Judge-Prometheus-8x7b-v2.0/evaluation_results.csv --validation")
    else:
        build_adjudication_file(key, sheets, Path(args.sheet_dir) / "adjudicated.csv")
        print("\n[merge] done (pass 1). Compute agreement BEFORE discussing:\n"
              "  python src/compute_iaa.py --eval_csv "
              "tests/llama-3.2-3b/Judge-Prometheus-8x7b-v2.0/evaluation_results.csv")


if __name__ == "__main__":
    main()
