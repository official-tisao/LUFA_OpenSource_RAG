#!/usr/bin/env python3
"""
sync_lufa_evaluation_csv.py

Bidirectional synchroniser for the two LUFA result ledgers:
    tests/lufa_out_data.csv        (retrieval + answers)
    tests/evaluation_results.csv   (retrieval + answers + metrics + judge scores)

The two files share a large block of columns (question, answer, base_model_used,
language, attempts, grounded, and every per-chunk source column). This script
reconciles them so neither drifts:

  * For every SHARED column, a value that is missing / blank / invalid in one
    file is back-filled from the other file (whichever holds a valid value).
  * A row that exists in only one file is created in the other.
  * `language` is always normalised to 'en' / 'fr' (inferred from question_id
    when the stored value is missing or corrupted).
  * `evaluation_results` mirror fields (id, rag_base_model) are back-filled from
    their lufa counterparts (question_id, base_model_used) when blank.
  * Old single-score schemas are migrated to the per-chunk schema first.

Rows are matched on (question_id, base_model_used). Both files are backed up
before being rewritten. Eval-only columns that cannot be inferred (metrics,
judge scores) are left blank on newly-created rows — run `metrics.py` afterwards
to fill them.

Usage
-----
    python src/sync_lufa_evaluation_csv.py
    python src/sync_lufa_evaluation_csv.py --lufa_csv tests/lufa_out_data.csv \
                                           --eval_csv tests/evaluation_results.csv
    python src/sync_lufa_evaluation_csv.py --dry-run
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from run_simulation import OUTPUT_COLUMNS as LUFA_COLUMNS
from evaluate import EVAL_COLUMNS
from retrieval import has_old_schema
from csv_utils import migrate_csv_schema, resolve_language

# eval columns that mirror a lufa column when blank
_EVAL_MIRRORS = {"id": "question_id", "rag_base_model": "base_model_used"}


def _blank(v) -> bool:
    """True when a value is missing / empty / a NaN-like placeholder."""
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return str(v).strip().lower() in ("", "nan", "none")


def _key(row: dict):
    """Match key: (question_id, base_model_used) — base falls back to rag_base_model."""
    qid = str(row.get("question_id", "")).strip()
    model = str(row.get("base_model_used", "") or row.get("rag_base_model", "")).strip()
    return qid, model


def _load(path, canonical):
    """Load a ledger, migrating an old single-score schema to per-chunk first."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        print(f"[Sync] {path.name} not found or empty — treating as no rows.")
        return pd.DataFrame(columns=canonical)

    df = pd.read_csv(path, on_bad_lines="skip")
    if has_old_schema(df):
        print(f"[Sync] Old single-score schema detected in {path.name} — migrating to per-chunk schema...")
        migrate_csv_schema(path, canonical)
        df = pd.read_csv(path, on_bad_lines="skip")

    for c in canonical:
        if c not in df.columns:
            df[c] = ""
    return df


def sync(lufa_path, eval_path, dry_run=False):
    shared = [c for c in LUFA_COLUMNS if c in EVAL_COLUMNS]

    lufa_df = _load(lufa_path, LUFA_COLUMNS)
    eval_df = _load(eval_path, EVAL_COLUMNS)
    print(f"[Sync] Loaded {len(lufa_df)} lufa rows and {len(eval_df)} evaluation rows.")

    lufa_map = {_key(r.to_dict()): r.to_dict() for _, r in lufa_df.iterrows()}
    eval_map = {_key(r.to_dict()): r.to_dict() for _, r in eval_df.iterrows()}

    all_keys = set(lufa_map) | set(eval_map)
    stats = {"filled_lufa": 0, "filled_eval": 0, "added_lufa": 0, "added_eval": 0, "lang_fixed": 0}
    needs_metrics = []

    for k in sorted(all_keys):
        l = lufa_map.get(k)
        e = eval_map.get(k)

        if l is not None and e is not None:
            # Both present -> back-fill blanks in either direction.
            for c in shared:
                lv, ev = l.get(c), e.get(c)
                if _blank(lv) and not _blank(ev):
                    l[c] = ev
                    stats["filled_lufa"] += 1
                elif _blank(ev) and not _blank(lv):
                    e[c] = lv
                    stats["filled_eval"] += 1

        elif l is not None:
            # Row missing from evaluation_results -> create it from lufa.
            e = {c: "" for c in EVAL_COLUMNS}
            for c in shared:
                e[c] = l.get(c, "")
            eval_map[k] = e
            stats["added_eval"] += 1
            needs_metrics.append(k[0])

        else:
            # Row missing from lufa_out -> create it from evaluation_results.
            l = {c: "" for c in LUFA_COLUMNS}
            for c in shared:
                l[c] = e.get(c, "")
            lufa_map[k] = l
            stats["added_lufa"] += 1

        # Normalise language on both using the question_id as the source of truth.
        qid = (l or e).get("question_id", "")
        if l is not None:
            fixed = resolve_language(l.get("language", ""), qid)
            if str(fixed) != str(l.get("language", "")):
                stats["lang_fixed"] += 1
            l["language"] = fixed
        if e is not None:
            e["language"] = resolve_language(e.get("language", ""), qid)
            # Back-fill eval mirror fields from their lufa counterparts.
            for mirror, src in _EVAL_MIRRORS.items():
                if _blank(e.get(mirror)) and not _blank(e.get(src)):
                    e[mirror] = e.get(src)

    out_lufa = pd.DataFrame([lufa_map[k] for k in sorted(lufa_map)], columns=LUFA_COLUMNS)
    out_eval = pd.DataFrame([eval_map[k] for k in sorted(eval_map)], columns=EVAL_COLUMNS)

    print("\n[Sync] Reconciliation summary")
    print(f"  * Blank cells filled in lufa_out       : {stats['filled_lufa']}")
    print(f"  * Blank cells filled in evaluation     : {stats['filled_eval']}")
    print(f"  * Rows added to lufa_out               : {stats['added_lufa']}")
    print(f"  * Rows added to evaluation_results     : {stats['added_eval']}")
    print(f"  * Language values normalised/repaired  : {stats['lang_fixed']}")
    print(f"  * Final counts -> lufa: {len(out_lufa)} | evaluation: {len(out_eval)}")

    if dry_run:
        print("\n[Sync] --dry-run set: no files written.")
        return

    for p, df in ((lufa_path, out_lufa), (eval_path, out_eval)):
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.stat().st_size > 0:
            bak = p.with_suffix(p.suffix + ".syncbak")
            pd.read_csv(p, on_bad_lines="skip").to_csv(bak, index=False)
            print(f"[Sync] Backed up {p.name} -> {bak.name}")
        df.to_csv(p, index=False)
        print(f"[Sync] Wrote synchronised {p.name} ({len(df)} rows).")

    if needs_metrics:
        print(f"\n[Sync] {len(needs_metrics)} eval rows were created from lufa without metrics.")
        print("[Sync] Run `python src/metrics.py` to compute their generation/retrieval metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronise lufa_out_data.csv and evaluation_results.csv (bidirectional back-fill).")
    parser.add_argument("--lufa_csv", default="tests/lufa_out_data.csv")
    parser.add_argument("--eval_csv", default="tests/evaluation_results.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing any files.")
    args = parser.parse_args()

    sync(args.lufa_csv, args.eval_csv, dry_run=args.dry_run)
