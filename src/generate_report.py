#!/usr/bin/env python3
"""
generate_report.py — combine per-model lufa_out / evaluation_results CSVs into
category-level reports (combined CSVs + an HTML dashboard each), plus one overall
"general" combination.

Categories (searched recursively; only real files named exactly
`lufa_out_data.csv` / `evaluation_results.csv` are used — anything ending in
"bak" or otherwise named is ignored):

  1. cloud         (prefix "cloud")        → tests/cloud/**
  2. naive         (prefix "naive")        → tests/naive-rag/**
  3. crosslingual  (prefix "crosslingual") → tests/cross-lingual-german/**
  4. agentic       (NO prefix)             → tests/llama-3.1-8b/**, tests/llama-3.2-3b/**,
                                             tests/mistral-7b/**
  5. general       (prefix "general")      → concatenation of all four above

Per-row column rewrites applied to every combined CSV:
  base_model_used  ->  "<prefix>/<base_model_used>"   (skipped when prefix is empty, i.e. agentic)
  rag_base_model   ->  "<prefix>/<rag_base_model>"    (eval only; skipped when prefix empty)
  judge_llm        ->  "tensortemplar/prometheus2:8x7b-Q4_K_S"   (eval only)

Outputs land in ./reports/ :
  cloud_lufa_out_data.csv / cloud_evaluation_results.csv / cloud.html
  naive_*                 / naive_*                       / naive.html
  crosslingual_*          / crosslingual_*                / crosslingual.html
  lufa_out_data.csv       / evaluation_results.csv        / agentic.html   (agentic: no prefix)
  general_lufa_out_data.csv / general_evaluation_results.csv / general_report.html

Usage:
  python src/generate_report.py
  python src/generate_report.py --root . --out reports
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from run_simulation import OUTPUT_COLUMNS as LUFA_COLUMNS
from evaluate import EVAL_COLUMNS
from csv_utils import normalize_legacy_columns
from dashboard_generator import generate_dashboard

JUDGE_LLM_CONST = "tensortemplar/prometheus2:8x7b-Q4_K_S"

# name, filename prefix ("" = none), directories to scan (relative to --root)
CATEGORIES = [
    {"name": "cloud",        "prefix": "cloud",        "dirs": ["tests/cloud"]},
    {"name": "naive",        "prefix": "naive",        "dirs": ["tests/naive-rag"]},
    {"name": "crosslingual", "prefix": "crosslingual", "dirs": ["tests/cross-lingual-german"]},
    {"name": "agentic",      "prefix": "",             "dirs": ["tests/llama-3.1-8b",
                                                                "tests/llama-3.2-3b",
                                                                "tests/mistral-7b"]},
]

LUFA_NAME = "lufa_out_data.csv"
EVAL_NAME = "evaluation_results.csv"


def _find_csvs(root: Path, dirs, filename):
    """Recursively find real CSVs named exactly `filename` under the given dirs
    (skips anything containing 'bak')."""
    found = []
    for d in dirs:
        base = root / d
        if not base.exists():
            print(f"   [skip] {base} not found")
            continue
        for p in base.rglob(filename):
            if "bak" in p.name.lower():
                continue
            if p.is_file():
                found.append(p)
    return sorted(set(found))


def _load_aligned(files, canonical):
    """Read each CSV, normalise legacy columns, reindex to `canonical`, and return
    a list of frames (empty/unreadable files skipped)."""
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, on_bad_lines="skip")
        except Exception as e:
            print(f"   [warn] could not read {f}: {e}")
            continue
        if df is None or df.empty:
            continue
        df = normalize_legacy_columns(df)          # rename source{n}_score, repair language
        for c in canonical:
            if c not in df.columns:
                df[c] = ""
        frames.append(df[canonical])
    return frames


def _combine(frames, canonical):
    if not frames:
        return pd.DataFrame(columns=canonical)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.where(pd.notna(combined), "")
    combined = combined.drop_duplicates(ignore_index=True)   # drop accidental exact dup rows
    return combined[canonical]


def _apply_rewrites(df, prefix, is_eval):
    """Prefix model columns and (for eval) stamp the judge model."""
    if df.empty:
        return df

    def _pref(v):
        s = str(v).strip()
        if not prefix or s in ("", "nan", "none"):
            return v
        return f"{prefix}/{s}"

    if "base_model_used" in df.columns:
        df["base_model_used"] = df["base_model_used"].map(_pref)

    if is_eval:
        if "rag_base_model" in df.columns:
            # ensure rag_base_model is populated (dashboard groups by it) before prefixing
            if "base_model_used" in df.columns:
                # NB: base_model_used is already prefixed above; use it as the fallback source
                df["rag_base_model"] = [
                    rb if str(rb).strip() not in ("", "nan", "none") else bm
                    for rb, bm in zip(df["rag_base_model"], df["base_model_used"])
                ]
            # prefix only the rows that weren't already filled from (already-prefixed) base_model_used
            df["rag_base_model"] = df["rag_base_model"].map(
                lambda v: v if (str(v).startswith(f"{prefix}/") or not prefix) else _pref(v)
            )
        if "judge_llm" in df.columns:
            df["judge_llm"] = JUDGE_LLM_CONST
    return df


def _build_dashboard(eval_df, lufa_df, out_html):
    """Generate an HTML dashboard, preferring the eval frame; never raises."""
    try:
        df = eval_df if (eval_df is not None and not eval_df.empty) else (lufa_df.copy() if lufa_df is not None else None)
        if df is None or df.empty:
            print(f"   [dashboard] nothing to render for {out_html.name}")
            return False
        if "rag_base_model" not in df.columns and "base_model_used" in df.columns:
            df["rag_base_model"] = df["base_model_used"]
        out_html.parent.mkdir(parents=True, exist_ok=True)
        generate_dashboard(df, str(out_html))
        return True
    except Exception as e:
        print(f"   [dashboard] failed for {out_html.name}: {e}")
        return False


def _fname(prefix, base):
    return f"{prefix}_{base}" if prefix else base


def main():
    parser = argparse.ArgumentParser(description="Combine per-model result CSVs into category + general reports.")
    parser.add_argument("--root", default=".", help="Repository root containing the tests/ tree.")
    parser.add_argument("--out", default="reports", help="Output directory for combined reports.")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_lufa_frames = []
    all_eval_frames = []

    for cat in CATEGORIES:
        name, prefix, dirs = cat["name"], cat["prefix"], cat["dirs"]
        print(f"\n=== Category: {name} (prefix={prefix or '<none>'}) ===")

        lufa_files = _find_csvs(root, dirs, LUFA_NAME)
        eval_files = _find_csvs(root, dirs, EVAL_NAME)
        print(f"   lufa_out files: {len(lufa_files)} | evaluation files: {len(eval_files)}")
        for f in lufa_files + eval_files:
            print(f"      - {f}")

        lufa_df = _apply_rewrites(_combine(_load_aligned(lufa_files, LUFA_COLUMNS), LUFA_COLUMNS), prefix, is_eval=False)
        eval_df = _apply_rewrites(_combine(_load_aligned(eval_files, EVAL_COLUMNS), EVAL_COLUMNS), prefix, is_eval=True)

        lufa_out = out_dir / _fname(prefix, LUFA_NAME)
        eval_out = out_dir / _fname(prefix, EVAL_NAME)
        html_out = out_dir / (f"{prefix}.html" if prefix else f"{name}.html")

        lufa_df.to_csv(lufa_out, index=False)
        eval_df.to_csv(eval_out, index=False)
        print(f"   -> {lufa_out.name} ({len(lufa_df)} rows) | {eval_out.name} ({len(eval_df)} rows)")
        if _build_dashboard(eval_df, lufa_df, html_out):
            print(f"   -> dashboard {html_out.name}")

        # accumulate the already-prefixed frames for the general combination
        if not lufa_df.empty:
            all_lufa_frames.append(lufa_df)
        if not eval_df.empty:
            all_eval_frames.append(eval_df)

    # ── 5. General combination of all categories ──
    print("\n=== General combination ===")
    gen_lufa = _combine(all_lufa_frames, LUFA_COLUMNS)
    gen_eval = _combine(all_eval_frames, EVAL_COLUMNS)
    gen_lufa_out = out_dir / "general_lufa_out_data.csv"
    gen_eval_out = out_dir / "general_evaluation_results.csv"
    gen_html = out_dir / "general_report.html"
    gen_lufa.to_csv(gen_lufa_out, index=False)
    gen_eval.to_csv(gen_eval_out, index=False)
    print(f"   -> {gen_lufa_out.name} ({len(gen_lufa)} rows) | {gen_eval_out.name} ({len(gen_eval)} rows)")
    if _build_dashboard(gen_eval, gen_lufa, gen_html):
        print(f"   -> dashboard {gen_html.name}")

    print(f"\n[Report] Done. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
