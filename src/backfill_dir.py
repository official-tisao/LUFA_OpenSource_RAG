#!/usr/bin/env python3
"""
backfill_dir.py — run the full Chapter-4 metric backfill for ONE test directory.

Chains the three batches the evaluation is split into, so a directory can be
launched unattended and resumed after an interruption:

  batch 1  retrieval          -> source{n}_*, retrieval_latency_s, warmup_applied
  batch 2  answer generation  -> answer, attempts, grounded, ttft_s,
                                 end_to_end_latency_s, gpu_vram_mb, system_ram_mb
  batch 2b deterministic eval -> lexical + retrieval metrics, precision@k,
                                 citation_accuracy_regex
  batch 3  LLM judge          -> answer_relevance, faithfulness, context_precision
                                 (SEPARATE prompt per metric, forced re-judge)

Every batch writes row-by-row, so killing the process loses at most one row.
Re-running resumes: batches 1 and 2 use --force only when --regenerate is passed.

Known directories (use --dir <key>):
  llama-3.2-3b | cross-lingual-german | llama-3.1-8b | mistral-7b

Examples:
  # full regeneration of the fastest directory (~21h)
  python src/backfill_dir.py --dir llama-3.2-3b --regenerate

  # metrics only, reusing existing answers (no generation)
  python src/backfill_dir.py --dir mistral-7b --batches 2b,3

  # see the commands without running them
  python src/backfill_dir.py --dir llama-3.1-8b --regenerate --dry_run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# "-u" keeps child stdout unbuffered so progress is visible live in a redirected log —
# essential when a single batch runs for many hours.
PY = sys.executable
PYU = [sys.executable, "-u"]
JUDGE = "tensortemplar/prometheus2:8x7b-Q4_K_S"

# key -> (directory holding the CSVs, ollama model tag, no_translate)
# no_translate=True is the TRUE cross-lingual pipeline: no English bridge, the answer is
# produced in the question's language and judged in that language; a separate rendering
# in the benchmark language is stored for the lexical metrics only.
# All generators use the "-gpu" Modelfile variants (num_gpu 33 + num_ctx 8192).
# Verified 100% GPU: llama3.2:3b-gpu 3.1 GB, mistral:7b-gpu 5.5 GB, llama3.1:8b-gpu
# 5.8 GB of 6.0 GB. The stock tags silently ran partly on CPU (llama3.2:3b at
# 77%/23% CPU/GPU because its 131k default context made the KV cache ~18 GB), which
# would have turned every latency measurement into a CPU measurement.
DIRECTORIES = {
    "llama-3.2-3b": (
        "tests/llama-3.2-3b/Judge-Prometheus-8x7b-v2.0",
        "llama3.2:3b-gpu", False,
    ),
    "cross-lingual-german": (
        "tests/cross-lingual-german/llama-3.1-8b/Judge-Prometheus-8x7b-v2.0",
        "llama3.1:8b-gpu", True,
    ),
    "llama-3.1-8b": (
        "tests/llama-3.1-8b/Judge-Prometheus-8x7b-v2.0",
        "llama3.1:8b-gpu", False,
    ),
    "mistral-7b": (
        "tests/mistral-7b/Judge-Prometheus-8x7b-v2.0",
        "mistral:7b-gpu", False,
    ),
}

# The three monolingual directories ask the SAME 426 questions of the SAME retriever, and
# `_retrieve_nodes` is deterministic — so retrieval is measured once and copied. The
# German set asks different questions in a different language and is always measured
# on its own.
RETRIEVAL_SOURCE = "llama-3.2-3b"
RETRIEVAL_SHARERS = {"llama-3.1-8b", "mistral-7b"}

# Recommended order: fastest / smallest first so complete results arrive early.
DEFAULT_ORDER = ["llama-3.2-3b", "cross-lingual-german", "llama-3.1-8b", "mistral-7b"]


def _run(label, cmd, dry_run=False):
    printable = " ".join(str(c) for c in cmd)
    print("\n" + "=" * 78)
    print(f"[{label}] {printable}")
    print("=" * 78, flush=True)
    if dry_run:
        return 0
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO))
    mins = (time.perf_counter() - start) / 60.0
    print(f"[{label}] finished rc={proc.returncode} in {mins:.1f} min", flush=True)
    if proc.returncode != 0:
        print(f"[{label}] NON-ZERO EXIT — stopping so the failure is not masked.")
    return proc.returncode


def backfill(key, batches, regenerate, max_retries, top_k, dry_run, no_reuse=False,
             resume_regen=False):
    rel_dir, model_tag, no_translate = DIRECTORIES[key]
    d = REPO / rel_dir
    lufa = d / "lufa_out_data.csv"
    evalr = d / "evaluation_results.csv"
    # Each directory ships its own ground truth — essential for cross-lingual-german,
    # whose 20 `test_de_*` questions are disjoint from the main 426-question set.
    test = d / "combined_test_data_and_ground_truth.csv"

    for p in (lufa, test):
        if not p.exists():
            print(f"[backfill] ERROR: missing {p}")
            return 1

    print(f"\n### {key}  ({rel_dir})")
    print(f"    model={model_tag}  regenerate={regenerate}  no_translate={no_translate}  "
          f"batches={','.join(batches)}")

    if "1" in batches:
        share = (key in RETRIEVAL_SHARERS) and not no_reuse
        if share:
            # Reuse the already-measured retrieval instead of recomputing an identical one.
            src_dir, _, _ = DIRECTORIES[RETRIEVAL_SOURCE]
            src_lufa = REPO / src_dir / "lufa_out_data.csv"
            if not src_lufa.exists():
                print(f"[{key}] retrieval source {src_lufa} missing — measuring locally instead.")
                share = False
            else:
                cmd = PYU + ["src/reuse_retrieval.py", "--source", str(src_lufa), "--target", str(lufa)]
                if _run(f"{key} batch1 retrieval (reused from {RETRIEVAL_SOURCE})", cmd, dry_run):
                    return 1
        if not share:
            cmd = PYU + ["src/retrieval.py", "--input", str(test), "--output", str(lufa),
                   "--top_k", str(top_k), "--no_dashboard"]
            if regenerate:
                cmd.append("--force")
            if _run(f"{key} batch1 retrieval", cmd, dry_run):
                return 1

    if "2" in batches:
        cmd = PYU + ["src/answer_generator.py", "--input", str(test), "--output", str(lufa),
               "--mode", "local", "--max_retries", str(max_retries),
               "--llm_model", model_tag, "--no_dashboard"]
        if resume_regen:
            # Correct way to continue an interrupted regeneration: skip only rows that
            # already carry generation telemetry, regenerate the rest.
            cmd.append("--resume_regen")
        elif regenerate:
            cmd.append("--force")
        if no_translate:
            cmd += ["--no_translate", "--metrics_language", "en"]
        if _run(f"{key} batch2 generation", cmd, dry_run):
            return 1

    if "2b" in batches:
        cmd = PYU + ["src/metrics.py", "--lufa_csv", str(lufa), "--test_csv", str(test),
               "--out_csv", str(evalr), "--no_judge", "--no_dashboard"]
        if _run(f"{key} batch2b deterministic", cmd, dry_run):
            return 1

    if "3" in batches:
        # Separate prompt per metric; forced because the answers are newly generated,
        # so any stored judge score refers to a previous answer.
        cmd = PYU + ["src/metrics.py", "--lufa_csv", str(lufa), "--test_csv", str(test),
               "--out_csv", str(evalr), "--judge_llm", JUDGE,
               "--separate_prompts", "--no_dashboard"]
        if regenerate:
            cmd.append("--force_judge")
        if _run(f"{key} batch3 judge", cmd, dry_run):
            return 1

    print(f"\n### {key} COMPLETE")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Run the Chapter-4 metric backfill for a test directory.")
    ap.add_argument("--dir", action="append", choices=list(DIRECTORIES) + ["all"],
                    help="Directory key to process (repeatable). 'all' uses the fast-first order.")
    ap.add_argument("--batches", default="1,2,2b,3",
                    help="Comma-separated subset of 1,2,2b,3 (default: all).")
    ap.add_argument("--regenerate", action="store_true",
                    help="Force re-retrieval, answer regeneration and re-judging. WITHOUT this "
                         "flag completed rows are skipped and existing answers are kept.")
    ap.add_argument("--max_retries", type=int, default=3,
                    help="Agentic corrective-loop depth for generation (default 3).")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--resume_regen", action="store_true",
                    help="Continue an interrupted --regenerate run: only rows lacking generation "
                         "telemetry (ttft_s) are regenerated. Safer than re-running --regenerate.")
    ap.add_argument("--no_reuse", action="store_true",
                    help="Measure retrieval separately per directory instead of copying the "
                         "shared, deterministic result from the reference run.")
    ap.add_argument("--dry_run", action="store_true", help="Print the commands without running them.")
    args = ap.parse_args()

    keys = args.dir or ["llama-3.2-3b"]
    if "all" in keys:
        keys = DEFAULT_ORDER
    batches = [b.strip() for b in args.batches.split(",") if b.strip()]

    if args.regenerate:
        print("*** --regenerate: existing answers WILL be overwritten. "
              "Snapshots live in backfill_backup/, and git holds the committed copies. ***")

    t0 = time.perf_counter()
    for key in keys:
        if backfill(key, batches, args.regenerate, args.max_retries, args.top_k,
                    args.dry_run, args.no_reuse, args.resume_regen):
            print(f"\n[backfill] Stopped on '{key}'.")
            return 1
    print(f"\n[backfill] All done in {(time.perf_counter() - t0) / 3600:.2f} h: {', '.join(keys)}")
    print("[backfill] Next: python src/generate_report.py && python src/generate_figures.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
