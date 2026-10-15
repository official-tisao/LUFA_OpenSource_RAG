#!/usr/bin/env python3
"""
run_regen.py: drive the post-fix regeneration in three phases, in the order that keeps the
expensive step last.

  Phase A  generation      every directory, retry rows only
  Phase B  metrics         every directory, deterministic only, judge skipped
  Phase C  judge           every directory, last

Why the phases are separated rather than run per directory. The judge is Prometheus 8x7B and is
by far the slowest step; running it per directory would interleave it with generation and delay
the point at which any deterministic result exists. Doing all generation first, then all
deterministic metrics, means MRR, Recall@k, citation accuracy and the lexical scores are
available for the whole corpus before a single judge call is made. If the judge then has to be
abandoned, everything except the three judged metrics is already complete.

Only the rows queued by mark_for_regen.py are regenerated: a query grounded on its first attempt
never entered the corrective loop, so the pipeline change cannot have altered it.

The pre-change results stay readable throughout, in reports/ and reports/frozen_pre_regen/, so
thesis writing continues against the old numbers until the new ones land.

Run:
  python src/mark_for_regen.py          # queue the affected rows first
  python src/run_regen.py               # then this
  python src/run_regen.py --phase A     # or one phase at a time
  python src/run_regen.py --dry-run
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = [sys.executable, "-u"]

# Keys as backfill_dir.py names them. Cheapest model first, so a usable result arrives early.
DIRS = ["llama-3.2-3b", "cross-lingual-german", "llama-3.1-8b", "mistral-7b"]

PHASES = {
    "A": ("generation",           "2"),
    "B": ("deterministic metrics", "2b"),
    "C": ("judge",                "3"),
}


def ollama_up(timeout=4):
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=timeout).read(1)
        return True
    except Exception:
        return False


def run(label, cmd, dry_run):
    print("\n" + "=" * 78, flush=True)
    print(f"[{datetime.now():%H:%M:%S}] {label}", flush=True)
    print("  " + " ".join(str(c) for c in cmd), flush=True)
    print("=" * 78, flush=True)
    if dry_run:
        return 0
    t = time.perf_counter()
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    print(f"[{datetime.now():%H:%M:%S}] {label} rc={rc} in {(time.perf_counter()-t)/60:.1f} min",
          flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", action="append", choices=list(PHASES),
                    help="run only these phases; default is A then B then C")
    ap.add_argument("--dir", action="append", dest="dirs", choices=DIRS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="abort the whole run on the first non-zero exit. Off by default, so one "
                         "failing directory does not cost the others their results.")
    args = ap.parse_args()

    phases = args.phase or ["A", "B", "C"]
    dirs = args.dirs or DIRS

    if not args.dry_run and not ollama_up():
        print("[regen] Ollama is not answering on http://localhost:11434.")
        print("[regen] Start it with:  ollama serve")
        return 2

    print(f"[regen] phases {' -> '.join(phases)} over {len(dirs)} directory(ies)")
    print("[regen] pre-change results remain in reports/ and reports/frozen_pre_regen/")

    failures = []
    for ph in phases:
        name, batch = PHASES[ph]
        print(f"\n\n{'#'*78}\n### PHASE {ph}: {name.upper()} ({len(dirs)} directories)\n{'#'*78}",
              flush=True)
        for key in dirs:
            cmd = PY + ["src/backfill_dir.py", "--dir", key, "--batches", batch]
            if ph == "A":
                # Regenerate exactly the rows mark_for_regen queued: those are the ones whose
                # ttft_s was blanked. Rows that still carry it are skipped.
                cmd.append("--resume_regen")
            if ph in ("B", "C"):
                # The answers are new, so every stored score describes a previous answer.
                # For B this becomes --force_det, for C --force_judge. Without it metrics.py
                # keeps any non-zero cell it already has and the regeneration is invisible.
                cmd.append("--regenerate")
            rc = run(f"phase {ph} / {name} / {key}", cmd, args.dry_run)
            if rc:
                failures.append(f"{ph}:{key}")
                if args.stop_on_error:
                    print(f"[regen] stopping: {ph}/{key} exited {rc}")
                    return rc

    print("\n" + "=" * 78)
    if failures:
        print(f"[regen] FINISHED WITH {len(failures)} FAILURE(S): {', '.join(failures)}")
        print("[regen] Re-run just those with --phase and --dir.")
    else:
        print("[regen] All phases complete.")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
