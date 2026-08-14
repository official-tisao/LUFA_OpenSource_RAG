#!/usr/bin/env python3
"""
thesis_stats.py: single source of truth for every number quoted in Chapters 5 and 6.

Reads reports/general_evaluation_results.csv (+ the lufa ledger for attempt counts) and
prints a complete, labelled statistics dump. Nothing here is hand-typed into the thesis:
the tables in Chapter 5 are generated from this output so a re-run cannot leave the prose
and the data disagreeing.

Rows whose judge cells are blank are EXCLUDED from judge means and the count of judged
rows is reported alongside every judge figure, because the naive-RAG rebuild is still
completing and its later rows are not yet judged.

Usage:
  python src/thesis_stats.py                 # human-readable dump
  python src/thesis_stats.py --json out.json # machine-readable, for the figure scripts
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from generate_figures import canon  # noqa: E402

EVAL_CSV = "reports/general_evaluation_results.csv"
LUFA_CSV = "reports/general_lufa_out_data.csv"

# label -> canonical model key as it appears after generate_report.py prefixing
SYSTEMS = [
    ("A: Llama 3.2 3B (agentic)",   "llama3.2:3b"),
    ("B: Llama 3.1 8B (agentic)",   "llama3.1:8b"),
    ("C: Mistral 7B (agentic)",     "mistral:7b"),
    ("D: GPT-5.4-mini (cloud)",     "cloud/gpt-5.4-mini"),
    ("E: Claude Haiku 4.5 (cloud)", "cloud/claude-haiku-4-5"),
    ("F: Gemini 3.1 FL (cloud)",    "cloud/gemini-3.1-flash-lite"),
    ("Naive: Llama 3.2 3B",         "naive/llama3.2:3b"),
    ("Naive: Llama 3.1 8B",         "naive/llama3.1:8b"),
    ("Naive: Mistral 7B",           "naive/mistral:7b"),
    ("X-ling: Llama 3.1 8B (DE)",   "crosslingual/llama3.1:8b"),
]

JUDGE = ["answer_relevance", "faithfulness", "context_precision"]
LEXICAL = ["token_f1_score", "sentence_bleu_score", "rouge1", "rouge2", "rougeL", "meteor"]
RETRIEVAL = ["mrr", "ndcg_at_k", "recall_1", "recall_3", "recall_5",
             "precision_1", "precision_3", "precision_5"]
PERF = ["retrieval_latency_s", "ttft_s", "end_to_end_latency_s",
        "gpu_vram_mb", "system_ram_mb"]
CITATION = ["citation_accuracy_regex"]

NUMERIC = JUDGE + LEXICAL + RETRIEVAL + PERF + CITATION


def load():
    ev = pd.read_csv(EVAL_CSV, low_memory=False)
    lu = pd.read_csv(LUFA_CSV, low_memory=False)
    for c in NUMERIC:
        if c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
        else:
            ev[c] = pd.NA

    # An UNJUDGED row carries 0.0 in all three judge columns: the deterministic pass writes
    # that placeholder and the judge pass later overwrites it. Verified that no genuinely
    # judged row in any completed directory scores 0 on even one of the three metrics, so
    # "all three are zero" is an unambiguous not-yet-judged signature. Mask those to NA,
    # otherwise the still-running naive rebuild silently halves its own judge means.
    unjudged = (ev[JUDGE].fillna(0) == 0).all(axis=1)
    ev.loc[unjudged, JUDGE] = pd.NA
    print(f"[stats] masked {int(unjudged.sum())} not-yet-judged rows "
          f"(all three judge cells zero) out of {len(ev)}", file=sys.stderr)

    ev["_key"] = ev["rag_base_model"].map(canon)
    ev["_lang"] = ev["language"].astype(str).str.lower().str[:2].replace({"ge": "de"})
    lu["_key"] = lu["base_model_used"].map(canon)
    lu["attempts"] = pd.to_numeric(lu.get("attempts"), errors="coerce")
    lu["end_to_end_latency_s"] = pd.to_numeric(lu.get("end_to_end_latency_s"), errors="coerce")
    lu["retrieval_latency_s"] = pd.to_numeric(lu.get("retrieval_latency_s"), errors="coerce")
    lu["ttft_s"] = pd.to_numeric(lu.get("ttft_s"), errors="coerce")
    return ev, lu


def block(ev, lu, label, key):
    e = ev[ev["_key"] == canon(key)]
    l_ = lu[lu["_key"] == canon(key)]
    if len(e) == 0 and len(l_) == 0:
        return None
    out = {"label": label, "key": key, "n_eval": int(len(e)), "n_lufa": int(len(l_))}

    for c in JUDGE:
        s = e[c].dropna()
        out[c] = round(float(s.mean()), 4) if len(s) else None
        out[c + "_n"] = int(len(s))
    for c in LEXICAL + RETRIEVAL + CITATION:
        s = e[c].dropna()
        out[c] = round(float(s.mean()), 4) if len(s) else None

    # latency lives in the lufa ledger (the retrieval/generation phase writes it there)
    for c in ["retrieval_latency_s", "ttft_s", "end_to_end_latency_s"]:
        s = l_[c].dropna() if c in l_.columns else pd.Series(dtype=float)
        if len(s):
            out[c + "_mean"] = round(float(s.mean()), 2)
            out[c + "_median"] = round(float(s.median()), 2)
            out[c + "_p95"] = round(float(s.quantile(0.95)), 2)
            out[c + "_n"] = int(len(s))
        else:
            out[c + "_mean"] = out[c + "_median"] = out[c + "_p95"] = None
            out[c + "_n"] = 0
    e2e = l_["end_to_end_latency_s"].dropna() if "end_to_end_latency_s" in l_.columns else pd.Series(dtype=float)
    out["pct_under_60s"] = round(100.0 * float((e2e < 60).mean()), 1) if len(e2e) else None

    for c in ["gpu_vram_mb", "system_ram_mb"]:
        s = pd.to_numeric(l_.get(c), errors="coerce").dropna() if c in l_.columns else pd.Series(dtype=float)
        out[c + "_mean"] = round(float(s.mean()), 0) if len(s) else None
        out[c + "_peak"] = round(float(s.max()), 0) if len(s) else None

    at = l_["attempts"].dropna()
    out["attempts_mean"] = round(float(at.mean()), 3) if len(at) else None
    out["attempts_gt1"] = int((at > 1).sum()) if len(at) else None
    g = l_.get("grounded")
    if g is not None:
        gs = g.astype(str).str.strip().str.lower()
        gs = gs[gs.isin(["true", "false", "1", "0"])]
        if len(gs):
            out["grounded_n"] = int(gs.isin(["true", "1"]).sum())
            out["grounded_total"] = int(len(gs))
            out["grounded_pct"] = round(100.0 * out["grounded_n"] / out["grounded_total"], 1)

    for lg in ("en", "fr", "de"):
        sub = e[e["_lang"] == lg]
        if len(sub):
            out[f"n_{lg}"] = int(len(sub))
            out[f"faithfulness_{lg}"] = round(float(sub["faithfulness"].dropna().mean()), 4) \
                if sub["faithfulness"].notna().any() else None
            out[f"token_f1_{lg}"] = round(float(sub["token_f1_score"].dropna().mean()), 4) \
                if sub["token_f1_score"].notna().any() else None
            out[f"mrr_{lg}"] = round(float(sub["mrr"].dropna().mean()), 4) \
                if sub["mrr"].notna().any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    ev, lu = load()
    rows = [b for lab, k in SYSTEMS if (b := block(ev, lu, lab, k))]

    def show(title, cols, fmt="{:.4f}", width=11):
        print(f"\n### {title}")
        print("system".ljust(30) + "n".rjust(6) + "".join(c[:width].rjust(width + 2) for c in cols))
        for r in rows:
            line = r["label"].ljust(30) + str(r["n_eval"]).rjust(6)
            for c in cols:
                v = r.get(c)
                line += (fmt.format(v) if isinstance(v, (int, float)) else "-").rjust(width + 2)
            print(line)

    print("=" * 100)
    print("THESIS STATISTICS DUMP")
    print(f"source: {EVAL_CSV} ({len(ev)} rows) + {LUFA_CSV} ({len(lu)} rows)")
    print("=" * 100)

    show("Judge metrics (Prometheus-2 8x7B, 3 separate prompts)", JUDGE)
    print("\njudged-row counts (blank judge cells excluded from the means above):")
    for r in rows:
        print(f"  {r['label']:<30} AR n={r.get('answer_relevance_n')}  "
              f"faith n={r.get('faithfulness_n')}  ctxp n={r.get('context_precision_n')}")

    show("Lexical overlap vs human gold answers", LEXICAL)
    show("Retrieval metrics", RETRIEVAL)
    show("Citation accuracy (deterministic regex)", CITATION)

    print("\n### Latency and hardware (from the lufa ledger)")
    hdr = ["e2e_med", "e2e_mean", "e2e_p95", "%<60s", "ttft_med", "retr_med", "vram_pk", "n"]
    print("system".ljust(30) + "".join(h.rjust(11) for h in hdr))
    for r in rows:
        vals = [r.get("end_to_end_latency_s_median"), r.get("end_to_end_latency_s_mean"),
                r.get("end_to_end_latency_s_p95"), r.get("pct_under_60s"),
                r.get("ttft_s_median"), r.get("retrieval_latency_s_median"),
                r.get("gpu_vram_mb_peak"), r.get("end_to_end_latency_s_n")]
        print(r["label"].ljust(30) + "".join(
            (f"{v:.1f}" if isinstance(v, float) else str(v) if v is not None else "-").rjust(11)
            for v in vals))

    print("\n### Agentic loop behaviour")
    print("system".ljust(30) + "attempts_mean".rjust(15) + "rows>1".rjust(9) +
          "grounded".rjust(11) + "grounded%".rjust(11))
    for r in rows:
        print(r["label"].ljust(30) +
              (f"{r['attempts_mean']:.3f}" if r.get("attempts_mean") is not None else "-").rjust(15) +
              str(r.get("attempts_gt1", "-")).rjust(9) +
              f"{r.get('grounded_n','-')}/{r.get('grounded_total','-')}".rjust(11) +
              (f"{r['grounded_pct']:.1f}" if r.get("grounded_pct") is not None else "-").rjust(11))

    print("\n### Per-language breakdown")
    print("system".ljust(30) + "".join(h.rjust(13) for h in
          ["n_en", "faith_en", "f1_en", "mrr_en", "n_fr", "faith_fr", "f1_fr", "mrr_fr"]))
    for r in rows:
        vals = [r.get("n_en"), r.get("faithfulness_en"), r.get("token_f1_en"), r.get("mrr_en"),
                r.get("n_fr"), r.get("faithfulness_fr"), r.get("token_f1_fr"), r.get("mrr_fr")]
        print(r["label"].ljust(30) + "".join(
            (f"{v:.4f}" if isinstance(v, float) else str(v) if v is not None else "-").rjust(13)
            for v in vals))

    # ------------------------------------------------------------------
    # Matched-pair agentic vs naive (RQ1).
    #
    # A plain mean-vs-mean comparison is INVALID here. The naive set was seeded from the
    # agentic rows that finished in a single attempt, i.e. the queries the reflector accepted
    # first time, which are the easy ones. Its freshly generated hard rows are not judged yet.
    # Comparing "naive over the easy subset" against "agentic over all 426" would flatter the
    # naive baseline by construction. Every comparison below is therefore restricted to the
    # question_ids that are judged on BOTH sides, and the pair count is always reported.
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("MATCHED-PAIR AGENTIC vs NAIVE (RQ1): same question_ids, judged on both sides")
    print("=" * 100)
    paired = {}
    for lab, ag, na in [("Llama 3.2 3B", "llama3.2:3b", "naive/llama3.2:3b"),
                        ("Llama 3.1 8B", "llama3.1:8b", "naive/llama3.1:8b"),
                        ("Mistral 7B",   "mistral:7b",  "naive/mistral:7b")]:
        A = ev[ev["_key"] == canon(ag)].set_index("question_id")
        N = ev[ev["_key"] == canon(na)].set_index("question_id")
        both = A.index.intersection(N.index)
        ok = [q for q in both if pd.notna(A.loc[q, "faithfulness"]) and pd.notna(N.loc[q, "faithfulness"])]
        if not ok:
            continue
        A, N = A.loc[ok], N.loc[ok]
        rec = {"label": lab, "n_pairs": len(ok), "coverage_pct": round(100.0 * len(ok) / 426, 1)}
        print(f"\n{lab}: {len(ok)} matched pairs ({rec['coverage_pct']}% of 426)")
        print("  metric".ljust(26) + "agentic".rjust(10) + "naive".rjust(10) +
              "delta".rjust(10) + "  (delta = agentic - naive)")
        for c in JUDGE + ["token_f1_score", "rougeL", "meteor", "mrr", "recall_5",
                          "citation_accuracy_regex"]:
            a, n = A[c].dropna().mean(), N[c].dropna().mean()
            if pd.isna(a) or pd.isna(n):
                continue
            rec[c] = {"agentic": round(float(a), 4), "naive": round(float(n), 4),
                      "delta": round(float(a - n), 4)}
            print(f"  {c:<24}{a:10.4f}{n:10.4f}{a - n:+10.4f}")
        # latency delta measures the cost of the retry loop only
        AL = lu[lu["_key"] == canon(ag)].set_index("question_id")
        NL = lu[lu["_key"] == canon(na)].set_index("question_id")
        lb = [q for q in ok if q in AL.index and q in NL.index]
        if lb:
            a = AL.loc[lb, "end_to_end_latency_s"].dropna()
            n = NL.loc[lb, "end_to_end_latency_s"].dropna()
            if len(a) and len(n):
                rec["e2e_median"] = {"agentic": round(float(a.median()), 2),
                                     "naive": round(float(n.median()), 2)}
                print(f"  {'e2e_latency_s (median)':<24}{a.median():10.2f}{n.median():10.2f}"
                      f"{a.median() - n.median():+10.2f}")
        paired[lab] = rec

    # ------------------------------------------------------------------
    # The RETRY subset is where RQ1 actually lives.
    #
    # The matched-pair block above returns deltas of exactly zero, and that is correct rather
    # than a defect: the naive rows for one-attempt questions were COPIED from the agentic run
    # (attempt 1 never invokes the rewriter, so it already IS the naive pipeline). On that
    # subset each row is compared with itself. The only questions where agentic and naive can
    # differ are those the reflector sent back for another pass, i.e. attempts > 1. Those rows
    # were regenerated at max_retries=1 for the naive arm, so THIS is the RQ1 contrast.
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("RETRY SUBSET (attempts > 1 in the agentic run): the only rows where RQ1 can differ")
    print("=" * 100)
    retry = {}
    for lab, ag, na in [("Llama 3.2 3B", "llama3.2:3b", "naive/llama3.2:3b"),
                        ("Llama 3.1 8B", "llama3.1:8b", "naive/llama3.1:8b"),
                        ("Mistral 7B",   "mistral:7b",  "naive/mistral:7b")]:
        AL = lu[lu["_key"] == canon(ag)]
        ids = set(AL.loc[AL["attempts"] > 1, "question_id"].astype(str))
        A = ev[(ev["_key"] == canon(ag)) & ev["question_id"].astype(str).isin(ids)].set_index("question_id")
        N = ev[(ev["_key"] == canon(na)) & ev["question_id"].astype(str).isin(ids)].set_index("question_id")
        both = list(A.index.intersection(N.index))
        if not both:
            print(f"\n{lab}: 0 of {len(ids)} retry rows regenerated on the naive side yet.")
            continue
        A, N = A.loc[both], N.loc[both]
        rec = {"label": lab, "n_retry_total": len(ids), "n_available": len(both)}
        print(f"\n{lab}: {len(both)} of {len(ids)} retry rows available on both sides")
        print("  metric".ljust(26) + "agentic".rjust(10) + "naive".rjust(10) +
              "delta".rjust(10) + "   n_ag/n_na")
        for c in JUDGE + ["token_f1_score", "rougeL", "meteor", "mrr", "recall_5",
                          "citation_accuracy_regex"]:
            a, n = A[c].dropna(), N[c].dropna()
            # A judge pass in flight leaves a handful of rows scored and the rest masked, which
            # yields a mean over n=3 that looks like a real result (values of exactly 1.0000 are
            # the giveaway). Require most of the subset on BOTH sides before reporting, and
            # always print the pair counts so a partial cell can never be mistaken for a final one.
            enough = len(a) >= 0.9 * len(both) and len(n) >= 0.9 * len(both)
            if not len(a) or not len(n):
                print(f"  {c:<24}{'-':>10}{'-':>10}{'':>10}"
                      f"   {len(a)}/{len(n)}  (judge pending)")
                continue
            if not enough:
                print(f"  {c:<24}{a.mean():10.4f}{n.mean():10.4f}{a.mean() - n.mean():+10.4f}"
                      f"   {len(a)}/{len(n)}  PARTIAL, DO NOT QUOTE")
                continue
            rec[c] = {"agentic": round(float(a.mean()), 4), "naive": round(float(n.mean()), 4),
                      "delta": round(float(a.mean() - n.mean()), 4), "n_ag": len(a), "n_na": len(n)}
            print(f"  {c:<24}{a.mean():10.4f}{n.mean():10.4f}{a.mean() - n.mean():+10.4f}"
                  f"   {len(a)}/{len(n)}")
        NL = lu[lu["_key"] == canon(na)].set_index("question_id")
        ALi = AL.set_index("question_id")
        lb = [q for q in both if q in NL.index]
        if lb:
            a = ALi.loc[lb, "end_to_end_latency_s"].dropna()
            n = NL.loc[lb, "end_to_end_latency_s"].dropna()
            if len(a) and len(n):
                rec["e2e_median"] = {"agentic": round(float(a.median()), 2),
                                     "naive": round(float(n.median()), 2)}
                print(f"  {'e2e_latency_s (median)':<24}{a.median():10.2f}{n.median():10.2f}"
                      f"{a.median() - n.median():+10.2f}")
                rec["attempts_mean_agentic"] = round(float(ALi.loc[lb, "attempts"].dropna().mean()), 2)
                print(f"  {'attempts (agentic)':<24}{rec['attempts_mean_agentic']:10.2f}"
                      f"{1.0:10.2f}")
        retry[lab] = rec

    if args.json:
        Path(args.json).write_text(
            json.dumps({"systems": rows, "paired_rq1": paired, "retry_subset": retry}, indent=2),
            encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
