#!/usr/bin/env python3
"""
generate_figures.py: publication-quality result figures for the thesis, computed directly
from reports/general_evaluation_results.csv and reports/general_lufa_out_data.csv (produced
by generate_report.py).

Outputs PNG (300 dpi) + PDF (vector) into thesis/figures/ (override with --out):
  fig5_1_answer_quality        judge metrics per system
  fig5_2_retrieval             MRR / nDCG@5 / Recall@5 per system
  fig5_3_agentic_vs_naive      RQ1 on the RETRY subset, the only rows where the two differ
  fig5_4_language              faithfulness and Token-F1, English vs French
  fig5_5_crosslingual          German cross-lingual vs monolingual retrieval
  fig5_6_latency               end-to-end latency distribution against the 60 s H3 threshold
  fig5_7_citation_precision    citation accuracy (regex) and Precision@1/3/5

Two data-integrity rules are enforced here, not left to the caller:

1. An UNJUDGED row carries 0.0 in all three judge columns (the deterministic pass writes that
   placeholder, the judge pass overwrites it). Those rows are masked to NaN, so a partially
   judged directory cannot silently halve its own mean.

2. RQ1 is plotted on the retry subset only. The naive rows for one-attempt questions were
   COPIED from the agentic run, because attempt 1 never invokes the query rewriter and is
   therefore already the naive pipeline. On that subset a row would be compared with itself
   and every delta is exactly zero. Only questions the reflector sent back for another pass
   can distinguish the two systems.

Run:  python src/generate_figures.py
      python src/generate_figures.py --out thesis/staging/figures
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

EVAL_CSV = "reports/general_evaluation_results.csv"
LUFA_CSV = "reports/general_lufa_out_data.csv"

# system label (A-F) -> the rag_base_model value in the combined CSV
SYSTEMS = [
    ("A: Llama 3.2 3B",      "llama3.2:3b"),
    ("B: Llama 3.1 8B",      "llama3.1:8b"),
    ("C: Mistral 7B",        "mistral:7b"),
    ("D: GPT-5.4-mini",      "cloud/gpt-5.4-mini"),
    ("E: Claude Haiku 4.5",  "cloud/claude-haiku-4-5"),
    ("F: Gemini 3.1 FL",     "cloud/gemini-3.1-flash-lite"),
]
LOCAL_SYSTEMS = SYSTEMS[:3]

JUDGE = ["answer_relevance", "faithfulness", "context_precision"]

# The local generators were re-run under the "-gpu" Modelfile variants, so the stored tag
# changed (llama3.2:3b-instruct-q4_K_M -> llama3.2:3b-gpu). Matching on the literal string
# silently produced EMPTY bars for systems A-C. Compare on a canonical key instead, so both
# the pre- and post-re-run tags resolve to the same system and any future retag also works.
_TAG_SUFFIXES = ("-gpu", "-instruct-q4_k_m", ":latest")


def canon(tag) -> str:
    """Canonical model key: lowercase, prefix preserved, known variant suffixes stripped."""
    s = str(tag).strip().lower()
    for suf in _TAG_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.rstrip(":-")


# colour-blind-safe palette (Okabe-Ito subset); local = blue, cloud = orange
LOCAL_C = "#0072B2"
CLOUD_C = "#D55E00"
NAIVE_C = "#999999"
ACCENT = ["#009E73", "#56B4E9", "#E69F00"]

plt.rcParams.update({
    "font.size": 10,
    # Times New Roman is the project default; serif fallback if the font is absent.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "figure.dpi": 120,
})

NUMERIC = JUDGE + [
    "token_f1_score", "sentence_bleu_score", "rouge1", "rouge2", "rougeL", "meteor",
    "mrr", "ndcg_at_k", "recall_1", "recall_3", "recall_5",
    "precision_1", "precision_3", "precision_5", "citation_accuracy_regex",
    "retrieval_latency_s", "ttft_s", "end_to_end_latency_s", "gpu_vram_mb",
]

OUT_DIR = Path("thesis/figures")


def _load():
    ev = pd.read_csv(EVAL_CSV, low_memory=False)
    lu = pd.read_csv(LUFA_CSV, low_memory=False)
    for c in NUMERIC:
        ev[c] = pd.to_numeric(ev[c], errors="coerce") if c in ev.columns else np.nan
    for c in ["attempts", "end_to_end_latency_s", "ttft_s", "retrieval_latency_s", "gpu_vram_mb"]:
        lu[c] = pd.to_numeric(lu[c], errors="coerce") if c in lu.columns else np.nan

    # Rule 1: mask not-yet-judged rows (all three judge cells zero) so a partially judged
    # directory cannot drag its own mean toward zero.
    unjudged = (ev[JUDGE].fillna(0) == 0).all(axis=1)
    ev.loc[unjudged, JUDGE] = np.nan
    if unjudged.any():
        print(f"[figures] masked {int(unjudged.sum())} not-yet-judged rows out of {len(ev)}")

    ev["lang"] = ev["language"].astype(str).str.lower().str[:2].replace({"ge": "de"})
    ev["_key"] = ev["rag_base_model"].map(canon)
    lu["_key"] = lu["base_model_used"].map(canon)
    ev["question_id"] = ev["question_id"].astype(str)
    lu["question_id"] = lu["question_id"].astype(str)
    return ev, lu


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"   wrote {OUT_DIR / name}.png / .pdf")


def _means(df, model, cols):
    s = df[df["_key"] == canon(model)]
    return [s[c].mean() for c in cols]


def _grouped_bars(labels, series, series_names, title, ylabel, colors=None,
                  ymax=1.0, figsize=(9, 4.5), note=None):
    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=figsize)
    for i, (vals, sname) in enumerate(zip(series, series_names)):
        col = (colors[i] if colors else None)
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=sname, color=col)
        for b, v in zip(bars, vals):
            if v == v:  # not NaN
                ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.012, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, ymax * 1.14)
    ax.legend(frameon=False, ncol=len(series_names), loc="upper left", fontsize=8)
    if note:
        ax.text(0.5, -0.32, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, style="italic", color="#444444")
    return fig


# ----------------------------------------------------------------------------------
def fig_answer_quality(ev):
    labels = [l for l, _ in SYSTEMS]
    models = [m for _, m in SYSTEMS]
    n = [int(ev[(ev._key == canon(m))]["faithfulness"].notna().sum()) for m in models]
    series = [[_means(ev, m, [c])[0] for m in models] for c in JUDGE]
    counts = ", ".join(f"{lab.split(':')[0]}={c}" for lab, c in zip(labels, n))
    fig = _grouped_bars(labels, series,
                        ["Answer relevance", "Faithfulness", "Context precision"],
                        "Answer quality by system: judge is Prometheus-2 8x7B",
                        "Mean score", colors=ACCENT,
                        note=f"Judged rows per system: {counts}. "
                             "All systems, local and cloud, were scored by the same judge model.")
    _save(fig, "fig5_1_answer_quality")


def fig_retrieval(ev):
    labels = [l for l, _ in SYSTEMS]
    models = [m for _, m in SYSTEMS]
    mrr = [_means(ev, m, ["mrr"])[0] for m in models]
    ndcg = [_means(ev, m, ["ndcg_at_k"])[0] for m in models]
    r5 = [_means(ev, m, ["recall_5"])[0] for m in models]
    fig = _grouped_bars(labels, [mrr, ndcg, r5], ["MRR", "nDCG@5", "Recall@5"],
                        "Retrieval performance by system", "Mean score",
                        colors=ACCENT, ymax=0.45,
                        note="Systems D-F share one retrieval pass: the cloud arms reuse the same local "
                             "hybrid retriever, so their retrieval scores are identical by construction.")
    _save(fig, "fig5_2_retrieval")


def fig_agentic_vs_naive(ev, lu):
    """RQ1 on the retry subset. See rule 2 in the module docstring."""
    pairs = [("A: Llama 3.2 3B", "llama3.2:3b", "naive/llama3.2:3b"),
             ("B: Llama 3.1 8B", "llama3.1:8b", "naive/llama3.1:8b"),
             ("C: Mistral 7B",   "mistral:7b",  "naive/mistral:7b")]
    metrics = ["mrr", "recall_5", "token_f1_score", "rougeL", "citation_accuracy_regex"]
    mnames = ["MRR", "Recall@5", "Token-F1", "ROUGE-L", "Citation acc."]

    # Only systems with data get a panel. An empty placeholder axis would waste most of the
    # canvas, and clearing its ticks propagates through sharey and strips the y-axis from the
    # panels that do have data.
    panels, pending = [], []
    for lab, ag, na in pairs:
        AL = lu[lu._key == canon(ag)]
        ids = set(AL.loc[AL["attempts"] > 1, "question_id"])
        A = ev[(ev._key == canon(ag)) & ev.question_id.isin(ids)].set_index("question_id")
        N = ev[(ev._key == canon(na)) & ev.question_id.isin(ids)].set_index("question_id")
        both = list(A.index.intersection(N.index))
        if both:
            panels.append((lab, A.loc[both], N.loc[both], len(both)))
        else:
            pending.append(f"{lab} ({len(ids)} retry queries)")

    if not panels:
        print("   skipped fig5_3: no retry rows available on both sides yet")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4), sharey=True,
                             squeeze=False)
    axes = axes[0]
    x = np.arange(len(metrics))
    w = 0.38
    for ax, (lab, A, N, nb) in zip(axes, panels):
        ag = [A[c].mean() for c in metrics]
        na = [N[c].mean() for c in metrics]
        ax.bar(x - w / 2, ag, w, label="Agentic (up to 3 passes)", color=LOCAL_C)
        ax.bar(x + w / 2, na, w, label="Naive (single pass)", color=NAIVE_C)
        for i, (a, b) in enumerate(zip(ag, na)):
            if a == a:
                ax.text(x[i] - w / 2, a + 0.008, f"{a:.2f}", ha="center", fontsize=7)
            if b == b:
                ax.text(x[i] + w / 2, b + 0.008, f"{b:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(mnames, rotation=25, ha="right")
        ax.set_title(f"{lab}  (n = {nb} retry queries)", fontsize=10)
        ax.set_ylim(0, 0.45)
    axes[0].set_ylabel("Mean score")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("RQ1: agentic loop vs naive single pass, on the queries the reflector sent back",
                 fontsize=11)
    note = ("Restricted to queries with more than one agentic attempt. Queries answered in a single "
            "attempt are excluded because attempt 1 does not invoke the query rewriter, so the two "
            "systems are identical there by construction. Judge metrics are omitted: the naive judge "
            "pass over these rows had not completed when this figure was produced.")
    if pending:
        note += " Awaiting naive regeneration: " + "; ".join(pending) + "."
    fig.text(0.5, -0.08, note, ha="center", va="top", fontsize=7.5, style="italic",
             color="#444444", wrap=True)
    _save(fig, "fig5_3_agentic_vs_naive")


def fig_language(ev):
    labels = [l for l, _ in SYSTEMS]
    models = [m for _, m in SYSTEMS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (col, ttl, ymax) in zip(axes, [("faithfulness", "Faithfulness", 1.0),
                                           ("token_f1_score", "Token-F1", 0.3)]):
        en = [ev[(ev._key == canon(m)) & (ev.lang == "en")][col].mean() for m in models]
        fr = [ev[(ev._key == canon(m)) & (ev.lang == "fr")][col].mean() for m in models]
        x = np.arange(len(labels)); w = 0.38
        ax.bar(x - w / 2, en, w, label="English (n = 226)", color=LOCAL_C)
        ax.bar(x + w / 2, fr, w, label="French (n = 200)", color=CLOUD_C)
        for i, (a, b) in enumerate(zip(en, fr)):
            ax.text(x[i] - w / 2, a + ymax * 0.012, f"{a:.2f}", ha="center", fontsize=7)
            ax.text(x[i] + w / 2, b + ymax * 0.012, f"{b:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(ttl); ax.set_ylim(0, ymax * 1.16); ax.set_ylabel(f"Mean {ttl.lower()}")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Performance by query language: no systematic French penalty", fontsize=11)
    _save(fig, "fig5_4_language")


def fig_crosslingual(ev):
    cols = ["mrr", "ndcg_at_k", "recall_5", "recall_1"]
    names = ["MRR", "nDCG@5", "Recall@5", "Recall@1"]
    mono = _means(ev, "llama3.1:8b", cols)
    cross = _means(ev, "crosslingual/llama3.1:8b", cols)
    n_cross = int(ev[ev._key == canon("crosslingual/llama3.1:8b")].shape[0])
    fig = _grouped_bars(names, [mono, cross],
                        [f"Monolingual EN and FR (n = 426)", f"Cross-lingual German (n = {n_cross})"],
                        "RQ2: cross-lingual vs monolingual retrieval, Llama 3.1 8B",
                        "Mean score", colors=[LOCAL_C, CLOUD_C], ymax=0.32, figsize=(8, 4.4),
                        note="The German condition runs without translation: the query is embedded as "
                             "written and matched against the English and French provisions directly.")
    _save(fig, "fig5_5_crosslingual")


def fig_latency(lu):
    """H3 evidence: end-to-end latency distribution against the 60 s threshold."""
    sets = [(l, m) for l, m in LOCAL_SYSTEMS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    data, labels, pcts = [], [], []
    for lab, m in sets:
        s = lu[lu._key == canon(m)]["end_to_end_latency_s"].dropna()
        if not len(s):
            continue
        data.append(s.values)
        labels.append(f"{lab}\n(n = {len(s)})")
        pcts.append(100.0 * float((s < 60).mean()))
    # matplotlib 3.9 renamed boxplot's `labels` to `tick_labels`.
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor(LOCAL_C)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black")
    ax.axhline(60, color=CLOUD_C, linestyle="--", linewidth=1.6)
    ax.text(0.02, 62, "H3 threshold: 60 s", color=CLOUD_C, fontsize=8.5, va="bottom")
    ax.set_ylabel("End-to-end latency (s)")
    ax.set_title("Latency distribution, local agentic systems")
    # Fit to the tallest whisker (Q3 + 1.5 IQR), not a percentile: an under-tall limit clips the
    # whisker line and makes the widest distribution look truncated rather than wide.
    def _whisker_top(d):
        q1, q3 = np.percentile(d, [25, 75])
        cap = q3 + 1.5 * (q3 - q1)
        inside = d[d <= cap]
        return float(inside.max()) if len(inside) else float(q3)
    ax.set_ylim(0, (max(_whisker_top(d) for d in data) * 1.06) if data else 300)

    ax = axes[1]
    x = np.arange(len(labels))
    cols = [LOCAL_C if p >= 75 else CLOUD_C for p in pcts]
    bars = ax.bar(x, pcts, 0.55, color=cols)
    for b, p in zip(bars, pcts):
        ax.text(b.get_x() + b.get_width() / 2, p + 1.5, f"{p:.1f}%", ha="center", fontsize=8.5)
    ax.axhline(75, color="black", linestyle="--", linewidth=1.4)
    ax.text(len(labels) - 0.45, 77, "H3 requires 75%", fontsize=8.5, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([l.split("\n")[0] for l in labels], rotation=20, ha="right")
    ax.set_ylabel("Queries answered in under 60 s (%)")
    ax.set_ylim(0, 105)
    ax.set_title("H3 latency criterion: not met by any local system")

    fig.suptitle("H3: end-to-end latency against the 60 s target on a 6 GB GPU", fontsize=11)
    fig.text(0.5, -0.05,
             "Latency covers the complete agentic pipeline: retrieval, generation, reflection and any "
             "corrective passes. Cloud systems D-F are excluded because their responses were collected "
             "without per-query timing instrumentation.",
             ha="center", va="top", fontsize=7.5, style="italic", color="#444444")
    _save(fig, "fig5_6_latency")


def fig_citation_precision(ev):
    labels = [l for l, _ in LOCAL_SYSTEMS]
    models = [m for _, m in LOCAL_SYSTEMS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    vals = [_means(ev, m, ["citation_accuracy_regex"])[0] for m in models]
    bars = ax.bar(np.arange(len(labels)), vals, 0.5, color=LOCAL_C)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean citation accuracy")
    ax.set_ylim(0, 0.45)
    ax.set_title("Citation accuracy: deterministic article and clause matching")

    ax = axes[1]
    x = np.arange(len(labels)); w = 0.26
    for i, (c, nm) in enumerate(zip(["precision_1", "precision_3", "precision_5"],
                                    ["P@1", "P@3", "P@5"])):
        v = [_means(ev, m, [c])[0] for m in models]
        bars = ax.bar(x + (i - 1) * w, v, w, label=nm, color=ACCENT[i])
        for b, val in zip(bars, v):
            if val == val:
                ax.text(b.get_x() + b.get_width() / 2, val + 0.004, f"{val:.3f}",
                        ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean precision")
    ax.set_ylim(0, 0.26)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.set_title("Precision@k")

    fig.suptitle("Citation grounding and precision at k, local agentic systems", fontsize=11)
    fig.text(0.5, -0.05,
             "Citation accuracy scores 1.0 when the answer cites the gold article and clause, 0.5 when "
             "the article matches but the clause does not, and 0.0 otherwise. Precision@k is low by "
             "construction: most questions have a single gold chunk, which caps P@5 at 0.20.",
             ha="center", va="top", fontsize=7.5, style="italic", color="#444444")
    _save(fig, "fig5_7_citation_precision")


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description="Regenerate the thesis result figures.")
    ap.add_argument("--out", default="thesis/figures")
    args = ap.parse_args()
    OUT_DIR = Path(args.out)

    if not Path(EVAL_CSV).exists():
        print(f"[figures] {EVAL_CSV} not found. Run: python src/generate_report.py")
        return 1
    ev, lu = _load()

    fig_answer_quality(ev)
    fig_retrieval(ev)
    fig_agentic_vs_naive(ev, lu)
    fig_language(ev)
    fig_crosslingual(ev)
    fig_latency(lu)
    fig_citation_precision(ev)

    print(f"\n[figures] Done. 7 figures (PNG + PDF) in {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
