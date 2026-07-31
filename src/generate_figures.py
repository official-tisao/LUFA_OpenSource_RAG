#!/usr/bin/env python3
"""
generate_figures.py — publication-quality result figures for the thesis, computed
directly from reports/general_evaluation_results.csv (produced by generate_report.py).

Outputs PNG (300 dpi) + PDF (vector) into thesis/figures/:
  fig5_1_answer_quality.*        grouped bars: faithfulness / answer relevance / context precision per system
  fig5_2_retrieval.*             grouped bars: MRR / nDCG@5 / Recall@5 per system
  fig5_3_agentic_vs_naive.*      agentic vs naive (Llama 3.1 8B, Mistral 7B)
  fig5_4_language.*              faithfulness EN vs FR per system
  fig5_5_crosslingual.*         cross-lingual vs monolingual retrieval

Run:  python src/generate_figures.py
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

EVAL_CSV = "reports/general_evaluation_results.csv"
OUT_DIR = Path("thesis/figures")

# system label (A-F) -> the rag_base_model value in the combined CSV
SYSTEMS = [
    ("A: Llama 3.2 3B",      "llama3.2:3b-gpu"),
    ("B: Llama 3.1 8B",      "llama3.1:8b-gpu"),
    ("C: Mistral 7B",        "mistral:7b-gpu"),
    ("D: GPT-5.4-mini",      "cloud/gpt-5.4-mini"),
    ("E: Claude Haiku 4.5",  "cloud/claude-haiku-4-5"),
    ("F: Gemini 3.1 FL",     "cloud/gemini-3.1-flash-lite"),
]


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

# colour-blind-safe palette (Okabe-Ito subset); local = blues/greens, cloud = oranges/greys
LOCAL_C = "#0072B2"
CLOUD_C = "#D55E00"
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


def _load():
    df = pd.read_csv(EVAL_CSV)
    for c in ["token_f1_score", "rougeL", "meteor", "mrr", "ndcg_at_k",
              "recall_1", "recall_3", "recall_5", "answer_relevance",
              "faithfulness", "context_precision"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["lang"] = df["language"].astype(str).str.lower().str[:2].replace({"ge": "de"})
    # canonical model key so a retagged model (e.g. "-gpu") still matches SYSTEMS
    df["_model_key"] = df["rag_base_model"].map(canon)
    return df


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"   wrote {OUT_DIR / name}.png / .pdf")


def _means(df, model, cols):
    s = df[df["_model_key"] == canon(model)]
    return [s[c].mean() for c in cols]


def _grouped_bars(labels, series, series_names, title, ylabel, name, colors=None, ymax=1.0):
    import numpy as np
    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (vals, sname) in enumerate(zip(series, series_names)):
        col = (colors[i] if colors else None)
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=sname, color=col)
        for b, v in zip(bars, vals):
            if v == v:  # not NaN
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, ymax * 1.12)
    ax.legend(frameon=False, ncol=len(series_names), loc="upper left", fontsize=8)
    return fig


def main():
    if not Path(EVAL_CSV).exists():
        print(f"[figures] {EVAL_CSV} not found — run: python src/generate_report.py")
        return
    df = _load()
    labels = [lab for lab, _ in SYSTEMS]
    models = [m for _, m in SYSTEMS]
    barcolors = [LOCAL_C, LOCAL_C, LOCAL_C, CLOUD_C, CLOUD_C, CLOUD_C]

    # Fig 5.1 — answer quality
    faith = [_means(df, m, ["faithfulness"])[0] for m in models]
    arel = [_means(df, m, ["answer_relevance"])[0] for m in models]
    ctxp = [_means(df, m, ["context_precision"])[0] for m in models]
    fig = _grouped_bars(labels, [faith, arel, ctxp],
                        ["Faithfulness", "Answer relevance", "Context precision"],
                        "Answer quality by system (judge: Prometheus-2 8x7B)",
                        "Mean score", "fig5_1_answer_quality", colors=ACCENT)
    _save(fig, "fig5_1_answer_quality")

    # Fig 5.2 — retrieval
    mrr = [_means(df, m, ["mrr"])[0] for m in models]
    ndcg = [_means(df, m, ["ndcg_at_k"])[0] for m in models]
    r5 = [_means(df, m, ["recall_5"])[0] for m in models]
    fig = _grouped_bars(labels, [mrr, ndcg, r5], ["MRR", "nDCG@5", "Recall@5"],
                        "Retrieval performance by system", "Mean score",
                        "fig5_2_retrieval", colors=ACCENT, ymax=0.5)
    _save(fig, "fig5_2_retrieval")

    # Fig 5.3 — agentic vs naive
    import numpy as np
    pairs = [("Llama 3.1 8B", "llama3.1:8b"), ("Mistral 7B", "mistral:7b")]
    metrics = ["faithfulness", "answer_relevance", "context_precision"]
    mnames = ["Faithfulness", "Answer rel.", "Context prec."]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (pname, gen) in zip(axes, pairs):
        ag = _means(df, gen, metrics)
        na = _means(df, "naive/" + gen, metrics)
        x = np.arange(len(metrics)); w = 0.38
        ax.bar(x - w / 2, ag, w, label="Agentic", color=LOCAL_C)
        ax.bar(x + w / 2, na, w, label="Naive", color="#999999")
        for i, (a, b) in enumerate(zip(ag, na)):
            ax.text(x[i] - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=7)
            ax.text(x[i] + w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(mnames, rotation=15, ha="right")
        ax.set_title(pname); ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Mean score")
    axes[0].legend(frameon=False)
    fig.suptitle("Agentic vs. naive single-pass (same generator)")
    _save(fig, "fig5_3_agentic_vs_naive")

    # Fig 5.4 — faithfulness EN vs FR
    en = [df[(df._model_key == canon(m)) & (df.lang == "en")]["faithfulness"].mean() for m in models]
    fr = [df[(df._model_key == canon(m)) & (df.lang == "fr")]["faithfulness"].mean() for m in models]
    fig = _grouped_bars(labels, [en, fr], ["English", "French"],
                        "Faithfulness by query language", "Mean faithfulness",
                        "fig5_4_language", colors=[LOCAL_C, CLOUD_C])
    _save(fig, "fig5_4_language")

    # Fig 5.5 — cross-lingual vs monolingual retrieval (System B)
    b_mono = _means(df, "llama3.1:8b", ["mrr", "ndcg_at_k", "recall_5"])
    b_cross = _means(df, "crosslingual/llama3.1:8b-gpu", ["mrr", "ndcg_at_k", "recall_5"])
    fig = _grouped_bars(["MRR", "nDCG@5", "Recall@5"], [b_mono, b_cross],
                        ["Monolingual (EN+FR)", "Cross-lingual (DE→EN/FR)"],
                        "Cross-lingual vs. monolingual retrieval (Llama 3.1 8B)",
                        "Mean score", "fig5_5_crosslingual",
                        colors=[LOCAL_C, CLOUD_C], ymax=0.5)
    _save(fig, "fig5_5_crosslingual")

    print(f"\n[figures] Done. 5 figures (PNG+PDF) in {OUT_DIR}/")


if __name__ == "__main__":
    main()
