#!/usr/bin/env python3
"""
generate_architecture_figures.py : Chapter 3 implementation diagrams.

Draws the two figures Chapter 3 needs directly from the implemented system. No external
diagram tooling is required (graphviz and mermaid-cli are not installed here), so the
diagrams are laid out with matplotlib primitives and exported in the same style as the
Chapter 5 result figures.

  fig3_1_architecture.*   layered system architecture: ingestion -> vector store ->
                          hybrid retrieval -> generation / agentic control -> evaluation,
                          plus the cross-cutting instrumentation strip
  fig3_2_query_flow.*     query-time flowchart including the corrective agentic loop

Outputs PNG (300 dpi) + PDF (vector) into thesis/figures/.
Run:  python src/generate_architecture_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).parent))

OUT_DIR = Path("thesis/figures")

# Colour-blind-safe palette (Okabe-Ito), consistent with the Chapter 5 figures.
INGEST = "#56B4E9"
STORE = "#009E73"
RETRIEVE = "#0072B2"
GENERATE = "#E69F00"
EVAL = "#8C8C8C"
INSTR = "#D55E00"
XLING = "#4C6EF5"
BAND = "#F2F2F2"

# Times New Roman is the project default (12 pt for body text). Figure labels sit below
# body size so they stay legible when the figure is scaled into the page.
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
})


def box(ax, x, y, w, h, text, color, fontsize=8.5, text_color="white", bold=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.1, edgecolor=color, facecolor=color, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold" if bold else "normal", zorder=4, linespacing=1.4)


def diamond(ax, cx, cy, w, h, text, color=GENERATE, fontsize=8):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(plt.Polygon(pts, closed=True, facecolor="white",
                             edgecolor=color, linewidth=1.7, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color="#111111", fontweight="bold", zorder=4, linespacing=1.3)


def arrow(ax, p1, p2, color="#333333", rad=0.0, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=12, linewidth=lw,
        color=color, linestyle=ls, zorder=2,
        connectionstyle="arc3,rad=%s" % rad, shrinkA=1.5, shrinkB=1.5))


def route(ax, pts, color="#333333", lw=1.3, ls="-"):
    """Orthogonally routed connector: plain segments, arrowhead on the final one."""
    for i in range(len(pts) - 1):
        style = "-|>" if i == len(pts) - 2 else "-"
        ax.add_patch(FancyArrowPatch(
            pts[i], pts[i + 1], arrowstyle=style, mutation_scale=12, linewidth=lw,
            color=color, linestyle=ls, zorder=2, shrinkA=0, shrinkB=0))


def band(ax, x, y, w, h, label, color):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=BAND, edgecolor="#D0D0D0",
                           linewidth=0.8, zorder=1))
    ax.text(x + 0.14, y + h - 0.15, label, ha="left", va="top", fontsize=8.6,
            color=color, fontweight="bold", zorder=4)


def numbered(ax, x, y, n, color="#222222"):
    ax.add_patch(plt.Circle((x, y), 0.185, facecolor=color, edgecolor="none", zorder=5))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=7.6,
            color="white", fontweight="bold", zorder=6)


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"   wrote {OUT_DIR / name}.png / .pdf")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3.1 : system architecture
# ─────────────────────────────────────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(13.4, 10.9))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 11.45)
    ax.axis("off")
    IW = 10.30                 # instrumentation strip starts here
    CR = IW - 0.30             # content-band width

    # ---- Band 1: ingestion (offline) -----------------------------------------
    band(ax, 0.15, 9.70, CR, 1.45,
         "SUBSYSTEM 1: DOCUMENT PREPROCESSING (offline, run once)", INGEST)
    box(ax, 0.42, 9.82, 1.70, 0.92,
        "Collective\nagreement PDFs\n(EN / FR)", INGEST, 7.8)
    box(ax, 2.32, 9.82, 2.35, 0.92,
        "ClauseBoundaryChunker\nsplit at ARTICLE / clause\nmerge <30 tok, split >512 tok",
        INGEST, 7.0)
    box(ax, 4.87, 9.82, 2.55, 0.92,
        "TextNode + metadata\narticle_number, clause_id,\n"
        "section_title, language, page_no,\nend_year, recency_weight", INGEST, 6.3)
    box(ax, 7.62, 9.82, 2.28, 0.92,
        "Embedding model\nnomic-embed-text-v2-moe\n(multilingual, via Ollama)", INGEST, 7.0)
    for a, b in [((2.12, 10.28), (2.32, 10.28)), ((4.67, 10.28), (4.87, 10.28)),
                 ((7.42, 10.28), (7.62, 10.28))]:
        arrow(ax, a, b)

    # ---- Band 2: vector store -------------------------------------------------
    band(ax, 0.15, 8.35, CR, 1.05, "SUBSYSTEM 2: VECTOR STORE", STORE)
    box(ax, 2.60, 8.46, 5.20, 0.62,
        'ChromaDB  (PersistentClient, collection "multilingual_docs")\n'
        'clause-level chunks + dense vectors + metadata', STORE, 7.4)
    route(ax, [(8.76, 9.82), (8.76, 9.55), (7.80, 9.55), (7.80, 9.08)])
    ax.text(8.86, 9.58, "index", fontsize=6.8, color="#555555", style="italic")

    # ---- Band 3: hybrid retrieval ---------------------------------------------
    band(ax, 0.15, 5.05, CR, 3.20, "SUBSYSTEM 3: HYBRID RETRIEVAL (per query)", RETRIEVE)
    box(ax, 0.60, 7.24, 2.00, 0.58, "Language\ndetection", RETRIEVE, 7.8)
    box(ax, 2.80, 7.24, 2.40, 0.58, "Query handler\nyear-range augmentation", RETRIEVE, 7.3)
    box(ax, 0.60, 6.32, 3.00, 0.62,
        "Dense retrieval: VectorIndexRetriever\ncosine similarity, top_k x 2", RETRIEVE, 7.1)
    box(ax, 3.90, 6.32, 3.00, 0.62,
        "Sparse retrieval: BM25Okapi\nover the full clause corpus", RETRIEVE, 7.1)
    box(ax, 0.60, 5.28, 2.70, 0.68,
        "Recency tie re-sort\ndense ties within 0.02 ordered\nby recency_weight", RETRIEVE, 6.9)
    box(ax, 3.60, 5.28, 2.40, 0.68, "RRF fusion\nsum of 1/(60 + rank)", RETRIEVE, 7.2)
    box(ax, 6.30, 5.28, 1.70, 0.68, "Top-k clause\nchunks", RETRIEVE, 7.5)
    arrow(ax, (2.60, 7.53), (2.80, 7.53))
    arrow(ax, (3.70, 7.24), (2.30, 6.94), rad=0.12)
    arrow(ax, (4.30, 7.24), (5.30, 6.94), rad=-0.12)
    arrow(ax, (2.10, 6.32), (1.95, 5.96))
    arrow(ax, (5.30, 6.32), (4.90, 5.96))
    arrow(ax, (3.30, 5.62), (3.60, 5.62))
    arrow(ax, (6.00, 5.62), (6.30, 5.62))
    # vector store -> dense retrieval: straight down through the 0.20 gap between the
    # language-detection and query-handler boxes, so it crosses nothing.
    route(ax, [(2.70, 8.46), (2.70, 6.94)])
    ax.text(2.80, 7.02, "dense vectors + chunk text", fontsize=6.5, color="#555555",
            style="italic", va="bottom", ha="left")

    # ---- Band 4: generation + agentic control loop -----------------------------
    band(ax, 0.15, 2.35, CR, 2.45,
         "SUBSYSTEM 4: GENERATION AND AGENTIC CONTROL LOOP", GENERATE)
    box(ax, 0.42, 3.58, 2.05, 0.84,
        "Prompt assembly\nsystem prompt(language)\n+ context + question", GENERATE, 7.1)
    box(ax, 2.67, 3.58, 1.95, 0.84,
        "Local LLM (Ollama)\nstreaming completion\n100% GPU offload", GENERATE, 7.1)
    box(ax, 4.82, 3.58, 1.70, 0.84, "Reflector\ngroundedness\nverdict", GENERATE, 7.3)
    box(ax, 6.72, 3.58, 1.80, 0.84, "Query rewriter\n(retries only)", GENERATE, 7.3)
    box(ax, 8.72, 3.58, 1.18, 0.84, "Answer", GENERATE, 8.0)
    arrow(ax, (2.47, 4.00), (2.67, 4.00))
    arrow(ax, (4.62, 4.00), (4.82, 4.00))
    arrow(ax, (6.52, 4.00), (6.72, 4.00))
    ax.text(6.62, 4.46, "ungrounded", fontsize=6.8, color="#B04A00", ha="center", style="italic")
    arrow(ax, (8.52, 4.00), (8.72, 4.00))
    ax.text(8.62, 4.46, "grounded", fontsize=6.8, color="#1B7F5A", ha="center", style="italic")
    # rewriter loops back into retrieval with a widened k, routed up the right margin
    route(ax, [(7.62, 4.42), (7.62, 4.90), (8.62, 4.90), (8.62, 5.62), (8.00, 5.62)],
          color=INSTR, ls="--")
    ax.text(8.72, 5.16, "re-retrieve, k + 1", fontsize=6.9, color=INSTR, style="italic")
    box(ax, 0.42, 2.50, 6.60, 0.74,
        "Cross-lingual handling.  bridge mode: translate query to English, translate answer back\n"
        "no-translation mode: retrieve with the raw query, answer in the question's own language",
        XLING, 7.0)
    route(ax, [(9.31, 3.58), (9.31, 1.65)])
    ax.text(9.41, 2.50, "answers +\ntelemetry", fontsize=6.8, color="#555555", style="italic")

    # ---- Band 5: evaluation harness -------------------------------------------
    band(ax, 0.15, 0.20, CR, 1.45,
         "SUBSYSTEM 5: EVALUATION HARNESS (offline, batched)", EVAL)
    box(ax, 0.42, 0.44, 1.60, 0.74, "retrieval.py\nbatch 1", EVAL, 7.5)
    box(ax, 2.22, 0.44, 1.90, 0.74, "answer_generator.py\nbatch 2", EVAL, 7.5)
    box(ax, 4.32, 0.44, 2.15, 0.74,
        "metrics.py\nbatch 2b deterministic\nbatch 3 LLM-as-judge", EVAL, 7.1)
    box(ax, 6.67, 0.44, 1.60, 0.74, "CSV ledgers\nlufa_out /\nevaluation_results", EVAL, 6.9)
    box(ax, 8.47, 0.44, 1.43, 0.74, "Dashboard,\nreports,\nfigures", EVAL, 7.3)
    for a, b in [((2.02, 0.81), (2.22, 0.81)), ((4.12, 0.81), (4.32, 0.81)),
                 ((6.47, 0.81), (6.67, 0.81)), ((8.27, 0.81), (8.47, 0.81))]:
        arrow(ax, a, b)

    # ---- Instrumentation strip (cross-cutting) --------------------------------
    ax.add_patch(Rectangle((IW + 0.05, 0.20), 2.95, 10.95, facecolor="#FDF1E7",
                           edgecolor=INSTR, linewidth=1.2, zorder=1))
    ax.text(IW + 1.53, 11.02, "INSTRUMENTATION\n(cross-cutting)", ha="center", va="top",
            fontsize=8.6, color=INSTR, fontweight="bold")
    items = [
        ("Latency",
         "retrieval_latency_s\nttft_s\nend_to_end_latency_s\n(time.perf_counter)"),
        ("Hardware",
         "gpu_vram_mb, gpu_util_percent\n(nvidia-smi)\nRAM + CPU scoped to the\nOllama processes (psutil)"),
        ("Context sizing",
         "context_window_used\nprompt_tokens_est\npredicted_output_tokens"),
        ("Warm-up protocol",
         "first query retrieved twice,\nthe cold pass discarded"),
        ("Judge",
         "Prometheus-2 8x7B\nseparate prompt per metric"),
    ]
    y = 9.95
    for title, body in items:
        box(ax, IW + 0.22, y - 1.05, 2.61, 1.05, f"{title}\n{body}", INSTR, 6.6)
        y -= 1.28

    ax.set_title("Figure 3.1: Architecture of the implemented bilingual agentic RAG system",
                 fontsize=12, fontweight="bold", pad=14)
    save(fig, "fig3_1_architecture")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3.2 : query-time flowchart
# ─────────────────────────────────────────────────────────────────────────────
def fig_query_flow():
    fig, ax = plt.subplots(figsize=(12.2, 12.8))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 12.8)
    ax.axis("off")
    cx = 4.30
    W = 5.80          # wide enough that no step label overflows its box

    def step(y, text, n, color=RETRIEVE, h=0.60, fs=8.1):
        box(ax, cx - W / 2, y, W, h, text, color, fs)
        numbered(ax, cx - W / 2 - 0.32, y + h / 2, n)

    box(ax, cx - 1.45, 12.05, 2.90, 0.46, "START: user query", "#222222", 8.6)
    arrow(ax, (cx, 12.05), (cx, 11.78))

    step(11.14, "Detect the query language", 1, RETRIEVE, 0.56)
    arrow(ax, (cx, 11.14), (cx, 10.76))

    diamond(ax, cx, 10.40, 3.30, 0.72, "Language is EN or FR?", GENERATE, 8.0)
    numbered(ax, cx - 1.87, 10.40, 2)
    # NO branch -> cross-lingual mode selection
    arrow(ax, (cx + 1.65, 10.40), (7.40, 10.40))
    ax.text(6.55, 10.50, "no", fontsize=7.4, color="#B04A00", style="italic")
    box(ax, 7.40, 9.92, 4.55, 0.96,
        "Select cross-lingual mode\nbridge: translate query to English\n"
        "no-translation: keep the source language", XLING, 7.2)
    route(ax, [(9.67, 9.92), (9.67, 9.58), (cx + 1.90, 9.58)])
    arrow(ax, (cx, 10.04), (cx, 9.72))
    ax.text(cx + 0.14, 9.86, "yes", fontsize=7.4, color="#1B7F5A", style="italic")

    step(9.06, "Augment the query with the agreement year range\nif no 4-digit year is present",
         3, RETRIEVE, 0.60, 7.9)
    arrow(ax, (cx, 9.06), (cx, 8.70))

    step(8.20, "Enter the corrective loop:  attempt = 1 .. 3", 4, "#555555", 0.50, 8.2)
    arrow(ax, (cx, 8.20), (cx, 7.84))

    diamond(ax, cx, 7.48, 2.85, 0.66, "attempt > 1 ?", GENERATE, 8.0)
    arrow(ax, (cx - 1.43, 7.48), (1.95, 7.48))
    ax.text(2.20, 7.58, "yes", fontsize=7.4, color="#B04A00", style="italic")
    box(ax, 0.18, 7.06, 1.77, 0.84,
        "Rewrite the query\nusing the rejected answer\nand the provision titles", INSTR, 6.9)
    route(ax, [(1.07, 7.06), (1.07, 6.60), (cx - W / 2, 6.60)])
    arrow(ax, (cx, 7.15), (cx, 6.86))
    ax.text(cx + 0.14, 7.00, "no", fontsize=7.4, color="#1B7F5A", style="italic")

    step(6.00, "Hybrid retrieval\ndense (cosine) + sparse (BM25)  ->  recency tie re-sort\n"
               "->  RRF fusion  ->  top-k clause chunks", 5, RETRIEVE, 0.86, 7.7)
    arrow(ax, (cx, 6.00), (cx, 5.66))

    step(4.90, "Assemble the prompt\nsystem prompt for the answer language + retrieved\n"
               "context + question + citation instruction", 6, GENERATE, 0.80, 7.7)
    arrow(ax, (cx, 4.90), (cx, 4.56))

    step(3.90, "Generate the answer with the local LLM\nstreamed; time-to-first-token recorded",
         7, GENERATE, 0.62, 7.9)
    arrow(ax, (cx, 3.90), (cx, 3.52))

    diamond(ax, cx, 3.14, 3.55, 0.76,
            "Reflector: is the answer grounded\nin the retrieved context?", GENERATE, 7.7)
    numbered(ax, cx - 2.00, 3.14, 8)
    # ungrounded -> loop back up the right margin
    arrow(ax, (cx + 1.78, 3.14), (9.85, 3.14), color=INSTR)
    ax.text(7.95, 3.24, "no,  and attempt < 3", fontsize=7.2, color=INSTR, style="italic")
    route(ax, [(9.85, 3.14), (9.85, 7.72), (cx + 1.43, 7.72)], color=INSTR)
    ax.text(9.99, 5.40, "attempt := attempt + 1", fontsize=7.2, color=INSTR,
            rotation=90, va="center", style="italic")
    arrow(ax, (cx, 2.76), (cx, 2.42))
    ax.text(cx + 0.14, 2.58, "yes,  or attempts exhausted", fontsize=7.2,
            color="#1B7F5A", style="italic")

    step(1.42, "Post-process according to mode\nbridge: translate the answer back to the query language\n"
               "no-translation: additionally render a copy in the benchmark\n"
               "language for the lexical metrics only (never shown to the judge)",
         9, XLING, 1.00, 7.2)
    arrow(ax, (cx, 1.42), (cx, 1.08))

    step(0.62, "Persist the row: answer, per-chunk cosine / recency-adjusted / RRF\n"
               "scores, attempts, grounded verdict, latency and hardware telemetry",
         10, EVAL, 0.66, 7.5)
    arrow(ax, (cx, 0.62), (cx, 0.48))
    box(ax, cx - 1.95, 0.02, 3.90, 0.44,
        "END: row written to lufa_out_data.csv", "#222222", 8.2)

    ax.set_title("Figure 3.2: Query-time processing flow, including the corrective agentic loop",
                 fontsize=12, fontweight="bold", pad=14)
    save(fig, "fig3_2_query_flow")


def main():
    print("[arch-figures] generating Chapter 3 diagrams...")
    fig_architecture()
    fig_query_flow()
    print(f"[arch-figures] done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
