#!/usr/bin/env python3
"""
generate_architecture_figures_split.py : Chapter 3 diagrams, split into small readable parts.

The single fig3_1_architecture / fig3_2_query_flow pair packed the whole system into two
page-sized canvases. Reduced to the width of a thesis page, the box labels fell to roughly
5 pt and became unreadable, which is the defect raised in the supervisor's review of
Sections 3.7.2 and 3.7.3.

This script replaces those two with eight figures, each covering one subsystem or one
stage of the query path. Every figure is drawn on a canvas no wider than 6.5 in, which is
the text width of a Letter page at 1 in margins, so the figure is placed at 100% scale and
the label sizes below are the sizes the reader actually sees. Nothing is scaled down.

No figure carries a title: captions belong below the figure in the Word document, not
baked into the image.

  fig3_1_overview          five subsystems end to end, no internals        -> Sec 3.2.2
  fig3_2_ingestion         PDF to indexed clause chunk (Subsystems 1, 2)   -> Sec 3.3.2
  fig3_3_chunk_metadata    the metadata record carried by every chunk      -> Sec 3.3.3
  fig3_4_hybrid_retrieval  dense + sparse + recency + RRF (Subsystem 3)    -> Sec 3.4.3
  fig3_5_agentic_loop      generate, reflect, rewrite, re-retrieve (Sub 4) -> Sec 3.5.3
  fig3_6_crosslingual      the two cross-lingual handling modes            -> Sec 3.5.5
  fig3_7_query_flow        the query-time path as a numbered flowchart     -> Sec 3.6.3
  fig3_8_instrumentation   telemetry and the offline harness (Subsystem 5) -> Sec 3.7.5

Outputs PNG (300 dpi) + PDF (vector) into thesis/figures/.
Run:  python src/generate_architecture_figures_split.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).parent))

from generate_architecture_figures import (  # noqa: E402  reuse the shared primitives
    OUT_DIR, INGEST, STORE, RETRIEVE, GENERATE, EVAL, INSTR, XLING,
    box, diamond, arrow, route, numbered, save,
)

# Labels are set two points above the old script's 8.5 pt. Because every canvas here is
# already page-width, no shrink is applied on insertion and 10.5 pt is what is printed.
plt.rcParams.update({
    "font.size": 10.5,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
})

W = 6.5          # page text width in inches: the hard cap for every canvas
FS = 10.5        # in-box label size
FSS = 9.5        # secondary label size
FSE = 9.0        # edge annotation size


def blank(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def edge(ax, x, y, text, color="#333333", ha="center", style="italic"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=FSE, color=color, style=style)


# ─────────────────────────────────────────────────────────────────────────────
#  3.1  the five subsystems, end to end, with no internals
# ─────────────────────────────────────────────────────────────────────────────
def fig_overview():
    fig, ax = blank(W, 3.9)
    bw, bh = 4.15, 0.46
    x = 1.35
    rows = [
        (3.28, "1. Document preprocessing  (offline, run once)", INGEST),
        (2.60, "2. Vector store: ChromaDB + BM25 index", STORE),
        (1.92, "3. Hybrid retrieval  (per query)", RETRIEVE),
        (1.24, "4. Generation and agentic control loop", GENERATE),
        (0.30, "5. Evaluation harness  (offline, batched)", EVAL),
    ]
    for y, label, colour in rows:
        box(ax, x, y, bw, bh, label, colour, fontsize=FS)
    for i in range(len(rows) - 1):
        y_top, y_bot = rows[i][0], rows[i + 1][0] + bh
        arrow(ax, (x + bw / 2, y_top), (x + bw / 2, y_bot))

    # the instrumentation strip spans every stage
    ax.add_patch(Rectangle((5.72, 0.30), 0.55, 3.44, facecolor="#FDF0E6",
                           edgecolor=INSTR, linewidth=1.2, zorder=1))
    ax.text(5.99, 2.02, "Instrumentation (cross-cutting)", ha="center", va="center",
            fontsize=FSS, color=INSTR, fontweight="bold", rotation=90)

    edge(ax, 1.28, 3.51, "PDFs", ha="right")
    arrow(ax, (0.18, 2.15), (x, 2.15))
    edge(ax, 0.76, 2.29, "user query")
    arrow(ax, (x, 1.47), (0.18, 1.47))
    edge(ax, 0.76, 1.61, "answer + citations")
    edge(ax, x + bw / 2 + 0.12, 0.98, "answers and telemetry, logged", ha="left")
    save(fig, "fig3_1_overview")


# ─────────────────────────────────────────────────────────────────────────────
#  3.2  ingestion: PDF to indexed clause chunk
# ─────────────────────────────────────────────────────────────────────────────
def fig_ingestion():
    fig, ax = blank(W, 3.2)
    bw, bh, x = 4.9, 0.5, 0.8
    steps = [
        (2.72, "Collective agreement PDFs  (EN / FR)", INGEST),
        (1.96, "ClauseBoundaryChunker\nsplit at ARTICLE and clause headers", INGEST),
        (1.20, "Merge clauses under 30 tokens,\nsplit clauses over 512 at sentence bounds", INGEST),
        (0.62, "Embed each chunk: nomic-embed-text-v2-moe", INGEST),
        (0.04, "ChromaDB collection \"multilingual_docs\"", STORE),
    ]
    heights = [0.42, 0.60, 0.60, 0.42, 0.42]
    for (y, label, colour), h in zip(steps, heights):
        box(ax, x, y, bw, h, label, colour, fontsize=FS)
    for i in range(len(steps) - 1):
        top = steps[i][0]
        bot = steps[i + 1][0] + heights[i + 1]
        arrow(ax, (x + bw / 2, top), (x + bw / 2, bot))
    edge(ax, x + bw + 0.10, 1.84, "4,591 clause\nchunks", ha="left")
    edge(ax, x + bw + 0.10, 0.54, "768-dim\nvectors", ha="left")
    save(fig, "fig3_2_ingestion")


# ─────────────────────────────────────────────────────────────────────────────
#  3.3  the per-chunk metadata record
# ─────────────────────────────────────────────────────────────────────────────
def fig_chunk_metadata():
    fig, ax = blank(W, 2.7)
    box(ax, 0.15, 1.60, 2.9, 0.85,
        "Clause text\n(the embedded payload)", INGEST, fontsize=FS)
    embedded = ["article_number", "clause_id", "section_title", "language"]
    excluded = ["page_no", "end_year", "recency_weight",
                "token_count", "chunk_index", "doc_source"]
    box(ax, 3.35, 1.60, 3.0, 0.85,
        "Embedded metadata\n" + ",  ".join(embedded[:2]) + "\n" + ",  ".join(embedded[2:]),
        STORE, fontsize=FSS)
    box(ax, 3.35, 0.32, 3.0, 1.00,
        "Retained but excluded from the vector\n"
        + ",  ".join(excluded[:2]) + "\n"
        + ",  ".join(excluded[2:4]) + "\n"
        + ",  ".join(excluded[4:]),
        EVAL, fontsize=FSS)
    box(ax, 0.15, 0.32, 2.9, 1.00,
        "Used for citation rendering,\nrecency tie-breaking and\nper-query telemetry,\n"
        "never for similarity", GENERATE, fontsize=FSS)
    arrow(ax, (3.05, 2.02), (3.35, 2.02))
    arrow(ax, (3.35, 0.82), (3.05, 0.82))
    save(fig, "fig3_3_chunk_metadata")


# ─────────────────────────────────────────────────────────────────────────────
#  3.4  hybrid retrieval and RRF fusion
# ─────────────────────────────────────────────────────────────────────────────
def fig_hybrid_retrieval():
    fig, ax = blank(W, 3.9)
    box(ax, 1.40, 3.38, 3.3, 0.40, "User query", RETRIEVE, fontsize=FS)
    box(ax, 1.40, 2.66, 3.3, 0.54,
        "Language detection, then\nyear-range augmentation", RETRIEVE, fontsize=FSS)
    box(ax, 0.10, 1.86, 2.85, 0.56,
        "Dense retrieval\ncosine similarity, top_k x 2", RETRIEVE, fontsize=FSS)
    box(ax, 3.20, 1.86, 3.10, 0.56,
        "Sparse retrieval\nBM25Okapi, full clause corpus", RETRIEVE, fontsize=FSS)
    box(ax, 0.10, 1.10, 2.85, 0.52,
        "Recency tie re-sort\nties within 0.02 by recency_weight", STORE, fontsize=FSS)
    box(ax, 1.05, 0.50, 4.0, 0.42,
        "RRF fusion:  score = sum of 1 / (60 + rank)", GENERATE, fontsize=FS)
    box(ax, 1.85, 0.02, 2.4, 0.36, "Top-k clause chunks", STORE, fontsize=FS)

    arrow(ax, (3.05, 3.38), (3.05, 3.20))
    arrow(ax, (2.60, 2.72), (1.52, 2.42))
    arrow(ax, (3.50, 2.72), (4.75, 2.42))
    arrow(ax, (1.52, 1.86), (1.52, 1.62))
    route(ax, [(1.52, 1.10), (1.52, 0.71), (1.05, 0.71)])
    route(ax, [(4.75, 1.86), (4.75, 0.71), (5.05, 0.71)])
    arrow(ax, (3.05, 0.50), (3.05, 0.38))
    save(fig, "fig3_4_hybrid_retrieval")


# ─────────────────────────────────────────────────────────────────────────────
#  3.5  the agentic control loop
# ─────────────────────────────────────────────────────────────────────────────
def fig_agentic_loop():
    fig, ax = blank(W, 3.4)
    box(ax, 0.30, 2.80, 2.5, 0.44, "Top-k clause chunks", RETRIEVE, fontsize=FSS)
    box(ax, 0.30, 2.05, 2.5, 0.52,
        "Prompt assembly\nsystem + context + question", GENERATE, fontsize=FSS)
    box(ax, 0.30, 1.32, 2.5, 0.50,
        "Local LLM via Ollama\nstreaming completion", GENERATE, fontsize=FSS)
    diamond(ax, 4.65, 1.57, 2.5, 1.30,
            "Reflector:\nis every claim\nsupported by the\nretrieved context?", GENERATE,
            fontsize=FSS)
    box(ax, 3.55, 0.10, 2.2, 0.46, "Answer + citations", STORE, fontsize=FSS)
    box(ax, 0.30, 0.42, 2.5, 0.52,
        "Query rewriter\nthen re-retrieve at k + 1", INSTR, fontsize=FSS)

    arrow(ax, (1.55, 2.80), (1.55, 2.57))
    arrow(ax, (1.55, 2.05), (1.55, 1.82))
    arrow(ax, (2.80, 1.57), (3.40, 1.57))
    route(ax, [(4.65, 0.92), (4.65, 0.56)])
    edge(ax, 4.95, 0.74, "grounded", color="#1B7F4B", ha="left")
    route(ax, [(3.40, 1.57), (3.10, 1.57), (3.10, 0.68), (2.80, 0.68)],
          color=INSTR, ls="--")
    edge(ax, 3.15, 1.05, "ungrounded", color=INSTR, ha="left")
    route(ax, [(0.55, 0.94), (0.12, 0.94), (0.12, 2.31), (0.30, 2.31)],
          color=INSTR, ls="--")
    ax.text(0.02, 1.62, "retry, max 3 attempts", ha="center", va="center",
            fontsize=FSE, color=INSTR, style="italic", rotation=90)
    save(fig, "fig3_5_agentic_loop")


# ─────────────────────────────────────────────────────────────────────────────
#  3.6  the two cross-lingual handling modes
# ─────────────────────────────────────────────────────────────────────────────
def fig_crosslingual():
    fig, ax = blank(W, 2.8)
    box(ax, 1.85, 2.30, 2.8, 0.40, "Query language detected", RETRIEVE, fontsize=FS)

    box(ax, 0.10, 1.35, 3.0, 0.62,
        "EN or FR: no-translation mode\nembed as written, answer in kind",
        STORE, fontsize=FSS)
    box(ax, 3.40, 1.35, 3.0, 0.62,
        "Any other language: bridge mode\ntranslate to EN, then translate back",
        XLING, fontsize=FSS)
    box(ax, 0.10, 0.52, 3.0, 0.56,
        "Retrieval crosses languages\nin the shared embedding space", RETRIEVE, fontsize=FSS)
    box(ax, 3.40, 0.52, 3.0, 0.56,
        "German evaluation set runs\nwithout the bridge (see Chapter 4)", INSTR, fontsize=FSS)
    arrow(ax, (3.25, 2.30), (1.60, 1.97))
    arrow(ax, (3.25, 2.30), (4.90, 1.97))
    arrow(ax, (1.60, 1.35), (1.60, 1.08))
    arrow(ax, (4.90, 1.35), (4.90, 1.08))
    save(fig, "fig3_6_crosslingual")


# ─────────────────────────────────────────────────────────────────────────────
#  3.7  the query-time path as a numbered flowchart
# ─────────────────────────────────────────────────────────────────────────────
def fig_query_flow():
    fig, ax = blank(W, 4.9)
    bw, bh, x = 4.55, 0.40, 1.05
    steps = [
        "Receive the user query",
        "Detect the query language",
        "Append the agreement year range if absent",
        "Rewrite the query   (attempts 2 and 3 only)",
        "Hybrid retrieval, then RRF fusion to top-k",
        "Assemble the prompt and generate the answer",
        "Reflector returns GROUNDED or UNGROUNDED",
        "Return the answer, log telemetry and scores",
    ]
    ys = [4.30 - i * 0.58 for i in range(len(steps))]
    for n, (y, label) in enumerate(zip(ys, steps), start=1):
        colour = GENERATE if n in (4, 6, 7) else RETRIEVE
        if n == 8:
            colour = STORE
        box(ax, x, y, bw, bh, label, colour, fontsize=FS)
        numbered(ax, x - 0.32, y + bh / 2, n)
    for i in range(len(ys) - 1):
        arrow(ax, (x + bw / 2, ys[i]), (x + bw / 2, ys[i + 1] + bh))
    # the corrective loop: step 7 back to step 4
    route(ax, [(x + bw, ys[6] + bh / 2), (6.30, ys[6] + bh / 2),
               (6.30, ys[3] + bh / 2), (x + bw, ys[3] + bh / 2)],
          color=INSTR, ls="--")
    ax.text(6.42, (ys[6] + ys[3]) / 2 + bh / 2, "UNGROUNDED: retry at k + 1",
            ha="center", va="center", fontsize=FSE, color=INSTR,
            style="italic", rotation=90)
    edge(ax, x - 0.42, ys[7] + bh + 0.09, "GROUNDED, or\nattempt 3 reached",
         color="#1B7F4B", ha="right")
    save(fig, "fig3_7_query_flow")


# ─────────────────────────────────────────────────────────────────────────────
#  3.8  instrumentation and the offline evaluation harness
# ─────────────────────────────────────────────────────────────────────────────
def fig_instrumentation():
    fig, ax = blank(W, 3.3)
    ax.text(1.65, 3.12, "Captured per query", ha="center", va="center",
            fontsize=FS, fontweight="bold", color=INSTR)
    ax.text(4.98, 3.12, "Offline harness", ha="center", va="center",
            fontsize=FS, fontweight="bold", color=EVAL)
    left = [
        ("Latency\nretrieval, TTFT, end to end", INSTR),
        ("Hardware\nGPU VRAM and utilisation, RAM, CPU", INSTR),
        ("Context sizing\nwindow used, prompt and output tokens", INSTR),
        ("Loop state\nattempts, groundedness verdict", INSTR),
    ]
    right = [
        ("answer_generator.py\ngenerates and logs answers", EVAL),
        ("metrics.py\nretrieval and lexical metrics", EVAL),
        ("metrics.py --judge_llm\nPrometheus-2 8x7B judge", EVAL),
        ("CSV ledgers, dashboards,\nfigures", STORE),
    ]
    for i, (label, colour) in enumerate(left):
        box(ax, 0.10, 2.32 - i * 0.72, 3.10, 0.58, label, colour, fontsize=FSS)
    for i, (label, colour) in enumerate(right):
        box(ax, 3.55, 2.32 - i * 0.72, 2.85, 0.58, label, colour, fontsize=FSS)
    for i in range(3):
        arrow(ax, (4.98, 2.32 - i * 0.72), (4.98, 2.32 - (i + 1) * 0.72 + 0.58))
    arrow(ax, (3.20, 2.61), (3.55, 2.61))
    save(fig, "fig3_8_instrumentation")


def main():
    print("[arch-split] generating the eight Chapter 3 diagrams...")
    fig_overview()
    fig_ingestion()
    fig_chunk_metadata()
    fig_hybrid_retrieval()
    fig_agentic_loop()
    fig_crosslingual()
    fig_query_flow()
    fig_instrumentation()
    print(f"[arch-split] done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
