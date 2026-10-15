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

Numbering follows the chapter after "Query process flow diagram" was moved from Sec 3.8 up
to Sec 3.2.3. That section carries TWO figures, the query path and the instrumentation, so
both move ahead of the old Figure 3.2 and everything from 3.2 through 3.6 shifts by two.

  fig3_1_overview          five subsystems end to end, no internals        -> Sec 3.2.2
  fig3_2_query_flow        the query-time path as a numbered flowchart     -> Sec 3.2.3
  fig3_3_instrumentation   telemetry and the offline harness (Subsystem 5) -> Sec 3.2.3
  fig3_4_ingestion         PDF to indexed clause chunk (Subsystems 1, 2)   -> Sec 3.3.2
  fig3_5_chunk_metadata    the metadata record carried by every chunk      -> Sec 3.3.3
  fig3_6_hybrid_retrieval  dense + sparse + recency + RRF (Subsystem 3)    -> Sec 3.4.3
  fig3_7_agentic_loop      generate, reflect, rewrite, re-retrieve (Sub 4) -> Sec 3.5.3
  fig3_8_crosslingual      the two cross-lingual handling modes            -> Sec 3.6.2

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
    save(fig, "fig3_4_ingestion")


# ─────────────────────────────────────────────────────────────────────────────
#  3.3  the per-chunk metadata record
# ─────────────────────────────────────────────────────────────────────────────
def fig_chunk_metadata():
    """
    Redrawn after the supervisor's review of Figure 3.3, which asked whether the two halves
    were connected, noted that no arrow ran from the upper boxes to the lower ones, and asked
    what "retained but excluded from the vector" meant.

    The previous version placed four boxes in a 2x2 grid with one arrow along the top row and
    one along the bottom row pointing the opposite way. Nothing represented the record itself,
    so the two rows read as two unrelated layers, and the reader was invited to look for a
    cycle that does not exist.

    This version makes the single record the origin and lets both paths descend from it, so
    the structure states the actual claim: ONE record, TEN fields, all stored, split only by
    whether a field is allowed to reach the encoder. Every arrow now points down.
    """
    fig, ax = blank(W, 4.15)

    # column geometry: two branches under one record
    lx, rx, cw = 0.20, 3.35, 2.95
    lc, rc = lx + cw / 2, rx + cw / 2          # branch centre lines
    rec_x, rec_w = 0.85, 4.80
    rec_c = rec_x + rec_w / 2

    # ── the record: the thing the whole figure is about ──────────────────────
    rec_y, rec_h = 3.46, 0.52
    box(ax, rec_x, rec_y, rec_w, rec_h,
        "Metadata record carried by every chunk  (10 fields)\n"
        "all of it written to ChromaDB, all of it returned at query time",
        STORE, fontsize=FSS)

    # ── row 2: where each field goes ─────────────────────────────────────────
    f2_y, f2_h = 2.16, 1.02
    box(ax, lx, f2_y, cw, f2_h,
        "ENTERS THE EMBEDDING\n\n"
        "clause text\n"
        "article_number,  clause_id\n"
        "section_title,  language",
        INGEST, fontsize=FSS)
    box(ax, rx, f2_y, cw, f2_h,
        "STORED, NEVER EMBEDDED\n\n"
        "page_no,  end_year\n"
        "recency_weight,  token_count\n"
        "chunk_index,  doc_source",
        EVAL, fontsize=FSS)

    # ── row 3: what that means downstream ────────────────────────────────────
    f3_y, f3_h = 0.30, 1.22
    box(ax, lx, f3_y, cw, f3_h,
        "Encoded into one\n768-dimension vector\n\n"
        "This is the only text\nsimilarity search compares",
        RETRIEVE, fontsize=FSS)
    box(ax, rx, f3_y, cw, f3_h,
        "Read only AFTER retrieval\n\n"
        "citation rendering, recency\ntie-breaking, per-query telemetry\n\n"
        "cannot influence the ranking",
        INSTR, fontsize=FSS)

    # ── the split: one record, two destinations. Both arrows descend. ────────
    bus_y = 3.30                                # horizontal bus below the record
    arrow(ax, (rec_c, rec_y), (rec_c, bus_y + 0.02))
    ax.plot([lc, rc], [bus_y, bus_y], color="#333333", linewidth=1.2, zorder=2)
    arrow(ax, (lc, bus_y), (lc, f2_y + f2_h))
    arrow(ax, (rc, bus_y), (rc, f2_y + f2_h))

    # ── each destination to its consequence ──────────────────────────────────
    arrow(ax, (lc, f2_y), (lc, f3_y + f3_h))
    arrow(ax, (rc, f2_y), (rc, f3_y + f3_h))

    mid = (f2_y + f3_y + f3_h) / 2
    edge(ax, lc - 0.16, mid, "concatenated,\nthen encoded", ha="right")
    edge(ax, rc + 0.16, mid, "withheld from\nthe encoded string", ha="left")

    save(fig, "fig3_5_chunk_metadata")


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
    save(fig, "fig3_6_hybrid_retrieval")


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
    # The rewriter box previously read "then re-retrieve at k + 1". k no longer grows with the
    # attempt: widening it admitted progressively weaker chunks, which gave a retry a second
    # way to end up worse than its own first pass and confounded the comparison against the
    # single-pass configuration. The rewriter is also no longer blind; from attempt 2 it is
    # given the answer the reflector rejected and the titles of the provisions that were
    # actually retrieved, so it can target a real provision instead of inventing one.
    box(ax, 0.30, 0.32, 2.85, 0.72,
        "Query rewriter, given the rejected\nanswer and the retrieved clause titles\n"
        "re-retrieve at the same k",
        INSTR, fontsize=FSS)

    arrow(ax, (1.55, 2.80), (1.55, 2.57))
    arrow(ax, (1.55, 2.05), (1.55, 1.82))
    arrow(ax, (2.80, 1.57), (3.40, 1.57))
    route(ax, [(4.65, 0.92), (4.65, 0.56)])
    edge(ax, 4.95, 0.74, "grounded", color="#1B7F4B", ha="left")
    # Leave the diamond on its lower-left EDGE, not its left vertex: the incoming arrow from
    # the LLM already terminates at that vertex, and two connectors meeting at one point read
    # as a single line.
    route(ax, [(4.02, 1.245), (4.02, 0.68), (3.15, 0.68)], color=INSTR, ls="--")
    edge(ax, 4.10, 0.95, "ungrounded", color=INSTR, ha="left")
    # The retry re-enters at RETRIEVAL, not at prompt assembly: a rewritten query is only
    # useful if it is used to search again.
    route(ax, [(0.55, 1.04), (0.12, 1.04), (0.12, 3.02), (0.30, 3.02)],
          color=INSTR, ls="--")
    ax.text(0.02, 2.03, "retry, max 3 attempts", ha="center", va="center",
            fontsize=FSE, color=INSTR, style="italic", rotation=90)
    save(fig, "fig3_7_agentic_loop")


# ─────────────────────────────────────────────────────────────────────────────
#  3.6  the two cross-lingual handling modes
# ─────────────────────────────────────────────────────────────────────────────
def fig_crosslingual():
    """
    The lower-right box used to read "German evaluation set runs without the bridge (see
    Chapter 4)". Both halves of that were wrong for a design chapter: it named a specific
    evaluation set, which is an experimental choice belonging to a later chapter, and it
    forwarded the reader to a chapter that has not been read yet. The supervisor's review
    asked for both to go.

    What the box should carry is the same thing its left-hand twin carries: the consequence
    of the branch above it. So the two lower boxes are now parallel statements of where
    retrieval and generation actually happen in each mode, which is a Chapter 3 fact and needs
    no reference to anything outside it.
    """
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
        "Retrieval and generation run in\nEnglish; the answer is translated back",
        XLING, fontsize=FSS)
    arrow(ax, (3.25, 2.30), (1.60, 1.97))
    arrow(ax, (3.25, 2.30), (4.90, 1.97))
    arrow(ax, (1.60, 1.35), (1.60, 1.08))
    arrow(ax, (4.90, 1.35), (4.90, 1.08))
    save(fig, "fig3_8_crosslingual")


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
    # k is fixed at 5 on every attempt; only the query changes between passes.
    ax.text(6.42, (ys[6] + ys[3]) / 2 + bh / 2, "UNGROUNDED: retry, same k",
            ha="center", va="center", fontsize=FSE, color=INSTR,
            style="italic", rotation=90)
    edge(ax, x - 0.42, ys[7] + bh + 0.09, "GROUNDED, or\nattempt 3 reached",
         color="#1B7F4B", ha="right")
    save(fig, "fig3_2_query_flow")


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
    save(fig, "fig3_3_instrumentation")


def main():
    print("[arch-split] generating the eight Chapter 3 diagrams...")
    # generated in the order the figures now appear in the chapter
    fig_overview()          # 3.1
    fig_query_flow()        # 3.2
    fig_instrumentation()   # 3.3
    fig_ingestion()         # 3.4
    fig_chunk_metadata()    # 3.5
    fig_hybrid_retrieval()  # 3.6
    fig_agentic_loop()      # 3.7
    fig_crosslingual()      # 3.8
    print(f"[arch-split] done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
