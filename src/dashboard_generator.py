#!/usr/bin/env python3
"""
LUFA RAG Evaluation Dashboard generator.

This module is the single source of truth for building the evaluation dashboard
HTML.  It produces a page that:

  * Works fully in a JavaScript-DISABLED browser.  Python pre-computes every
    average ("KPI") block and every basic bar chart (rendered as pure-CSS/SVG
    bars), and writes the complete detailed results table in the project theme
    colours.  No JS is required to read any of that.

  * Progressively ENHANCES into a richer interactive report when JavaScript is
    available.  The richer version adds Chart.js graphics (all 9+ metrics),
    a per-column filter UI on the detailed table, and a "Normalize Data" gear
    toggle.  Filtering and normalisation recompute every KPI, chart and the
    table live, and the two filters compose (they can be used together or
    independently).

Public API (kept stable so existing call sites only need to change the import):

    from dashboard_generator import generate_dashboard, df_to_js_data
    generate_dashboard(df, "dashboard/index.html")

`generate_dashboard(df, output_path)` keeps the exact same signature the rest of
the codebase already calls, so updating those modules is a one-line import swap.
"""

import json
from datetime import datetime

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  METRIC DEFINITIONS  (single source of truth, shared by Python + embedded JS)
# ─────────────────────────────────────────────────────────────────────────────
# Per-row numeric metric columns that the dashboard knows how to aggregate.
NUMERIC_METRICS = [
    "token_f1_score", "sentence_bleu_score", "rouge1", "rouge2", "rougeL",
    "meteor", "mrr", "ndcg_at_k", "recall_1", "recall_3", "recall_5",
    "answer_relevance", "faithfulness", "context_precision",
]

GEN_METRICS = ["token_f1_score", "sentence_bleu_score", "rouge1", "rouge2", "rougeL", "meteor"]
RET_METRICS = ["mrr", "ndcg_at_k", "recall_1", "recall_3", "recall_5"]
JUDGE_METRICS = ["answer_relevance", "faithfulness", "context_precision"]

# Columns carried into each embedded row.  The JS layer recomputes *everything*
# from these rows, so any column listed here can be used as a table filter.
ROW_COLUMNS = [
    "question_id", "question", "answer", "rag_base_model", "language",
    "category", "difficulty",
    "token_f1_score", "sentence_bleu_score", "rouge1", "rouge2", "rougeL",
    "meteor", "mrr", "ndcg_at_k", "recall_1", "recall_3", "recall_5",
    "answer_relevance", "faithfulness", "context_precision",
    "grounded", "attempts",
]

# The 13 headline averages the supervisor asked to always see (req. 1).
# (key, label, kind)  kind drives formatting: "score" 0-1, "pct" %, "num" raw.
KPI_DEFS = [
    ("__count__",          "Questions",   "count"),
    ("token_f1_score",     "Avg F1",      "score"),
    ("sentence_bleu_score","Avg BLEU",    "score"),
    ("rougeL",             "Avg ROUGE-L", "score"),
    ("meteor",             "Avg METEOR",  "score"),
    ("mrr",                "Avg MRR",     "score"),
    ("recall_1",           "Recall@1",    "score"),
    ("recall_3",           "Recall@3",    "score"),
    ("recall_5",           "Recall@5",    "score"),
    ("answer_relevance",   "Relevance",   "score"),
    ("faithfulness",       "Faithful",    "score"),
    ("context_precision",  "Precision",   "score"),
    ("__grounded__",       "Grounded",    "pct"),
    ("attempts",           "Avg Attempts","num"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _to_num(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _is_grounded(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def _row_has_answer(row):
    """A row is considered 'answered' when the model produced a non-empty,
    non-error answer.  Used by the JS normalize toggle (mirrored here for the
    static fallback)."""
    a = row.get("answer", "")
    if a is None or (isinstance(a, float) and pd.isna(a)):
        return False
    a = str(a).strip()
    return a != "" and a.upper() != "ERROR" and a.lower() != "nan"


def _row_has_retrieval(row):
    """True if the row retrieved at least a top-1 source id."""
    sid = row.get("source1_id", "")
    if sid is None or (isinstance(sid, float) and pd.isna(sid)):
        # Fall back to recall/mrr signal when source ids are not embedded.
        return _to_num_scalar(row.get("recall_1")) > 0 or _to_num_scalar(row.get("mrr")) > 0
    return str(sid).strip() not in ("", "nan")


def _to_num_scalar(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(v)
    except Exception:
        return default


def _avg_by(df, group_col, metric):
    if group_col not in df.columns or metric not in df.columns:
        return {}
    tmp = df.copy()
    tmp[metric] = _to_num(tmp[metric])
    return {
        str(k): round(float(v), 4)
        for k, v in tmp.groupby(group_col)[metric].mean().items()
        if str(k).strip() != "" and str(k).strip().lower() != "nan"
    }


def _overall_avgs(df):
    out = {}
    for m in NUMERIC_METRICS:
        if m in df.columns:
            out[m] = round(float(_to_num(df[m]).mean()), 4) if len(df) else 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  DATA PREP
# ─────────────────────────────────────────────────────────────────────────────
def df_to_js_data(df):
    """Build the JSON payload embedded in the page.

    Includes the full per-row records (so the JS layer can recompute and filter
    by ANY column) plus the server pre-computed aggregates used to render the
    JS-free fallback view.
    """
    cleaned = df.copy()
    if "question_id" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["question_id", "rag_base_model"], keep="last")
        cleaned = cleaned[cleaned["question_id"].astype(str).str.strip() != ""]

    for m in NUMERIC_METRICS:
        if m in cleaned.columns:
            cleaned[m] = _to_num(cleaned[m])

    models = [m for m in cleaned.get("rag_base_model", pd.Series(dtype=str)).dropna().unique().tolist()
              if str(m).strip() != ""]

    # Per-row records carrying every filterable / recomputable column.
    present_cols = [c for c in ROW_COLUMNS if c in cleaned.columns]
    rows_df = cleaned[present_cols].copy()
    # answer / source columns may be huge; trim answer text for transport only
    if "answer" in rows_df.columns:
        rows_df["answer"] = rows_df["answer"].astype(str).str.slice(0, 400)
    # Keep an explicit "has answer" / "has retrieval" precomputed flag per row so
    # the JS normalize filter is cheap and the static fallback can mirror it.
    has_answer_flags, has_ret_flags = [], []
    for _, r in cleaned.iterrows():
        has_answer_flags.append(bool(_row_has_answer(r)))
        has_ret_flags.append(bool(_row_has_retrieval(r)))
    rows = rows_df.fillna("").to_dict(orient="records")
    for rec, ha, hr in zip(rows, has_answer_flags, has_ret_flags):
        rec["_has_answer"] = ha
        rec["_has_retrieval"] = hr

    grounded_rate = 0.0
    if "grounded" in cleaned.columns and len(cleaned):
        grounded_rate = round(float(_is_grounded(cleaned["grounded"]).mean()), 4)
    avg_attempts = 0.0
    if "attempts" in cleaned.columns and len(cleaned):
        avg_attempts = round(float(_to_num(cleaned["attempts"]).mean()), 2)

    data = {
        "models": models,
        "overall": _overall_avgs(cleaned),
        "by_model": {m: _avg_by(cleaned, "rag_base_model", m)
                     for m in NUMERIC_METRICS if m in cleaned.columns},
        "by_language": {m: _avg_by(cleaned, "language", m)
                        for m in GEN_METRICS + JUDGE_METRICS if m in cleaned.columns},
        "by_difficulty": {m: _avg_by(cleaned, "difficulty", m)
                          for m in GEN_METRICS + RET_METRICS if m in cleaned.columns},
        "by_category": {m: _avg_by(cleaned, "category", m)
                        for m in GEN_METRICS + JUDGE_METRICS if m in cleaned.columns},
        "grounded_rate": grounded_rate,
        "avg_attempts": avg_attempts,
        "total_questions": int(len(cleaned)),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "rows": rows,
        "metric_meta": {
            "numeric": NUMERIC_METRICS,
            "gen": GEN_METRICS,
            "ret": RET_METRICS,
            "judge": JUDGE_METRICS,
            "kpis": KPI_DEFS,
            "row_columns": present_cols,
        },
    }
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  STATIC (NO-JS) FRAGMENT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_score(v):
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v) if v not in (None, "") else ""


def _score_class(v):
    try:
        v = float(v)
    except Exception:
        return ""
    return "score-high" if v >= 0.7 else "score-mid" if v >= 0.4 else "score-low"


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _build_kpi_html(data):
    """Static KPI cards — all 13 headline averages, readable without JS."""
    ov = data["overall"]
    cards = []
    for key, label, kind in KPI_DEFS:
        if kind == "count":
            value = str(data["total_questions"])
        elif kind == "pct":
            value = f"{data['grounded_rate'] * 100:.1f}%"
        elif kind == "num":
            value = f"{data['avg_attempts']:.2f}"
        else:
            value = _fmt_score(ov.get(key, 0.0))
        cards.append(
            f'<div class="metric-card">'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{_esc(label)}</div>'
            f'</div>'
        )
    return "\n".join(cards)


def _bars(pairs, palette):
    """Render a horizontal-ish bar group as inline SVG/CSS (no JS).
    `pairs` is a list of (label, value 0..maxval)."""
    rows = []
    for i, (label, val) in enumerate(pairs):
        v = max(0.0, float(val))
        pct = min(100.0, v * 100.0)  # metrics are 0..1
        color = palette[i % len(palette)]
        rows.append(
            f'<div class="sbar-row">'
            f'<span class="sbar-label">{_esc(label)}</span>'
            f'<span class="sbar-track"><span class="sbar-fill" '
            f'style="width:{pct:.1f}%;background:{color}"></span></span>'
            f'<span class="sbar-val">{v:.3f}</span>'
            f'</div>'
        )
    return '<div class="sbar-group">' + "".join(rows) + "</div>"


def _build_static_charts_html(data):
    """Pre-rendered, JS-free versions of every chart, using the embedded
    server-side aggregates."""
    GEN_LABELS = [("token_f1_score", "F1"), ("sentence_bleu_score", "BLEU"),
                  ("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"),
                  ("rougeL", "ROUGE-L"), ("meteor", "METEOR")]
    RET_LABELS = [("mrr", "MRR"), ("ndcg_at_k", "NDCG"), ("recall_1", "Recall@1"),
                  ("recall_3", "Recall@3"), ("recall_5", "Recall@5")]
    JUDGE_LABELS = [("answer_relevance", "Answer Relevance"),
                    ("faithfulness", "Faithfulness"),
                    ("context_precision", "Context Precision")]

    blue = ["#3b82f6"]
    light = ["#93c5fd"]
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4"]
    ov = data["overall"]

    gen = _bars([(lbl, ov.get(k, 0.0)) for k, lbl in GEN_LABELS], blue * 6)
    ret = _bars([(lbl, ov.get(k, 0.0)) for k, lbl in RET_LABELS], light * 6)
    judge = _bars([(lbl, ov.get(k, 0.0)) for k, lbl in JUDGE_LABELS], palette)

    # language: F1 / ROUGE-L / METEOR grouped
    langs = list((data["by_language"].get("token_f1_score") or {}).keys())
    lang_rows = []
    for lang in langs:
        lang_rows.append(f'<div class="sbar-subhead">{_esc(lang)}</div>')
        lang_rows.append(_bars([
            ("F1", (data["by_language"].get("token_f1_score") or {}).get(lang, 0.0)),
            ("ROUGE-L", (data["by_language"].get("rougeL") or {}).get(lang, 0.0)),
            ("METEOR", (data["by_language"].get("meteor") or {}).get(lang, 0.0)),
        ], palette))
    lang = "".join(lang_rows) if lang_rows else '<div class="sbar-empty">No data</div>'

    diffs = list((data["by_difficulty"].get("token_f1_score") or {}).keys())
    diff = _bars([(d, (data["by_difficulty"].get("token_f1_score") or {}).get(d, 0.0))
                  for d in diffs], palette) if diffs else '<div class="sbar-empty">No data</div>'

    cats = list((data["by_category"].get("rougeL") or {}).keys())
    cat = _bars([(c, (data["by_category"].get("rougeL") or {}).get(c, 0.0))
                 for c in cats], palette) if cats else '<div class="sbar-empty">No data</div>'

    return {"gen": gen, "ret": ret, "judge": judge, "lang": lang, "diff": diff, "cat": cat}


def _build_static_table_html(data):
    """Full detailed results table, theme-coloured, JS-free."""
    rows = data["rows"]
    body = []
    for i, r in enumerate(rows, start=1):
        lang = str(r.get("language", ""))
        badge = ("badge-en" if lang.lower().startswith("en")
                 else "badge-fr" if lang.lower().startswith("fr") else "badge-other")
        q = _esc(r.get("question", ""))[:90]
        grounded = str(r.get("grounded", "")).strip().lower() in ("true", "1", "1.0", "yes")
        gicon = '<span class="ok">&#10003;</span>' if grounded else '<span class="no">&#10007;</span>'
        body.append(
            "<tr>"
            f'<td class="muted">{i}</td>'
            f'<td class="qcell" title="{_esc(r.get("question",""))}">{q}</td>'
            f'<td class="model">{_esc(r.get("rag_base_model",""))}</td>'
            f'<td><span class="badge {badge}">{_esc(lang)}</span></td>'
            f'<td>{_esc(r.get("category",""))}</td>'
            f'<td>{_esc(r.get("difficulty",""))}</td>'
            f'<td class="{_score_class(r.get("token_f1_score"))}">{_fmt_score(r.get("token_f1_score"))}</td>'
            f'<td class="{_score_class(r.get("sentence_bleu_score"))}">{_fmt_score(r.get("sentence_bleu_score"))}</td>'
            f'<td class="{_score_class(r.get("rougeL"))}">{_fmt_score(r.get("rougeL"))}</td>'
            f'<td class="{_score_class(r.get("meteor"))}">{_fmt_score(r.get("meteor"))}</td>'
            f'<td class="{_score_class(r.get("mrr"))}">{_fmt_score(r.get("mrr"))}</td>'
            f'<td class="{_score_class(r.get("recall_1"))}">{_fmt_score(r.get("recall_1"))}</td>'
            f'<td class="{_score_class(r.get("recall_3"))}">{_fmt_score(r.get("recall_3"))}</td>'
            f'<td class="{_score_class(r.get("recall_5"))}">{_fmt_score(r.get("recall_5"))}</td>'
            f'<td class="{_score_class(r.get("answer_relevance"))}">{_fmt_score(r.get("answer_relevance"))}</td>'
            f'<td class="{_score_class(r.get("faithfulness"))}">{_fmt_score(r.get("faithfulness"))}</td>'
            f'<td class="{_score_class(r.get("context_precision"))}">{_fmt_score(r.get("context_precision"))}</td>'
            f'<td class="center">{gicon}</td>'
            f'<td class="center muted">{_esc(r.get("attempts",""))}</td>'
            "</tr>"
        )
    return "\n".join(body)


# ─────────────────────────────────────────────────────────────────────────────
#  GENERATE
# ─────────────────────────────────────────────────────────────────────────────
def generate_dashboard(df, output_path):
    """Build and write the dashboard HTML to `output_path`.

    Signature is intentionally identical to the legacy function so every
    existing caller works after only swapping the import.
    """
    data = df_to_js_data(df)
    data_json = json.dumps(data, ensure_ascii=False, default=str)

    static_kpis = _build_kpi_html(data)
    static_charts = _build_static_charts_html(data)
    static_table = _build_static_table_html(data)

    html = (DASHBOARD_TEMPLATE
            .replace("__DATA_PLACEHOLDER__", data_json)
            .replace("__GEN_AT__", _esc(data["generated_at"]))
            .replace("__STATIC_KPIS__", static_kpis)
            .replace("__STATIC_GEN__", static_charts["gen"])
            .replace("__STATIC_RET__", static_charts["ret"])
            .replace("__STATIC_JUDGE__", static_charts["judge"])
            .replace("__STATIC_LANG__", static_charts["lang"])
            .replace("__STATIC_DIFF__", static_charts["diff"])
            .replace("__STATIC_CAT__", static_charts["cat"])
            .replace("__STATIC_TABLE__", static_table))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def refresh_dashboard(out_path="dashboard/index.html",
                      eval_csv="tests/evaluation_results.csv",
                      lufa_csv="tests/lufa_out_data.csv"):
    """
    Best-effort regeneration of the HTML dashboard from whatever CSVs exist.

    Prefers evaluation_results.csv (has metrics); falls back to lufa_out_data.csv
    so the dashboard still refreshes during retrieval/answer-only phases. Never
    raises — a dashboard hiccup must not interrupt row-by-row processing.
    """
    try:
        from pathlib import Path
        ev, lf = Path(eval_csv), Path(lufa_csv)
        if ev.exists() and ev.stat().st_size > 0:
            df = pd.read_csv(ev, on_bad_lines="skip")
        elif lf.exists() and lf.stat().st_size > 0:
            df = pd.read_csv(lf, on_bad_lines="skip")
        else:
            return False
        if df is None or df.empty:
            return False
        # Dashboard code keys models off rag_base_model; lufa uses base_model_used.
        if "rag_base_model" not in df.columns and "base_model_used" in df.columns:
            df["rag_base_model"] = df["base_model_used"]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        generate_dashboard(df, out_path)
        return True
    except Exception as e:
        print(f"      [Dashboard] refresh skipped: {e}")
        return False


# The HTML template lives in dashboard_template.py to keep this file readable.
from dashboard_template import DASHBOARD_TEMPLATE  # noqa: E402
