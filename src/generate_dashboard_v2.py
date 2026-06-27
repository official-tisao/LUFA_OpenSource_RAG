#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


CANONICAL_COLUMN_CANDIDATES = {
    "question_id": ["question_id", "id"],
    "question": ["question"],
    "answer": ["answer"],
    "expected_answer": ["expected_answer"],
    "rag_base_model": ["rag_base_model", "base_model_used"],
    "language": ["language"],
    "category": ["category"],
    "difficulty": ["difficulty"],
    "token_f1_score": ["token_f1_score"],
    "sentence_bleu_score": ["sentence_bleu_score"],
    "rougeL": ["rougeL", "rouge_l"],
    "meteor": ["meteor"],
    "mrr": ["mrr"],
    "ndcg_at_k": ["ndcg_at_k", "ndcg"],
    "recall_1": ["recall_1", "recall1"],
    "recall_3": ["recall_3", "recall3"],
    "recall_5": ["recall_5", "recall5"],
    "answer_relevance": ["answer_relevance", "relevance"],
    "faithfulness": ["faithfulness", "faithful"],
    "context_precision": ["context_precision", "precision"],
    "grounded": ["grounded"],
    "attempts": ["attempts"],
    "source1_id": ["source1_id", "source1id"],
    "source1_text": ["source1_text", "source1text"],
    "source1_score": ["source1_score", "source1score"],
    "source2_id": ["source2_id", "source2id"],
    "source2_text": ["source2_text", "source2text"],
    "source2_score": ["source2_score", "source2score"],
    "source3_id": ["source3_id", "source3id"],
    "source3_text": ["source3_text", "source3text"],
    "source3_score": ["source3_score", "source3score"],
    "source4_id": ["source4_id", "source4id"],
    "source4_text": ["source4_text", "source4text"],
    "source4_score": ["source4_score", "source4score"],
    "source5_id": ["source5_id", "source5id"],
    "source5_text": ["source5_text", "source5text"],
    "source5_score": ["source5_score", "source5score"],
    "original_cosine_score": ["original_cosine_score"],
    "recency_adjusted_score": ["recency_adjusted_score"],
    "RRF": ["RRF"],
    "judge_llm": ["judge_llm"],
    "ground_source_truth_id": ["ground_source_truth_id", "ground_source_truthid", "ground_truth_source_ids"],
    "ground_source_truth": ["ground_source_truth", "ground_truth_source_text"],
}

NUMERIC_COLUMNS = [
    "token_f1_score",
    "sentence_bleu_score",
    "rougeL",
    "meteor",
    "mrr",
    "ndcg_at_k",
    "recall_1",
    "recall_3",
    "recall_5",
    "answer_relevance",
    "faithfulness",
    "context_precision",
    "attempts",
    "source1_score",
    "source2_score",
    "source3_score",
    "source4_score",
    "source5_score",
    "original_cosine_score",
    "recency_adjusted_score",
    "RRF",
]

SUMMARY_METRICS = [
    ("token_f1_score", "Avg F1"),
    ("sentence_bleu_score", "Avg BLEU"),
    ("rougeL", "Avg ROUGE-L"),
    ("meteor", "Avg METEOR"),
    ("mrr", "Avg MRR"),
    ("recall_1", "Avg Recall@1"),
    ("recall_3", "Avg Recall@3"),
    ("recall_5", "Avg Recall@5"),
    ("answer_relevance", "Avg Relevance"),
    ("faithfulness", "Avg Faithful"),
    ("context_precision", "Avg Precision"),
]

PREFERRED_DETAIL_ORDER = [
    "question_id", "question", "answer", "expected_answer", "rag_base_model", "judge_llm",
    "language", "category", "difficulty",
    "token_f1_score", "sentence_bleu_score", "rougeL", "meteor",
    "mrr", "ndcg_at_k", "recall_1", "recall_3", "recall_5",
    "answer_relevance", "faithfulness", "context_precision",
    "grounded", "attempts",
    "source1_id", "source1_score", "source1_text",
    "source2_id", "source2_score", "source2_text",
    "source3_id", "source3_score", "source3_text",
    "source4_id", "source4_score", "source4_text",
    "source5_id", "source5_score", "source5_text",
    "original_cosine_score", "recency_adjusted_score", "RRF",
    "ground_source_truth_id", "ground_source_truth",
]

CHART_GROUPS = {
    "generation": ["token_f1_score", "sentence_bleu_score", "rougeL", "meteor"],
    "retrieval": ["mrr", "recall_1", "recall_3", "recall_5", "ndcg_at_k"],
    "judge": ["answer_relevance", "faithfulness", "context_precision"],
}

DISPLAY_LABELS = {
    "token_f1_score": "F1",
    "sentence_bleu_score": "BLEU",
    "rougeL": "ROUGE-L",
    "meteor": "METEOR",
    "mrr": "MRR",
    "ndcg_at_k": "NDCG",
    "recall_1": "Recall@1",
    "recall_3": "Recall@3",
    "recall_5": "Recall@5",
    "answer_relevance": "Relevance",
    "faithfulness": "Faithful",
    "context_precision": "Precision",
    "grounded_rate": "Grounded Rate",
    "attempts": "Attempts",
}


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _to_bool(value) -> bool:
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return 0.0
    return round(float(s.mean()), 4)


def _prepare_dataframe(
    eval_df: pd.DataFrame,
    lufa_df: Optional[pd.DataFrame] = None,
    gt_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = eval_df.copy()

    if "question_id" not in df.columns and "id" in df.columns:
        df["question_id"] = df["id"]

    if lufa_df is not None and not lufa_df.empty:
        lufa = lufa_df.copy()
        if "question_id" not in lufa.columns and "id" in lufa.columns:
            lufa["question_id"] = lufa["id"]
        lufa = lufa.drop_duplicates(subset=["question_id"], keep="last")
        extra_cols = [c for c in lufa.columns if c not in df.columns or c in {"source1_id", "source1_text", "source1_score", "source2_id", "source2_text", "source2_score", "source3_id", "source3_text", "source3_score", "source4_id", "source4_text", "source4_score", "source5_id", "source5_text", "source5_score", "answer"}]
        df = df.merge(lufa[["question_id"] + [c for c in extra_cols if c != "question_id"]], on="question_id", how="left", suffixes=("", "__lufa"))
        for col in list(df.columns):
            if col.endswith("__lufa"):
                base = col[:-6]
                if base not in df.columns:
                    df[base] = df[col]
                else:
                    df[base] = df[base].where(df[base].notna() & (df[base].astype(str) != ""), df[col])
                df = df.drop(columns=[col])

    if gt_df is not None and not gt_df.empty:
        gt = gt_df.copy()
        if "question_id" not in gt.columns and "id" in gt.columns:
            gt["question_id"] = gt["id"]
        gt = gt.drop_duplicates(subset=["question_id"], keep="last")
        gt_keep = [c for c in ["question_id", "expected_answer", "ground_source_truth_id", "ground_source_truth", "ground_truth_source_ids"] if c in gt.columns]
        if gt_keep:
            df = df.merge(gt[gt_keep], on="question_id", how="left", suffixes=("", "__gt"))
            for col in list(df.columns):
                if col.endswith("__gt"):
                    base = col[:-4]
                    if base not in df.columns:
                        df[base] = df[col]
                    else:
                        df[base] = df[base].where(df[base].notna() & (df[base].astype(str) != ""), df[col])
                    df = df.drop(columns=[col])

    normalized = pd.DataFrame()
    for canon, candidates in CANONICAL_COLUMN_CANDIDATES.items():
        src = _pick_column(df, candidates)
        normalized[canon] = df[src] if src else ""

    for col in NUMERIC_COLUMNS:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    normalized["grounded"] = normalized["grounded"].apply(_to_bool)
    normalized["attempts"] = pd.to_numeric(normalized["attempts"], errors="coerce").fillna(0)

    for col in normalized.columns:
        if col not in NUMERIC_COLUMNS and col != "grounded":
            normalized[col] = normalized[col].fillna("").astype(str)

    normalized["__has_answer__"] = normalized["answer"].astype(str).str.strip() != ""
    normalized["__has_top1__"] = (
        (normalized["source1_id"].astype(str).str.strip() != "") |
        (normalized["source1_text"].astype(str).str.strip() != "")
    )
    normalized["__normalizable__"] = normalized["__has_answer__"] & normalized["__has_top1__"]

    normalized = normalized.drop_duplicates(subset=["question_id"], keep="last")
    return normalized


def _metric_dict(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for key, _label in SUMMARY_METRICS:
        out[key] = _mean(df[key]) if key in df.columns else 0.0
    out["grounded_rate"] = round(float(df["grounded"].mean()) if len(df) else 0.0, 4)
    out["attempts"] = round(float(pd.to_numeric(df["attempts"], errors="coerce").fillna(0).mean()) if len(df) else 0.0, 4)
    out["rows"] = int(len(df))
    out["normalized_rows"] = int(df["__normalizable__"].sum()) if "__normalizable__" in df.columns else 0
    return out


def _group_mean(df: pd.DataFrame, group_col: str, metric_col: str) -> Dict[str, float]:
    if group_col not in df.columns or metric_col not in df.columns or df.empty:
        return {}
    temp = df[[group_col, metric_col]].copy()
    temp[metric_col] = pd.to_numeric(temp[metric_col], errors="coerce")
    temp[group_col] = temp[group_col].replace("", "Unknown")
    grouped = temp.groupby(group_col, dropna=False)[metric_col].mean().fillna(0)
    return {str(k): round(float(v), 4) for k, v in grouped.items()}


def _aggregate(df: pd.DataFrame) -> Dict:
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "overall": _metric_dict(df),
        "models": sorted([m for m in df["rag_base_model"].replace("", "Unknown").unique().tolist() if m != ""]) if "rag_base_model" in df.columns else [],
        "languages": sorted([m for m in df["language"].replace("", "Unknown").unique().tolist() if m != ""]) if "language" in df.columns else [],
        "categories": sorted([m for m in df["category"].replace("", "Unknown").unique().tolist() if m != ""]) if "category" in df.columns else [],
        "difficulties": sorted([m for m in df["difficulty"].replace("", "Unknown").unique().tolist() if m != ""]) if "difficulty" in df.columns else [],
        "by_model": {},
        "by_language": {},
        "by_category": {},
        "by_difficulty": {},
    }

    all_metrics = [m for m, _ in SUMMARY_METRICS] + ["ndcg_at_k"]
    for metric in all_metrics:
        payload["by_model"][metric] = _group_mean(df, "rag_base_model", metric)
        payload["by_language"][metric] = _group_mean(df, "language", metric)
        payload["by_category"][metric] = _group_mean(df, "category", metric)
        payload["by_difficulty"][metric] = _group_mean(df, "difficulty", metric)

    payload["by_language"]["grounded_rate"] = (
        df.assign(_grounded_num=df["grounded"].astype(int))
          .groupby(df["language"].replace("", "Unknown"))["_grounded_num"]
          .mean().round(4).to_dict()
        if len(df) else {}
    )

    return payload


def _score_class(v) -> str:
    try:
        x = float(v)
    except Exception:
        return "score-na"
    if x >= 0.70:
        return "score-high"
    if x >= 0.40:
        return "score-mid"
    return "score-low"


def _fmt(v) -> str:
    try:
        return f"{float(v):.4f}"
    except Exception:
        return html.escape(str(v))


def _svg_bar_card(title: str, metric_map: Dict[str, float], height: int = 220) -> str:
    items = list(metric_map.items())
    if not items:
        items = [("No data", 0.0)]
    width = 860
    margin_left = 56
    margin_bottom = 48
    chart_w = width - margin_left - 24
    chart_h = height - 50 - margin_bottom
    bar_w = max(24, int(chart_w / max(len(items), 1) * 0.55))
    gap = chart_w / max(len(items), 1)

    bars = []
    labels = []
    grids = []
    for i in range(6):
        y = 28 + chart_h - (chart_h * i / 5.0)
        grids.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-16}" y2="{y:.1f}" stroke="#334155" stroke-width="1"/>')
        grids.append(f'<text x="18" y="{y+4:.1f}" fill="#ffffff" font-size="11" font-weight="700">{i/5:.1f}</text>')

    for idx, (label, value) in enumerate(items):
        x = margin_left + idx * gap + (gap - bar_w) / 2
        h = max(0, min(1.0, float(value))) * chart_h
        y = 28 + chart_h - h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="6" fill="#3b82f6" stroke="#93c5fd" stroke-width="1.2"/>'
        )
        labels.append(
            f'<text x="{x + bar_w/2:.1f}" y="{height-18}" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="700">{html.escape(str(label))}</text>'
        )

    return f"""
    <div class="card static-chart-card">
      <div class="section-title">{html.escape(title)}</div>
      <svg viewBox="0 0 {width} {height}" class="static-svg" role="img" aria-label="{html.escape(title)}">
        {''.join(grids)}
        {''.join(bars)}
        {''.join(labels)}
      </svg>
    </div>
    """


def _svg_radar_card(title: str, values: Dict[str, float]) -> str:
    labels = ["Relevance", "Faithful", "Precision"]
    nums = [
        float(values.get("answer_relevance", 0.0)),
        float(values.get("faithfulness", 0.0)),
        float(values.get("context_precision", 0.0)),
    ]
    cx, cy, r = 180, 140, 88

    def point(angle_deg, scale):
        import math
        ang = math.radians(angle_deg)
        return (
            cx + (r * scale * math.cos(ang)),
            cy + (r * scale * math.sin(ang))
        )

    base_angles = [-90, 30, 150]
    rings = []
    for s in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{point(a, s)[0]:.1f},{point(a, s)[1]:.1f}" for a in base_angles)
        rings.append(f'<polygon points="{pts}" fill="none" stroke="#334155" stroke-width="1"/>')

    spokes = []
    for a in base_angles:
        x, y = point(a, 1.0)
        spokes.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#475569" stroke-width="1"/>')

    pts = " ".join(f"{point(a, max(0.0, min(1.0, v)))[0]:.1f},{point(a, max(0.0, min(1.0, v)))[1]:.1f}" for a, v in zip(base_angles, nums))
    labels_svg = [
        f'<text x="{point(-90, 1.15)[0]:.1f}" y="{point(-90, 1.15)[1]:.1f}" text-anchor="middle" fill="#ffffff" font-size="12" font-weight="700">Relevance</text>',
        f'<text x="{point(30, 1.22)[0]:.1f}" y="{point(30, 1.22)[1]:.1f}" text-anchor="middle" fill="#ffffff" font-size="12" font-weight="700">Faithful</text>',
        f'<text x="{point(150, 1.25)[0]:.1f}" y="{point(150, 1.25)[1]:.1f}" text-anchor="middle" fill="#ffffff" font-size="12" font-weight="700">Precision</text>',
    ]

    return f"""
    <div class="card static-chart-card">
      <div class="section-title">{html.escape(title)}</div>
      <svg viewBox="0 0 360 260" class="static-svg" role="img" aria-label="{html.escape(title)}">
        {''.join(rings)}
        {''.join(spokes)}
        <polygon points="{pts}" fill="rgba(59,130,246,0.28)" stroke="#93c5fd" stroke-width="2"/>
        {''.join(labels_svg)}
      </svg>
    </div>
    """


def _build_static_table(df: pd.DataFrame, detail_columns: List[str]) -> str:
    thead = "".join(f"<th>{html.escape(c)}</th>" for c in detail_columns)
    body_rows = []

    for _, row in df.iterrows():
        cells = []
        for col in detail_columns:
            val = row.get(col, "")
            if col == "grounded":
                rendered = "TRUE" if _to_bool(val) else "FALSE"
            elif col in NUMERIC_COLUMNS:
                rendered = _fmt(val) if str(val) != "" else ""
            else:
                rendered = html.escape(str(val))
            cells.append(f"<td class='{_score_class(val) if col in NUMERIC_COLUMNS else ''}'>{rendered}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
    <div class="card">
      <div class="section-title">Detailed Results</div>
      <div class="table-wrap">
        <table id="detailsTable">
          <thead>
            <tr>{thead}</tr>
          </thead>
          <tbody id="detailsBody">
            {''.join(body_rows)}
          </tbody>
        </table>
      </div>
    </div>
    """


def _build_kpi_cards(stats: Dict[str, float]) -> str:
    metric_blocks = []
    ordered = SUMMARY_METRICS + [("grounded_rate", "Grounded Rate"), ("attempts", "Avg Attempts")]
    for key, label in ordered:
        val = stats.get(key, 0.0)
        metric_blocks.append(
            f"""
            <div class="metric-card">
              <div class="metric-value">{float(val):.4f}</div>
              <div class="metric-label">{html.escape(label)}</div>
            </div>
            """
        )
    metric_blocks.insert(
        0,
        f"""
        <div class="metric-card">
          <div class="metric-value">{int(stats.get('rows', 0))}</div>
          <div class="metric-label">Rows</div>
        </div>
        """
    )
    metric_blocks.insert(
        1,
        f"""
        <div class="metric-card">
          <div class="metric-value">{int(stats.get('normalized_rows', 0))}</div>
          <div class="metric-label">Normalized Rows</div>
        </div>
        """
    )
    return "".join(metric_blocks)


def _build_rows_payload(df: pd.DataFrame, detail_columns: List[str]) -> List[Dict]:
    rows = []
    for _, row in df.iterrows():
        item = {}
        for col in detail_columns:
            v = row.get(col, "")
            if col == "grounded":
                item[col] = bool(_to_bool(v))
            elif col in NUMERIC_COLUMNS:
                item[col] = None if pd.isna(v) else round(float(v), 6)
            else:
                item[col] = "" if pd.isna(v) else str(v)
        item["__normalizable__"] = bool(row.get("__normalizable__", False))
        rows.append(item)
    return rows


def generate_dashboard(
    df: pd.DataFrame,
    output_path: str = "dashboard/index.html",
    lufa_df: Optional[pd.DataFrame] = None,
    gt_df: Optional[pd.DataFrame] = None,
) -> None:
    prepared = _prepare_dataframe(df, lufa_df=lufa_df, gt_df=gt_df)

    detail_columns = [c for c in PREFERRED_DETAIL_ORDER if c in prepared.columns]
    for c in prepared.columns:
        if c not in detail_columns and not c.startswith("__"):
            detail_columns.append(c)

    aggregate = _aggregate(prepared)
    static_stats = aggregate["overall"]
    rows_payload = _build_rows_payload(prepared, detail_columns)

    static_generation = {
        "F1": static_stats.get("token_f1_score", 0.0),
        "BLEU": static_stats.get("sentence_bleu_score", 0.0),
        "ROUGE-L": static_stats.get("rougeL", 0.0),
        "METEOR": static_stats.get("meteor", 0.0),
    }
    static_retrieval = {
        "MRR": static_stats.get("mrr", 0.0),
        "Recall@1": static_stats.get("recall_1", 0.0),
        "Recall@3": static_stats.get("recall_3", 0.0),
        "Recall@5": static_stats.get("recall_5", 0.0),
        "NDCG": static_stats.get("ndcg_at_k", 0.0),
    }
    static_judge = {
        "answer_relevance": static_stats.get("answer_relevance", 0.0),
        "faithfulness": static_stats.get("faithfulness", 0.0),
        "context_precision": static_stats.get("context_precision", 0.0),
    }
    static_support = {
        "Grounded": static_stats.get("grounded_rate", 0.0),
        "Attempts": min(1.0, static_stats.get("attempts", 0.0) / 5.0),
    }

    data_payload = {
        "generated_at": aggregate["generated_at"],
        "detail_columns": detail_columns,
        "rows": rows_payload,
        "display_labels": DISPLAY_LABELS,
    }

    static_table_html = _build_static_table(prepared, detail_columns)
    kpi_html = _build_kpi_cards(static_stats)
    static_chart_html = (
        _svg_bar_card("Static Generation Averages", static_generation) +
        _svg_bar_card("Static Retrieval Averages", static_retrieval) +
        _svg_radar_card("Static Judge Metrics", static_judge) +
        _svg_bar_card("Static Support Metrics", static_support)
    )

    template = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>LUFA RAG Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root{
      --bg:#0b1220;
      --surface:#162234;
      --surface2:#1d2b40;
      --line:#334155;
      --line2:#4b5f7a;
      --text:#ffffff;
      --muted:#dbeafe;
      --blue:#60a5fa;
      --blue2:#93c5fd;
      --green:#22c55e;
      --amber:#f59e0b;
      --red:#ef4444;
      --purple:#a78bfa;
      --cyan:#22d3ee;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      background:var(--bg);
      color:var(--text);
      font-family:Segoe UI,Arial,sans-serif;
      font-weight:700;
    }
    body *{
      color:var(--text);
      font-weight:700;
    }
    .page{
      padding:20px;
      max-width:1900px;
      margin:0 auto;
    }
    .banner{
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:18px;
      margin-bottom:18px;
      flex-wrap:wrap;
    }
    .title{
      font-size:32px;
      line-height:1.15;
      margin:0;
      color:#ffffff;
      font-weight:800;
      letter-spacing:.03em;
    }
    .subtitle{
      margin-top:6px;
      color:var(--muted);
      font-size:14px;
    }
    .banner-actions{
      display:flex;
      align-items:center;
      gap:12px;
      flex-wrap:wrap;
    }
    .card{
      background:var(--surface);
      border:1px solid var(--line);
      border-radius:14px;
      padding:16px;
      box-shadow:0 8px 22px rgba(0,0,0,.18);
    }
    .metric-grid{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
      gap:12px;
      margin-bottom:18px;
    }
    .metric-card{
      background:linear-gradient(135deg,#1f3555,#162234);
      border:1px solid #3b82f655;
      border-radius:12px;
      padding:14px;
      min-height:92px;
    }
    .metric-value{
      font-size:24px;
      color:#ffffff;
      margin-bottom:8px;
      font-weight:800;
    }
    .metric-label{
      font-size:13px;
      color:var(--muted);
    }
    .section-title{
      font-size:16px;
      margin:0 0 12px 0;
      letter-spacing:.08em;
      text-transform:uppercase;
      color:#ffffff;
      font-weight:800;
    }
    .charts-grid{
      display:grid;
      grid-template-columns:repeat(2,minmax(0,1fr));
      gap:16px;
      margin-bottom:18px;
    }
    .static-grid{
      display:grid;
      grid-template-columns:repeat(2,minmax(0,1fr));
      gap:16px;
      margin-bottom:18px;
    }
    .canvas-wrap{
      min-height:320px;
      height:320px;
      position:relative;
    }
    .canvas-wrap canvas{
      width:100% !important;
      height:100% !important;
    }
    .filter-card{
      margin-bottom:18px;
    }
    .filter-grid{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
      gap:10px;
    }
    .filter-item{
      display:flex;
      flex-direction:column;
      gap:6px;
    }
    .filter-item label{
      font-size:12px;
      color:var(--muted);
      text-transform:uppercase;
      letter-spacing:.04em;
    }
    .filter-item input{
      width:100%;
      border:1px solid var(--line2);
      background:#0f172a;
      color:#fff;
      border-radius:10px;
      padding:9px 10px;
      font-weight:700;
    }
    .table-wrap{
      overflow:auto;
      max-height:520px;
      border:1px solid var(--line);
      border-radius:12px;
    }
    table{
      width:100%;
      border-collapse:collapse;
      min-width:1700px;
      background:transparent;
    }
    thead th{
      position:sticky;
      top:0;
      background:#101828;
      color:#fff;
      z-index:2;
      border-bottom:1px solid var(--line2);
    }
    th,td{
      padding:10px 12px;
      border-bottom:1px solid #243247;
      vertical-align:top;
      font-size:12px;
      color:#fff;
      font-weight:700;
      text-align:left;
      white-space:nowrap;
    }
    td{
      max-width:340px;
      overflow:hidden;
      text-overflow:ellipsis;
    }
    tr:hover td{
      background:#1b2c44;
    }
    .score-high{color:#86efac}
    .score-mid{color:#fcd34d}
    .score-low{color:#fca5a5}
    .score-na{color:#fff}
    .small-note{font-size:12px;color:var(--muted)}
    .static-svg{width:100%;height:220px}
    .footer{
      margin-top:16px;
      text-align:center;
      color:var(--muted);
      font-size:12px;
    }
    .checkbox-wrap{
      display:inline-flex;
      align-items:center;
      gap:8px;
      padding:10px 12px;
      border-radius:12px;
      border:1px solid var(--line2);
      background:var(--surface2);
    }
    .checkbox-wrap input{
      width:16px;
      height:16px;
      accent-color:#60a5fa;
    }
    .gear-btn{
      display:none;
      align-items:center;
      gap:8px;
      border:1px solid var(--line2);
      background:var(--surface2);
      padding:10px 12px;
      border-radius:12px;
      cursor:pointer;
      color:#fff;
      font-weight:800;
    }
    .gear-btn.active{
      border-color:#60a5fa;
      box-shadow:0 0 0 2px rgba(96,165,250,.18) inset;
    }
    .js-enabled .gear-btn{display:inline-flex}
    .js-enabled .no-js-normalize{display:none}
    .js-only{display:none}
    .js-enabled .js-only{display:block}
    .js-enabled .static-only{display:none}
    .pill{
      display:inline-block;
      border:1px solid var(--line2);
      border-radius:999px;
      padding:4px 9px;
      background:#0f172a;
      color:#fff;
      font-size:11px;
      font-weight:800;
    }
    @media (max-width:1100px){
      .charts-grid,.static-grid{grid-template-columns:1fr}
      .canvas-wrap{height:300px}
    }
  </style>
</head>
<body>
<div class="page">
  <div class="banner">
    <div>
      <h1 class="title">LUFA RAG Dashboard</h1>
      <div class="subtitle">Static metrics and detail table are precomputed in Python. JavaScript enhances filtering, recomputation, and chart interactivity.</div>
    </div>
    <div class="banner-actions">
      <label class="checkbox-wrap no-js-normalize">
        <input type="checkbox" id="normalizeStaticCheckbox"/>
        <span>Normalize Data</span>
      </label>
      <button class="gear-btn" id="normalizeGearBtn" type="button" title="Normalize Data">
        <span style="font-size:18px;">⚙</span>
        <span>Normalize Data</span>
      </button>
      <span class="pill" id="generatedAtPill">__GENERATED_AT__</span>
    </div>
  </div>

  <div class="metric-grid" id="kpiGrid">
    __KPI_HTML__
  </div>

  <div class="static-only">
    <div class="static-grid">
      __STATIC_CHARTS__
    </div>
  </div>

  <div class="js-only">
    <div class="charts-grid">
      <div class="card"><div class="section-title">Average Metrics</div><div class="canvas-wrap"><canvas id="avgMetricsChart"></canvas></div></div>
      <div class="card"><div class="section-title">Generation Metrics by Model</div><div class="canvas-wrap"><canvas id="generationByModelChart"></canvas></div></div>
      <div class="card"><div class="section-title">Retrieval Metrics by Model</div><div class="canvas-wrap"><canvas id="retrievalByModelChart"></canvas></div></div>
      <div class="card"><div class="section-title">Judge Metrics Radar</div><div class="canvas-wrap"><canvas id="judgeRadarChart"></canvas></div></div>
      <div class="card"><div class="section-title">Performance by Language</div><div class="canvas-wrap"><canvas id="languageChart"></canvas></div></div>
      <div class="card"><div class="section-title">F1 by Difficulty</div><div class="canvas-wrap"><canvas id="difficultyChart"></canvas></div></div>
      <div class="card"><div class="section-title">ROUGE-L by Category</div><div class="canvas-wrap"><canvas id="categoryChart"></canvas></div></div>
      <div class="card"><div class="section-title">Grounded Rate and Attempts</div><div class="canvas-wrap"><canvas id="supportChart"></canvas></div></div>
      <div class="card"><div class="section-title">Completeness and Normalization</div><div class="canvas-wrap"><canvas id="completenessChart"></canvas></div></div>
    </div>
  </div>

  <div class="card filter-card">
    <div class="section-title">Detailed Result Filters</div>
    <div class="small-note">Every column in the detailed section is filterable. In JavaScript-enabled browsers, all KPI cards and all charts recompute immediately after each filter change and after Normalize Data is toggled.</div>
    <div class="filter-grid" id="filterGrid"></div>
  </div>

  __STATIC_TABLE__

  <div class="footer">Generated at __GENERATED_AT__</div>
</div>

<script>
const DASHBOARD_DATA = __DATA_JSON__;

document.documentElement.classList.add('js-enabled');

const CHARTS = {};
const state = {
  normalize: false,
  filters: {}
};

const KPI_METRICS = [
  ["token_f1_score","Avg F1"],
  ["sentence_bleu_score","Avg BLEU"],
  ["rougeL","Avg ROUGE-L"],
  ["meteor","Avg METEOR"],
  ["mrr","Avg MRR"],
  ["recall_1","Avg Recall@1"],
  ["recall_3","Avg Recall@3"],
  ["recall_5","Avg Recall@5"],
  ["answer_relevance","Avg Relevance"],
  ["faithfulness","Avg Faithful"],
  ["context_precision","Avg Precision"],
  ["grounded_rate","Grounded Rate"],
  ["attempts","Avg Attempts"]
];

const palette = ["#60a5fa","#34d399","#f59e0b","#f87171","#a78bfa","#22d3ee","#f472b6","#fb7185"];
const pale = ["rgba(96,165,250,0.55)","rgba(52,211,153,0.55)","rgba(245,158,11,0.55)","rgba(248,113,113,0.55)","rgba(167,139,250,0.55)","rgba(34,211,238,0.55)","rgba(244,114,182,0.55)","rgba(251,113,133,0.55)"];

function toNumber(v){
  if(v === null || v === undefined || v === "") return NaN;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

function mean(rows, key){
  const vals = rows.map(r => toNumber(r[key])).filter(v => !Number.isNaN(v));
  if(!vals.length) return 0;
  return vals.reduce((a,b)=>a+b,0) / vals.length;
}

function groundedRate(rows){
  if(!rows.length) return 0;
  return rows.filter(r => !!r.grounded).length / rows.length;
}

function hasAnswer(row){
  return String(row.answer || "").trim() !== "";
}

function hasTop1(row){
  return String(row.source1_id || "").trim() !== "" || String(row.source1_text || "").trim() !== "";
}

function normalizePass(row){
  if(!state.normalize) return true;
  return hasAnswer(row) && hasTop1(row);
}

function filterPass(row){
  for(const [col, raw] of Object.entries(state.filters)){
    const term = String(raw || "").trim().toLowerCase();
    if(!term) continue;
    const val = String(row[col] ?? "").toLowerCase();
    if(!val.includes(term)) return false;
  }
  return true;
}

function getFilteredRows(){
  return DASHBOARD_DATA.rows.filter(r => normalizePass(r) && filterPass(r));
}

function fmt(v){
  return Number.isFinite(v) ? v.toFixed(4) : "0.0000";
}

function scoreClass(v){
  const n = toNumber(v);
  if(Number.isNaN(n)) return "";
  if(n >= 0.70) return "score-high";
  if(n >= 0.40) return "score-mid";
  return "score-low";
}

function buildKpis(rows){
  const grid = document.getElementById("kpiGrid");
  const cards = [];
  cards.push(cardHtml(rows.length, "Rows"));
  cards.push(cardHtml(rows.filter(r => hasAnswer(r) && hasTop1(r)).length, "Normalized Rows"));
  for(const [key,label] of KPI_METRICS){
    const val = key === "grounded_rate" ? groundedRate(rows) : mean(rows, key);
    cards.push(cardHtml(fmt(val), label));
  }
  grid.innerHTML = cards.join("");
}

function cardHtml(value, label){
  return `<div class="metric-card"><div class="metric-value">${value}</div><div class="metric-label">${label}</div></div>`;
}

function groupedMean(rows, groupKey, metricKey){
  const map = {};
  rows.forEach(r => {
    const g = String(r[groupKey] || "Unknown").trim() || "Unknown";
    const n = toNumber(r[metricKey]);
    if(!map[g]) map[g] = [];
    if(!Number.isNaN(n)) map[g].push(n);
  });
  const out = {};
  Object.entries(map).forEach(([k, vals]) => {
    out[k] = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : 0;
  });
  return out;
}

function destroyCharts(){
  Object.values(CHARTS).forEach(ch => { try{ ch.destroy(); }catch(e){} });
  Object.keys(CHARTS).forEach(k => delete CHARTS[k]);
}

function chartOptions(max=1){
  return {
    responsive:true,
    maintainAspectRatio:false,
    plugins:{
      legend:{labels:{color:"#ffffff", font:{weight:"700"}}}
    },
    scales:{
      x:{ticks:{color:"#ffffff", font:{weight:"700"}}, grid:{color:"#334155"}},
      y:{beginAtZero:true, suggestedMax:max, ticks:{color:"#ffffff", font:{weight:"700"}}, grid:{color:"#334155"}}
    }
  };
}

function renderCharts(rows){
  destroyCharts();

  CHARTS.avg = new Chart(document.getElementById("avgMetricsChart"), {
    type:"bar",
    data:{
      labels:["F1","BLEU","ROUGE-L","METEOR","MRR","Recall@1","Recall@3","Recall@5","Relevance","Faithful","Precision"],
      datasets:[{
        label:"Averages",
        data:[
          mean(rows,"token_f1_score"),
          mean(rows,"sentence_bleu_score"),
          mean(rows,"rougeL"),
          mean(rows,"meteor"),
          mean(rows,"mrr"),
          mean(rows,"recall_1"),
          mean(rows,"recall_3"),
          mean(rows,"recall_5"),
          mean(rows,"answer_relevance"),
          mean(rows,"faithfulness"),
          mean(rows,"context_precision")
        ],
        backgroundColor:pale,
        borderColor:palette,
        borderWidth:1.5
      }]
    },
    options: chartOptions(1)
  });

  const models = [...new Set(rows.map(r => String(r.rag_base_model || "Unknown").trim() || "Unknown"))];

  CHARTS.genByModel = new Chart(document.getElementById("generationByModelChart"), {
    type:"bar",
    data:{
      labels:["F1","BLEU","ROUGE-L","METEOR"],
      datasets:models.map((m,i)=>({
        label:m,
        data:[
          groupedMean(rows,"rag_base_model","token_f1_score")[m] || 0,
          groupedMean(rows,"rag_base_model","sentence_bleu_score")[m] || 0,
          groupedMean(rows,"rag_base_model","rougeL")[m] || 0,
          groupedMean(rows,"rag_base_model","meteor")[m] || 0,
        ],
        backgroundColor:pale[i % pale.length],
        borderColor:palette[i % palette.length],
        borderWidth:1.5
      }))
    },
    options: chartOptions(1)
  });

  CHARTS.retByModel = new Chart(document.getElementById("retrievalByModelChart"), {
    type:"bar",
    data:{
      labels:["MRR","NDCG","Recall@1","Recall@3","Recall@5"],
      datasets:models.map((m,i)=>({
        label:m,
        data:[
          groupedMean(rows,"rag_base_model","mrr")[m] || 0,
          groupedMean(rows,"rag_base_model","ndcg_at_k")[m] || 0,
          groupedMean(rows,"rag_base_model","recall_1")[m] || 0,
          groupedMean(rows,"rag_base_model","recall_3")[m] || 0,
          groupedMean(rows,"rag_base_model","recall_5")[m] || 0,
        ],
        backgroundColor:pale[(i+1) % pale.length],
        borderColor:palette[(i+1) % palette.length],
        borderWidth:1.5
      }))
    },
    options: chartOptions(1)
  });

  CHARTS.radar = new Chart(document.getElementById("judgeRadarChart"), {
    type:"radar",
    data:{
      labels:["Relevance","Faithful","Precision"],
      datasets:models.map((m,i)=>({
        label:m,
        data:[
          groupedMean(rows,"rag_base_model","answer_relevance")[m] || 0,
          groupedMean(rows,"rag_base_model","faithfulness")[m] || 0,
          groupedMean(rows,"rag_base_model","context_precision")[m] || 0,
        ],
        backgroundColor:pale[i % pale.length].replace("0.55","0.25"),
        borderColor:palette[i % palette.length],
        pointBackgroundColor:palette[i % palette.length],
        borderWidth:2
      }))
    },
    options:{
      responsive:true,
      maintainAspectRatio:false,
      plugins:{legend:{labels:{color:"#ffffff", font:{weight:"700"}}}},
      scales:{r:{min:0,max:1,ticks:{color:"#ffffff", backdropColor:"transparent"}, grid:{color:"#334155"}, pointLabels:{color:"#ffffff", font:{weight:"700"}}}}
    }
  });

  const langs = [...new Set(rows.map(r => String(r.language || "Unknown").trim() || "Unknown"))];
  CHARTS.lang = new Chart(document.getElementById("languageChart"), {
    type:"bar",
    data:{
      labels:langs,
      datasets:[
        {label:"F1", data:langs.map(l => groupedMean(rows,"language","token_f1_score")[l] || 0), backgroundColor:pale[0], borderColor:palette[0], borderWidth:1.5},
        {label:"ROUGE-L", data:langs.map(l => groupedMean(rows,"language","rougeL")[l] || 0), backgroundColor:pale[1], borderColor:palette[1], borderWidth:1.5},
        {label:"METEOR", data:langs.map(l => groupedMean(rows,"language","meteor")[l] || 0), backgroundColor:pale[2], borderColor:palette[2], borderWidth:1.5},
        {label:"Relevance", data:langs.map(l => groupedMean(rows,"language","answer_relevance")[l] || 0), backgroundColor:pale[3], borderColor:palette[3], borderWidth:1.5},
      ]
    },
    options: chartOptions(1)
  });

  const diffs = [...new Set(rows.map(r => String(r.difficulty || "Unknown").trim() || "Unknown"))];
  CHARTS.diff = new Chart(document.getElementById("difficultyChart"), {
    type:"bar",
    data:{
      labels:diffs,
      datasets:[
        {label:"F1", data:diffs.map(d => groupedMean(rows,"difficulty","token_f1_score")[d] || 0), backgroundColor:pale[0], borderColor:palette[0], borderWidth:1.5},
        {label:"MRR", data:diffs.map(d => groupedMean(rows,"difficulty","mrr")[d] || 0), backgroundColor:pale[4], borderColor:palette[4], borderWidth:1.5},
      ]
    },
    options: chartOptions(1)
  });

  const cats = [...new Set(rows.map(r => String(r.category || "Unknown").trim() || "Unknown"))];
  CHARTS.cat = new Chart(document.getElementById("categoryChart"), {
    type:"bar",
    data:{
      labels:cats,
      datasets:[
        {label:"ROUGE-L", data:cats.map(c => groupedMean(rows,"category","rougeL")[c] || 0), backgroundColor:pale[1], borderColor:palette[1], borderWidth:1.5},
        {label:"Recall@5", data:cats.map(c => groupedMean(rows,"category","recall_5")[c] || 0), backgroundColor:pale[2], borderColor:palette[2], borderWidth:1.5},
      ]
    },
    options: chartOptions(1)
  });

  CHARTS.support = new Chart(document.getElementById("supportChart"), {
    type:"bar",
    data:{
      labels:["Grounded Rate","Avg Attempts"],
      datasets:[{
        label:"Support",
        data:[groundedRate(rows), mean(rows,"attempts")],
        backgroundColor:[pale[5], pale[6]],
        borderColor:[palette[5], palette[6]],
        borderWidth:1.5
      }]
    },
    options: chartOptions(Math.max(1, mean(rows,"attempts") + 1))
  });

  const totalRows = DASHBOARD_DATA.rows.length;
  const normalizedRows = rows.filter(r => hasAnswer(r) && hasTop1(r)).length;
  const answerableRows = rows.filter(r => hasAnswer(r)).length;
  CHARTS.complete = new Chart(document.getElementById("completenessChart"), {
    type:"bar",
    data:{
      labels:["Visible Rows","With Answer","Normalized Visible"],
      datasets:[{
        label:"Counts",
        data:[rows.length, answerableRows, normalizedRows],
        backgroundColor:[pale[0], pale[1], pale[2]],
        borderColor:[palette[0], palette[1], palette[2]],
        borderWidth:1.5
      }]
    },
    options: {
      responsive:true,
      maintainAspectRatio:false,
      plugins:{legend:{labels:{color:"#ffffff", font:{weight:"700"}}}},
      scales:{
        x:{ticks:{color:"#ffffff", font:{weight:"700"}}, grid:{color:"#334155"}},
        y:{beginAtZero:true, ticks:{color:"#ffffff", font:{weight:"700"}}, grid:{color:"#334155"}}
      }
    }
  });
}

function renderTable(rows){
  const tbody = document.getElementById("detailsBody");
  tbody.innerHTML = rows.map(r => {
    const tds = DASHBOARD_DATA.detail_columns.map(col => {
      let val = r[col];
      if(col === "grounded"){
        val = val ? "TRUE" : "FALSE";
      }else if(val === null || val === undefined){
        val = "";
      }
      const cls = KPI_METRICS.map(x=>x[0]).includes(col) || ["ndcg_at_k"].includes(col) ? scoreClass(val) : "";
      return `<td class="${cls}">${escapeHtml(String(val))}</td>`;
    }).join("");
    return `<tr>${tds}</tr>`;
  }).join("");
}

function escapeHtml(str){
  return str
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#39;");
}

function buildFilterControls(){
  const wrap = document.getElementById("filterGrid");
  wrap.innerHTML = DASHBOARD_DATA.detail_columns.map(col => {
    const safeId = `flt_${col}`;
    return `
      <div class="filter-item">
        <label for="${safeId}">${escapeHtml(col)}</label>
        <input id="${safeId}" data-col="${escapeHtml(col)}" type="text" placeholder="Filter ${escapeHtml(col)}"/>
      </div>
    `;
  }).join("");

  wrap.querySelectorAll("input[data-col]").forEach(inp => {
    inp.addEventListener("input", (e) => {
      state.filters[e.target.dataset.col] = e.target.value || "";
      rerender();
    });
  });
}

function rerender(){
  const rows = getFilteredRows();
  buildKpis(rows);
  renderCharts(rows);
  renderTable(rows);
}

function initNormalize(){
  const btn = document.getElementById("normalizeGearBtn");
  btn.addEventListener("click", () => {
    state.normalize = !state.normalize;
    btn.classList.toggle("active", state.normalize);
    rerender();
  });
}

buildFilterControls();
initNormalize();
rerender();
</script>
</body>
</html>
"""

    final_html = (
        template
        .replace("__GENERATED_AT__", aggregate["generated_at"])
        .replace("__KPI_HTML__", kpi_html)
        .replace("__STATIC_CHARTS__", static_chart_html)
        .replace("__STATIC_TABLE__", static_table_html)
        .replace("__DATA_JSON__", json.dumps(data_payload, ensure_ascii=False))
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(final_html, encoding="utf-8")


def build_dashboard_from_csvs(
    lufa_csv: Optional[str],
    evaluation_csv: str,
    gt_csv: Optional[str],
    output_html: str = "dashboard/index.html",
) -> None:
    eval_df = pd.read_csv(evaluation_csv, on_bad_lines="skip")
    lufa_df = pd.read_csv(lufa_csv, on_bad_lines="skip") if lufa_csv and Path(lufa_csv).exists() else None
    gt_df = pd.read_csv(gt_csv, on_bad_lines="skip") if gt_csv and Path(gt_csv).exists() else None
    generate_dashboard(eval_df, output_html, lufa_df=lufa_df, gt_df=gt_df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build LUFA interactive dashboard HTML.")
    parser.add_argument("--lufa_csv", default="tests/lufa_out_data.csv")
    parser.add_argument("--evaluation_csv", default="tests/evaluation_results.csv")
    parser.add_argument("--gt_csv", default="tests/combined_test_data_and_ground_truth.csv")
    parser.add_argument("--output_html", default="dashboard/index.html")
    args = parser.parse_args()

    build_dashboard_from_csvs(
        lufa_csv=args.lufa_csv,
        evaluation_csv=args.evaluation_csv,
        gt_csv=args.gt_csv,
        output_html=args.output_html,
    )