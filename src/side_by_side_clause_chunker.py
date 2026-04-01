# python
"""
side_by_side_clause_chunker.py
Extract bilingual side-by-side EN/FR PDFs into clause-level TextNodes and
optionally inject into LlamaIndex/Chroma (local).
Deps: pip install pdfplumber langdetect llama-index-core llama-index-vector-stores-chroma
      llama-index-embeddings-huggingface chromadb pandas
"""
from __future__ import annotations

import re
import json
import uuid
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
from langdetect import detect, DetectorFactory
from llama_index.core.schema import TextNode

DetectorFactory.seed = 42
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CLAUSE_RE = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')
NUM_RE = re.compile(r'^[\d\W]+$')


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    language: str
    clause_no: Optional[str]
    title: Optional[str]
    text: str
    partner_chunk_id: Optional[str] = None
    x0: Optional[float] = None
    x1: Optional[float] = None
    y0: Optional[float] = None
    y1: Optional[float] = None

def clean_text(s: str) -> str:
    s = s.replace('\u00a0', ' ').replace('\u200b', ' ')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def extract_words_split(page) -> Tuple[List[dict], List[dict]]:
    words = page.extract_words(extra_attrs=["x0", "x1", "top", "bottom"], keep_blank_chars=False, use_text_flow=True) or []
    if not words:
        return [], []
    mid = float(page.width) / 2.0
    left = [w for w in words if (w["x0"] + w["x1"]) / 2.0 < mid]
    right = [w for w in words if (w["x0"] + w["x1"]) / 2.0 >= mid]
    return left, right


def cluster_lines(words, y_tol=3.5):
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines = []
    cur = [words[0]]
    cur_y = words[0]["top"]
    for w in words[1:]:
        if abs(w["top"] - cur_y) <= y_tol:
            cur.append(w)
            cur_y = (cur_y * (len(cur)-1) + w["top"]) / len(cur)
        else:
            lines.append(cur)
            cur = [w]
            cur_y = w["top"]
    lines.append(cur)
    return lines


def line_text(line):
    return clean_text(" ".join(w["text"] for w in sorted(line, key=lambda x: x["x0"])))


def merge_lines(lines):
    merged = []
    buf = []
    for line in lines:
        t = line_text(line)
        if not t:
            continue
        if buf and (t[0].islower() or t.startswith('—') or t.startswith('-')):
            buf.append(t)
        else:
            if buf:
                merged.append(' '.join(buf))
            buf = [t]
    if buf:
        merged.append(' '.join(buf))
    return [clean_text(x) for x in merged if clean_text(x)]


def parse_clause_header(text: str):
    m = CLAUSE_RE.match(text)
    if m:
        return m.group(1), m.group(2).strip()
    return None, None


def infer_lang_for_column(text: str) -> str:
    sample = (text or "")[:500].strip()
    try:
        return detect(sample) if sample else "en"
    except Exception:
        return "en"


def extract_column_chunks(doc_id: str, page_num: int, language: str, words) -> List[Chunk]:
