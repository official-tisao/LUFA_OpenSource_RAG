"""
clause_chunker.py
=================
Subsystem 1 - Document Preprocessing Pipeline
Clause-Boundary Chunking for the LUFA Collective Agreement (EN + FR)

Thesis: Designing a Locally Deployed Cross-Lingual Agentic RAG System
        for Bilingual Institutional Legal Documents
Author: Saheed Oluwatosin Tiamiyu | Laurentian University
Supervisor: Prof. Kalpdrum Passi
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pdfplumber
from langdetect import detect, DetectorFactory
from llama_index.core.schema import TextNode

DetectorFactory.seed = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ===========================================================================
# 0. CORPUS-WIDE RECENCY CONSTANTS
# ===========================================================================

CORPUS_MIN_YEAR: int = 1998
CORPUS_MAX_YEAR: int = 2026

YEAR_RE = re.compile(r'\b(?:19|20)\d{2}\b')

# ===========================================================================
# 1. REGEX PATTERNS
# ===========================================================================

ARTICLE_HEADER_RE = re.compile(
    r"^(?:ARTICLE|Article)[\s\xa0]+(\d{1,3})[\s\xa0.\-\u2013\u2014:]*"
    r"([A-Z\u00C0-\u00FF][^\n]*)?$",
    re.MULTILINE
)

CLAUSE_HEADER_RE = re.compile(
    r"^(\d{1,3}(?:\.\d+)+(?:\([a-zA-Z0-9]+\))?)[\s\xa0.\-\u2013\u2014:]+(.+)?$",
    re.MULTILINE
)

SECTION_DIVIDER_RE = re.compile(
    r"^(?:PART|SECTION|SCHEDULE|APPENDIX|ANNEXE|PARTIE)[\s\xa0]+[IVX\d]+",
    re.IGNORECASE | re.MULTILINE
)

# FIXED: Generalized pattern handles variations like Page - 1, PAGE 2, page-10, or <<page_1>>
PAGE_MARKER_RE = re.compile(
    r"page(?:[\s\xa0]*[-_]+[\s\xa0]*|[\s\xa0]+)?(\d+)",
    re.IGNORECASE
)

def token_count(text: str) -> int:
    """Whitespace-based token count (fast, consistent)."""
    if not text:
        return 0
    return len(text.split())


def clean_text(s: str) -> str:
    """Normalise whitespace and remove non-breaking/zero-width spaces."""
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\xa0", " ").replace("\u200b", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ===========================================================================
# 2. RECENCY WEIGHTING HELPERS
# ===========================================================================

def extract_end_year(filename: str) -> Optional[int]:
    """Return the highest 4-digit year found in the filename stem, or None."""
    years = [int(y) for y in YEAR_RE.findall(Path(filename).stem)]
    return max(years) if years else None


def compute_weight(
    end_year: Optional[int],
    min_year: int = CORPUS_MIN_YEAR,
    max_year: int = CORPUS_MAX_YEAR,
) -> float:
    if end_year is None or max_year == min_year:
        return 1.0
    raw = max(0.0, min(1.0, (end_year - min_year) / (max_year - min_year)))
    return round(0.30 + 0.70 * raw, 4)


# ===========================================================================
# 3. DATA STRUCTURES
# ===========================================================================

@dataclass
class RawClause:
    """Intermediate container before finalising as a LlamaIndex TextNode."""
    article_number: str
    clause_id:      str
    section_title:  str
    text:           str
    language:       str
    page_no:        int
    end_year:       Optional[int] = None
    recency_weight: float        = 1.0
