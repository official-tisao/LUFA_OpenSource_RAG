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
    tokens:         int          = 0

    def __post_init__(self) -> None:
        self.tokens = token_count(self.text)


# ===========================================================================
# 4. PDF TEXT EXTRACTION
# ===========================================================================

def extract_pages(pdf_path: Path) -> List[dict]:
    """Extract text per page using pdfplumber with safe default layout resolution tolerances."""
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # FIXED: Restored core defaults to stop character dropping bugs
            text = page.extract_text() or ""
            pages.append({"page_no": i, "text": text})
    logger.info("Extracted %d pages from %s", len(pages), pdf_path.name)
    return pages


def build_marked_text(pages: List[dict]) -> str:
    parts = []
    for p in pages:
        parts.append(f"<<page_{p['page_no']}>>")
        parts.append(p["text"])
    return "\n".join(parts)


def detect_language(text: str) -> str:
    try:
        sample = text[:500].strip()
        return detect(sample) if sample else "en"
    except Exception:
        return "en"


# ===========================================================================
# 5. CLAUSE BOUNDARY CHUNKER
# ===========================================================================

class ClauseBoundaryChunker:

    def __init__(
        self,
        min_tokens: int = 30,
        max_tokens: int = 512,
        language_override: Optional[str] = None,
        min_year: int = CORPUS_MIN_YEAR,
        max_year: int = CORPUS_MAX_YEAR,
    ) -> None:
        self.min_tokens        = min_tokens
        self.max_tokens        = max_tokens
        self.language_override = language_override
        self.min_year          = min_year
        self.max_year          = max_year

    def chunk_pdf(self, pdf_path: Path) -> List[TextNode]:
        end_year = extract_end_year(str(pdf_path))
        weight   = compute_weight(end_year, self.min_year, self.max_year)
        logger.info(
            "Recency | end_year=%s | weight=%.4f | file=%s",
            end_year, weight, pdf_path.name,
        )

        pages     = extract_pages(pdf_path)
        full_text = build_marked_text(pages)
        raw       = self._split_into_clauses(full_text)

        for clause in raw:
            clause.end_year       = end_year
            clause.recency_weight = weight

        merged = self._merge_short_clauses(raw)
        final  = self._split_long_clauses(merged)
        nodes  = self._to_text_nodes(final)

        avg = sum(int(n.metadata["token_count"]) for n in nodes) / max(len(nodes), 1)
        logger.info(
            "Chunking done | chunks=%d | avg_tokens=%.1f | lang=%s",
            len(nodes), avg, self.language_override or "auto",
        )
        return nodes

    # ------------------------------------------------------------------
    # Step A: Split at article/clause boundaries
    # ------------------------------------------------------------------

    def _split_into_clauses(self, full_text: str) -> List[RawClause]:
        lines   = full_text.splitlines()
        clauses: List[RawClause] = []

        current_article  = "0"
        current_clause_id = "0"
        current_title    = "PREAMBLE"
        current_page     = 1
        buffer: List[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            # FIXED: Clear out the entire text tag safely during flush using the regex
            text = PAGE_MARKER_RE.sub("", text).strip()
            if text:
                lang = (
                    self.language_override
                    if self.language_override
                    else detect_language(text)
                )
                clauses.append(RawClause(
                    article_number=current_article,
                    clause_id=current_clause_id,
                    section_title=current_title,
                    text=text,
                    language=lang,
                    page_no=current_page,
                ))
            buffer.clear()

        # ===========================================================================
        # RECOGNITION LOOP CORRECTION (Inside _split_into_clauses method)
        # ===========================================================================

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 1. FIXED: Changed .match() to .search() to find page targets anywhere on the line
            pm = PAGE_MARKER_RE.search(stripped)
            if pm:
                current_page = int(pm.group(1))
                buffer.append(line)
                continue

            # --- Section divider
            if SECTION_DIVIDER_RE.search(stripped):  # FIXED: Changed to .search()
                current_title = stripped
                buffer.append(line)
                continue

            # --- Top-level ARTICLE header
            art_m = ARTICLE_HEADER_RE.search(stripped)  # FIXED: Changed to .search()
            if art_m:
                flush()
                current_article = art_m.group(1)
                title_part = (art_m.group(2) or "").strip()
                if title_part:
                    current_title = title_part
                current_clause_id = current_article
                buffer.append(line)
                continue

            # --- Sub-clause header
            cl_m = CLAUSE_HEADER_RE.search(stripped)  # FIXED: Changed to .search()
            if cl_m:
                candidate = cl_m.group(1)
                if candidate.startswith(current_article + "."):
                    flush()
                    current_clause_id = self._normalise_id(candidate)
                    buffer.append(line)
                    continue

            buffer.append(line)
        flush()
        logger.info("Initial split: %d raw clauses", len(clauses))
        return clauses

    # ------------------------------------------------------------------
    # Step B: Merge clauses below min_tokens
    # ------------------------------------------------------------------

    def _merge_short_clauses(self, clauses: List[RawClause]) -> List[RawClause]:
        if not clauses:
            return clauses

        pending_texts: List[str] = []
        pending_page  = 0
        merged: List[RawClause] = []

        for clause in clauses:
            if clause.tokens < self.min_tokens:
                pending_texts.append(clause.text)
                if not pending_page:
                    pending_page = clause.page_no
            else:
                if pending_texts:
                    combined = "\n".join(pending_texts) + "\n" + clause.text
                    clause = RawClause(
                        article_number=clause.article_number,
                        clause_id=clause.clause_id,
                        section_title=clause.section_title,
                        text=combined.strip(),
                        language=clause.language,
                        page_no=pending_page or clause.page_no,
                        end_year=clause.end_year,
                        recency_weight=clause.recency_weight,
                    )
                    pending_texts.clear()
                    pending_page = 0
                merged.append(clause)

        if pending_texts and merged:
            last     = merged[-1]
            combined = last.text + "\n" + "\n".join(pending_texts)
            merged[-1] = RawClause(
                article_number=last.article_number,
                clause_id=last.clause_id,
                section_title=last.section_title,
                text=combined.strip(),
                language=last.language,
                page_no=last.page_no,
                end_year=last.end_year,
                recency_weight=last.recency_weight,
            )

        logger.info(
            "After merge: %d clauses (removed %d short)",
            len(merged), len(clauses) - len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # Step C: Split clauses above max_tokens
    # ------------------------------------------------------------------

    def _split_long_clauses(self, clauses: List[RawClause]) -> List[RawClause]:
        result: List[RawClause] = []
        sent_boundary = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u00FF])")

        for clause in clauses:
            if clause.tokens <= self.max_tokens:
                result.append(clause)
                continue

            sentences = sent_boundary.split(clause.text)
            part_buf: List[str] = []
            part_tok  = 0
            part_idx  = 1

            for sent in sentences:
                st = token_count(sent)
                if part_tok + st > self.max_tokens and part_buf:
                    suffix = ("__p" + str(part_idx)) if part_idx > 1 else ""
                    result.append(RawClause(
                        article_number=clause.article_number,
                        clause_id=clause.clause_id + suffix,
                        section_title=clause.section_title,
                        text=" ".join(part_buf).strip(),
                        language=clause.language,
                        page_no=clause.page_no,
                        end_year=clause.end_year,
                        recency_weight=clause.recency_weight,
                    ))
                    part_idx += 1
                    part_buf  = [sent]
                    part_tok  = st
                else:
                    part_buf.append(sent)
                    part_tok += st

            if part_buf:
                suffix = ("__p" + str(part_idx)) if part_idx > 1 else ""
                result.append(RawClause(
                    article_number=clause.article_number,
                    clause_id=clause.clause_id + suffix,
                    section_title=clause.section_title,
                    text=" ".join(part_buf).strip(),
                    language=clause.language,
                    page_no=clause.page_no,
                    end_year=clause.end_year,
                    recency_weight=clause.recency_weight,
                ))

        logger.info("After long-split: %d final clauses", len(result))
        return result

    # ------------------------------------------------------------------
    # Step D: Convert to LlamaIndex TextNode list
    # ------------------------------------------------------------------

    def _to_text_nodes(self, clauses: List[RawClause]) -> List[TextNode]:
        nodes: List[TextNode] = []
        for idx, clause in enumerate(clauses):
            node = TextNode(
                id_=str(uuid.uuid4()),
                text=clause.text,
                metadata={
                    "article_number":  clause.article_number,
                    "clause_id":       clause.clause_id,
                    "section_title":   clause.section_title,
                    "language":        clause.language,
                    "page_no":         str(clause.page_no),
                    "token_count":     str(clause.tokens),
                    "chunk_index":     str(idx),
                    "doc_source":      "lufa_collective_agreement",
                    "end_year":        str(clause.end_year) if clause.end_year else "",
                    "recency_weight":  str(clause.recency_weight),
                },
            )
            node.excluded_embed_metadata_keys = [
                "token_count", "chunk_index", "doc_source",
                "page_no", "end_year", "recency_weight",
            ]
            node.excluded_llm_metadata_keys = [
                "token_count", "chunk_index",
            ]
            nodes.append(node)
        return nodes

    @staticmethod
    def _normalise_id(raw: str) -> str:
        normalised = re.sub(r"\(([a-z]+)\)", r".\1", raw)
        return normalised.strip(" .")


# ===========================================================================
# 6. CONVENIENCE WRAPPER
# ===========================================================================

def ingest_pdf(
    pdf_path,
    language_override: Optional[str] = None,
    min_tokens: int = 30,
    max_tokens: int = 512,
    min_year: int = CORPUS_MIN_YEAR,
    max_year: int = CORPUS_MAX_YEAR,
) -> List[TextNode]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError("PDF not found: " + str(path.resolve()))

    chunker = ClauseBoundaryChunker(
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        language_override=language_override,
        min_year=min_year,
        max_year=max_year,
    )
    return chunker.chunk_pdf(path)


# ===========================================================================
# 7. CSV / JSONL OUTPUT HELPER
# ===========================================================================

def save_outputs(nodes: List[TextNode], out_prefix: str) -> tuple:
    import pandas as pd

    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "chunk_id":       n.node_id,
            "text":           n.text,
            **n.metadata,
        }
        for n in nodes
    ]

    csv_path   = f"{out_prefix}.csv"
    jsonl_path = f"{out_prefix}.jsonl"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Saved %d chunks -> %s  |  %s", len(nodes), csv_path, jsonl_path)
    return csv_path, jsonl_path


# ===========================================================================
# 8. FULL PIPELINE RUNNER
# ===========================================================================

def run_ingestion_pipeline(
    en_pdf_path,
    fr_pdf_path,
    chroma_persist_dir: str = "./chroma_db",
    collection_en: str = "lufa_en",
    collection_fr: str = "lufa_fr",
    min_year: int = CORPUS_MIN_YEAR,
    max_year: int = CORPUS_MAX_YEAR,
) -> dict:
    try:
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        raise ImportError("Missing optional dependencies for vector storage.") from exc

    logger.info("Loading BGE-M3 embedding model (local)...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device="cuda", max_length=512)
    chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
    results: dict = {}

    for lang, pdf, col_name in [("en", en_pdf_path, collection_en), ("fr", fr_pdf_path, collection_fr)]:
        logger.info("Indexing %s -> collection '%s'", lang.upper(), col_name)
        nodes = ingest_pdf(pdf, language_override=lang, min_year=min_year, max_year=max_year)

        chroma_col   = chroma_client.get_or_create_collection(col_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_col)
        storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)

        VectorStoreIndex(nodes, storage_context=storage_ctx, embed_model=embed_model, show_progress=True)

        avg    = sum(int(n.metadata["token_count"]) for n in nodes) / max(len(nodes), 1)
        weight = float(nodes[0].metadata["recency_weight"]) if nodes else 1.0
        results[lang + "_chunks"]          = len(nodes)
        results["avg_tokens_" + lang]      = round(avg, 1)
        results["recency_weight_" + lang]  = weight

    return results


# ===========================================================================
# 9. DIAGNOSTIC UTILITY
# ===========================================================================

def print_chunk_report(nodes: List[TextNode], max_rows: int = 25) -> None:
    header = "{:<6} {:<10} {:<20} {:<7} {:<7} {:<8} {:<14} {}".format("#", "article", "clause_id", "lang", "page", "tokens", "recency_wt", "preview")
    print(header)
    print("-" * 110)
    for i, node in enumerate(nodes[:max_rows]):
        m       = node.metadata
        preview = node.text.replace("\n", " ")[:60].strip()
        print("{:<6} {:<10} {:<20} {:<7} {:<7} {:<8} {:<14} {}...".format(i, m["article_number"], m["clause_id"], m["language"], m["page_no"], m["token_count"], m.get("recency_weight", ""), preview))
    print(f"\nTotal chunks: {len(nodes)}")
    avg = sum(int(n.metadata["token_count"]) for n in nodes) / max(len(nodes), 1)
    print(f"Average tokens/chunk: {avg:.1f}")


# ===========================================================================
# 10. CLI ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Clause-boundary chunker for LUFA agreements.")
    parser.add_argument("pdf", help="Path to the PDF file to chunk")
    parser.add_argument("--lang", choices=["en", "fr"], default=None)
    parser.add_argument("--min-tokens", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--min-year", type=int, default=CORPUS_MIN_YEAR)
    parser.add_argument("--max-year", type=int, default=CORPUS_MAX_YEAR)
    parser.add_argument("--report", action="store_true")

    args = parser.parse_args()

    output_nodes = ingest_pdf(
        args.pdf,
        language_override=args.lang,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        min_year=args.min_year,
        max_year=args.max_year,
    )

    if args.report:
        print_chunk_report(output_nodes)

    if args.out_prefix:
        csv_p, jsonl_p = save_outputs(output_nodes, args.out_prefix)
    else:
        print(f"Chunks produced: {len(output_nodes)}")

    sys.exit(0)