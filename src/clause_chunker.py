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
