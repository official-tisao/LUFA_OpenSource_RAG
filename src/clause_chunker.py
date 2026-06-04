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
