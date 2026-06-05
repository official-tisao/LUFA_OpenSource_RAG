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
