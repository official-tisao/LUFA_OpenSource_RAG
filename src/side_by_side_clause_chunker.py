    lines = cluster_lines(words)
    para_texts = merge_lines(lines)
    chunks: List[Chunk] = []
    running = []
    prev_clause_no = None
    prev_title = None
    idx = 1
    for t in para_texts:
        clause_no, title = parse_clause_header(t)
        if clause_no and running:
            text = clean_text("\n".join(running))
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_p{page_num:03d}_{language}_{idx:03d}",
                doc_id=doc_id,
                page=page_num,
                language=language,
                clause_no=prev_clause_no,
                title=prev_title,
                text=text,
            ))
            idx += 1
            running = []
        prev_clause_no, prev_title = clause_no, title or (t if len(t) < 90 else None)
        running.append(t)
    if running:
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_p{page_num:03d}_{language}_{idx:03d}",
            doc_id=doc_id,
            page=page_num,
            language=language,
            clause_no=prev_clause_no,
            title=prev_title,
            text=clean_text("\n".join(running)),
        ))
    return chunks


def chunk_side_by_side_pdf(pdf_path: str, doc_id: Optional[str] = None) -> List[TextNode]:
    pdf_path = str(pdf_path)
    doc_id = doc_id or Path(pdf_path).stem
    out_chunks: List[Chunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = clean_text(page.extract_text() or "")
            if not page_text:
                continue
            left_words, right_words = extract_words_split(page)
            left_preview = clean_text(" ".join(w['text'] for w in left_words[:40])) if left_words else ''
            right_preview = clean_text(" ".join(w['text'] for w in right_words[:40])) if right_words else ''
            left_lang = infer_lang_for_column(left_preview or page_text[:500])
            right_lang = 'fr' if left_lang == 'en' else 'en'
            left_chunks = extract_column_chunks(doc_id, i, left_lang, left_words)
            right_chunks = extract_column_chunks(doc_id, i, right_lang, right_words)
            # pair by index
            pair_count = min(len(left_chunks), len(right_chunks))
            for k in range(pair_count):
                left_chunks[k].partner_chunk_id = right_chunks[k].chunk_id
                right_chunks[k].partner_chunk_id = left_chunks[k].chunk_id
            out_chunks.extend(left_chunks)
            out_chunks.extend(right_chunks)

    # convert to TextNode with metadata
    nodes: List[TextNode] = []
    for idx, c in enumerate(out_chunks):
        node = TextNode(
            id_=str(uuid.uuid4()),
            text=c.text,
            metadata={
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "page_no": str(c.page),
                "language": c.language,
                "clause_no": c.clause_no or "",
                "title": c.title or "",
                "partner_chunk_id": c.partner_chunk_id or "",
                "doc_source": doc_id,
            },
        )
        # hide low-level fields from embedding by listing keys (kept for llama-index compatibility)
