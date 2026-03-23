    lang_code = "fr" if "fr" in row_lang or "fran" in row_lang else "en"
    art_no = extract_article_number(question_text) or extract_article_number(expected_text)

    # Pathway 1: Complex Metadata SQL WHERE Route
    if art_no:
        where_clause = {
            "$and": [
                {"article_number": str(art_no)},
                {"language": lang_code}
            ]
        }
        print(f"   [Dual Strategy] Executing SQL WHERE route -> Article: {art_no} | Lang: {lang_code}")
        db_results = collection.get(
            where=where_clause,
            include=["documents"]
        )
    # Pathway 2: Fallback Embedding Index Route
    else:
        where_clause = {"language": lang_code}
        print(f"   [Dual Strategy] No Article found. Executing fallback Neural Vector Index Query route -> Lang: {lang_code}")
        query_embedding = embed_model.get_text_embedding(question_text)
        query_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=20,
            where=where_clause,
            include=["documents"]
        )
        db_results = {
            "ids": query_results["ids"][0] if query_results["ids"] else [],
            "documents": query_results["documents"][0] if query_results["documents"] else []
        }

    ids = db_results.get("ids", [])
    docs = db_results.get("documents", [])

    if not ids or not docs:
        print("   ⚠️  Warning: No database records matched query filters.")
        return "", "", 0.0, 0.0

    best_id = ids[0]
    best_doc = docs[0]
    max_overlap = -1.0

    for cid, doc_text in zip(ids, docs):
        overlap = calculate_token_overlap(doc_text, expected_text)
        if overlap > max_overlap:
            max_overlap = overlap
            best_id = cid
            best_doc = doc_text

    # Cross-verify using answer column token footprint ratio
    answer_overlap_score = calculate_answer_based_overlap(expected_text, best_doc)
    return best_id, best_doc, max_overlap, answer_overlap_score


def run_pipeline(csv_path, db_path, collection_name, output_path):
    print(f"[Initialization] Loading base source text dataset from: {csv_path}")
    if not Path(csv_path).exists():
        print(f"Error: Source file {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    total_records = len(df)
    print(f"[Initialization] Total entries detected to process: {total_records}")

    df["ground_source_truth_id"] = ""
    df["ground_source_truth"] = ""

    print(f"[Initialization] Establishing persistent connection to ChromaDB at: {db_path}")
