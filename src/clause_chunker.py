
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