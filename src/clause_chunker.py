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
