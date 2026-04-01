        node.excluded_embed_metadata_keys = ["doc_id", "doc_source", "partner_chunk_id"]
        node.excluded_llm_metadata_keys = ["doc_id"]
        nodes.append(node)
    logger.info("Produced %d bilingual chunks from %s", len(nodes), pdf_path)
    return nodes


def index_nodes_to_chroma(nodes: List[TextNode], collection_name: str, chroma_persist_dir: str = "./chroma_db", model_device: str = "cuda"):
    try:
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        raise ImportError("Missing optional deps for indexing. Install: chromadb llama-index-vector-stores-chroma llama-index-embeddings-huggingface") from exc

    logger.info("Loading embedding model: BGE-M3 (local) on %s", model_device)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device=model_device, max_length=512)

    chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
    chroma_col = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_col)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes,
        storage_context=storage_ctx,
        embed_model=embed_model,
        show_progress=True,
    )
    logger.info("Indexed %d nodes into collection '%s' (path=%s)", len(nodes), collection_name, chroma_persist_dir)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    p = argparse.ArgumentParser(description="Side-by-side bilingual clause chunker -> LlamaIndex/Chroma")
    p.add_argument("pdf", help="Path to side-by-side bilingual PDF")
    p.add_argument("--doc-id", default=None, help="Document id override")
    p.add_argument("--out-prefix", default=None, help="If set, write <prefix>.csv and .jsonl")
    p.add_argument("--index", action="store_true", help="Also index into Chroma collections")
    p.add_argument("--collection", default="lufa_bilingual", help="Chroma collection name")
    p.add_argument("--chroma-dir", default="./chroma_db", help="Chroma persistent directory")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Embedding device")
    args = p.parse_args()

    nodes = chunk_side_by_side_pdf(args.pdf, args.doc_id)

    if args.out_prefix:
        rows = [{"chunk_id": n.metadata.get("chunk_id"), "text": n.text, **n.metadata} for n in nodes]
        Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(f"{args.out_prefix}.csv", index=False)
        with open(f"{args.out_prefix}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote outputs: %s.csv / %s.jsonl", args.out_prefix, args.out_prefix)

    if args.index:
        index_nodes_to_chroma(nodes, args.collection, chroma_persist_dir=args.chroma_dir, model_device=args.device)

    print(f"Done. Produced {len(nodes)} chunks.")
