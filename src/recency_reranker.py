    """
    return max(0.0, 1.0 - distance)


def rerank(
    chroma_results: Dict[str, Any],
    top_k: int = 5,
    tie_threshold: float = TIE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Re-rank ChromaDB query results using similarity as primary sort and
    recency_weight as a tie-breaker.

    Parameters
    ----------
    chroma_results : dict
        Raw output from chroma_collection.query(include=[...]).
    top_k : int
        Number of chunks to return after reranking.
    tie_threshold : float
        Score difference below which two chunks are considered tied and
        recency_weight decides the winner.

    Returns
    -------
    List of dicts with keys: chunk_id, text, metadata, score, recency_weight
    """
    docs      = chroma_results.get("documents", [[]])[0]
    metas     = chroma_results.get("metadatas", [[]])[0]
    distances = chroma_results.get("distances", [[]])[0]
    ids       = chroma_results.get("ids", [[]])[0]

    if not docs:
        return []

    candidates = []
    for doc, meta, dist, cid in zip(docs, metas, distances, ids):
        sim    = _chroma_distance_to_similarity(dist)
        weight = float(meta.get("recency_weight", 1.0))
        candidates.append({
            "chunk_id":       cid,
            "text":           doc,
            "metadata":       meta,
            "score":          round(sim, 6),
            "recency_weight": weight,
        })

    # ── Two-key sort ─────────────────────────────────────────────────────────
