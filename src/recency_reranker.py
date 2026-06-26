"""
recency_reranker.py — tie-breaking reranker using recency_weight.

Drop this next to your rag_engine.py and call rerank() on ChromaDB results
before returning them to the user.

Rules
-----
- Primary sort  : cosine similarity score  (descending, higher = better)
- Tie-breaker   : recency_weight           (descending, newer = better)
  applied ONLY when two chunks are within TIE_THRESHOLD of each other

Default TIE_THRESHOLD = 0.02  (i.e. scores within 2 percentage points are
considered a tie and recency decides the winner).

Why 0.02?
  Typical nomic-embed cosine scores for relevant chunks cluster in a narrow
  band (e.g. 0.72–0.81).  A 0.02 window lets genuinely different-quality
  matches keep their natural order while still boosting a newer agreement
  clause over an identically-worded older one.

Usage
-----
  from recency_reranker import rerank

  results = chroma_collection.query(
      query_embeddings=[embedding],
      n_results=20,                  # fetch more than you need
      include=["documents", "metadatas", "distances"],
  )
  top_k = rerank(results, top_k=5)
  # top_k is a list of dicts: {text, metadata, score, recency_weight}
"""

from __future__ import annotations
from typing import List, Dict, Any

TIE_THRESHOLD: float = 0.02


def _chroma_distance_to_similarity(distance: float) -> float:
    """
    ChromaDB returns L2 distance by default when using cosine space;
    convert to a 0-1 similarity.  If you use cosine distance directly
    (distance_function="cosine") it is already 1 - cosine_similarity,
    so we do 1 - distance.  Adjust if your collection uses inner_product.
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
    # We want: higher score first, then higher recency_weight first.
    # But only apply weight as a tiebreaker, not a multiplier.
    #
    # Implementation: bucket scores into groups where every member is within
    # `tie_threshold` of the group's best score, then sort within each bucket
    # by recency_weight.
    candidates.sort(key=lambda x: x["score"], reverse=True)

    reranked: list = []
    i = 0
    while i < len(candidates):
        bucket_score = candidates[i]["score"]
        bucket = []
        j = i
        while j < len(candidates) and (bucket_score - candidates[j]["score"]) < tie_threshold:
            bucket.append(candidates[j])
            j += 1
        # Within the tie bucket: newer agreement wins
        bucket.sort(key=lambda x: x["recency_weight"], reverse=True)
        reranked.extend(bucket)
        i = j

    return reranked[:top_k]


# ── Convenience: pretty-print for debugging ───────────────────────────────────

def explain_ranking(results: List[Dict[str, Any]]) -> None:
    """Print a ranked summary to stdout for inspection."""
    print(f"{'Rank':<5} {'Score':>7} {'Weight':>7} {'EndYear':>8}  chunk_id")
    print("-" * 70)
    for rank, r in enumerate(results, start=1):
        end_year = r["metadata"].get("end_year", "?")
        print(
            f"{rank:<5} {r['score']:>7.4f} {r['recency_weight']:>7.4f} "
            f"{str(end_year):>8}  {r['chunk_id']}"
        )
