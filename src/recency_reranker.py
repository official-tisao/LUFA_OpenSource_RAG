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
