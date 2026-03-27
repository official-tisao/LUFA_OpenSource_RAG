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
