    df["answer_relevance"] = pd.to_numeric(df["answer_relevance"], errors="coerce").fillna(0.0)
    df["faithfulness"] = pd.to_numeric(df["faithfulness"], errors="coerce").fillna(0.0)

    # Track true boolean array values for grounding check matching
    df["grounded_bool"] = df["grounded"].astype(str).str.strip().str.lower() == "true"

    # Identify bad rows based on advanced invalidation metrics criteria
    bad_rows_condition = (
            ((df["attempts"] > 2) & (~df["grounded_bool"])) |
            ((df["grounded_bool"]) & (df["faithfulness"] < 0.40)) |
            ((df["mrr"] > 0.0) & (df["token_f1_score"] == 0.0)) |
            ((df["attempts"] >= 2) & (df["answer_relevance"] < 0.40))
    )

    # Tag rows temporarily to isolate them during sorting
    df["is_corrupted_metric"] = bad_rows_condition

    # Sorting Strategy to bubble the absolute best records to the top:
    # 1. Valid metrics come first (is_corrupted_metric: False before True)
    # 2. Highest text matching score comes next (token_f1_score descending)
    # 3. Highest retrieval positioning comes next (mrr descending)
    # 4. Highest judge accuracy comes next (faithfulness descending)
    df_sorted = df.sort_values(
        by=["is_corrupted_metric", "token_f1_score", "mrr", "faithfulness"],
        ascending=[True, False, False, False]
