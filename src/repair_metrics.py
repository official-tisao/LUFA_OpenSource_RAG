    gt_df = pd.read_csv(gt_csv) if Path(gt_csv).exists() else None

    if lufa_df is None or eval_df is None or gt_df is None:
        print("Missing required CSV files. Repair aborted.")
        return

    gt_lookup = {}
    for _, row in gt_df.iterrows():
        qid = str(row.get("id", "")).strip()
        gt_col = "ground_source_truth_id" if "ground_source_truth_id" in row else "ground_truth_source_ids"
        gt_lookup[qid] = [s.strip() for s in str(row.get(gt_col, "")).split("|") if s.strip()]

    import chromadb
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("multilingual_docs")
    chroma_data = collection.get(include=["documents"])
    db_ids = chroma_data.get("ids", [])
    db_docs = chroma_data.get("documents", [])

    updates = 0
    for idx, eval_row in eval_df.iterrows():
        qid = str(eval_row.get("question_id", "")).strip()

        repaired_ids = []
        for i in range(1, 6):
            text_val = eval_row.get(f"source{i}_text", "")
            if pd.isna(text_val) or str(text_val).strip() == "":
                continue

            source_clean = str(text_val).strip().lower()
            matched_id = eval_row.get(f"source{i}_id", "")

            if "_chunk" in str(matched_id):
                max_overlap = -1.0
                for cid, doc_text in zip(db_ids, db_docs):
                    if source_clean in str(doc_text).lower():
                        matched_id = cid
                        break
                    overlap = calculate_token_overlap(source_clean, str(doc_text).lower())
                    if overlap > max_overlap:
                        max_overlap = overlap
                        matched_id = cid

            if matched_id:
                repaired_ids.append(matched_id)
                eval_df.at[idx, f"source{i}_id"] = matched_id
