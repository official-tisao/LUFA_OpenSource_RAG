    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection(collection_name)
        db_results = collection.get(include=["documents"])
        print(f" -> Successfully loaded {len(db_results.get('ids', []))} total text chunks from database.")
    except Exception as e:
        print(f" 💥 Error connecting to collection '{collection_name}': {e}")
        return

    print("\n================================================================================")
    print("STAGE 2: Executing Source ID Repair Loop")
    print("================================================================================")

    total_repaired = 0
    total_fields = 0
    total_records = len(lufa_df)

    for idx, row in lufa_df.iterrows():
        current_counter = idx + 1
        q_id = str(row.get("question_id", "")).strip()
        print(f"\n[{current_counter}/{total_records}] Analyzing Question ID: {q_id}")

        gt_row = gt_lookup.get(q_id, {})

        for i in range(1, 6):
            text_col = f"source{i}_text"
            id_col = f"source{i}_id"

            source_text = row.get(text_col, "")
            if pd.isna(source_text) or str(source_text).strip() == "":
                continue

            total_fields += 1
            source_clean = str(source_text).strip().lower()
            repaired_id = ""
            method_used = ""

            # Strategy 1: Cross-verify Source 1 text with ground truth registry text directly
            if i == 1 and gt_row:
                gt_text = str(gt_row.get("ground_source_truth", "")).lower()
                gt_id = str(gt_row.get("ground_source_truth_id", "")).strip()
                if gt_text and gt_id:
                    overlap = calculate_token_overlap(source_clean, gt_text)
                    if overlap > 0.85 or source_clean in gt_text:
                        repaired_id = gt_id
                        method_used = f"Ground Truth Registry Alignment (Score: {overlap:.2%})"

            # Strategy 2: Scan full ChromaDB document pool for matching substrings or overlap token sets
            if not repaired_id:
                best_match_id = ""
                max_overlap = -1.0
                exact_found = False

                for cid, doc_text in zip(db_results["ids"], db_results["documents"]):
                    doc_clean = str(doc_text).lower()
                    if source_clean in doc_clean:
