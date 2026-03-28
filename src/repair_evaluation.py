
    chroma_cached_data = None
    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection("multilingual_docs")
        chroma_cached_data = collection.get(include=["documents"])
    except Exception as dberr:
        print(f"[Warning] Chroma connection failed: {dberr}")

    counter = 0
    for qid, reason in invalidated_qids.items():
        counter += 1
        print(f"\n[{counter}/{len(invalidated_qids)}] Healing Question ID: {qid}")
        print(f"   -> Reason for Repair: {reason}")

        gt_matches = gt_df[gt_df["id"].astype(str).str.strip() == qid]
        if gt_matches.empty:
            print(f"   ❌ Abort: Could not locate metadata for ID {qid} in master data file.")
            continue

        gt_row = gt_matches.iloc[0]

        print("   -> Dispatched to run_simulation framework for inference pass...")
        sim_output = query_single_record(gt_row.to_dict(), sim_mode, cfg_base_model, llm_model, api_url, counter)

        prediction = str(sim_output.get("answer", ""))
        reference = str(gt_row.get("expected_answer", ""))
        retrieved_ids = build_retrieved_ids(sim_output)

        gt_col = "ground_source_truth_id" if "ground_source_truth_id" in gt_df.columns else "ground_truth_source_ids"
        ground_truth_ids = parse_source_ids(gt_row.get(gt_col, ""))

        context = build_context_from_row(sim_output)
        question = str(gt_row.get("question", ""))
        language_val = str(gt_row.get("language", "en"))

        print("   -> Re-calculating text generation metrics...")
        f1_val = round(token_f1(prediction, reference), 4)
        bleu_val = round(compute_bleu(prediction, reference), 4)
        rouge_scores = compute_rouge(prediction, reference)
        meteor_val = round(compute_meteor(prediction, reference), 4)
        print(f"      * Recalculated F1: {f1_val} | BLEU: {bleu_val} | ROUGE-L: {rouge_scores['rougeL']}")

        print("   -> Re-calculating vector position ranks...")
        mrr_val = round(mrr(retrieved_ids, ground_truth_ids), 4)
        ndcg_val = ndcg_at_k(retrieved_ids, ground_truth_ids, k=5)
        rec1 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=1), 4)
        rec3 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=3), 4)
        rec5 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=5), 4)

        if mrr_val == 0.0 and ndcg_val == 0.0:
            print("      ⚠️  Warning: Rescored retrieval returned 0.0. Attempting embedded text match recovery...")
            try:
                from evaluate import repair_single_row_sources
                fixed_ids = repair_single_row_sources(sim_output, chroma_cached_data, db_path, "multilingual_docs")
                if fixed_ids:
                    retrieved_ids = fixed_ids
                    mrr_val = round(mrr(retrieved_ids, ground_truth_ids), 4)
                    ndcg_val = ndcg_at_k(retrieved_ids, ground_truth_ids, k=5)
                    rec1 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=1), 4)
                    rec3 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=3), 4)
                    rec5 = round(recall_at_k(retrieved_ids, ground_truth_ids, k=5), 4)
                    print(f"         * Healed Ranks Successfully -> MRR: {mrr_val} | NDCG@5: {ndcg_val}")
                    for i, cid in enumerate(retrieved_ids, start=1):
                        sim_output[f"source{i}_id"] = cid
            except Exception as repair_err:
                print(f"         [Live Repair Error] Single row recovery pass failed: {repair_err}")

        judge_relevance = 0.0
        judge_faithfulness = 0.0
        judge_precision = 0.0
        if prediction and prediction != "ERROR":
            print(f"   -> Dispatching evaluation prompts to local Judge Model ({llm_model})...")
            try:
                judge = llm_judge_scores(question, prediction, context, llm_model)
                judge_relevance = judge.get("answer_relevance", 0.0)
                judge_faithfulness = judge.get("faithfulness", 0.0)
                judge_precision = judge.get("context_precision", 0.0)
                print(
                    f"      * Recalculated Judge Scores -> Relevance: {judge_relevance} | Faithfulness: {judge_faithfulness}")
