            except Exception as j_err:
                print(f"      [Judge Error] Connection dropped: {j_err}")

        # Sync back to frames
        lufa_row_dict = {}
        for col in LUFA_COLUMNS:
            lufa_row_dict[col] = sim_output.get(col, "")
        lufa_row_dict["question_id"] = qid
        lufa_df = pd.concat([lufa_df, pd.DataFrame([lufa_row_dict], columns=LUFA_COLUMNS)], ignore_index=True)

        eval_row_dict = {}
        for col in LUFA_COLUMNS:
            eval_row_dict[col] = sim_output.get(col, "")

        eval_row_dict["id"] = qid
        eval_row_dict["question_id"] = qid
        eval_row_dict["question"] = question
        eval_row_dict["language"] = language_val
        eval_row_dict["rag_base_model"] = str(sim_output.get("base_model_used", llm_model))
        eval_row_dict["judge_llm"] = llm_model
        eval_row_dict["category"] = str(gt_row.get("category", ""))
        eval_row_dict["difficulty"] = str(gt_row.get("difficulty", ""))

        eval_row_dict["token_f1_score"] = f1_val
        eval_row_dict["sentence_bleu_score"] = bleu_val
        eval_row_dict["rouge1"] = rouge_scores["rouge1"]
        eval_row_dict["rouge2"] = rouge_scores["rouge2"]
        eval_row_dict["rougeL"] = rouge_scores["rougeL"]
        eval_row_dict["meteor"] = meteor_val

        eval_row_dict["mrr"] = mrr_val
        eval_row_dict["ndcg_at_k"] = ndcg_val
        eval_row_dict["recall_1"] = rec1
        eval_row_dict["recall_3"] = rec3
        eval_row_dict["recall_5"] = rec5

        eval_row_dict["answer_relevance"] = judge_relevance
        eval_row_dict["faithfulness"] = judge_faithfulness
        eval_row_dict["context_precision"] = judge_precision

        primary_score = safe_float(sim_output.get("source1_score", 0.0))
        eval_row_dict["original_cosine_score"] = safe_float(sim_output.get("original_cosine_score", primary_score))
        eval_row_dict["recency_adjusted_score"] = safe_float(sim_output.get("recency_adjusted_score", primary_score))
        eval_row_dict["RRF"] = safe_float(sim_output.get("RRF", primary_score))

        eval_df = pd.concat([eval_df, pd.DataFrame([eval_row_dict], columns=EVAL_COLUMNS)], ignore_index=True)
        print("   ✅ Row repaired successfully and updated inside data matrices.")

    print("\n" + "=" * 80)
    print("STAGE 3: Synchronizing Ledger Checkpoints & Compiling Dashboard UI")
    print("=" * 80)

    lufa_df = lufa_df.drop_duplicates(subset=["question_id"], keep="last")
    eval_df = eval_df.drop_duplicates(subset=["question_id"], keep="last")

    lufa_df.to_csv(lufa_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    print(f" -> Synchronized {lufa_path} records.")
    print(f" -> Synchronized {eval_path} scorecards.")

    try:
        generate_dashboard(eval_df, dash_path)
        print(f" -> Real-time HTML dashboard refreshed at: {dash_path}")
    except Exception as uierr:
        print(f" [Dashboard Warning] Live UI build skipped: {uierr}")

    print("================================================================================")
    print(" REPAIR RUN METRIC SEQUENCE COMPLETE")
    print("================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System healing script for RAG evaluations.")
    parser.add_argument("--lufa_csv", default="tests/lufa_out_data.csv")
    parser.add_argument("--eval_csv", default="tests/evaluation_results.csv")
    parser.add_argument("--test_csv", default="tests/combined_test_data_and_ground_truth.csv")
    parser.add_argument("--db", default="db/chroma_db")
    parser.add_argument("--dashboard", default="dashboard/index.html")
    parser.add_argument("--llm_model", default="llama3.2:3b-instruct-q4_K_M")
    parser.add_argument("--sim_mode", choices=["local", "api", "frontier"], default="local")
    parser.add_argument("--api_url", default="http://localhost:8000")
    args = parser.parse_args()

    process_healing_cycle(
        lufa_path=args.lufa_csv,
        eval_path=args.eval_csv,
        gt_path=args.test_csv,
        db_path=args.db,
        dash_path=args.dashboard,
        llm_model=args.llm_model,
        sim_mode=args.sim_mode,
        api_url=args.api_url
    )