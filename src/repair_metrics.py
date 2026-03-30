
                lufa_idx = lufa_df.index[lufa_df['question_id'] == qid].tolist()
                if lufa_idx:
                    lufa_df.at[lufa_idx[0], f"source{i}_id"] = matched_id

        gt_ids = gt_lookup.get(qid, [])
        if repaired_ids and gt_ids:
            new_mrr = round(mrr(repaired_ids, gt_ids), 4)
            new_ndcg = ndcg_at_k(repaired_ids, gt_ids, 5)

            if eval_row.get("mrr") != new_mrr or eval_row.get("ndcg_at_k") != new_ndcg:
                eval_df.at[idx, "mrr"] = new_mrr
                eval_df.at[idx, "ndcg_at_k"] = new_ndcg
                eval_df.at[idx, "recall_1"] = round(recall_at_k(repaired_ids, gt_ids, 1), 4)
                eval_df.at[idx, "recall_3"] = round(recall_at_k(repaired_ids, gt_ids, 3), 4)
                eval_df.at[idx, "recall_5"] = round(recall_at_k(repaired_ids, gt_ids, 5), 4)
                updates += 1
                print(f"Repaired Question {qid} -> MRR: {new_mrr} | NDCG: {new_ndcg}")

    if updates > 0:
        lufa_df.to_csv(lufa_csv, index=False)
        eval_df.to_csv(eval_csv, index=False)
        print(f"Saved {updates} row updates to CSVs.")

        generate_dashboard(eval_df, dash_out)
        print("Dashboard HTML successfully updated.")
    else:
        print("No repairs needed. Files are aligned.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lufa_csv", default="tests/lufa_out_data.csv")
    parser.add_argument("--eval_csv", default="tests/evaluation_results.csv")
    parser.add_argument("--gt_csv", default="tests/combined_test_data_and_ground_truth.csv")
    parser.add_argument("--db", default="db/chroma_db")
    parser.add_argument("--dash", default="dashboard/index.html")
    args = parser.parse_args()

    run_repair(args.lufa_csv, args.eval_csv, args.gt_csv, args.db, args.dash)