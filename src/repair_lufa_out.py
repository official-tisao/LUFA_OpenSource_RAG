                        best_match_id = cid
                        exact_found = True
                        break

                    overlap = calculate_token_overlap(source_clean, doc_clean)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_match_id = cid

                if exact_found:
                    repaired_id = best_match_id
                    method_used = "ChromaDB Substring Perfect Fit"
                elif max_overlap > 0.5:
                    repaired_id = best_match_id
                    method_used = f"ChromaDB Vector Token Overlap (Score: {max_overlap:.2%})"

            if repaired_id:
                old_id = row.get(id_col, "")
                if str(old_id) != str(repaired_id):
                    lufa_df.at[idx, id_col] = repaired_id
                    total_repaired += 1
                    print(f"   ✅ Repaired {id_col} -> {repaired_id} [{method_used}]")
                else:
                    print(f"   · Verified {id_col} is already correct -> {repaired_id}")
            else:
                print(f"   ❌ Warning: Could not locate matching chunk ID for {text_col}")

    print("\n================================================================================")
    print("STAGE 3: Saving Repaired Records and Summary")
    print("================================================================================")

    lufa_df.to_csv(lufa_path, index=False)
    print(f" -> Repaired file saved cleanly back to: {lufa_path}")
    print(f" -> Total source text fields evaluated: {total_fields}")
    print(f" -> Total source identifier columns updated: {total_repaired}")
    print("================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair missing or misplaced source chunk IDs in lufa output logs.")
    parser.add_argument("--lufa_csv", default=DEFAULT_LUFA_CSV, help="Path to lufa output log sheet")
    parser.add_argument("--gt_csv", default=DEFAULT_GROUND_TRUTH_CSV, help="Path to ground truth reference sheet")
    parser.add_argument("--db", default=DEFAULT_DB, help="ChromaDB persistent directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="ChromaDB collection name")
    args = parser.parse_args()

    repair_dataset(args.lufa_csv, args.gt_csv, args.db, args.collection)