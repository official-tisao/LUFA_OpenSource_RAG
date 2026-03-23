    client = chromadb.PersistentClient(path=db_path)

    print("[Initialization] Instantiating neural layout embedding index models...")
    engine = create_rag_engine(db_path=db_path)
    embed_model = engine.embed_model

    print("\n" + "=" * 80)
    print("STARTING INTERACTIVE BILINGUAL PROCESSING LOOP")
    print("=" * 80)

    for idx, row in df.iterrows():
        current_counter = idx + 1
        question_id = row.get("id", f"row_{idx}")

        print(f"\n[{current_counter}/{total_records}] Processing Question ID: {question_id}")
        print(f"   -> Query Preview: \"{str(row['question'])[:65]}...\"")

        try:
            gt_id, gt_text, overlap_pct, answer_overlap_pct = find_exact_ground_truth(
                row=row,
                client=client,
                collection_name=collection_name,
                embed_model=embed_model
            )

            df.at[idx, "ground_source_truth_id"] = gt_id
            df.at[idx, "ground_source_truth"] = gt_text

            if gt_id:
                print(f"   ✅ Success: Chunk linked successfully.")
                print(f"      - Database Node UUID: {gt_id}")
                print(f"      - Text Alignment Score (Doc-to-Expected): {overlap_pct:.2%}")
                print(f"      - Answer Footprint Cross-Check (Expected-in-Doc): {answer_overlap_pct:.2%}")
            else:
                print(f"   ❌ Null Extraction: Found no underlying text blocks.")

        except Exception as e:
            print(f"   💥 Pipeline Exception on index row index [{idx}]: {e}")

    print("\n" + "=" * 80)
    print("PROCESSING CYCLE COMPLETED")
    print("=" * 80)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[Export] Integrated data structured table saved cleanly to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find true database ground truth mappings with runtime console feedback.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to evaluation test dataset")
    parser.add_argument("--db", default=DEFAULT_DB, help="ChromaDB persistent file location")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="ChromaDB target collection")
    parser.add_argument("--out", default=OUTPUT_CSV, help="Path for the output joined results")
    args = parser.parse_args()

    run_pipeline(args.csv, args.db, args.collection, args.out)