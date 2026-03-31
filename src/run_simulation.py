    parser.add_argument("--model", default=None, help="Frontier model ID (for --mode frontier)")
    parser.add_argument("--api_url", default="http://localhost:8000", help="API base URL for --mode api")
    parser.add_argument("--input", default=INPUT_CSV, help="Input CSV path")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    cfg = load_config()
    base_model = cfg.get("models", {}).get("llm", {}).get("name", "llama3.2:3b-instruct-q4_K_M")
    model_name = args.model or base_model

    ensure_ground_truth(args.input)

    print(f"[Sim] Loading input master file: {args.input}")
    df = pd.read_csv(args.input)
    print(f"[Sim] {len(df)} total target questions configured in master dataset.")

    completed_ids = set()
    all_logged_ids = set()
    out_path = Path(args.output)

    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(args.output)
            if "question_id" in existing_df.columns:
                all_logged_ids = set(existing_df["question_id"].dropna().astype(str).tolist())

                # Filter strictly for rows that have valid string content and no ERROR flag
                successful_df = existing_df[
                    existing_df["question_id"].notna() &
                    existing_df["answer"].notna() &
                    (existing_df["answer"].astype(str).str.strip() != "ERROR") &
                    (existing_df["answer"].astype(str).str.strip() != "")
                    ]
                completed_ids = set(successful_df["question_id"].dropna().astype(str).tolist())

                error_count = len(all_logged_ids) - len(completed_ids)
                print(f"[Resumption] Located active output log file: {args.output}")
                print(f"[Resumption] Found {len(completed_ids)} successfully processed rows.")
                if error_count > 0:
                    print(
                        f"[Resumption] Found {error_count} rows containing ERROR flags. These will be automatically re-run.")
        except Exception as err:
            print(f"[Warning] Error parsing simulation file checkpoint: {err}")
    else:
        print(f"[Sim] Output path '{args.output}' is empty or new. Starting execution pass.")

    print("\n" + "=" * 80)
    print("STARTING UNIFIED SIMULATION PASS (REPAIRING ERRORS + ADDING NEW QUESTIONS)")
    print("=" * 80)

    for idx, record in df.iterrows():
        current_counter = idx + 1
        q_id = str(record["id"])

        if q_id in completed_ids:
            print(f"[{current_counter}/{len(df)}] Skipping Question ID {q_id} (Valid answer already stored)")
            continue

        # Verbose message informing whether it is fixing an error or processing a brand new row
        if q_id in all_logged_ids:
            print(f"\n[{current_counter}/{len(df)}] 🛠️  Re-running failed record -> Question ID: {q_id}")
        else:
            print(f"\n[{current_counter}/{len(df)}] 🚀 Processing unvisited record -> Question ID: {q_id}")

        row_res = query_single_record(record, args.mode, base_model, model_name, args.api_url, current_counter)

        single_row_df = pd.DataFrame([row_res], columns=OUTPUT_COLUMNS)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        file_is_new = not out_path.exists() or out_path.stat().st_size == 0
        single_row_df.to_csv(str(out_path), mode="a", index=False, header=file_is_new)
        print("   ✅ Row appended cleanly to simulation output log.")
        time.sleep(0.5)

    print("\n" + "=" * 80)
    print(f"[Sim] Execution pass complete. All rows verified and logged to: {args.output}")
    print("=" * 80 + "\n")