    )

    # Drop duplicates on the question_id keeping first, which locks in the best entry
    cleaned_df = df_sorted.drop_duplicates(subset=["question_id"], keep="first")

    # Clean up working columns before writing back to disk
    cleaned_df = cleaned_df.drop(columns=["grounded_bool", "is_corrupted_metric"])

    # Re-sort chronologically by ID for visual presentation alignment
    cleaned_df = cleaned_df.sort_values(by=["question_id"])

    cleaned_df.to_csv(path, index=False)
    final_count = len(cleaned_df)

    print("Cleanup pass completed successfully.")
    print(f"Total duplicate or bad metric records eliminated: {initial_count - final_count}")
    print(f"Final clean unique record count saved to disk: {final_count}")

    # Automatically re-compile the user dashboard metrics view
    try:
        sys.path.insert(0, str(Path(file_path).parent.parent / "src"))
        from evaluate import generate_dashboard
        generate_dashboard(cleaned_df, dashboard_path)
        print(f"Dashboard interface successfully updated with unique clean records at {dashboard_path}")
    except Exception as d_err:
        print(f"Note: Dashboard live UI compile step skipped: {d_err}")


if __name__ == "__main__":
    clean_evaluation_file()