from pathlib import Path
import pandas as pd

# ============================================================
# CONSTANTS - EDIT THESE
# ============================================================

# Source-of-truth CSV (file 1)
SOURCE_OF_TRUTH_CSV = "./tests/combined_test_data_and_ground_truth.csv"

# Column used to intersect/filter all other CSVs
INTERSECT_COLUMN = "question_id"

# One or more column combinations used to remove duplicates
# from the source-of-truth file before filtering others.
#
# Examples:
#   [("question_id",)]
#   [("question_id",), ("question", "answer")]
#   [("id",)]
#
# Duplicates are removed sequentially in the order listed below.
SOURCE_DUPLICATE_COLUMN_COMBINATIONS = [
    ("question_id") #,"base_model_used"),
    # ("question", "answer"),
]

# CSVs to filter against the source-of-truth.
# You can put any number of files here.
INPUT_CSVS_TO_FILTER = [
    "./tests/combined_test_data.csv",
]

# Optional per-file mapping for the intersect column name.
# Use this when a file does not use the same key column name as INTERSECT_COLUMN.
# Example: combined_test_data_and_ground_truth.csv uses "id" instead of "question_id"
INTERSECT_COLUMN_BY_FILE = {
    #"combined_test_data_and_ground_truth.csv": "question_id",
     #"combined_test_data.csv": "id",
}

# Output folder
OUTPUT_DIR = "filtered_output"

# If True, overwrite source file with de-duplicated version in OUTPUT_DIR
WRITE_CLEANED_SOURCE = True

# CSV write options
CSV_ENCODING = "utf-8"
CSV_INDEX = False

# ============================================================
# LOGIC
# ============================================================

def ensure_columns_exist(df: pd.DataFrame, required_columns: list[str], file_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing column(s) {missing} in file '{file_name}'. "
            f"Available columns: {list(df.columns)}"
        )

def normalize_key_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
    )

def remove_duplicates_from_source(
        df: pd.DataFrame,
        duplicate_column_combinations: list[tuple[str, ...]],
        file_name: str
) -> pd.DataFrame:
    cleaned_df = df.copy()

    for cols in duplicate_column_combinations:
        cols = list(cols)
        ensure_columns_exist(cleaned_df, cols, file_name)
        before = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)
        after = len(cleaned_df)
        print(f"[SOURCE DEDUPE] {file_name}: removed {before - after} duplicate row(s) using columns {cols}")

    return cleaned_df

def filter_csv_by_source_keys(
        source_keys: set,
        input_csv: str,
        input_key_column: str,
        output_dir: Path
) -> None:
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    ensure_columns_exist(df, [input_key_column], input_csv)

    before = len(df)
    key_series = normalize_key_series(df[input_key_column])
    filtered_df = df[key_series.isin(source_keys)].copy()
    after = len(filtered_df)

    output_path = output_dir / Path(input_csv).name
    filtered_df.to_csv(output_path, index=CSV_INDEX, encoding=CSV_ENCODING)

    print(
        f"[FILTERED] {input_csv}: kept {after}/{before} row(s), "
        f"removed {before - after} row(s) not present in source '{SOURCE_OF_TRUTH_CSV}'"
    )

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(SOURCE_OF_TRUTH_CSV, dtype=str, keep_default_na=False)
    ensure_columns_exist(source_df, [INTERSECT_COLUMN], SOURCE_OF_TRUTH_CSV)

    # Step 1: remove duplicates from source-of-truth using configured column combinations
    source_df = remove_duplicates_from_source(
        source_df,
        SOURCE_DUPLICATE_COLUMN_COMBINATIONS,
        SOURCE_OF_TRUTH_CSV
    )

    # Step 2: build source key set from the cleaned source file
    source_keys = set(normalize_key_series(source_df[INTERSECT_COLUMN]))
    print(f"[SOURCE] {SOURCE_OF_TRUTH_CSV}: loaded {len(source_df)} unique row(s) after dedupe")
    print(f"[SOURCE] unique intersect keys in '{INTERSECT_COLUMN}': {len(source_keys)}")

    # Step 3: optionally write cleaned source file
    if WRITE_CLEANED_SOURCE:
        cleaned_source_output = output_dir / Path(SOURCE_OF_TRUTH_CSV).name
        source_df.to_csv(cleaned_source_output, index=CSV_INDEX, encoding=CSV_ENCODING)
        print(f"[SOURCE WRITTEN] cleaned source saved to: {cleaned_source_output}")

    # Step 4: filter all input CSVs based on source keys
    for input_csv in INPUT_CSVS_TO_FILTER:
        input_key_column = INTERSECT_COLUMN_BY_FILE.get(input_csv, INTERSECT_COLUMN)
        filter_csv_by_source_keys(
            source_keys=source_keys,
            input_csv=input_csv,
            input_key_column=input_key_column,
            output_dir=output_dir
        )

if __name__ == "__main__":
    main()