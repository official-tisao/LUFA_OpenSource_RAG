Command: python src/run_simulation.py
Input: data/combined_test_data.csv
Output: tests/lufa_out_data.csv
Note: If the script crashes or is stopped, running it again will automatically resume from the last successful non-error row.

### Step 3: Evaluate Performance
This script compares the simulation outputs against the ground truth. It calculates Generation NLP metrics (BLEU, ROUGE, F1) and Retrieval position metrics (MRR, NDCG). It also calls the LLM-as-a-Judge.
Command: python src/evaluate.py
Input: tests/lufa_out_data.csv & tests/combined_test_data_and_ground_truth.csv
Output: tests/evaluation_results.csv & dashboard/index.html
Note: If simulation gaps exist, evaluate.py will dynamically run the simulation for that missing row inline.

### Step 4: System Repair (If Necessary)
If older logs generated _chunkX placeholders instead of real database UUIDs, this script will read the raw text snippets, search ChromaDB for the exact match, and overwrite both CSV files. It will then recalculate the retrieval metrics and refresh the dashboard.
Command: python src/repair_metrics.py
Input: tests/lufa_out_data.csv & tests/evaluation_results.csv
Output: Overwrites both CSVs and dashboard/index.html natively.

Need help? Check the full [README.md](README.md) or open an issue!
