#!/usr/bin/env python3
"""
Performance evaluation script for LUFA Agentic RAG system with row-by-row
checkpoint saving, crash-resumption support, verbose metric logging,
and live real-time HTML dashboard incremental updates.
Includes inline single-pass retrieval repair hooks to prevent zero metric tracking bugs.
"""

import sys
import json
import argparse
import math
import re
import warnings
from pathlib import Path
from datetime import datetime
from dashboard_generator import generate_dashboard

import pandas as pd

# NLP metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Dynamic simulation hooks
from evaluate import run_evaluation

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LUFA RAG system performance")
    parser.add_argument("--lufa_csv", default="tests/lufa_out_data.csv")
    parser.add_argument("--test_csv", default="tests/combined_test_data_and_ground_truth.csv")
    parser.add_argument("--out_csv", default="tests/evaluation_results.csv")
    parser.add_argument("--dashboard", default="dashboard/index.html")
    parser.add_argument("--judge_llm", default=None)
    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--sim_mode", choices=["local", "local-naive", "api", "frontier"], default="local-naive")
    parser.add_argument("--api_url", default="http://localhost:8000")
    args = parser.parse_args()

    run_evaluation(
        lufa_csv=args.lufa_csv,
        test_csv=args.test_csv,
        out_csv=args.out_csv,
        dashboard_out=args.dashboard,
        judge_llm_model=args.judge_llm,
        llm_model=args.llm_model,
        sim_mode=args.sim_mode,
        api_url=args.api_url
    )