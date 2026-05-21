#!/usr/bin/env python3
"""
Consolidate PDF and TXT corpus documents from english and french folders.
Dynamically applies OCR only when an image-only layer is identified.
"""

import os
from pathlib import Path
import re
import sys

try:
    import pdfplumber
    import ocrmypdf
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please run: pip install pdfplumber ocrmypdf")
    sys.exit(1)

TARGET_DIRS = [
    Path("data/english"),
    Path("data/french")
]
OUTPUT_FILE = Path("all_corpus_text.txt")
TEMP_OCR_FILE = Path("temp_processed_ocr.pdf")

# Regex pattern to identify a valid year range (e.g., 1998-2003 or 2020-2025)
YEAR_RANGE_RE = re.compile(r'\b\d{4}[-—]\d{4}\b')


def is_image_based_pdf(pdf_path):
    """Inspects the first few pages of a PDF to check if it lacks a native text layer."""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            text_sample = ""
            for page in pdf.pages[:3]:
                text_sample += page.extract_text() or ""
            return len(text_sample.strip()) < 50
    except Exception:
        return True


def sanitize_filename(stem_name):
    """
    Sanitizes filenames by replacing underscores and non-year hyphens with spaces,
    preserving the year-range hyphens intact.
    """
    sanitized = stem_name.replace('_', ' ')

    # Isolate and protect valid year-range hyphenated strings
    protected_ranges = YEAR_RANGE_RE.findall(sanitized)
    placeholders = []

    for idx, year_range in enumerate(protected_ranges):
        placeholder = f"__YEAR_RANGE_PH_{idx}__"
        sanitized = sanitized.replace(year_range, placeholder)
        placeholders.append((placeholder, year_range))

    # Safely convert other structural hyphens to spaces
    sanitized = sanitized.replace('-', ' ')

    # Restore the protected year ranges
    for placeholder, year_range in placeholders:
        sanitized = sanitized.replace(placeholder, year_range)

    return sanitized.strip()


def run_consolidation():
    print("================================================================================")
    print("STAGE 1: Scanning Target Directories and Discovery")
    print("================================================================================")

    all_files = []
    for target_dir in TARGET_DIRS:
        if not target_dir.exists():
            print(f" -> Info: Path '{target_dir}' does not exist. Skipping.")
            continue
        print(f" -> Scanning directory: {target_dir}")
        for item in target_dir.iterdir():
            if item.is_file() and item.suffix.lower() in [".pdf", ".txt"]:
                all_files.append(item)

    total_files = len(all_files)
