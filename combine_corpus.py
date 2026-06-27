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
    print(f" -> Discovery Complete. Found {total_files} total corpus files to process.")

    if total_files == 0:
        print(" -> Warning: No valid files found to process. Exiting script execution.")
        return

    print("\n================================================================================")
    print("STAGE 2: Beginning Text Compilation and Layer Analysis")
    print("================================================================================")

    total_pages = 0
    error_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for idx, file_path in enumerate(all_files):
            current_counter = idx + 1
            print(f"\n[{current_counter}/{total_files}] Current Document: {file_path.name}")

            try:
                # Execution Branch for PDF Documents
                if file_path.suffix.lower() == ".pdf":
                    active_pdf_path = file_path

                    # Run image-layer detection threshold check
                    if is_image_based_pdf(file_path):
                        print("   -> Layer Check: Image layer detected. Executing conditional OCR layer injection...")
                        inferred_lang = "fra" if "french" in str(file_path).lower() else "eng"

                        ocrmypdf.ocr(
                            str(file_path),
                            str(TEMP_OCR_FILE),
                            deskew=True,
                            force_ocr=True,
                            language=[inferred_lang]
                        )
                        active_pdf_path = TEMP_OCR_FILE
                    else:
                        print("   -> Layer Check: Native text layer detected. Skipping OCR execution completely.")

                    # Extract text content layout
                    with pdfplumber.open(str(active_pdf_path)) as pdf:
                        for page_idx, page in enumerate(pdf.pages, start=1):
                            total_pages += 1
                            raw_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

                            if not lines:
                                continue

                            last_line = lines[-1]

                            # Validate whether the footer contains the required agreement pattern
                            if YEAR_RANGE_RE.search(last_line):
                                processed_last_line = last_line
                            else:
                                sanitized_base = sanitize_filename(file_path.stem)
                                processed_last_line = f"{sanitized_base} Page {page_idx}"

                            lines[-1] = processed_last_line
                            page_content = "\n".join(lines)
                            outfile.write(page_content + "\n\n")

                    # Clean up temporary file if generated
                    if TEMP_OCR_FILE.exists():
                        os.remove(TEMP_OCR_FILE)

                # Execution Branch for Text Documents
                elif file_path.suffix.lower() == ".txt":
                    print("   -> Strategy: Parsing raw text streams directly")
                    with open(file_path, "r", encoding="utf-8") as txt_file:
                        content = txt_file.read()

                    pages_text = content.split('\x0c') if '\x0c' in content else [content]
                    for page_idx, page_text in enumerate(pages_text, start=1):
                        total_pages += 1
                        lines = [line.strip() for line in page_text.splitlines() if line.strip()]

                        if not lines:
                            continue

                        last_line = lines[-1]

                        if YEAR_RANGE_RE.search(last_line):
                            processed_last_line = last_line
                        else:
                            sanitized_base = sanitize_filename(file_path.stem)
                            processed_last_line = f"{sanitized_base} Page {page_idx}"

                        lines[-1] = processed_last_line
                        page_content = "\n".join(lines)
                        outfile.write(page_content + "\n\n")

                print("   ✅ Success: Content processed and compiled smoothly.")

            except Exception as exc:
                error_count += 1
                print(f"   💥 Major Exception Encountered processing file '{file_path.name}': {exc}")
                # Ensure safety cleanup on crash
                if TEMP_OCR_FILE.exists():
                    os.remove(TEMP_OCR_FILE)

    print("\n================================================================================")
    print("STAGE 3: Consolidation Summary Results")
    print("================================================================================")
    print(f" -> Completed processing: {total_files - error_count} out of {total_files} files successfully.")
    print(f" -> Total parsed virtual pages appended: {total_pages}")
    print(f" -> Critical execution failures recorded: {error_count}")
    print(f" -> Master compilation text document stored at: {OUTPUT_FILE}")
    print("================================================================================")


if __name__ == "__main__":
    run_consolidation()