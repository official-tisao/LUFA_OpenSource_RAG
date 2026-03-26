    print("No options specified. Initiating automated repository directory scans...")

    # 1a. Scan ./data/english/
    eng_dir = Path("data/english")
    print("\nScanning directory: ./data/english")
    if eng_dir.exists():
        files = [p for p in eng_dir.iterdir() if p.suffix.lower() == ".pdf"]
        if not files:
            print("  No PDF files found in this directory.")
        for p in files:
            print(f"  [Evaluating] File: {p.name}")
            if p.name.endswith("_english_ocr.pdf") or p.name.endswith("_french_ocr.pdf"):
                print("    -> Skipped: Already a processed OCR output file.")
                continue

            if is_image_based_pdf(p):
                print(f"    -> [Match] Image layer detected. Running English OCR...")
                out_path = eng_dir / f"{p.stem}_english_ocr.pdf"
                if run_ocr_on_file(p, out_path, ["eng"]):
                    print(f"    -> Saved: {out_path}")
                    archive_original_file(p, "english")
            else:
                print("    -> Skipped: PDF already contains native text layer.")
    else:
        print("  Directory does not exist. Skipping.")

    # 1b. Scan ./data/french/
    fr_dir = Path("data/french")
    print("\nScanning directory: ./data/french")
    if fr_dir.exists():
        files = [p for p in fr_dir.iterdir() if p.suffix.lower() == ".pdf"]
        if not files:
            print("  No PDF files found in this directory.")
        for p in files:
            print(f"  [Evaluating] File: {p.name}")
            if p.name.endswith("_english_ocr.pdf") or p.name.endswith("_french_ocr.pdf"):
                print("    -> Skipped: Already a processed OCR output file.")
                continue

            if is_image_based_pdf(p):
                print(f"    -> [Match] Image layer detected. Running French OCR...")
                out_path = fr_dir / f"{p.stem}_french_ocr.pdf"
                if run_ocr_on_file(p, out_path, ["fra"]):
                    print(f"    -> Saved: {out_path}")
                    archive_original_file(p, "french")
            else:
                print("    -> Skipped: PDF already contains native text layer.")
    else:
        print("  Directory does not exist. Skipping.")

    # 1c. Scan ./data/english_and_french/
    dual_dir = Path("data/english_and_french")
    print("\nScanning directory: ./data/english_and_french")
    if dual_dir.exists():
        files = [p for p in dual_dir.iterdir() if p.suffix.lower() == ".pdf"]
        if not files:
            print("  No PDF files found in this directory.")
        for p in files:
            if p.name.startswith("temp_"):
                continue
            print(f"  [Evaluating] File: {p.name}")
            if p.name.endswith("_english_ocr.pdf") or p.name.endswith("_french_ocr.pdf"):
                print("    -> Skipped: Already a processed OCR output file.")
                continue

            print("    -> [Match] Dual-column bilingual file target identified.")
            if process_single_file(p, mode="dual", output_opt=None):
                archive_original_file(p, "english_and_french")
    else:
        print("  Directory does not exist. Skipping.")


if __name__ == "__main__":
    from argparse import RawTextHelpFormatter

    tutorial_guide = """
EXAMPLES OF HOW TO RUN THIS SCRIPT:
-----------------------------------
1. Run default batch scan operation (Scans all target paths and organizes files automatically):
   python pdf_ocr_converter.py

2. Process a targeted side-by-side bilingual document file:
   python pdf_ocr_converter.py -i ./data/english_and_french/document.pdf --mode dual

3. Process a side-by-side file and force outputs to go to a unified specific workspace folder:
   python pdf_ocr_converter.py -i ./data/english_and_french/document.pdf --mode dual -o ./my_output_dir

4. Process a standard monolingual file manually:
   python pdf_ocr_converter.py -i ./data/scanned_image.pdf --mode single
    """

    parser = argparse.ArgumentParser(
        description="Universal OCR Processing Pipeline and Dual-Column Layout Splitter Engine.",
        epilog=tutorial_guide,
        formatter_class=RawTextHelpFormatter
    )

    parser.add_argument("-i", "--input", default=None, help="Path to an individual target PDF file resource.")
    parser.add_argument("-o", "--output", default=None,
                        help="Custom output path choice (Directory for dual mode, File/Directory for single mode).")
    parser.add_argument("--mode", choices=["single", "dual"], default="single",
                        help="Layout processing matrix strategy structure selection.")
    parser.add_argument("--lang", choices=["eng", "fra"], default=None,
                        help="Force specific language optimization configuration overrides.")

    args = parser.parse_args()

    if args.input is None and args.output is None and args.mode == "single" and args.lang is None:
        run_automated_batch_pipeline()
    else:
        if args.input is None:
            print(
                "Error: The -i/--input parameter resource definition attribute is required when applying explicit filter flags.")
            sys.exit(1)
        process_single_file(Path(args.input), args.mode, args.output, args.lang)

    print("\nProcessing workflow execution cycle completed.")