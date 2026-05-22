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
