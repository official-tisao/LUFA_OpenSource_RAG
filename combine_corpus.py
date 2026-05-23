
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