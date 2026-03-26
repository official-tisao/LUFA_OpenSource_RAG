                    c.drawText(text_obj)
                    c.showPage()
                    if page_sizes and page_idx < len(page_sizes):
                        c.setPageSize(page_sizes[page_idx])
                    text_obj = c.beginText(x, height - top_margin)
                    text_obj.setFont(font_name, font_size)
                    text_obj.setLeading(leading)
            if cur:
                text_obj.textLine(cur)

            if text_obj.getY() < bottom_margin + leading:
                c.drawText(text_obj)
                c.showPage()
                if page_sizes and page_idx < len(page_sizes):
                    c.setPageSize(page_sizes[page_idx])
                text_obj = c.beginText(x, height - top_margin)
                text_obj.setFont(font_name, font_size)
                text_obj.setLeading(leading)

        c.drawText(text_obj)
        c.showPage()
    c.save()


def is_image_based_pdf(pdf_path: Path) -> bool:
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            text_sample = ""
            for page in pdf.pages[:3]:
                text_sample += page.extract_text() or ""
            return len(text_sample.strip()) < 50
    except Exception:
        return True


def run_ocr_on_file(input_path: Path, output_path: Path, langs: list) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ocrmypdf.ocr(
            str(input_path),
            str(output_path),
            deskew=True,
            force_ocr=True,
            language=langs
        )
        return True
    except Exception as e:
        print(f"    -> OCR execution failed for {input_path.name}: {e}")
        return False


def process_single_file(pdf_path: Path, mode: str, output_opt: str = None, force_lang: str = None) -> bool:
    if not pdf_path.exists():
        print(f"Error: File {pdf_path} does not exist.")
        return False

    stem = pdf_path.stem

    if mode == "dual":
        if output_opt:
            out_dir = Path(output_opt)
            en_out_path = out_dir / f"{stem}_english_ocr.pdf"
            fr_out_path = out_dir / f"{stem}_french_ocr.pdf"
        else:
            en_out_path = Path("data/english") / f"{stem}_english_ocr.pdf"
            fr_out_path = Path("data/french") / f"{stem}_french_ocr.pdf"

        temp_ocr_path = pdf_path.parent / f"temp_{stem}_searchable.pdf"

        if is_image_based_pdf(pdf_path):
            print(f"    -> Image layer detected. Initializing dual-language OCR translation...")
            success = run_ocr_on_file(pdf_path, temp_ocr_path, ["eng", "fra"])
            if not success:
                return False
            active_pdf = temp_ocr_path
        else:
            active_pdf = pdf_path

        try:
            with pdfplumber.open(str(active_pdf)) as pdf:
                page_sizes = [(p.width, p.height) for p in pdf.pages]

            eng_pages, fr_pages, _, _ = extract_column_pages_text(active_pdf, "dual")
            write_pdf_from_pages(eng_pages, en_out_path, page_sizes)
            write_pdf_from_pages(fr_pages, fr_out_path, page_sizes)
            print(f"    -> Saved extracted channels:\n       {en_out_path}\n       {fr_out_path}")
            return True
        except Exception as e:
            print(f"    -> Error processing layout split metrics: {e}")
            return False
        finally:
            if temp_ocr_path.exists():
                os.remove(temp_ocr_path)

    else:
        inferred_lang = force_lang if force_lang else ("fra" if "french" in str(pdf_path).lower() else "eng")
        suffix = "_french_ocr.pdf" if inferred_lang == "fra" else "_english_ocr.pdf"

        if output_opt:
            out_p = Path(output_opt)
            if out_p.is_dir() or not out_p.suffix:
                final_out = out_p / f"{stem}{suffix}"
            else:
                final_out = out_p
        else:
            target_folder = "./data/french" if inferred_lang == "fra" else "./data/english"
            final_out = Path(target_folder) / f"{stem}{suffix}"

        if is_image_based_pdf(pdf_path):
            print(f"    -> Image layer detected. Injecting missing searchable text layer ({inferred_lang})...")
            if run_ocr_on_file(pdf_path, final_out, [inferred_lang]):
                print(f"    -> Saved clean document output target to: {final_out}")
                return True
            return False
        else:
            with pdfplumber.open(str(pdf_path)) as pdf:
                page_sizes = [(p.width, p.height) for p in pdf.pages]
            primary_pages, _, _, _ = extract_column_pages_text(pdf_path, "single")
            write_pdf_from_pages(primary_pages, final_out, page_sizes)
            print(f"    -> Saved searchable structured file reference to: {final_out}")
            return True


def archive_original_file(file_path: Path, subfolder_name: str):
    try:
        archive_base = Path("data/processed")
        target_dir = archive_base / subfolder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_path = target_dir / file_path.name
        shutil.move(str(file_path), str(dest_path))
        print(f"    -> Archived original file cleanly to: {dest_path}")
    except Exception as e:
        print(f"    -> Failed to archive processed file {file_path.name}: {e}")


def run_automated_batch_pipeline():
