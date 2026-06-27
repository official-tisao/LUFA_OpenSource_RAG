import os
import argparse
from pathlib import Path
import re
import sys
import shutil
from typing import List

try:
    import pdfplumber
    import ocrmypdf
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please run: pip install pdfplumber ocrmypdf reportlab")
    sys.exit(1)

try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
    DEFAULT_FONT = "DejaVuSans"
except Exception:
    DEFAULT_FONT = "Helvetica"


def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\u200b", " ")
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


def extract_words_split(page, mode="single"):
    words = page.extract_words(
        extra_attrs=["x0", "x1", "top", "bottom"],
        keep_blank_chars=False,
        use_text_flow=True) or []

    if not words:
        return [], []

    if mode == "dual":
        mid = float(page.width) / 2.0
        left = [w for w in words if (w["x0"] + w["x1"]) / 2.0 < mid]
        right = [w for w in words if (w["x0"] + w["x1"]) / 2.0 >= mid]
        return left, right
    else:
        return words, []


def cluster_lines(words, y_tol=3.5):
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines = []
    cur = [words[0]]
    cur_y = words[0]["top"]

    for w in words[1:]:
        if abs(w["top"] - cur_y) <= y_tol:
            cur.append(w)
            cur_y = (cur_y * (len(cur) - 1) + w["top"]) / len(cur)
        else:
            lines.append(cur)
            cur = [w]
            cur_y = w["top"]

    lines.append(cur)
    return lines


def line_text(line):
    if not line:
        return ""
    line = sorted(line, key=lambda x: x["x0"])
    parts = [line[0]["text"]]
    prev_x1 = line[0]["x1"]

    for w in line[1:]:
        gap = w["x0"] - prev_x1
        if gap > 1.5:
            parts.append(" ")
        parts.append(w["text"])
        prev_x1 = w["x1"]

    return clean_text("".join(parts))


def merge_lines(lines):
    merged = []
    buf = []
    for line in lines:
        t = line_text(line)
        if not t:
            if buf:
                merged.append(" ".join(buf))
                buf = []
            merged.append("")
            continue
        if buf and (t[0].islower() or t.startswith("—") or t.startswith("-")):
            buf.append(t)
        else:
            if buf:
                merged.append(" ".join(buf))
            buf = [t]
    if buf:
        merged.append(" ".join(buf))

    cleaned = []
    for item in merged:
        if item == "":
            cleaned.append("")
        else:
            ct = clean_text(item)
            if ct:
                cleaned.append(ct)
    return cleaned


def extract_column_pages_text(pdf_path: Path, mode: str):
    primary_pages = []
    secondary_pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = clean_text(page.extract_text() or "")
            if not page_text:
                primary_pages.append([])
                if mode == "dual":
                    secondary_pages.append([])
                continue

            left_words, right_words = extract_words_split(page, mode)

            primary_lines = merge_lines(cluster_lines(left_words))
            primary_pages.append(primary_lines)

            if mode == "dual":
                secondary_lines = merge_lines(cluster_lines(right_words))
                secondary_pages.append(secondary_lines)

    return primary_pages, secondary_pages, pdf.pages[0].width if len(pdf.pages) else 612, pdf.pages[0].height if len(
        pdf.pages) else 792


def write_pdf_from_pages(pages_lines: List[List[str]], out_path: Path, page_sizes: List[tuple] = None):
    if not pages_lines:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    default_w, default_h = (612, 792)
    first_size = page_sizes[0] if (page_sizes and len(page_sizes) >= 1) else (default_w, default_h)

    c = canvas.Canvas(str(out_path), pagesize=first_size)
    font_name = DEFAULT_FONT
    font_size = 10
    leading = font_size * 1.25
    left_margin, right_margin, top_margin, bottom_margin = 36, 36, 36, 36

    for page_idx, lines in enumerate(pages_lines):
        if page_sizes and page_idx < len(page_sizes):
            c.setPageSize(page_sizes[page_idx])

        width, height = c._pagesize
        x = left_margin
        y_start = height - top_margin
        text_obj = c.beginText(x, y_start)
        text_obj.setFont(font_name, font_size)
        text_obj.setLeading(leading)

        for line in lines:
            if line == "":
                text_obj.textLine("")
                continue

            max_width = width - left_margin - right_margin
            words = line.split(" ")
            cur = ""
            for w in words:
                trial = cur + (" " if cur else "") + w
                if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
                    cur = trial
                else:
                    text_obj.textLine(cur)
                    cur = w

                if text_obj.getY() < bottom_margin + leading:
                    # Finish current page and start a new one
                    c.drawText(text_obj)
                    c.showPage()
                    if page_sizes and page_idx < len(page_sizes):
                        c.setPageSize(page_sizes[page_idx])
                    width, height = c._pagesize
                    text_obj = c.beginText(x, height - top_margin)
                    text_obj.setFont(font_name, font_size)
                    text_obj.setLeading(leading)

            # Flush any remaining wrapped text for this line
            if cur:
                text_obj.textLine(cur)

        # Finish page
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
        inferred_lang = force_lang if force_lang else (
            "fra" if "french" in str(pdf_path).lower() else "eng")
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
            print(
                f"    -> Image layer detected. Injecting missing searchable text layer ({inferred_lang})...")
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