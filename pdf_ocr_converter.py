import os
import argparse
from pathlib import Path
import re
import sys
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

CLAUSE_RE = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')


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


def ensure_searchable_pdf(pdf_path: Path, mode: str) -> Path:
    with pdfplumber.open(str(pdf_path)) as pdf:
        text_sample = ""
        for page in pdf.pages[:3]:
            text_sample += page.extract_text() or ""

        if len(text_sample.strip()) > 50:
            return pdf_path

    print(f"No text layer detected in {pdf_path.name}. Running OCR...")
    ocr_path = pdf_path.parent / f"{pdf_path.stem}_ocr.pdf"

    langs = ["eng", "fra"] if mode == "dual" else ["eng"]

    try:
        # Pass the input and output paths directly as the first two positional arguments
        ocrmypdf.ocr(
            str(pdf_path),
            str(ocr_path),
            deskew=True,
            force_ocr=True,
            language=langs
        )
        return ocr_path
    except Exception as e:
        print(f"OCR failed for {pdf_path.name}. Error: {e}")
        return pdf_path


def process_input(input_path_str: str, mode: str):
    """Handles both single files and directories."""
    input_path = Path(input_path_str)

    if not input_path.exists():
        print(f"Error: {input_path} does not exist.")
        return

    # DETECT IF INPUT IS A FILE OR A DIRECTORY
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            print(f"Error: Input file must be a PDF. You provided: {input_path.name}")
            return

        pdf_paths = [input_path]
        out_base = input_path.parent / f"{input_path.stem}_processed"
        print(f"Detected single file input. Output will be saved to: {out_base}/")

    elif input_path.is_dir():
        pdf_paths = sorted(
            [p for p in input_path.iterdir() if p.suffix.lower() == ".pdf" and not p.stem.endswith("_ocr")])
        out_base = input_path.parent / f"{input_path.name}_processed"
        print(f"Detected directory input. Found {len(pdf_paths)} PDFs. Output will be saved to: {out_base}/")

    else:
        print(f"Error: {input_path} is neither a file nor a directory.")
        return

    if not pdf_paths:
        print("No valid PDFs found to process.")
        return

    out_base.mkdir(parents=True, exist_ok=True)

    for idx, original_pdf_path in enumerate(pdf_paths, start=1):
        print(f"\n[{idx}/{len(pdf_paths)}] Processing: {original_pdf_path.name} (Mode: {mode.upper()})")

        active_pdf_path = ensure_searchable_pdf(original_pdf_path, mode)
        stem = original_pdf_path.stem

        try:
            with pdfplumber.open(str(active_pdf_path)) as pdf:
                page_sizes = [(p.width, p.height) for p in pdf.pages]

            primary_pages, secondary_pages, _, _ = extract_column_pages_text(active_pdf_path, mode)

            if mode == "dual":
                out_en = out_base / "english"
                out_fr = out_base / "french"
                out_en.mkdir(exist_ok=True)
                out_fr.mkdir(exist_ok=True)

                write_pdf_from_pages(primary_pages, out_en / f"{stem}_english.pdf", page_sizes)
                write_pdf_from_pages(secondary_pages, out_fr / f"{stem}_french.pdf", page_sizes)
                print(f"Saved dual-column outputs to {out_base.name}/")
            else:
                out_single = out_base / "clean_documents"
                out_single.mkdir(exist_ok=True)

                write_pdf_from_pages(primary_pages, out_single / f"{stem}_clean.pdf", page_sizes)
                print(f"Saved clean document to {out_single.name}/{stem}_clean.pdf")

        except Exception as e:
            print(f"Error processing {active_pdf_path.name}: {e}")

if __name__ == "__main__":
    from argparse import RawTextHelpFormatter

    tutorial_text = """
EXAMPLES OF HOW TO USE THIS SCRIPT:
-----------------------------------
Prerequisites
Before running the script, you must install the required Python library and the system engines that actually perform the text recognition:

Python Library:
pip install ocrmypdf

System Dependencies:

Windows: Download and install Tesseract OCR (from UB-Mannheim's GitHub) and Ghostscript. Make sure to check the boxes for "French" language data during the Tesseract installation, and ensure both programs are added to your system PATH.

macOS: brew install tesseract ghostscript

Linux (Ubuntu): sudo apt-get install tesseract-ocr tesseract-ocr-fra ghostscript

1. Process a standard, single-column PDF document:
   python universal_pdf_processor.py -i ./data/my_document.pdf --mode single

2. Process an entire folder of standard PDFs:
   python universal_pdf_processor.py -i ./data/my_pdfs/ --mode single

3. Process a bilingual side-by-side PDF (Splits left/right into English/French):
   python universal_pdf_processor.py -i ./data/lufa_agreement.pdf --mode dual

HOW IT WORKS:
- The script automatically checks if your PDF is an image. 
- If it is an image, it runs Dual-Language OCR (English & French) to make it readable.
- Outputs are saved in a new '_processed' folder right next to your original file.
    """

    parser = argparse.ArgumentParser(
        description="Universal OCR and Layout Splitter for PDFs.",
        epilog=tutorial_text,
        formatter_class=RawTextHelpFormatter  # This allows us to use nice spacing and newlines
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to a single PDF file OR a directory containing PDFs."
    )

    parser.add_argument(
        "--mode",
        choices=["single", "dual"],
        default="single",
        help="Layout mode:\n  'single' = standard PDFs (Default)\n  'dual'   = side-by-side bilingual PDFs (Splits vertically)"
    )

    args = parser.parse_args()

    process_input(args.input, args.mode)