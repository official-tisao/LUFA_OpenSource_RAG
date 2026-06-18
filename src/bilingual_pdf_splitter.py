# python
import os
from pathlib import Path
import re
from typing import List
import pdfplumber
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register a standard font (fallback)
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
    DEFAULT_FONT = "DejaVuSans"
except Exception:
    DEFAULT_FONT = "Helvetica"

INPUT_DIR = Path("./data/english_and_french")
OUT_EN = Path("./data/english")
OUT_FR = Path("./data/french")

CLAUSE_RE = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')


def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\u200b", " ")
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


def extract_words_split(page):
    words = page.extract_words(extra_attrs=["x0", "x1", "top", "bottom"], keep_blank_chars=False, use_text_flow=True) or []
    if not words:
        return [], []
    mid = float(page.width) / 2.0
    left = [w for w in words if (w["x0"] + w["x1"]) / 2.0 < mid]
    right = [w for w in words if (w["x0"] + w["x1"]) / 2.0 >= mid]
    return left, right


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
    return clean_text(" ".join(w["text"] for w in sorted(line, key=lambda x: x["x0"])))


def merge_lines(lines):
    merged = []
    buf = []
    for line in lines:
        t = line_text(line)
        if not t:
            # preserve blank lines as paragraph break
            if buf:
                merged.append(" ".join(buf))
                buf = []
            merged.append("")  # explicit blank line
            continue
        if buf and (t[0].islower() or t.startswith('—') or t.startswith('-')):
            buf.append(t)
        else:
            if buf:
                merged.append(' '.join(buf))
            buf = [t]
    if buf:
        merged.append(' '.join(buf))
    # Keep blank lines and trim others
    cleaned = []
    for item in merged:
        if item == "":
            cleaned.append("")  # preserve empty paragraph separator
        else:
            ct = clean_text(item)
            if ct:
                cleaned.append(ct)
    return cleaned


def extract_column_pages_text(pdf_path: Path):
    """
    Returns two lists (english_pages, french_pages), where each is a list of pages,
    and each page is a list of lines (strings) to render preserving paragraph/line breaks.
    """
    english_pages = []
    french_pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = clean_text(page.extract_text() or "")
            if not page_text:
                # append empty pages to keep page alignment
                english_pages.append([])
                french_pages.append([])
                continue
            left_words, right_words = extract_words_split(page)
            left_lines = merge_lines(cluster_lines(left_words))
            right_lines = merge_lines(cluster_lines(right_words))
            # If either column returned no text, fall back to page text split heuristics:
            english_pages.append(left_lines)
            french_pages.append(right_lines)
    return english_pages, french_pages, pdf.pages[0].width if len(pdf.pages) else 612, pdf.pages[0].height if len(pdf.pages) else 792


def write_pdf_from_pages(pages_lines: List[List[str]], out_path: Path, page_sizes: List[tuple] = None):
    """
    pages_lines: list of pages, each page is list of lines ('' represents blank line)
    page_sizes: optional list of (w,h) per page; if None, use default letter
    """
    if not pages_lines:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use first page size or default 612x792
    default_w, default_h = (612, 792)
    if page_sizes and len(page_sizes) >= 1:
        first_size = page_sizes[0]
    else:
        first_size = (default_w, default_h)

    c = canvas.Canvas(str(out_path), pagesize=first_size)
    font_name = DEFAULT_FONT
    font_size = 10
    leading = font_size * 1.25
    left_margin = 36
    right_margin = 36
    top_margin = 36
    bottom_margin = 36

    for page_idx, lines in enumerate(pages_lines):
        # set or change page size per original if provided
        if page_sizes and page_idx < len(page_sizes):
            c.setPageSize(page_sizes[page_idx])

        width, height = c._pagesize
        x = left_margin
        y_start = height - top_margin
        text_obj = c.beginText(x, y_start)
        text_obj.setFont(font_name, font_size)
        text_obj.setLeading(leading)

        for line in lines:
            # If an explicit blank line: add an empty text line (preserve paragraph break)
            if line == "":
                text_obj.textLine("")  # consumes leading space
                continue
            # Add the line; ReportLab will not wrap automatically with textLine
            # Do a simple wrap at page width
            max_width = width - left_margin - right_margin
            words = line.split(" ")
            cur = ""
            for w in words:
                trial = cur + (" " if cur else "") + w
                if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
                    cur = trial
                else:
                    # flush current
                    text_obj.textLine(cur)
                    cur = w
                # if text goes below bottom, force new page (keep one-to-one page mapping as much as possible)
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
            # Check for space to next paragraph; if not, flush page
            if text_obj.getY() < bottom_margin + leading:
                c.drawText(text_obj)
                c.showPage()
                if page_sizes and page_idx < len(page_sizes):
                    c.setPageSize(page_sizes[page_idx])
                text_obj = c.beginText(x, height - top_margin)
                text_obj.setFont(font_name, font_size)
                text_obj.setLeading(leading)
        # Finish the page
        c.drawText(text_obj)
        c.showPage()
    c.save()


def process_all(input_dir: Path = INPUT_DIR, out_en: Path = OUT_EN, out_fr: Path = OUT_FR):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist. Exiting.")
        return

    out_en.mkdir(parents=True, exist_ok=True)
    out_fr.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdf_paths:
        print(f"No PDFs found in {input_dir}. Nothing to do.")
        return

    for pdf_path in pdf_paths:
        stem = pdf_path.stem
        print(f"Processing {pdf_path.name} ...")
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                page_sizes = [(p.width, p.height) for p in pdf.pages]
            eng_pages, fr_pages, default_w, default_h = extract_column_pages_text(pdf_path)
            # Write english PDF
            eng_out = out_en / f"{stem}_english.pdf"
            write_pdf_from_pages(eng_pages, eng_out, page_sizes)
            # Write french PDF
            fr_out = out_fr / f"{stem}_french.pdf"
            write_pdf_from_pages(fr_pages, fr_out, page_sizes)
            print(f"Saved: {eng_out} and {fr_out}")
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")


if __name__ == "__main__":
    process_all()
