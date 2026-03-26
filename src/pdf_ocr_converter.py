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
