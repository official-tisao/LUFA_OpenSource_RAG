#!/usr/bin/env python3
from pathlib import Path
import re
import pdfplumber
import pandas as pd
import argparse
import chromadb
import importlib.metadata
import urllib.request
import json
import pandas as pd
import sys


ARTICLE_RE = re.compile(r'^(?:ARTICLE|Article)\s+(\d{1,3})\b', re.I)
CLAUSE_RE = re.compile(r'^(\d{1,3}(?:\.\d+)+(?:\([a-z]+\))?)\b', re.I)
SECTION_RE = re.compile(r'^(?:PART|SECTION|SCHEDULE|APPENDIX|ANNEXE|PARTIE)\s+[IVX\d]+', re.I)
PAGE_MARK_RE = re.compile(r'<<<PAGE:(\d+)>>>')

def get_pkg_version(pkg_name):
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return "Not Installed"


def get_ollama_info():
    try:
        req = urllib.request.Request("http://localhost:11434/api/version")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read())
            return data.get("version", "Unknown")
    except Exception:
        return "Ollama unreachable"


def get_ollama_model_info(model_substring):
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read())
            models = data.get("models", [])
            for m in models:
                if model_substring.lower() in m.get("name", "").lower():
                    details = m.get("details", {})
                    quant = details.get("quantization_level", "Unknown")
                    return f"Installed (GGUF {quant})"
            return "Not Installed"
    except Exception:
        return "Ollama API unreachable"


def generate_system_table():
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    data = [
        {
            "Component": "LLM serving",
            "Framework/Library": "Ollama",
            "Version": get_ollama_info()
        },
        {
            "Component": "Agent orchestration",
            "Framework/Library": "LlamaIndex",
            "Version": get_pkg_version("llama-index")
        },
        {
            "Component": "Vector database",
            "Framework/Library": "ChromaDB",
            "Version": get_pkg_version("chromadb")
        },
        {
            "Component": "Embedding model",
            "Framework/Library": "BGE-M3 (via Ollama)",
            "Version": get_ollama_model_info("bge-m3")
        },
        {
            "Component": "Generator model",
            "Framework/Library": "Llama 3.2 3B (via Ollama)",
            "Version": get_ollama_model_info("llama3.2")
        },
        {
            "Component": "RAG evaluation",
            "Framework/Library": "RAGAS",
            "Version": get_pkg_version("ragas")
        },
        {
            "Component": "BM25 implementation",
            "Framework/Library": "rank-bm25",
            "Version": get_pkg_version("rank-bm25")
        },
        {
            "Component": "PDF parsing",
            "Framework/Library": "PyMuPDF",
            "Version": get_pkg_version("PyMuPDF")
        },
        {
            "Component": "Experiment tracking",
            "Framework/Library": "Python logging + JSON",
            "Version": f"Built-in (Python {python_version})"
        }
    ]

    df = pd.DataFrame(data)

    print("\n" + "=" * 85)
    print(df.to_string(index=False, justify='left'))
    print("=" * 85 + "\n")

    output_file = "system_components_detected.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved dynamically detected system data to {output_file}")

def tokens(text):
    return len(text.split())


def extract_pages(pdf_path):
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            pages.append((i, p.extract_text(x_tolerance=3, y_tolerance=3) or ''))
    return pages


def build_marked(pages):
    parts = []
    for n, txt in pages:
        parts.append(f'<<<PAGE:{n}>>>')
        parts.append(txt)
    return '\n'.join(parts)


def detect_lang(text, forced=None):
    if forced:
        return forced
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        return detect(text[:500]) if text.strip() else 'en'
    except Exception:
        return 'en'


def chunk_pdf(pdf_path, forced_lang=None, min_tokens=30):
    pages = extract_pages(pdf_path)
