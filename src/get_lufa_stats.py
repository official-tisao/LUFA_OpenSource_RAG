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
    text = build_marked(pages)
    lines = text.splitlines()
    out = []
    cur_article = '0';
    cur_clause = '0';
    cur_title = 'PREAMBLE';
    cur_page = 1;
    buf = []

    def flush():
        nonlocal buf
        t = '\n'.join(buf).strip()
        t = PAGE_MARK_RE.sub('', t).strip()
        if t:
            out.append({'article_number': cur_article, 'clause_id': cur_clause, 'section_title': cur_title, 'text': t,
                        'language': detect_lang(t, forced_lang), 'page_no': cur_page, 'tokens': tokens(t)})
        buf = []

    for line in lines:
        m = PAGE_MARK_RE.match(line)
        if m:
            cur_page = int(m.group(1));
            buf.append(line);
            continue
        s = line.strip()
        if not s:
            buf.append(line);
            continue
        if SECTION_RE.match(s):
            cur_title = s;
            buf.append(line);
            continue
        am = ARTICLE_RE.match(s)
        if am:
            flush();
            cur_article = am.group(1);
            cur_clause = cur_article;
            buf.append(line);
            continue
        cm = CLAUSE_RE.match(s)
        if cm and cm.group(1).startswith(cur_article + '.'):
            flush();
            cur_clause = cm.group(1).replace('(', ' .').replace(')', '').replace(' ', '').replace('..', '.').replace(
                '. ', ' .').replace(' .', '.')
            cur_clause = re.sub(r'\(([a-z]+)\)', r'.\1', cur_clause)
            buf.append(line);
            continue
        buf.append(line)
    flush()

    # merge short clauses into next clause
    merged = [];
    pending = [];
    pending_page = None
    for c in out:
        if c['tokens'] < min_tokens:
            pending.append(c)
            pending_page = pending_page or c['page_no']
        else:
            if pending:
                prefix = '\n'.join(x['text'] for x in pending)
                c['text'] = (prefix + '\n' + c['text']).strip()
                c['tokens'] = tokens(c['text'])
                c['page_no'] = pending_page or c['page_no']
                pending = [];
                pending_page = None
            merged.append(c)
    if pending and merged:
        last = merged[-1]
        last['text'] = (last['text'] + '\n' + '\n'.join(x['text'] for x in pending)).strip()
        last['tokens'] = tokens(last['text'])
    return merged


def get_pdf_stats(pdf_path, lang):
    if not pdf_path or not Path(pdf_path).exists():
        return {'articles': 0, 'clauses': 0, 'chunks': 0, 'avg_tokens': 0, 'min_tokens': 0, 'max_tokens': 0,
                'total_tokens': 0, 'appendices': 0}

    nodes = chunk_pdf(Path(pdf_path), lang)
    df = pd.DataFrame(nodes)
    if len(df) == 0:
        return {'articles': 0, 'clauses': 0, 'chunks': 0, 'avg_tokens': 0, 'min_tokens': 0, 'max_tokens': 0,
                'total_tokens': 0, 'appendices': 0}

    return {
        'articles': df['article_number'].replace('0', pd.NA).dropna().nunique(),
        'clauses': df['clause_id'].replace('0', pd.NA).dropna().nunique(),
        'chunks': len(df),
        'avg_tokens': round(df['tokens'].mean()),
        'min_tokens': int(df['tokens'].min()),
        'max_tokens': int(df['tokens'].max()),
        'total_tokens': int(df['tokens'].sum()),
        'appendices': int(
            df['section_title'].str.contains('SCHEDULE|APPENDIX|ANNEXE|PARTIE', case=False, na=False).sum())
    }


def scan_chroma_db_bilingual(db_path):
    db_dir = Path(db_path)
    if not db_dir.exists():
        print(f"ChromaDB directory not found at {db_path}")
        return None, None

    print(f"Scanning ChromaDB SQLite database at {db_path}...")
    client = chromadb.PersistentClient(path=str(db_path))

    stats_en = {'chunks': 0, 'total_tokens': 0, 'articles': set(), 'clauses': set(), 'token_list': [], 'appendices': 0}
    stats_fr = {'chunks': 0, 'total_tokens': 0, 'articles': set(), 'clauses': set(), 'token_list': [], 'appendices': 0}

    collections = client.list_collections()
    if not collections:
        print("No collections found in the database.")
        return None, None

    for coll_meta in collections:
        coll_name = coll_meta.name if hasattr(coll_meta, 'name') else coll_meta
        coll = client.get_collection(coll_name)
        data = coll.get(include=['documents', 'metadatas'])

        docs = data.get('documents', [])
        metas = data.get('metadatas', [])

        for doc, meta in zip(docs, metas):
            if not doc or not meta: continue

            # Determine target dictionary based on language metadata
            lang = meta.get('language', 'en').lower()
            target = stats_fr if lang == 'fr' else stats_en

            target['chunks'] += 1
            tok_count = len(doc.split())
            target['total_tokens'] += tok_count
            target['token_list'].append(tok_count)

            # Track unique articles and clauses (ignoring '0')
            art_no = str(meta.get('article_number', '0'))
            cl_id = str(meta.get('clause_id', '0'))

            if art_no != '0': target['articles'].add(art_no)
            if cl_id != '0': target['clauses'].add(cl_id)

            # Track Appendices and schedules
            sec_title = str(meta.get('section_title', ''))
            if re.search(r'SCHEDULE|APPENDIX|ANNEXE|PARTIE', sec_title, re.IGNORECASE):
                target['appendices'] += 1

    def summarize(s):
        t_list = s['token_list']
        return {
            'articles': len(s['articles']),
            'clauses': len(s['clauses']),
            'chunks': s['chunks'],
            'avg_tokens': round(sum(t_list) / len(t_list)) if t_list else 0,
            'min_tokens': min(t_list) if t_list else 0,
            'max_tokens': max(t_list) if t_list else 0,
            'total_tokens': s['total_tokens'],
            'appendices': s['appendices']
        }

    return summarize(stats_en), summarize(stats_fr)


def generate_comparison_table(stats_en, stats_fr):
    data = [
        {
            'Property': 'Total articles',
            'English Version': stats_en['articles'],
            'French Version': stats_fr['articles']
        },
        {
            'Property': 'Total numbered clauses',
            'English Version': stats_en['clauses'],
            'French Version': stats_fr['clauses']
        },
        {
            'Property': 'Total chunks after clause-boundary segmentation',
            'English Version': stats_en['chunks'],
            'French Version': stats_fr['chunks']
        },
        {
            'Property': 'Average tokens per chunk',
            'English Version': stats_en['avg_tokens'],
            'French Version': stats_fr['avg_tokens']
        },
        {
            'Property': 'Minimum tokens per chunk',
            'English Version': stats_en['min_tokens'],
            'French Version': stats_fr['min_tokens']
        },
        {
            'Property': 'Maximum tokens per chunk',
            'English Version': stats_en['max_tokens'],
            'French Version': stats_fr['max_tokens']
        },
        {
            'Property': 'Total corpus tokens',
            'English Version': f"{stats_en['total_tokens']:,}",
            'French Version': f"{stats_fr['total_tokens']:,}"
        },
        {
            'Property': 'Appendices and schedules',
            'English Version': stats_en['appendices'],
            'French Version': stats_fr['appendices']
        }
    ]
    return pd.DataFrame(data)


def main():
    ap = argparse.ArgumentParser(description="Generate bilingual stats from PDFs or ChromaDB")
    ap.add_argument('--en-pdf', help='Path to the English PDF file')
    ap.add_argument('--fr-pdf', help='Path to the French PDF file')
    ap.add_argument('--db-path', default='./db/chroma_db',
                    help='Path to ChromaDB sqlite directory (used if PDFs are not provided)')
    ap.add_argument('--out', default='lufa_stats_bilingual.csv', help='Output CSV file path')
    args = ap.parse_args()

    if args.en_pdf or args.fr_pdf:
        print("Processing PDFs directly...")
        stats_en = get_pdf_stats(args.en_pdf, 'en')
        stats_fr = get_pdf_stats(args.fr_pdf, 'fr')
    else:
        stats_en, stats_fr = scan_chroma_db_bilingual(args.db_path)
        if stats_en is None:
            return

    # Generate the requested comparative DataFrame
    out_df = generate_comparison_table(stats_en, stats_fr)

    # Save to CSV
    out_df.to_csv(args.out, index=False)

    # Print the table to terminal matching the screenshot
    print("\n" + "=" * 80)
    print(out_df.to_string(index=False, justify='left'))
    print("=" * 80 + "\n")
    print(f"Saved comparative data to {args.out}")


if __name__ == '__main__':
    main()
    generate_system_table()