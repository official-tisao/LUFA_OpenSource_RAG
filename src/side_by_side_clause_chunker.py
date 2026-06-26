# python
"""
side_by_side_clause_chunker.py
Extract bilingual side-by-side EN/FR PDFs into clause-level TextNodes and
optionally inject into LlamaIndex/Chroma (local).
Deps: pip install pdfplumber langdetect llama-index-core llama-index-vector-stores-chroma
      llama-index-embeddings-huggingface chromadb pandas
"""
from __future__ import annotations

import re
import json
import uuid
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
from langdetect import detect, DetectorFactory
from llama_index.core.schema import TextNode

DetectorFactory.seed = 42
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CLAUSE_RE = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')
NUM_RE = re.compile(r'^[\d\W]+$')


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    language: str
    clause_no: Optional[str]
    title: Optional[str]
    text: str
    partner_chunk_id: Optional[str] = None
    x0: Optional[float] = None
    x1: Optional[float] = None
    y0: Optional[float] = None
    y1: Optional[float] = None


def clean_text(s: str) -> str:
    s = s.replace('\u00a0', ' ').replace('\u200b', ' ')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


def extract_words_split(page) -> Tuple[List[dict], List[dict]]:
    words = page.extract_words(extra_attrs=["x0", "x1", "top", "bottom"], keep_blank_chars=False,
                               use_text_flow=True) or []
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
            continue
        if buf and (t[0].islower() or t.startswith('—') or t.startswith('-')):
            buf.append(t)
        else:
            if buf:
                merged.append(' '.join(buf))
            buf = [t]
    if buf:
        merged.append(' '.join(buf))
    return [clean_text(x) for x in merged if clean_text(x)]


def parse_clause_header(text: str):
    m = CLAUSE_RE.match(text)
    if m:
        return m.group(1), m.group(2).strip()
    return None, None


def infer_lang_for_column(text: str) -> str:
    sample = (text or "")[:500].strip()
    try:
        return detect(sample) if sample else "en"
    except Exception:
        return "en"


def extract_column_chunks(doc_id: str, page_num: int, language: str, words) -> List[Chunk]:
    lines = cluster_lines(words)
    para_texts = merge_lines(lines)
    chunks: List[Chunk] = []
    running = []
    prev_clause_no = None
    prev_title = None
    idx = 1
    for t in para_texts:
        clause_no, title = parse_clause_header(t)
        if clause_no and running:
            text = clean_text("\n".join(running))
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_p{page_num:03d}_{language}_{idx:03d}",
                doc_id=doc_id,
                page=page_num,
                language=language,
                clause_no=prev_clause_no,
                title=prev_title,
                text=text,
            ))
            idx += 1
            running = []
        prev_clause_no, prev_title = clause_no, title or (t if len(t) < 90 else None)
        running.append(t)
    if running:
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_p{page_num:03d}_{language}_{idx:03d}",
            doc_id=doc_id,
            page=page_num,
            language=language,
            clause_no=prev_clause_no,
            title=prev_title,
            text=clean_text("\n".join(running)),
        ))
    return chunks


def chunk_side_by_side_pdf(pdf_path: str, doc_id: Optional[str] = None) -> List[TextNode]:
    pdf_path = str(pdf_path)
    doc_id = doc_id or Path(pdf_path).stem
    out_chunks: List[Chunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = clean_text(page.extract_text() or "")
            if not page_text:
                continue
            left_words, right_words = extract_words_split(page)
            left_preview = clean_text(" ".join(w['text'] for w in left_words[:40])) if left_words else ''
            right_preview = clean_text(" ".join(w['text'] for w in right_words[:40])) if right_words else ''
            left_lang = infer_lang_for_column(left_preview or page_text[:500])
            right_lang = 'fr' if left_lang == 'en' else 'en'
            left_chunks = extract_column_chunks(doc_id, i, left_lang, left_words)
            right_chunks = extract_column_chunks(doc_id, i, right_lang, right_words)
            # pair by index
            pair_count = min(len(left_chunks), len(right_chunks))
            for k in range(pair_count):
                left_chunks[k].partner_chunk_id = right_chunks[k].chunk_id
                right_chunks[k].partner_chunk_id = left_chunks[k].chunk_id
            out_chunks.extend(left_chunks)
            out_chunks.extend(right_chunks)

    # convert to TextNode with metadata
    nodes: List[TextNode] = []
    for idx, c in enumerate(out_chunks):
        node = TextNode(
            id_=str(uuid.uuid4()),
            text=c.text,
            metadata={
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "page_no": str(c.page),
                "language": c.language,
                "clause_no": c.clause_no or "",
                "title": c.title or "",
                "partner_chunk_id": c.partner_chunk_id or "",
                "doc_source": doc_id,
            },
        )
        # hide low-level fields from embedding by listing keys (kept for llama-index compatibility)
        node.excluded_embed_metadata_keys = ["doc_id", "doc_source", "partner_chunk_id"]
        node.excluded_llm_metadata_keys = ["doc_id"]
        nodes.append(node)
    logger.info("Produced %d bilingual chunks from %s", len(nodes), pdf_path)
    return nodes


def index_nodes_to_chroma(nodes: List[TextNode], collection_name: str, chroma_persist_dir: str = "./chroma_db",
                          model_device: str = "cuda"):
    try:
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        raise ImportError(
            "Missing optional deps for indexing. Install: chromadb llama-index-vector-stores-chroma llama-index-embeddings-huggingface") from exc

    logger.info("Loading embedding model: BGE-M3 (local) on %s", model_device)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device=model_device, max_length=512)

    chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
    chroma_col = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_col)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes,
        storage_context=storage_ctx,
        embed_model=embed_model,
        show_progress=True,
    )
    logger.info("Indexed %d nodes into collection '%s' (path=%s)", len(nodes), collection_name, chroma_persist_dir)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    p = argparse.ArgumentParser(description="Side-by-side bilingual clause chunker -> LlamaIndex/Chroma")
    p.add_argument("pdf", help="Path to side-by-side bilingual PDF")
    p.add_argument("--doc-id", default=None, help="Document id override")
    p.add_argument("--out-prefix", default=None, help="If set, write <prefix>.csv and .jsonl")
    p.add_argument("--index", action="store_true", help="Also index into Chroma collections")
    p.add_argument("--collection", default="lufa_bilingual", help="Chroma collection name")
    p.add_argument("--chroma-dir", default="./chroma_db", help="Chroma persistent directory")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Embedding device")
    args = p.parse_args()

    nodes = chunk_side_by_side_pdf(args.pdf, args.doc_id)

    if args.out_prefix:
        rows = [{"chunk_id": n.metadata.get("chunk_id"), "text": n.text, **n.metadata} for n in nodes]
        Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(f"{args.out_prefix}.csv", index=False)
        with open(f"{args.out_prefix}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote outputs: %s.csv / %s.jsonl", args.out_prefix, args.out_prefix)

    if args.index:
        index_nodes_to_chroma(nodes, args.collection, chroma_persist_dir=args.chroma_dir, model_device=args.device)

    print(f"Done. Produced {len(nodes)} chunks.")
