"""
FastAPI REST API for LUFA Bilingual Agentic RAG System.
Timeout configured for LLM inference which can take 30–120s per query.

Start with:  python src/api.py
Test with:   curl -X POST http://localhost:8000/agentic-query \
               -H "Content-Type: application/json" \
               -d '{"query":"What is the salary grid for 2024?","return_sources":true}'
"""

import sys, yaml
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from rag_engine import BilingualRAGEngine


def load_config(path: str = "config/config.yaml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

_cfg          = load_config()
LLM_MODEL     = _cfg.get("models", {}).get("llm",       {}).get("name", "llama3.2:3b-instruct-q4_K_M")
EMBED_MODEL   = _cfg.get("models", {}).get("embedding", {}).get("name", "nomic-embed-text-v2-moe")
DB_PATH       = _cfg.get("database",  {}).get("path",       "db/chroma_db")
COLLECTION    = _cfg.get("database",  {}).get("collection",  "multilingual_docs")
DEFAULT_TOP_K = _cfg.get("retrieval", {}).get("top_k", 5)

engine_instance: Optional[BilingualRAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_instance
    print("[API] Loading RAG engine...")
    engine_instance = BilingualRAGEngine(
        db_path=DB_PATH, collection_name=COLLECTION,
        llm_model=LLM_MODEL, embedding_model=EMBED_MODEL,
        similarity_top_k=DEFAULT_TOP_K,
    )
    print("[API] Ready.")
    yield
    print("[API] Shutdown.")


app = FastAPI(
    title="LUFA Bilingual Agentic RAG API",
    description="REST API for cross-lingual retrieval of LUFA collective agreements.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
