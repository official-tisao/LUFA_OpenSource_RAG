"""
FastAPI REST API for LUFA Bilingual Agentic RAG System.
Timeout configured for LLM inference which can take 30–120s per query.

Start with:  python src/api.py
Test with:   curl -X POST http://localhost:8000/agentic-query \
               -H "Content-Type: application/json" \
               -d '{"query":"What is the salary grid for 2024?","return_sources":true}'
"""

import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from rag_engine import BilingualRAGEngine
from config_loader import cfg

LLM_MODEL     = cfg("models.llm.name")
EMBED_MODEL   = cfg("models.embedding.name")
DB_PATH       = cfg("database.path")
COLLECTION    = cfg("database.collection_name")
DEFAULT_TOP_K = cfg("retrieval.top_k")

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


# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query:          str  = Field(..., example="What is the salary grid for 2024?")
    return_sources: bool = Field(True)
    top_k:          int  = Field(5, ge=1, le=20)

class AgenticQueryRequest(BaseModel):
    query:          str  = Field(..., example="What are the office hours requirements?")
    return_sources: bool = Field(True)
    top_k:          int  = Field(5, ge=1, le=20)
    max_retries:    int  = Field(3, ge=1, le=5)

class CopilotQueryRequest(BaseModel):
    query:          str  = Field(..., example="What is the academic freedom policy?")
    model:          str  = Field("gpt-4o", description="GitHub Models model ID")
    return_sources: bool = Field(True)
    top_k:          int  = Field(5, ge=1, le=20)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy" if engine_instance else "degraded",
        "engine_loaded": engine_instance is not None,
        "llm_model": LLM_MODEL,
        "embedding_model": EMBED_MODEL,
    }

@app.get("/models", tags=["System"])
async def list_models():
    return {"llm_model": LLM_MODEL, "embedding_model": EMBED_MODEL,
            "db_path": DB_PATH, "collection": COLLECTION}

@app.post("/query", tags=["Standard RAG"])
async def standard_query(req: QueryRequest):
    """Single-pass RAG query — responds in query language."""
    if not engine_instance:
        raise HTTPException(503, "RAG engine not initialized.")
    try:
        engine_instance.set_similarity_top_k(req.top_k)
        return engine_instance.query(req.query, return_sources=req.return_sources)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/agentic-query", tags=["Agentic RAG"])
async def agentic_query(req: AgenticQueryRequest):
    """
    Full Agentic RAG: translate → rewrite → retrieve → generate → reflect → re-retrieve.
    Set a high client timeout (≥ 300s) because the agent loop may run 3 LLM passes.
    """
    if not engine_instance:
        raise HTTPException(503, "RAG engine not initialized.")
    try:
        engine_instance.set_similarity_top_k(req.top_k)
        return engine_instance.agentic_query(
            query_text=req.query,
            return_sources=req.return_sources,
            max_retries=req.max_retries,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/copilot-query", tags=["Frontier Models"])
async def copilot_query(req: CopilotQueryRequest):
    """
    Retrieve locally with ChromaDB, then generate with a GitHub Models frontier model.
    Requires GITHUB_TOKEN environment variable.
    """
    if not engine_instance:
        raise HTTPException(503, "RAG engine not initialized.")
    try:
        from copilot_engine import CopilotEngine
        engine_instance.set_similarity_top_k(req.top_k)
        nodes = engine_instance._retrieve_nodes(req.query, top_k=req.top_k)
        copilot = CopilotEngine(model=req.model)
        lang    = engine_instance.detect_query_language(req.query)
        answer  = copilot.generate_from_nodes(req.query, nodes, lang)
        result  = {"response": answer, "model": req.model,
                   "detected_language": lang, "query": req.query}
        if req.return_sources:
            result["sources"] = [
                {"text": n.node.text[:200] + "...", "score": n.score,
                 "metadata": n.node.metadata, "id": n.node.node_id}
                for n in nodes
            ]
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        timeout_keep_alive=600,   # 10 min — LLM agent loop can be slow
    )