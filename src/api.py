

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
