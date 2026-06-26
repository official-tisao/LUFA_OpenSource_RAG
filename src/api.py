
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