"""
Query rewriting module for Agentic RAG.
Rewrites vague or short queries into precise, retrieval-optimised questions.
"""

from llama_index.llms.ollama import Ollama


REWRITE_PROMPT = {
    "en": """You are a query rewriting assistant for a university collective agreement 
document retrieval system.

Rewrite the following query to be more specific and precise so it retrieves the 
most relevant clauses or articles from the agreement. Output ONLY the rewritten 
query — do not answer it or explain yourself.

Original query: {query}
Rewritten query:""",

    "fr": """Tu es un assistant de réécriture de requêtes pour un système de 
récupération de convention collective universitaire.

Réécris la requête suivante pour qu'elle soit plus précise et récupère les 
clauses ou articles les plus pertinents. Retourne UNIQUEMENT la requête réécrite 
— ne réponds pas à la question.

Requête originale: {query}
Requête réécrite:"""
}


def rewrite_query(query: str, lang: str, llm: Ollama) -> str:
    """
    Rewrite a user query to be more retrieval-friendly.

    Args:
        query: Original user query
        lang:  Detected language code ('en' or 'fr')
        llm:   Shared Ollama LLM instance from BilingualRAGEngine

    Returns:
        Rewritten query string (falls back to original if rewrite is empty)
    """
    prompt = REWRITE_PROMPT.get(lang, REWRITE_PROMPT["en"]).format(query=query)
    try:
        response = llm.complete(prompt)
        rewritten = str(response).strip()
        # Reject rewrite if it's empty or suspiciously long (model went off-script)
        if rewritten and len(rewritten) < 400:
            return rewritten
    except Exception as e:
        print(f"[QueryRewriter] Rewrite failed: {e}")
    return query  # safe fallback
def rewrite_single_question(query: str, lang: str, llm: Ollama) -> str:
    """
    Standalone version of rewrite_query for use with standalone modules.
    This provides a consistent interface for the modular RAG pipeline.

    Args:
        query: Original user query
        lang:  Detected language code ('en' or 'fr')
        llm:   Shared Ollama LLM instance from BilingualRAGEngine

    Returns:
        Rewritten query string (falls back to original if rewrite is empty)
    """
    return rewrite_query(query, lang, llm)
