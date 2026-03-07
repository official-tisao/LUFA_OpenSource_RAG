"""
Reflection module for Agentic RAG.
Checks whether a generated answer is grounded in the retrieved source chunks.
"""

from llama_index.llms.ollama import Ollama


REFLECT_PROMPT = """You are a grounding verification assistant for a legal document 
question-answering system.

Given the answer and the retrieved source chunks below, decide whether EVERY claim 
in the answer is directly supported by the source chunks.

Answer:
{answer}

Retrieved source chunks:
{chunks}

Reply with ONLY one word — either GROUNDED or UNGROUNDED."""


def reflect(answer: str, chunks: list[str], llm: Ollama) -> bool:
    """
    Check whether the generated answer is grounded in retrieved chunks.

    Args:
        answer: Generated answer text
        chunks: List of retrieved text chunks
        llm:    Shared Ollama LLM instance from BilingualRAGEngine

    Returns:
        True  → answer is grounded (safe to return to user)
        False → answer is not grounded (trigger re-retrieval)
    """
    if not chunks:
        return False

    chunks_text = "\n\n---\n\n".join(chunks[:5])
    prompt = REFLECT_PROMPT.format(answer=answer, chunks=chunks_text)

    try:
        result = str(llm.complete(prompt)).strip().upper()
        tokens = result.split()
        return bool(tokens) and tokens[0] == "GROUNDED"
    except Exception as e:
        print(f"[Reflector] Reflection failed: {e}")
        return True  # fail-open: don't loop forever on LLM errors
