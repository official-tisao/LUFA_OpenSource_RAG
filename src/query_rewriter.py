"""
Query rewriting module for Agentic RAG.

The rewriter runs only AFTER a first attempt has failed. On attempt 1 it returns the query
untouched, which makes the agentic pipeline's first pass identical to the naive pipeline and
gives the retry comparison a single variable.

Why context matters. The original prompt received only the query, so the only way it could
make a query "more specific and precise" was to invent specificity: article names and clause
wording that may not exist in the agreement. Invented terms pull both the dense and the BM25
arm away from the real provision. From attempt 2 the rewriter therefore sees what the previous
pass actually found and said.

What it is given, and what it is deliberately NOT given:
  - the ORIGINAL question, never the previous rewrite, so rewrites cannot compound;
  - the previous answer, labelled as having failed the grounding check;
  - the article/clause HEADERS of the previously retrieved chunks, not their bodies.

Headers rather than bodies because chunk bodies are the noisiest artefact in the corpus
(median ~3,000 characters, most spanning more than one article), so five of them would swamp
the prompt. The headers cost ~10 tokens each, carry no prose noise, and do the one job that
matters: they show which provisions exist and what the agreement calls them, so the rewriter
cannot invent an article number.

Passing the previous answer alone would be worse than passing nothing: that answer is the
output the reflector just rejected, so a hallucinated article number in it would become the
rewriter's search target. Hence the explicit instruction not to reuse a number that appears
only in the answer.
"""

import re
from typing import List, Optional, Sequence

from llama_index.llms.ollama import Ollama


# Attempt 1 uses no rewrite at all, so this prompt is only ever reached from attempt 2.
REWRITE_PROMPT = {
    "en": """You are a query rewriting assistant for a university collective agreement
document retrieval system.

A previous search for this question returned passages that did not support a grounded answer.
Rewrite the question so the next search retrieves the correct provision.

Original question: {query}
{context}
Rules:
- Use only wording that appears in the original question or in the provision titles listed above.
- Do NOT invent an article or clause number, and do NOT reuse any number that appears only in
  the rejected answer.
- Do not restate or summarise the rejected answer. Rewrite the QUESTION.
- Output ONLY the rewritten question. Do not answer it and do not explain yourself.

Rewritten question:""",

    "fr": """Tu es un assistant de réécriture de requêtes pour un système de récupération de
convention collective universitaire.

Une recherche précédente pour cette question a retourné des passages qui ne permettaient pas
de fonder une réponse. Réécris la question afin que la prochaine recherche récupère la bonne
disposition.

Question originale : {query}
{context}
Règles :
- N'utilise que des termes présents dans la question originale ou dans les titres de
  dispositions listés ci-dessus.
- N'invente AUCUN numéro d'article ou de clause, et ne réutilise aucun numéro qui n'apparaît
  que dans la réponse rejetée.
- Ne reformule pas et ne résume pas la réponse rejetée. Réécris la QUESTION.
- Retourne UNIQUEMENT la question réécrite. N'y réponds pas et ne t'explique pas.

Question réécrite :"""
}

_CONTEXT_LABELS = {
    "en": {
        "headers": "Provision titles found by the previous search:",
        "answer":  "Answer produced from those passages, which FAILED the grounding check:",
    },
    "fr": {
        "headers": "Titres des dispositions trouvées par la recherche précédente :",
        "answer":  "Réponse produite à partir de ces passages, qui a ÉCHOUÉ à la vérification "
                   "de fondement :",
    },
}

# The answer is only a hint about how the question was misread. More than this is noise, and a
# long rejected answer would dominate the prompt.
ANSWER_PREVIEW_CHARS = 600
MAX_HEADERS = 5
# A rewrite longer than this means the model went off-script and started answering instead.
MAX_REWRITE_CHARS = 400


_SPLIT_SUFFIX = re.compile(r"^(.*?)__p(\d+)$")


def _ident(value) -> str:
    """
    Strip the chunker's internal split marker and reject the "unidentified" sentinel.

    `_split_long_clauses` suffixes continuation pieces with `__p2`, `__p3` and so on
    (clause_chunker.py:362,381). That suffix is bookkeeping, not part of the provision number,
    and must never reach a prompt: "ARTICLE 13__p2" invites the model to cite a provision that
    does not exist. A base of "0" is the chunker's sentinel for text it could not attribute to
    any article (clause_chunker.py:212-213), which is true of 2,754 of the 4,591 chunks in the
    current index, so it yields no identifier at all.
    """
    s = str(value or "").strip()
    m = _SPLIT_SUFFIX.match(s)
    if m:
        s = m.group(1)
    return "" if s in ("", "0") else s


def build_headers(nodes: Sequence) -> List[str]:
    """
    Turn retrieved nodes into one short 'ARTICLE x.y - Title' line each.

    Reads metadata rather than text on purpose: the body is what we are trying to keep out of
    the prompt. `clause_id` is preferred over `article_number` because it is more precise.
    A node with neither contributes its section title alone, or nothing.
    """
    out, seen = [], set()
    for n in nodes:
        meta = getattr(getattr(n, "node", n), "metadata", None) or {}
        title = str(meta.get("section_title", "") or "").strip()
        ident = _ident(meta.get("clause_id")) or _ident(meta.get("article_number"))

        label = f"ARTICLE {ident}" if ident else ""
        if title:
            label = f"{label} - {title}" if label else title
        label = label.strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
        if len(out) >= MAX_HEADERS:
            break
    return out


def _context_block(lang: str, headers: Optional[Sequence[str]],
                   prev_answer: Optional[str]) -> str:
    """
    Assemble the context section. Returns "" when there is nothing to add, which keeps the
    prompt identical to a no-context rewrite.

    The header list is built as its own paragraph so it can be dropped in one place if an
    answer-only variant is ever needed for comparison.
    """
    lab = _CONTEXT_LABELS.get(lang, _CONTEXT_LABELS["en"])
    parts = []

    if headers:
        lines = "\n".join(f"- {h}" for h in headers)
        parts.append(f"{lab['headers']}\n{lines}")

    if prev_answer:
        snippet = " ".join(str(prev_answer).split())[:ANSWER_PREVIEW_CHARS]
        if snippet:
            parts.append(f"{lab['answer']}\n{snippet}")

    return "\n" + "\n\n".join(parts) + "\n" if parts else ""


def rewrite_query(
    query:       str,
    lang:        str,
    llm:         Ollama,
    headers:     Optional[Sequence[str]] = None,
    prev_answer: Optional[str]           = None,
    attempt:     int                     = 1,
) -> str:
    """
    Rewrite a user query to be more retrieval-friendly, using what the last attempt found.

    Args:
        query:       The ORIGINAL user question. Never pass a previous rewrite, or rewrites
                     compound across attempts and drift from what was actually asked.
        lang:        Detected language code ('en' or 'fr').
        llm:         Shared Ollama LLM instance from BilingualRAGEngine.
        headers:     Provision titles from the previous retrieval, via `build_headers`.
        prev_answer: The previous answer, which the reflector judged UNGROUNDED.
        attempt:     1-based attempt number.

    Returns:
        On attempt 1, or with no context supplied, the query unchanged: there is nothing to
        condition a rewrite on, and leaving it alone keeps the first pass identical to the
        naive pipeline. Otherwise the rewritten query, falling back to the input if the model
        returns nothing usable.
    """
    if attempt <= 1:
        return query

    context = _context_block(lang, headers, prev_answer)
    if not context:
        # No context means this would be the old blind rewrite, which is the behaviour we are
        # removing. Leaving the query alone is the safer of the two.
        print("[QueryRewriter] No context available — leaving query unchanged.")
        return query

    prompt = REWRITE_PROMPT.get(lang, REWRITE_PROMPT["en"]).format(
        query=query, context=context
    )
    try:
        from llm_utils import stream_complete
        rewritten = stream_complete(llm, prompt)
        rewritten = (rewritten or "").strip()
        # Reject an empty rewrite, or one long enough that the model started answering.
        if rewritten and len(rewritten) < MAX_REWRITE_CHARS:
            return rewritten
        if rewritten:
            print(f"[QueryRewriter] Rewrite rejected ({len(rewritten)} chars) — using original.")
    except Exception as e:
        print(f"[QueryRewriter] Rewrite failed: {e}")
    return query  # safe fallback


def rewrite_single_question(query: str, lang: str, llm: Ollama) -> str:
    """
    Standalone version for the modular RAG pipeline, which has no attempt loop and therefore
    no previous pass to learn from.

    Kept for interface compatibility. It now returns the query unchanged, because a rewrite
    with no context is exactly the blind rewrite this module no longer performs.
    """
    return rewrite_query(query, lang, llm, attempt=1)
