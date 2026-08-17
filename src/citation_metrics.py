#!/usr/bin/env python3
"""
citation_metrics.py — citation-accuracy scoring for the LUFA RAG evaluation
(Ch4 §4.6.3 rubric, §4.8.4 computation).

Two independent scores on the SAME 1 / 0.5 / 0 rubric:

  * citation_accuracy_regex  — DETERMINISTIC. Parses the article number and clause
    identifier from the generated answer with regexes, compares them to the gold
    citation parsed from the ground-truth provision text. Fully objective; matches
    Ch4 §4.8.4's "regular expression identifier" description.

  * citation_accuracy_judge  — LLM-JUDGE. A dedicated prompt asks the judge model
    to grade the answer's citation against the gold article/clause. Uses whatever
    judge model the evaluator is running (get_llm_client). Meant to be reviewed and
    corrected by hand afterwards.

Rubric (Ch4 §4.6.3):
  1   = article number AND clause identifier both correct
  0.5 = article correct but clause wrong/missing, OR a correct secondary clause
        while the primary is missed
  0   = cited article/clause not the correct one, or no citation at all

`gold_text` is the `ground_source_truth` column of the benchmark, e.g.
"ARTICLE 7.20 – ABSENCE - GENERAL\n7.20.1 The University ...", from which the gold
article ("7") and the most specific clause id ("7.20.1") are extracted.
"""

import re

__all__ = [
    "extract_citations", "extract_gold_citation",
    "citation_accuracy_regex", "citation_accuracy_judge",
    "parse_citation_score", "CITATION_JUDGE_PROMPT",
]

# "Article 7", "article no. 7", "art. 7", "articles 7" → article number 7.
# Multilingual: EN/FR "article", DE "Artikel" — answers in the cross-lingual run are
# written in the question's language, so the article keyword may not be English.
_ARTICLE_RE = re.compile(
    r'\b(?:article[s]?|artikel|art)\.?\s*(?:no\.?|n[o°]\.?|nr\.?|#)?\s*(\d{1,3})\b',
    re.IGNORECASE)
# Clause-style references in EN / FR / DE
_CLAUSE_WORD_RE = re.compile(
    r'\b(?:clause|section|paragraph[e]?|abschnitt|absatz|ziffer)\s*(\d{1,3}(?:\.\d+)*)\b',
    re.IGNORECASE)
# Dotted clause identifiers: 7.20, 7.20.1, 7.20.1(a) / 7.20.1(a)
_DOTTED_ID_RE = re.compile(r'\b(\d{1,3}(?:\.\d+)+(?:\s*\([a-zA-Z0-9]+\))?)')


def _norm_clause(cid: str) -> str:
    """Normalise a clause id: strip spaces, lowercase parenthetical letters."""
    return re.sub(r'\s+', '', str(cid)).lower().strip(' .')


def extract_citations(text):
    """
    Return (articles:set[str], clauses:set[str]) found in `text`.

    Articles are the bare integer identifiers (e.g. "7"); clauses are the dotted
    identifiers (e.g. "7.20.1"). Every clause's leading integer is also added to
    the article set, so "7.20.1" implies article "7".
    """
    text = str(text or "")
    articles = {m.group(1) for m in _ARTICLE_RE.finditer(text)}
    clauses = {_norm_clause(m.group(1)) for m in _DOTTED_ID_RE.finditer(text)}
    for m in _CLAUSE_WORD_RE.finditer(text):
        val = _norm_clause(m.group(1))
        if "." in val:
            clauses.add(val)
        else:
            articles.add(val)
    for c in list(clauses):
        head = c.split(".")[0]
        if head.isdigit():
            articles.add(head)
    return articles, clauses


def extract_gold_citation(gold_text):
    """
    Extract (gold_article, gold_clause) from a ground-truth provision string.

    gold_article = the smallest explicit article integer (typically the header,
    e.g. "7"). gold_clause = the MOST SPECIFIC dotted id (most dot-separated
    components), e.g. "7.20.1" over "7.20". Either may be "" when not present.
    """
    articles, clauses = extract_citations(gold_text)
    gold_article = sorted(articles, key=lambda a: (len(a), a))[0] if articles else ""
    gold_clause = max(clauses, key=lambda c: (c.count("."), len(c))) if clauses else ""
    return gold_article, gold_clause


def citation_accuracy_regex(answer, gold_text):
    """
    Deterministic 1 / 0.5 / 0 citation score (Ch4 §4.6.3), or "" when there is no
    gold citation to compare against (cannot be scored).

      - no citation in the answer at all               → 0.0
      - answer clause matches a gold clause            → 1.0
      - answer article matches gold article (no clause)→ 0.5
      - otherwise                                       → 0.0
    """
    g_art, g_cl = extract_gold_citation(gold_text)
    if not g_art and not g_cl:
        return ""  # nothing to grade against
    a_art, a_cl = extract_citations(answer)
    if not a_art and not a_cl:
        return 0.0  # answer includes no citation

    gold_clauses = {c for c in [g_cl] if c}
    gold_articles = {a for a in [g_art] if a}

    clause_match = bool(a_cl & gold_clauses)
    article_match = bool(a_art & gold_articles)

    if clause_match:
        return 1.0
    if article_match:
        return 0.5
    return 0.0


CITATION_JUDGE_PROMPT = """You are grading whether a generated answer cites the CORRECT provision of a collective agreement.

The correct (ground-truth) citation is:
  Article number: {gold_article}
  Clause identifier: {gold_clause}
  Provision text: {gold_text}

The generated answer is:
{answer}

Grade the answer's citation on this rubric and reply with ONLY one number — 1, 0.5, or 0:
  1   = the answer cites BOTH the correct article number AND the correct clause identifier.
  0.5 = the answer cites the correct article number but the wrong or missing clause identifier,
        OR cites a correct secondary clause while missing the primary one.
  0   = the answer cites an article/clause that is not the correct one, or includes no citation at all.

Score:"""


def parse_citation_score(text):
    """Parse a judge reply into 1.0 / 0.5 / 0.0 (defaults 0.0 on no match)."""
    s = str(text)
    if re.search(r'0?\.5', s):
        return 0.5
    m = re.search(r'\b([01])(?:\.0+)?\b', s)
    if m:
        return float(m.group(1))
    return 0.0


def citation_accuracy_judge(llm, answer, gold_text, gold_article=None, gold_clause=None):
    """
    LLM-judge citation score (1/0.5/0). `llm` is a pre-loaded LlamaIndex-style
    client (see model_api_auth.get_llm_client). Returns "" when there is no gold
    citation to grade against. Never raises — returns "" on judge failure.
    """
    if gold_article is None or gold_clause is None:
        gold_article, gold_clause = extract_gold_citation(gold_text)
    if not gold_article and not gold_clause:
        return ""
    try:
        from llm_utils import stream_complete
        prompt = CITATION_JUDGE_PROMPT.format(
            gold_article=gold_article or "(none)",
            gold_clause=gold_clause or "(none)",
            gold_text=str(gold_text or "")[:1200],
            answer=str(answer or "")[:1200],
        )
        raw = stream_complete(llm, prompt)
        return parse_citation_score(raw)
    except Exception as e:
        print(f"      [Citation Judge Warning] {e}")
        return ""


if __name__ == "__main__":
    gold = "ARTICLE 7.20 – ABSENCE - GENERAL\n7.20.1 The University as a community of scholars ..."
    print("gold citation:", extract_gold_citation(gold))
    print("both match:", citation_accuracy_regex("Per clause 7.20.1 of Article 7, ...", gold))
    print("article only:", citation_accuracy_regex("Article 7 covers this.", gold))
    print("wrong:", citation_accuracy_regex("See Article 12.3.4.", gold))
    print("no cite:", citation_accuracy_regex("The university bears the cost.", gold))
    print("no gold:", citation_accuracy_regex("Article 7.", "some text with no citation"))
