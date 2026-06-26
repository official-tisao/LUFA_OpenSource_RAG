import csv
import random
import re
from pathlib import Path

random.seed(42)

INPUT_FILE = "all_corpus_text.txt"
OUTPUT_FILE = "generated_test_questions.csv"

EN_START = 1
EN_END = 84754
FR_START = 84755
FR_END = 149635

EN_TARGET = 500
FR_TARGET = 500

EN_Q_TEMPLATES = [
    "What does the agreement say about {topic}?",
    "How is {topic} addressed in the collective agreement?",
    "What are the rules concerning {topic}?",
    "What does the collective agreement provide regarding {topic}?",
    "How does the agreement define or describe {topic}?",
    "What conditions apply to {topic}?",
    "What is stated about {topic}?",
    "According to the agreement, what applies to {topic}?",
]

FR_Q_TEMPLATES = [
    "Que dit la convention collective au sujet de {topic} ?",
    "Comment la convention collective traite-t-elle de {topic} ?",
    "Quelles sont les règles concernant {topic} ?",
    "Que prévoit la convention collective concernant {topic} ?",
    "Comment la convention définit-elle ou décrit-elle {topic} ?",
    "Quelles conditions s’appliquent à {topic} ?",
    "Qu’est-il indiqué au sujet de {topic} ?",
    "Selon la convention, qu’est-ce qui s’applique à {topic} ?",
]

EN_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "shall", "will", "have", "has", "are", "been",
    "may", "any", "all", "not", "such", "their", "they", "them", "his", "her", "its", "but", "into",
    "where", "when", "what", "which", "under", "between", "among", "each", "every", "there", "than",
    "faculty", "university", "agreement", "collective", "article", "clause", "section", "member",
    "members", "employee", "employees"
}

FR_STOP = {
    "les", "des", "une", "un", "dans", "pour", "avec", "que", "qui", "sur", "par", "est", "sont", "ont",
    "être", "été", "aux", "ses", "leurs", "leur", "tout", "toute", "tous", "toutes", "ainsi", "entre",
    "selon", "comme", "plus", "moins", "convention", "collective", "université", "article", "clause",
    "section", "membre", "membres", "employé", "employés", "faculté"
}

EN_CATEGORY_KEYWORDS = {
    "Salary": [
        "salary", "salaries", "wage", "wages", "pay", "paid", "stipend",
        "increment", "increments", "allowance", "allowances", "compensation",
        "hourly rate", "floor", "starting salary"
    ],
    "Promotion": [
        "promotion", "promoted", "promotion procedures", "associate professor",
        "full professor", "associate librarian", "full librarian", "rank"
    ],
    "Tenure": [
        "tenure", "tenured", "probationary", "probation", "tenure-track",
        "tenure evaluation"
    ],
    "Grievance": [
        "grievance", "arbitration", "arbitrator", "joint grievance committee",
        "complaint", "dispute", "grievance procedure"
    ],
    "Leave": [
        "leave", "sabbatical", "absence", "vacation", "maternity", "parental",
        "medical leave", "sick leave", "study leave", "compassionate leave"
    ],
    "Benefits": [
        "benefit", "benefits", "insurance", "pension", "health", "dental",
        "tuition exemption", "reimbursement", "library card"
    ],
    "Workload": [
        "workload", "teaching load", "credits", "course load", "hours of work",
        "overtime", "assignment of duties", "office hours", "overload"
    ],
    "Academic Freedom": [
        "academic freedom", "freedom of discussion", "freedom", "classroom"
    ],
    "Appointments": [
        "appointment", "appointments", "appointment and renewal", "hiring",
        "advertised", "letter of appointment", "search committee",
        "appointee", "reappointment"
    ],
    "Discipline": [
        "discipline", "dismissal", "suspension", "just cause", "penalty",
        "disciplinary"
    ],
    "Research": [
        "research", "scholarly activity", "scholarship", "publication",
        "conference", "creative activity", "research fund"
    ],
    "Teaching": [
        "teaching", "course", "students", "grading", "instruction",
        "evaluation methods", "exam", "syllabus"
    ],
    "Seniority": [
        "seniority", "years of service", "priority of hire", "recall rights"
    ],
    "Termination": [
        "termination", "redundancy", "financial exigency", "lay-off",
        "severance", "non-renewal", "dismissal procedures"
    ],
    "Health and Safety": [
        "health and safety", "safe", "safety", "security", "workplace"
    ],
    "Intellectual Property": [
        "intellectual property", "patents", "copyright", "proprietary",
        "personal notes", "course material"
    ],
    "Union Recognition": [
        "union", "association", "bargaining agent", "lufa", "rights and privileges of the union"
    ],
    "Bargaining Unit": [
        "bargaining unit", "recognition of the bargaining unit", "unit"
    ],
    "Bilingualism": [
        "bilingual", "bilingualism", "french", "english", "official languages", "translation"
    ],
    "Professional Development": [
        "professional development", "training", "workshop", "conference"
    ],
    "Governance": [
        "governance", "committee", "senate", "administrative duties"
    ],
    "General Provisions": [
        "purpose", "scope", "application", "rights", "responsibilities", "duties",
        "definitions", "official languages"
    ]
}

FR_CATEGORY_KEYWORDS = {
    "Salaire": [
        "salaire", "salaires", "traitement", "rémunération", "paie", "taux horaire",
        "allocation", "indemnité", "augmentation", "échelon", "salaire de départ"
    ],
    "Promotion": [
        "promotion", "procédure de promotion", "professeur agrégé",
        "professeur titulaire", "rang"
    ],
    "Permanence": [
        "permanence", "titularisation", "titulaire", "probation", "probatoire",
        "évaluation de permanence"
    ],
    "Grief": [
        "grief", "arbitrage", "arbitre", "plainte", "différend"
    ],
    "Congés": [
        "congé", "congés", "sabbatique", "absence", "vacances", "maladie",
        "parental", "maternité", "adoption"
    ],
    "Avantages sociaux": [
        "avantages sociaux", "assurance", "pension", "retraite", "santé",
        "dentaire", "remboursement"
    ],
    "Charge de travail": [
        "charge de travail", "charge d'enseignement", "heures de travail",
        "heures supplémentaires", "attribution des tâches", "crédits"
    ],
    "Liberté académique": [
        "liberté académique", "liberté", "discussion", "salle de classe"
    ],
    "Nominations": [
        "nomination", "nominations", "renouvellement", "lettre de nomination",
        "comité de sélection", "réaffectation"
    ],
    "Discipline": [
        "discipline", "congédiement", "suspension", "cause juste"
    ],
    "Recherche": [
        "recherche", "activité savante", "publication", "érudition",
        "conférence", "activité créatrice"
    ],
    "Enseignement": [
        "enseignement", "cours", "étudiants", "notes", "évaluation",
        "examen", "plan de cours"
    ],
    "Ancienneté": [
        "ancienneté", "années de service", "priorité", "rappel"
    ],
    "Résiliation": [
        "résiliation", "redondance", "exigence financière", "mise à pied",
        "indemnité de départ", "non-renouvellement"
    ],
    "Santé et sécurité": [
        "santé et sécurité", "sécurité", "milieu de travail"
    ],
    "Propriété intellectuelle": [
        "propriété intellectuelle", "brevets", "droit d’auteur", "notes personnelles", "matériel de cours"
    ],
    "Reconnaissance syndicale": [
        "syndicat", "association", "agent négociateur", "lufa"
    ],
    "Unité de négociation": [
        "unité de négociation", "unité"
    ],
    "Bilinguisme": [
        "bilingue", "bilinguisme", "français", "anglais", "langues officielles", "traduction"
    ],
    "Perfectionnement professionnel": [
        "perfectionnement professionnel", "formation", "atelier", "conférence"
    ],
    "Gouvernance": [
        "gouvernance", "comité", "sénat", "fonctions administratives"
    ],
    "Dispositions générales": [
        "objet", "portée", "application", "droits", "responsabilités", "devoirs",
        "définitions", "langues officielles"
    ]
}

EN_ARTICLE_CATEGORY_HINTS = {
    "official languages": "Bilingualism",
    "bilingualism": "Bilingualism",
    "purpose of the collective agreement": "General Provisions",
    "definitions": "General Provisions",
    "recognition of the bargaining unit": "Bargaining Unit",
    "rights and privileges of the union": "Union Recognition",
    "academic freedom": "Academic Freedom",
    "health, safety and security": "Health and Safety",
    "appointment and renewal": "Appointments",
    "employment equity": "Appointments",
    "academic workload": "Workload",
    "tenure evaluation procedures": "Tenure",
    "promotion procedures": "Promotion",
    "discipline": "Discipline",
    "dismissal procedures": "Termination",
    "resignation": "Termination",
    "retirement": "Termination",
    "research and creativity": "Research",
    "professional development expenditures": "Professional Development",
    "recognized and other holidays": "Leave",
    "vacation": "Leave",
    "absence - general": "Leave",
    "sabbatical leaves": "Leave",
    "study leaves": "Leave",
    "physical and mental illness": "Leave",
    "pregnancy, parental and adoption leaves": "Leave",
    "salary structure": "Salary",
    "benefits": "Benefits",
    "grievance procedure": "Grievance",
    "arbitration procedure": "Grievance",
    "intellectual property": "Intellectual Property",
    "patents": "Intellectual Property",
    "copyright": "Intellectual Property"
}

FR_ARTICLE_CATEGORY_HINTS = {
    "langues officielles": "Bilinguisme",
    "bilinguisme": "Bilinguisme",
    "objet de la convention collective": "Dispositions générales",
    "définitions": "Dispositions générales",
    "reconnaissance de l’unité de négociation": "Unité de négociation",
    "reconnaissance de l'unité de négociation": "Unité de négociation",
    "liberté académique": "Liberté académique",
    "santé et sécurité": "Santé et sécurité",
    "nomination et renouvellement": "Nominations",
    "charge de travail": "Charge de travail",
    "évaluation de permanence": "Permanence",
    "procédures de promotion": "Promotion",
    "discipline": "Discipline",
    "résiliation": "Résiliation",
    "recherche": "Recherche",
    "perfectionnement professionnel": "Perfectionnement professionnel",
    "congés": "Congés",
    "structure salariale": "Salaire",
    "avantages sociaux": "Avantages sociaux",
    "grief": "Grief",
    "arbitrage": "Grief",
    "propriété intellectuelle": "Propriété intellectuelle",
    "brevets": "Propriété intellectuelle",
    "droit d’auteur": "Propriété intellectuelle",
    "droit d'auteur": "Propriété intellectuelle"
}

ARTICLE_RE = re.compile(
    r'^\s*ARTICLE\s+(\d+(?:\.\d+)*)\s*[-–—:\.]?\s*(.+?)\s*$',
    re.IGNORECASE
)

ARTICLE_INLINE_RE = re.compile(
    r'\bARTICLE\s+(\d+(?:\.\d+)*)\s*[-–—:\.]?\s*([A-ZÀ-Ÿ][A-ZÀ-Ÿ0-9 ,/&\-\']{4,})',
    re.IGNORECASE
)


def clean_line(line):
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    return line


def detect_language(text):
    fr_markers = [" le ", " la ", " les ", " des ", " une ", " un ", " de ", " et ", " est ", " au "]
    en_markers = [" the ", " and ", " of ", " to ", " is ", " are ", " with ", " for "]
    t = f" {text.lower()} "
    fr_score = sum(m in t for m in fr_markers)
    en_score = sum(m in t for m in en_markers)
    return "French" if fr_score > en_score else "English"


def normalize_article_title(title):
    title = re.sub(r"\s+", " ", title).strip(" -–—:.")
    return title


def detect_article_heading(line):
    line = clean_line(line)
    m = ARTICLE_RE.match(line)
    if m:
        return {
            "article_number": m.group(1).strip(),
            "article_title": normalize_article_title(m.group(2))
        }

    m2 = ARTICLE_INLINE_RE.search(line)
    if m2:
        return {
            "article_number": m2.group(1).strip(),
            "article_title": normalize_article_title(m2.group(2))
        }

    return None


def reconstruct_paragraphs_with_articles(lines, expected_language):
    blocks = []
    current_lines = []
    current_article_number = None
    current_article_title = None

    def flush_block():
        nonlocal current_lines
        if not current_lines:
            return
        block = " ".join(current_lines).strip()
        block = re.sub(r"\s+", " ", block)
        if len(block) > 80 and detect_language(block) == expected_language:
            blocks.append({
                "text": block,
                "article_number": current_article_number,
                "article_title": current_article_title
            })
        current_lines = []

    for raw in lines:
        line = clean_line(raw)
        if not line:
            flush_block()
            continue

        article = detect_article_heading(line)
        if article:
            flush_block()
            current_article_number = article["article_number"]
            current_article_title = article["article_title"]
            continue

        short_noise = (
                len(line) < 25 or
                re.fullmatch(r"page\s+\d+", line.lower()) or
                re.fullmatch(r"\d+", line)
        )

        hard_boundary = (
                short_noise or
                re.search(r"\bPage\s+\d+\b", line) or
                re.match(r"^(APPENDIX|Appendix|ANNEXE|SCHEDULE|TABLE OF CONTENTS|INTRODUCTION)\b", line)
        )

        if hard_boundary:
            flush_block()
            continue

        current_lines.append(line)

        if len(current_lines) >= 8:
            flush_block()

    flush_block()
    return blocks


def split_sentences(text):
    parts = re.split(r'(?<=[\.\?!;:])\s+', text)
    out = []
    for p in parts:
        p = p.strip()
        p = re.sub(r"\s+", " ", p)
        if 45 <= len(p) <= 450:
            out.append(p)
    return out


def extract_topic(text, language, article_title=None):
    if article_title and len(article_title.split()) <= 7:
        return article_title.lower()

    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-']{3,}", text.lower())
    stop = FR_STOP if language == "French" else EN_STOP
    words = [w for w in words if w not in stop]

    if not words:
        return "this matter" if language == "English" else "cette question"

    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    top = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))[:4]
    phrase = " ".join([w for w, _ in top[:2]])
    return phrase if phrase else ("this matter" if language == "English" else "cette question")


def classify_difficulty(answer):
    wc = len(answer.split())
    if wc < 18:
        return "easy"
    elif wc < 35:
        return "medium"
    return "hard"


def category_from_article(article_title, language):
    if not article_title:
        return None

    title = article_title.lower()
    hint_map = FR_ARTICLE_CATEGORY_HINTS if language == "French" else EN_ARTICLE_CATEGORY_HINTS

    best = None
    best_len = -1
    for hint, category in hint_map.items():
        if hint in title and len(hint) > best_len:
            best = category
            best_len = len(hint)
    return best


def map_category(text, language, article_title=None):
    article_category = category_from_article(article_title, language)
    if article_category:
        return article_category

    text_l = text.lower()
    keyword_map = FR_CATEGORY_KEYWORDS if language == "French" else EN_CATEGORY_KEYWORDS
    fallback = "Dispositions générales" if language == "French" else "General Provisions"

    scores = {}
    for category, keywords in keyword_map.items():
        score = 0
        for kw in keywords:
            if kw.lower() in text_l:
                score += 2 if " " in kw else 1
        if score > 0:
            scores[category] = score

    if scores:
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[0][0]

    return fallback


def make_question(answer, language, article_title=None):
    topic = extract_topic(answer, language, article_title=article_title)
    template = random.choice(FR_Q_TEMPLATES if language == "French" else EN_Q_TEMPLATES)
    q = template.format(topic=topic)
    return q[0].upper() + q[1:] if q else q


def normalize_answer(text):
    text = text.strip(' "\'')
    text = re.sub(r"\s+", " ", text)
    return text[:450].rstrip()


def dedupe_key(question, answer):
    q = re.sub(r"\W+", "", question.lower())
    a = re.sub(r"\W+", "", answer.lower())[:160]
    return q + "|" + a


def load_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f]


def build_candidates(blocks, expected_language):
    candidates = []

    for block in blocks:
        text = block["text"]
        article_number = block["article_number"]
        article_title = block["article_title"]

        sents = split_sentences(text)

        for sent in sents:
            sent = normalize_answer(sent)
            if 50 <= len(sent) <= 300:
                candidates.append({
                    "answer": sent,
                    "article_number": article_number,
                    "article_title": article_title,
                    "language": expected_language
                })

        if len(sents) >= 2:
            for i in range(len(sents) - 1):
                combined = normalize_answer(" ".join(sents[i:i + 2]))
                if 60 <= len(combined) <= 420:
                    candidates.append({
                        "answer": combined,
                        "article_number": article_number,
                        "article_title": article_title,
                        "language": expected_language
                    })

    return candidates


def sample_qa(candidates, n, language, prefix):
    rows = []
    seen = set()
    attempts = 0
    idx = 1

    while len(rows) < n and attempts < n * 100:
        attempts += 1
        item = random.choice(candidates)

        answer = item["answer"]
        article_number = item["article_number"]
        article_title = item["article_title"]

        question = make_question(answer, language, article_title=article_title)
        category = map_category(answer + " " + question, language, article_title=article_title)
        key = dedupe_key(question, answer)

        if key in seen:
            continue

        seen.add(key)
        row_id = f"{prefix}_{idx:03d}"

        rows.append([
            row_id,
            question,
            answer,
            category,
            classify_difficulty(answer),
            language
        ])
        idx += 1

    return rows


def main():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"Could not find {INPUT_FILE}")

    all_lines = load_lines(INPUT_FILE)

    en_lines = all_lines[EN_START - 1:EN_END]
    fr_lines = all_lines[FR_START - 1:FR_END]

    en_blocks = reconstruct_paragraphs_with_articles(en_lines, "English")
    fr_blocks = reconstruct_paragraphs_with_articles(fr_lines, "French")

    en_candidates = build_candidates(en_blocks, "English")
    fr_candidates = build_candidates(fr_blocks, "French")

    if len(en_candidates) < EN_TARGET:
        raise ValueError(f"Not enough English candidates: {len(en_candidates)}")
    if len(fr_candidates) < FR_TARGET:
        raise ValueError(f"Not enough French candidates: {len(fr_candidates)}")

    en_rows = sample_qa(en_candidates, EN_TARGET, "English", "test_en")
    fr_rows = sample_qa(fr_candidates, FR_TARGET, "French", "test_fr")

    rows = en_rows + fr_rows

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "expected_answer", "category", "difficulty", "language"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_FILE}")
    print(f"English blocks: {len(en_blocks)}")
    print(f"French blocks: {len(fr_blocks)}")
    print(f"English candidates: {len(en_candidates)}")
    print(f"French candidates: {len(fr_candidates)}")


if __name__ == "__main__":
    main()