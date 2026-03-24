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
