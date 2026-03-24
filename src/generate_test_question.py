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
    "the","and","for","with","that","this","from","shall","will","have","has","are","been",
    "may","any","all","not","such","their","they","them","his","her","its","but","into",
    "where","when","what","which","under","between","among","each","every","there","than",
    "faculty","university","agreement","collective","article","clause","section","member",
    "members","employee","employees"
}

FR_STOP = {
    "les","des","une","un","dans","pour","avec","que","qui","sur","par","est","sont","ont",
    "être","été","aux","ses","leurs","leur","tout","toute","tous","toutes","ainsi","entre",
    "selon","comme","plus","moins","convention","collective","université","article","clause",
    "section","membre","membres","employé","employés","faculté"
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
