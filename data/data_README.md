0--# Data Directory

This directory contains collective agreement PDFs organised into three subdirectories by language layout.  Both chunking scripts now embed **recency weights** into every chunk so that retrieval can rank newer agreements higher.

---

## Recency Weighting

Both `clause_chunker.py` and `side_by_side_clause_chunker.py` read the latest 4-digit year from each **filename** (e.g. `…-June-2014-FINAL.pdf` → end year **2014**) and store two extra fields in every chunk:

| Field | Type | Description |
|---|---|---|
| `end_year` | int | Latest year extracted from the filename |
| `recency_weight` | float 0.30–1.00 | Normalised recency score for this document |

### Weight formula

```
raw    = (end_year − CORPUS_MIN_YEAR) / (CORPUS_MAX_YEAR − CORPUS_MIN_YEAR)
weight = 0.30 + 0.70 × raw          # floor 0.30 · ceiling 1.00
```

Default corpus boundaries: `CORPUS_MIN_YEAR = 2005`, `CORPUS_MAX_YEAR = 2025`.
Override them at runtime with `--min-year` / `--max-year` if you add older or newer documents.

### Weight table — all corpus documents

| ['File                                                                 | End year | recency_weight |
|-----------------------------------------------------------------------|---|---|
| LUFA-Collective-Agreement-July-2002-June-2005-FINAL.pdf               | 2005 | 0.3000 |
| LUFA-Collective-Agreement-July-2005-June-2008-FINAL.pdf               | 2008 | 0.4050 |
| LUFA-Collective-Agreement-July-2008-June-2011-FINAL.pdf               | 2011 | 0.5100 |
| LUFA-Collective-Agreement-July-2011-June-2014-FINAL.pdf               | 2014 | 0.6150 |
| Huntington-2012-2016-CA-Lufa-collective-agreement.pdf                 | 2016 | 0.6850 |
| Thorneloe-LUFA-CA-2014-2017-Collective-Agreement-Signed.pdf           | 2017 | 0.7200 |
| University-of-Sudbury-Collective-Agreement-2015-2018-FinalVersion.pdf | 2018 | 0.7550 |
| Huntington-2016-2020-CA-Lufa-collective-agreement.pdf                 | 2020 | 0.8250 |
| Thorneloe-LUFA-CA-2017-2020-Collective-Agreement-Signed.pdf           | 2020 | 0.8250 |
| LUFA-Collective-Agreement-2017-2020-FINAL-Feb-8.pdf                   | 2020 | 0.8250 |
| University-of-Sudbury-Collective-Agreement-2018-2021-FinalVersion.pdf | 2021 | 0.8600 |
| Collective Agreement 2020-2025 English (1).pdf                        | 2025 | 1.0000 |
| Forced Contract 2020-2025 English.pdf                                 | 2025 | 1.0000 |
| Laurentian-University-LUFA-2020-2025.pdf                              | 2025 | 1.0000 |
| LUFA-Term-Sheet-Fully-Executed-Mediation-Term-Sheet-2020-2025.pdf     | 2025 | 1.0000 |
| Contrat imposé 2020-2025 Francais.pdf                                 | 2025 | 1.0000 |
| Convention Collective 2020-2025 Francais.pdf                          | 2025 | 1.0000 |

### Using `recency_weight` in retrieval

When querying ChromaDB, multiply the cosine similarity score by `recency_weight` to boost newer agreements:

```python
combined_score = cosine_similarity * chunk_metadata["recency_weight"]
```

Store `recency_weight` and `end_year` in the Chroma metadata dict at ingestion time.

---

## Directory Structure

```
data/
├── english/                   # English-only PDFs  → clause_chunker.py
├── english_and_french/        # Side-by-side bilingual PDFs → side_by_side_clause_chunker.py
├── french/                    # French-only PDFs   → clause_chunker.py
├── metadata.json
└── README.md
```

### Script selection at a glance

| Directory | Page layout | Script |
|---|---|---|
| `english/` | Single-column English | `src/clause_chunker.py` |
| `french/` | Single-column French | `src/clause_chunker.py` |
| `english_and_french/` | Two parallel columns EN + FR | `src/side_by_side_clause_chunker.py` |

---

## english/ — English-Only Documents

### Files and assigned doc IDs

| File | Doc ID | End year | Weight |
|---|---|---|---|
| Collective Agreement 2020-2025 English (1).pdf | ca_2020_2025_en | 2025 | 1.0000 |
| Forced Contract 2020-2025 English.pdf | forced_contract_2020_2025_en | 2025 | 1.0000 |
| Huntington-2012-2016-CA-Lufa-collective-agreement.pdf | huntington_ca_2012_2016_en | 2016 | 0.6850 |
| Huntington-2016-2020-CA-Lufa-collective-agreement.pdf | huntington_ca_2016_2020_en | 2020 | 0.8250 |
| Laurentian-University-LUFA-2020-2025.pdf | lufa_2020_2025_en | 2025 | 1.0000 |
| LUFA-Collective-Agreement-July-2002-June-2005-FINAL.pdf | lufa_ca_2002_2005_en | 2005 | 0.3000 |
