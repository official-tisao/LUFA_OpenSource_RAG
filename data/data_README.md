| University-of-Sudbury-Collective-Agreement-2015-2018-FinalVersion.pdf | sudbury_ca_2015_2018_bilingual | 2018 | 0.7550 |
| University-of-Sudbury-Collective-Agreement-2018-2021-FinalVersion.pdf | sudbury_ca_2018_2021_bilingual | 2021 | 0.8600 |

### Commands

```bash
python src/side_by_side_clause_chunker.py \
  "data/english_and_french/LUFA-Collective-Agreement-2017-2020-FINAL-Feb-8.pdf" \
  --doc-id lufa_ca_2017_2020_bilingual \
  --out-prefix output/bilingual/lufa_ca_2017_2020_bilingual

python src/side_by_side_clause_chunker.py \
  "data/english_and_french/LUFA-Collective-Agreement-July-2008-June-2011-FINAL.pdf" \
  --doc-id lufa_ca_2008_2011_bilingual \
  --out-prefix output/bilingual/lufa_ca_2008_2011_bilingual

python src/side_by_side_clause_chunker.py \
  "data/english_and_french/LUFA-Collective-Agreement-July-2011-June-2014-FINAL.pdf" \
  --doc-id lufa_ca_2011_2014_bilingual \
  --out-prefix output/bilingual/lufa_ca_2011_2014_bilingual

python src/side_by_side_clause_chunker.py \
  "data/english_and_french/University-of-Sudbury-Collective-Agreement-2015-2018-FinalVersion.pdf" \
  --doc-id sudbury_ca_2015_2018_bilingual \
  --out-prefix output/bilingual/sudbury_ca_2015_2018_bilingual

python src/side_by_side_clause_chunker.py \
  "data/english_and_french/University-of-Sudbury-Collective-Agreement-2018-2021-FinalVersion.pdf" \
  --doc-id sudbury_ca_2018_2021_bilingual \
  --out-prefix output/bilingual/sudbury_ca_2018_2021_bilingual
```

### Batch command

```bash
mkdir -p output/bilingual
for pdf in data/english_and_french/*.pdf; do
  doc_id=$(basename "$pdf" .pdf | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
  python src/side_by_side_clause_chunker.py "$pdf" \
    --doc-id "${doc_id}" --out-prefix "output/bilingual/${doc_id}"
done
```

### Chunk schema (bilingual output)

| Field | Description |
|---|---|
| `chunk_id` | `{doc_id}_p{page}_{lang}_{index}` |
| `language` | `en` or `fr` |
| `clause_no` | Detected clause number e.g. `12.3` |
| `title` | Detected clause title |
| `text` | Chunk body text |
| `partner_chunk_id` | Aligned translation chunk on the same page |
| `end_year` | Latest year from filename |
| `recency_weight` | Normalised 0.30–1.00 |

Filter to one language in Python:

```python
import pandas as pd
df = pd.read_csv("output/bilingual/lufa_ca_2017_2020_bilingual.csv")
en_df = df[df["language"] == "en"]
fr_df = df[df["language"] == "fr"]
```

---

## Process All Documents — Full Pipeline

```bash
mkdir -p output/en output/fr output/bilingual

# English-only
for pdf in data/english/*.pdf; do
  doc_id=$(basename "$pdf" .pdf | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | sed 's/(1)//')
  python src/clause_chunker.py "$pdf" \
    --doc-id "${doc_id}" --out-prefix "output/en/${doc_id}"
done

# French-only
for pdf in data/french/*.pdf; do
  doc_id=$(basename "$pdf" .pdf | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
  python src/clause_chunker.py "$pdf" \
    --doc-id "${doc_id}" --out-prefix "output/fr/${doc_id}"
done

# Side-by-side bilingual
for pdf in data/english_and_french/*.pdf; do
  doc_id=$(basename "$pdf" .pdf | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
  python src/side_by_side_clause_chunker.py "$pdf" \
    --doc-id "${doc_id}" --out-prefix "output/bilingual/${doc_id}"
done

# Rebuild ChromaDB index
python src/ingestion.py
```

---

## Overriding Corpus Year Boundaries

If you add a document older than 2005 or newer than 2025, pass updated boundaries so weights stay consistent across the whole corpus:

```bash
python src/clause_chunker.py "data/english/some-new-2026-agreement.pdf" \
  --doc-id new_2026_en \
  --out-prefix output/en/new_2026_en \
  --min-year 2005 --max-year 2026
```

---

## Sample Documents

- `english/sample_ai_document.txt` — English test document
- `french/sample_ai_document.txt` — French test document

Feel free to delete these and substitute your own. The ingestion script uses `LlamaIndex SimpleDirectoryReader` to discover all supported files in each subdirectory.

---

## Adding Your Own Documents

1. **English-only PDFs** → `data/english/` → `clause_chunker.py`
2. **French-only PDFs** → `data/french/` → `clause_chunker.py`
3. **Side-by-side bilingual PDFs** → `data/english_and_french/` → `side_by_side_clause_chunker.py`
4. Run `python src/ingestion.py` to rebuild ChromaDB
