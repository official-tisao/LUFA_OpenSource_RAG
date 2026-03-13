| LUFA-Collective-Agreement-July-2005-June-2008-FINAL.pdf | lufa_ca_2005_2008_en | 2008 | 0.4050 |
| LUFA-Term-Sheet-Fully-Executed-Mediation-Term-Sheet-2020-2025.pdf | lufa_term_sheet_2020_2025_en | 2025 | 1.0000 |
| Thorneloe-LUFA-CA-2014-2017-Collective-Agreement-Signed.pdf | thorneloe_ca_2014_2017_en | 2017 | 0.7200 |
| Thorneloe-LUFA-CA-2017-2020-Collective-Agreement-Signed.pdf | thorneloe_ca_2017_2020_en | 2020 | 0.8250 |

### Individual commands

```bash
python src/clause_chunker.py \
  "data/english/Collective Agreement 2020-2025 English (1).pdf" \
  --doc-id ca_2020_2025_en --out-prefix output/en/ca_2020_2025_en

python src/clause_chunker.py \
  "data/english/Forced Contract 2020-2025 English.pdf" \
  --doc-id forced_contract_2020_2025_en --out-prefix output/en/forced_contract_2020_2025_en

python src/clause_chunker.py \
  "data/english/Huntington-2012-2016-CA-Lufa-collective-agreement.pdf" \
  --doc-id huntington_ca_2012_2016_en --out-prefix output/en/huntington_ca_2012_2016_en

python src/clause_chunker.py \
  "data/english/Huntington-2016-2020-CA-Lufa-collective-agreement.pdf" \
  --doc-id huntington_ca_2016_2020_en --out-prefix output/en/huntington_ca_2016_2020_en

python src/clause_chunker.py \
  "data/english/Laurentian-University-LUFA-2020-2025.pdf" \
  --doc-id lufa_2020_2025_en --out-prefix output/en/lufa_2020_2025_en

python src/clause_chunker.py \
  "data/english/LUFA-Collective-Agreement-July-2002-June-2005-FINAL.pdf" \
  --doc-id lufa_ca_2002_2005_en --out-prefix output/en/lufa_ca_2002_2005_en

python src/clause_chunker.py \
  "data/english/LUFA-Collective-Agreement-July-2005-June-2008-FINAL.pdf" \
  --doc-id lufa_ca_2005_2008_en --out-prefix output/en/lufa_ca_2005_2008_en

python src/clause_chunker.py \
  "data/english/LUFA-Term-Sheet-Fully-Executed-Mediation-Term-Sheet-2020-2025.pdf" \
  --doc-id lufa_term_sheet_2020_2025_en --out-prefix output/en/lufa_term_sheet_2020_2025_en

python src/clause_chunker.py \
  "data/english/Thorneloe-LUFA-CA-2014-2017-Collective-Agreement-Signed.pdf" \
  --doc-id thorneloe_ca_2014_2017_en --out-prefix output/en/thorneloe_ca_2014_2017_en

python src/clause_chunker.py \
  "data/english/Thorneloe-LUFA-CA-2017-2020-Collective-Agreement-Signed.pdf" \
  --doc-id thorneloe_ca_2017_2020_en --out-prefix output/en/thorneloe_ca_2017_2020_en
```

### Batch command

```bash
mkdir -p output/en
for pdf in data/english/*.pdf; do
  doc_id=$(basename "$pdf" .pdf | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | sed 's/(1)//')
  python src/clause_chunker.py "$pdf" \
    --doc-id "${doc_id}" --out-prefix "output/en/${doc_id}"
done
```

---

## french/ — French-Only Documents

### Files and assigned doc IDs

| File | Doc ID | End year | Weight |
|---|---|---|---|
| Contrat imposé 2020-2025 Francais.pdf | contrat_impose_2020_2025_fr | 2025 | 1.0000 |
| Convention Collective 2020-2025 Francais.pdf | convention_collective_2020_2025_fr | 2025 | 1.0000 |

### Commands

```bash
python src/clause_chunker.py \
  "data/french/Contrat imposé 2020-2025 Francais.pdf" \
  --doc-id contrat_impose_2020_2025_fr --out-prefix output/fr/contrat_impose_2020_2025_fr

python src/clause_chunker.py \
  "data/french/Convention Collective 2020-2025 Francais.pdf" \
  --doc-id convention_collective_2020_2025_fr --out-prefix output/fr/convention_collective_2020_2025_fr
```

### Batch command

```bash
mkdir -p output/fr
for pdf in data/french/*.pdf; do
  doc_id=$(basename "$pdf" .pdf | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
  python src/clause_chunker.py "$pdf" \
    --doc-id "${doc_id}" --out-prefix "output/fr/${doc_id}"
done
```

---

## english_and_french/ — Side-by-Side Bilingual Documents

These PDFs have English and French in **two parallel columns**.  `side_by_side_clause_chunker.py` splits each page at the midpoint, extracts each column independently, links paired chunks via `partner_chunk_id`, and tags every chunk with `end_year` / `recency_weight`.

### Files and assigned doc IDs

| File | Doc ID | End year | Weight |
|---|---|---|---|
| LUFA-Collective-Agreement-2017-2020-FINAL-Feb-8.pdf | lufa_ca_2017_2020_bilingual | 2020 | 0.8250 |
| LUFA-Collective-Agreement-July-2008-June-2011-FINAL.pdf | lufa_ca_2008_2011_bilingual | 2011 | 0.5100 |
| LUFA-Collective-Agreement-July-2011-June-2014-FINAL.pdf | lufa_ca_2011_2014_bilingual | 2014 | 0.6150 |
