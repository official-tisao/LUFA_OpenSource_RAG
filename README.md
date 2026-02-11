# LUFA_OpenSource_RAG
LUFA_OpenSource_RAG


## Refined Bilingual Open-Source RAG Technical Implementation Plan

### Phase 1: Environment \& Hardware Setup

**Hardware Requirements:**

- Minimum 16GB RAM (32GB recommended for multilingual models)
- NVIDIA GPU with 6GB+ VRAM (or M1/M2/M3 Mac)

**Software Installation:**

- [Ollama](https://ollama.com)
- Python 3.10+
- Visual Studio Code
- Anaconda/Miniconda


### Phase 2: Bilingual Model Selection

**Multilingual Embedding Model**:[^1]

1. **BGE-M3** : Supports 100+ languages with strong French/English performance[^2]

```bash
ollama pull bge-m3
```

**Multilingual LLM** :[^1]

1. **Llama 3.2** : Officially supports French, English, and 6 other languages[^4]

```bash
ollama pull llama3.2:3b-instruct-q4_K_M
```

### Phase 3: Enhanced Project Structure

```
LUFA_OpenSource_RAG/
├── data/
│   ├── english/          # English collective agreements
│   ├── french/           # French collective agreements
│   └── metadata.json     # Track document language tags
├── db/
│   └── chroma_db/        # Single multilingual vector store
├── src/
│   ├── ingestion.py      # Bilingual document ingestion
│   ├── language_detector.py  # Auto-detect document language
│   ├── rag_engine.py     # Multilingual RAG pipeline
│   ├── query_handler.py  # Language-aware query processing
│   └── app.py            # Bilingual Streamlit interface
├── config/
│   └── config.yaml       # Language and model settings
├── tests/
│   ├── test_data_en.json # English test Q&A pairs
│   └── test_data_fr.json # French test Q&A pairs
└── requirements.txt
```


### Phase 4: Updated Dependencies

```
llama-index
llama-index-llms-ollama
llama-index-embeddings-ollama
llama-index-vector-stores-chroma
chromadb
streamlit
pypdf
langdetect                # Language detection
pycountry                 # Language code handling
```


### Phase 5: Bilingual Ingestion Strategy

**Key Features:**[^5][^1]

1. **Language Detection**: Automatically detect whether each PDF is English or French using `langdetect`
2. **Metadata Tagging**: Store language metadata with each chunk
3. **Unified Vector Store**: Both languages in same ChromaDB using multilingual embeddings[^6]
4. **Document Structure**:
    - Read both English and French collective agreements from respective folders
    - Chunk with 1024 tokens, 200 overlap
    - Tag each chunk with: `{language: "en/fr", source_doc: "filename", page: N}`

### Phase 6: Multilingual RAG Engine

**Core Capabilities:**[^1]

1. **Query Language Detection**: Detect if user asks in English or French
2. **Cross-Lingual Retrieval**: Multilingual embeddings enable:
    - English query → retrieves relevant French documents[^6]
    - French query → retrieves relevant English documents
    - Same-language retrieval
3. **Response Generation**: LLM responds in the same language as the query[^4]
4. **Retrieval Settings**: Top 5 chunks with similarity threshold 0.7

**System Prompts:**

```python
SYSTEM_PROMPTS = {
    "en": "You are a helpful assistant answering questions about the Laurentian University Faculty Association collective agreement. Respond in English.",
    "fr": "Tu es un assistant utile qui répond aux questions sur la convention collective de l'Association des professeur(e)s de l'Université Laurentienne. Réponds en français."
}
```


### Phase 7: Bilingual User Interface

**Streamlit Features:**[^1]

- Language toggle (EN/FR) for UI labels
- Automatic query language detection
- Display retrieved chunks with language tags
- Show source document and page numbers
- Option to translate the answer into another language

**UI Elements:**

```python
# Sidebar: Language preference selector
language = st.sidebar.selectbox("Language / Langue", ["English", "Français"])

# Main chat: Question input (accepts EN/FR)
query = st.chat_input("Ask a question / Posez une question")

# Response area: Answer + source citations
st.write(response)
st.caption(f"Sources: {source_docs}")

# Metadata display: Show which language documents were retrieved
st.info(f"Retrieved {len(chunks)} chunks ({lang_counts})")
```


### Phase 8: Evaluation Framework

**Bilingual Test Dataset:**[^1]

1. Create 20 English Q\&A pairs from English collective agreements
2. Create 20 French Q\&A pairs from French collective agreements
3. Test cross-lingual scenarios (English query, French source)

**Metrics:**

- Retrieval accuracy per language
- Answer quality (manual evaluation)
- Cross-lingual retrieval performance
- Response time per language

**Optional: RAGAS Evaluation** with local model as judge[^1]

### Phase 9: Technical Documentation Structure

Your thesis documentation should cover:

1. **Architecture Overview**
    - System diagram showing bilingual data flow
    - Component interactions
2. **Multilingual Challenges**
    - Embedding space alignment for EN/FR
    - Query-document language mismatch handling
    - Token counting differences between languages
3. **Implementation Details**
    - Model selection rationale (why BGE-M3 or Llama 3.2)
    - Chunking strategy for French vs English text
    - Performance optimization for local deployment
4. **Evaluation Results**
    - Monolingual performance (EN→EN, FR→FR)
    - Cross-lingual performance (EN→FR, FR→EN)
    - Comparison with baseline approaches

### Immediate Next Steps

1. Install Ollama and pull bilingual models:

```bash
ollama pull llama3.2
ollama pull bge-m3  # or mxbai-embed-large
```

2. Set up enhanced project structure with language folders
3. Implement `language_detector.py` for automatic language detection
4. Modify `ingestion.py` to handle both English and French PDFs with metadata tagging

This refined plan maintains your open-source approach while adding robust bilingual capabilities. The key advantage is that multilingual embedding models map semantically similar content across languages to nearby vectors, enabling true cross-lingual retrieval without translation overhead.[^7][^8]

---

## Footnotes

[^1]: This approach leverages the inherent multilingual capabilities of modern embedding and LLM models to provide seamless bilingual support without requiring separate pipelines or translation services.

[^2]: BGE-M3 (BAAI General Embedding - Multilingual, Multifunctionality, Multi-Granularity) is specifically designed for cross-lingual retrieval tasks and has been shown to perform well on French and English document pairs.

[^3]: The multilingual-e5-large model from Microsoft is trained on multilingual data and provides strong semantic search capabilities across different languages.

[^4]: Llama 3.2 officially supports 8 languages, including English and French, making it suitable for generating responses in either language while maintaining context and accuracy.

[^5]: Language detection and metadata tagging ensure that the system can track document provenance while still enabling cross-lingual retrieval through shared embedding space.

[^6]: Using a single unified vector store with multilingual embeddings is more efficient than maintaining separate stores per language and naturally enables cross-lingual retrieval.

[^7]: Cross-lingual retrieval allows users to query in one language (e.g., English) and retrieve relevant documents in another language (e.g., French) based on semantic similarity.

[^8]: Multilingual embedding models are trained to map semantically similar phrases across languages to nearby points in the embedding space, enabling natural cross-lingual information retrieval without explicit translation.
