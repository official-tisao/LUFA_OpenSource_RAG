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
