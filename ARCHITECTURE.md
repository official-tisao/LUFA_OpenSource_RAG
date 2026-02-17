# System Architecture and Flow

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Bilingual RAG System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │   English    │         │   French     │                      │
│  │  Documents   │         │  Documents   │                      │
│  └──────┬───────┘         └──────┬───────┘                      │
│         │                        │                               │
│         └────────┬───────────────┘                               │
│                  │                                               │
│         ┌────────▼────────┐                                      │
│         │  ingestion.py   │                                      │
│         │  - Load docs    │                                      │
│         │  - Detect lang  │                                      │
│         │  - Tag chunks   │                                      │
│         │  - Embed        │                                      │
│         └────────┬────────┘                                      │
│                  │                                               │
│         ┌────────▼────────┐                                      │
│         │   ChromaDB      │                                      │
│         │ (nomic-embed-text-v2-moe embed)  │                                      │
│         └────────┬────────┘                                      │
│                  │                                               │
│         ┌────────▼────────┐                                      │
│         │  rag_engine.py  │                                      │
│         │  - Query        │                                      │
│         │  - Retrieve     │                                      │
│         │  - Generate     │                                      │
│         └────────┬────────┘                                      │
│                  │                                               │
│         ┌────────▼────────┐                                      │
│         │    app.py       │                                      │
│         │  (Streamlit)    │                                      │
│         └─────────────────┘                                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Document Ingestion Flow

```
1. Load Documents
   ├─ English docs from data/english/
   └─ French docs from data/french/

2. Language Detection
   ├─ Analyze text sample
   └─ Tag with language metadata

3. Text Chunking
   ├─ Split into 512-token chunks
   └─ 50-token overlap for context

4. Embedding Generation
   ├─ Use nomic-embed-text-v2-moe (multilingual)
   └─ Generate dense vectors

5. Store in ChromaDB
   └─ Persist to db/chroma_db
```

## Query Processing Flow

```
User Query (EN or FR)
        │
        ▼
┌───────────────┐
│ Detect Query  │
│  Language     │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Embed Query   │
│  (nomic-embed-text-v2-moe)     │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Similarity    │
│ Search        │
│ (ChromaDB)    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Retrieve      │
│ Top-K Chunks  │
│ (cross-lingual)│
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Generate      │
│ Response      │
│ (Llama 3.2)   │
└───────┬───────┘
        │
        ▼
Response in Query Language
```

## Component Details

### 1. ingestion.py

**Purpose**: Process and index documents

**Key Functions**:
- `detect_language(text)`: Detect EN/FR
- `load_documents_from_directory(dir)`: Load files
- `tag_documents_with_language(docs)`: Add language metadata
- `create_multilingual_index()`: Build vector store

**Technologies**:
- LlamaIndex: Document loading and processing
- langdetect: Language identification
- nomic-embed-text-v2-moe: Multilingual embeddings
- ChromaDB: Vector storage

### 2. rag_engine.py

**Purpose**: Query processing and response generation

**Key Classes**:
- `BilingualRAGEngine`: Main RAG engine

**Key Functions**:
- `detect_query_language(query)`: Identify query language
- `create_language_aware_prompt()`: Add language instruction
- `query(text)`: Process query and generate response

**Technologies**:
- LlamaIndex: RAG orchestration
- Llama 3.2: Text generation
- ChromaDB: Vector retrieval

### 3. app.py

**Purpose**: User interface

**Features**:
- Chat interface
- Language detection display
- Source document viewer
- Usage statistics
- Configuration panel

**Technologies**:
- Streamlit: Web UI framework

## Data Flow

### Ingestion Phase
```
PDF/TXT Files → Document Objects → Text Chunks → 
Embeddings → Vector Store (ChromaDB)
```

### Query Phase
```
User Question → Query Embedding → Similar Chunks → 
Context + LLM → Answer in Query Language
```

## Language Support

### English (EN)
- Auto-detected from Latin alphabet patterns
- Default language if detection uncertain

### French (FR)
- Auto-detected from French-specific patterns
- Accented characters help identification

### Cross-lingual Retrieval
- nomic-embed-text-v2-moe embeddings map both languages to same vector space
- Queries in one language can retrieve docs in either language
- Semantic similarity transcends language barriers

## Models

### nomic-embed-text-v2-moe (Embeddings)
- **Purpose**: Multilingual dense embeddings
- **Languages**: 100+ including EN/FR
- **Dimension**: 1024
- **Use**: Document and query encoding

### Llama 3.2 (LLM)
- **Purpose**: Text generation
- **Context**: 8K tokens
- **Use**: Response synthesis
- **Feature**: Instruction following for language-specific responses

## Performance Considerations

### Chunking Strategy
- **Size**: 512 tokens
  - Small enough for precise retrieval
  - Large enough for context
- **Overlap**: 50 tokens
  - Prevents information loss at boundaries
  - Improves context continuity

### Retrieval Configuration
- **Top-K**: 3 documents
  - Balance between context and relevance
  - Adjustable based on needs
- **Similarity**: Cosine similarity in embedding space

### Response Generation
- **Mode**: Compact
  - Efficient token usage
  - Coherent synthesis
- **Timeout**: 120 seconds
  - Handles complex queries

## Extensibility

### Adding New Languages
1. Update `detect_language()` in ingestion.py
2. Add language-specific prompts in rag_engine.py
3. nomic-embed-text-v2-moe already supports 100+ languages

### Custom Documents
- Place in appropriate language directory
- Run ingestion to update index
- No code changes needed

### UI Customization
- Modify app.py
- Adjust Streamlit components
- Add custom features

## Security Notes

- All processing is local (no external APIs)
- Documents stay on your machine
- Ollama runs locally
- No data transmitted externally
