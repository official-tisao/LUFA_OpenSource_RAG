- Performs cross-lingual semantic search
- Generates responses in query language
- Includes source attribution

**Key Features:**
- `BilingualRAGEngine` class with full RAG pipeline
- `detect_query_language()` - Query language detection
- `create_language_aware_prompt()` - Language-specific prompts
- `query()` - Complete query processing

#### 3. Streamlit App (`src/app.py`)
- Interactive chat interface
- Real-time language detection display
- Source document viewer
- Usage statistics and analytics
- Configurable retrieval parameters

**UI Features:**
- Chat history with language indicators
- Source document display with scores
- Language distribution statistics
- Adjustable top-k retrieval

#### 4. Bootstrap Script (`bootstrap.sh`)
- Automated environment setup
- Dependency installation
- Ollama model checking
- Helper commands for common tasks

**Commands:**
- `./bootstrap.sh` - Initial setup
- `./bootstrap.sh ingest` - Run ingestion
- `./bootstrap.sh run` - Start application

### 📚 Documentation

| Document | Purpose | Status |
|----------|---------|--------|
| README.md | Complete user guide | ✅ 245 lines |
| QUICKSTART.md | Quick start guide | ✅ 90 lines |
| ARCHITECTURE.md | System design | ✅ 347 lines |
| TROUBLESHOOTING.md | Problem solving | ✅ 301 lines |
| CONTRIBUTING.md | Development guide | ✅ 68 lines |
| data/README.md | Data directory guide | ✅ 27 lines |

### 🧪 Testing

#### Basic Tests (`test_basic.py`)
- Directory structure validation
- Sample document verification
- Dependency checking
- Bootstrap script validation
- No external dependencies required

#### Integration Tests (`test_integration.py`)
- Full system integration
- Language detection with real module
- Document loading and tagging
- RAG engine initialization
- Requires full environment setup

### 🎨 Features Implemented

#### Language Detection
- Automatic detection of document language during ingestion
- Query language detection for appropriate responses
- Fallback to English for ambiguous cases

#### Cross-lingual Retrieval
- nomic-embed-text-v2-moe multilingual embeddings
- Semantic similarity across languages
- Retrieves relevant content regardless of language

#### Language-aware Responses
- Detects query language
- Adds language-specific instructions to LLM
- Returns answers in the same language as query

#### Persistent Storage
- ChromaDB vector database
- Persistent across sessions
- Efficient similarity search

#### User Interface
- Clean, modern Streamlit design
- Real-time chat interaction
- Language indicators (🇬🇧/🇫🇷)
- Source document display
- Usage statistics

### 🚀 Usage Workflow

1. **Setup** (one time):
   ```bash
   ./bootstrap.sh
   ollama pull llama3.2
   ollama pull nomic-embed-text-v2-moe
   ```

2. **Add Documents**:
   ```bash
   cp your-files.pdf data/english/
   cp vos-fichiers.pdf data/french/
   ```

3. **Ingest**:
   ```bash
   conda activate LUFA_OpenSource_RAG
   python src/ingestion.py
