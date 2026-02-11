# LUFA_OpenSource_RAG

A bilingual (English/French) Retrieval-Augmented Generation (RAG) system built with Ollama, LlamaIndex, ChromaDB, and Streamlit.

> **🔒 Security Update (2026-02-11)**: Dependencies have been updated to address critical vulnerabilities. See [SECURITY.md](SECURITY.md) for details.

## 🌟 Features

- **Bilingual Support**: Automatically detects and processes English and French documents
- **Language Detection**: Auto-detects query language and responds accordingly
- **Cross-lingual Retrieval**: Search across documents in both languages
- **Multilingual Embeddings**: Uses BGE-M3 for high-quality multilingual embeddings
- **User-friendly Interface**: Interactive Streamlit web application
- **Persistent Storage**: ChromaDB vector store for efficient document retrieval

## 🏗️ Architecture

```
LUFA_OpenSource_RAG/
├── data/
│   ├── english/          # Place English documents here (PDF, TXT, etc.)
│   └── french/           # Place French documents here (PDF, TXT, etc.)
├── src/
│   ├── ingestion.py      # Document ingestion with language detection
│   ├── rag_engine.py     # RAG engine with multilingual support
│   └── app.py            # Streamlit web application
├── db/                   # ChromaDB vector store (auto-generated)
├── requirements.txt      # Python dependencies
└── bootstrap.sh          # Setup and run script
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**: Make sure Python is installed
2. **Ollama**: Install from [ollama.ai](https://ollama.ai/)
3. **Required Models**: 
   - `ollama pull llama3.2` - LLM for text generation
   - `ollama pull bge-m3` - Multilingual embeddings

### Installation

1. Clone the repository:
```bash
git clone https://github.com/official-tisao/LUFA_OpenSource_RAG.git
cd LUFA_OpenSource_RAG
```

2. Run the bootstrap script:
```bash
./bootstrap.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Set up necessary directories
- Check for required Ollama models

### Usage

#### Step 1: Add Documents

Place your documents in the appropriate directories:
- English documents → `data/english/`
- French documents → `data/french/`

Supported formats: PDF, TXT, MD, and more.

#### Step 2: Ingest Documents

Process and index your documents:
```bash
./bootstrap.sh ingest
# or manually:
source venv/bin/activate
python src/ingestion.py
```

This will:
- Load documents from both directories
- Detect language of each document
- Tag chunks with language metadata
- Create embeddings using BGE-M3
- Store in ChromaDB vector store

#### Step 3: Run the Application

Start the Streamlit web interface:
```bash
./bootstrap.sh run
# or manually:
source venv/bin/activate
streamlit run src/app.py
```

The app will open in your browser at `http://localhost:8501`

## 💬 Using the Application

1. **Ask Questions**: Type your question in English or French
2. **Auto-Detection**: The system detects your query language
3. **Cross-lingual Search**: Retrieves relevant information from both language collections
4. **Natural Responses**: Get answers in your query language

### Example Queries

**English:**
- "What is this document about?"
- "Summarize the main points"
- "What are the key findings?"

**French:**
- "De quoi parle ce document?"
- "Résume les points principaux"
- "Quelles sont les principales conclusions?"

## 🛠️ Technical Details

### Models

- **LLM**: `llama3.2:latest` - Local language model via Ollama
- **Embeddings**: `bge-m3:latest` - Multilingual embeddings (BAAI BGE-M3)

### Technologies

- **LlamaIndex**: RAG framework and orchestration
- **ChromaDB**: Vector database for embeddings
- **Ollama**: Local LLM inference
- **Streamlit**: Web UI framework
- **langdetect**: Language detection
- **pypdf**: PDF processing

### How It Works

1. **Document Ingestion**:
   - Load documents from English and French directories
   - Detect language using langdetect
   - Split documents into chunks (512 tokens, 50 overlap)
   - Generate multilingual embeddings with BGE-M3
   - Store in ChromaDB with language metadata

2. **Query Processing**:
   - Detect query language
   - Generate query embedding
   - Retrieve top-k similar chunks (cross-lingual)
   - Synthesize response using LLM
   - Return answer in query language

3. **Cross-lingual Retrieval**:
   - BGE-M3 embeddings enable semantic search across languages
   - Language metadata helps filter and contextualize results

## 📦 Dependencies

Core dependencies (see `requirements.txt`):
- `llama-index` - RAG framework
- `llama-index-llms-ollama` - Ollama LLM integration
- `llama-index-embeddings-ollama` - Ollama embeddings
- `llama-index-vector-stores-chroma` - ChromaDB integration
- `chromadb` - Vector database
- `streamlit` - Web UI
- `pypdf` - PDF processing
- `langdetect` - Language detection

## 🔧 Configuration

### Environment Variables

You can customize the configuration by modifying the source files or setting environment variables:

- **Ollama URL**: Default is `http://localhost:11434`
- **Database Path**: Default is `db/chroma_db`
- **Collection Name**: Default is `multilingual_docs`
- **Chunk Size**: Default is 512 tokens
- **Top-K Retrieval**: Default is 3 documents

### Advanced Configuration

Edit the source files to customize:
- `src/ingestion.py`: Modify chunking strategy, embedding model
- `src/rag_engine.py`: Adjust retrieval parameters, prompt templates
- `src/app.py`: Customize UI, add features

## 🐛 Troubleshooting

### Ollama not running
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

### Models not found
```bash
# Pull required models
ollama pull llama3.2
ollama pull bge-m3
```

### No documents found
- Ensure documents are placed in `data/english/` or `data/french/`
- Run ingestion: `python src/ingestion.py`
- Check for error messages during ingestion

### Import errors
```bash
# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

## 📄 License

This project is open source and available under the terms specified in the LICENSE file.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

Built with ❤️ using Ollama, LlamaIndex, ChromaDB, and Streamlit
