# LUFA_OpenSource_RAG

A production-grade **Bilingual (English/French) Agentic RAG System** for querying the Laurentian University Faculty Association (LUFA) collective agreements. Combines local LLMs with frontier model support, clause-aware chunking, and multi-pass agentic retrieval for accurate legal document processing.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Core Python Modules](#core-python-modules)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [Evaluation & Testing](#evaluation--testing)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**LUFA_OpenSource_RAG** is an advanced retrieval-augmented generation (RAG) system purpose-built for university faculty collective agreements. The system:

- ✅ **Bilingual Support**: Native English/French query handling with automatic language detection
- ✅ **Clause-Aware Chunking**: Extracts semantic clause boundaries from PDFs (not naive fixed-size chunks)
- ✅ **Agentic Retrieval**: Multi-pass query rewriting, reflection, and re-retrieval for improved accuracy
- ✅ **Frontier Model Integration**: Optional use of GitHub Copilot models (GPT-5, Claude, Grok, Gemini) alongside local Ollama
- ✅ **Production API**: FastAPI REST interface with health checks, streaming, and timeout management
- ✅ **Evaluation Dashboard**: Automated RAGAS-style evaluation with interactive HTML dashboard
- ✅ **Local-First**: No cloud dependencies—runs entirely on-premise with Ollama + ChromaDB

**Target Users**: Legal researchers, faculty representatives, HR departments, and university administrators needing semantic search over bilingual policy documents.

---

## 🏗️ System Architecture
┌─────────────────────────────────────────────────────────────────┐ 
│ USER INTERFACES │ ├──────────────────────┬──────────────────┬──────────────────────┤ 
│ Streamlit Web UI │ REST API (Port │ CLI / Python SDK │ │ (app.py) │ 8000, api.py) │ 
│ └──────────────────────┴──────────────────┴──────────────────────┘ 
│ ┌───────────┼────────────┐ │ │ │ ┌───────▼─────┐ ┌──▼────────┐ ┌─▼──────────┐ 
│ Standard │ │ Agentic │ │ Frontier │ │ RAG │ │ RAG │ │ Model │ │ (1-pass) │ │ (3-pass) │ │ (GitHub) │
└───────┬─────┘ └──┬────────┘ └─┬──────────┘ │ │ │ └───────────┼────────────┘ │ 
┌───────────────────▼────────────────────┐ │ BilingualRAGEngine (rag_engine.py) │ │ - Language detection │ 
