# Data Directory

This directory contains sample documents to help you get started with the bilingual RAG system.

## Directory Structure

- `english/` - Place English documents here
- `french/` - Place French documents here

## Supported Formats

The system supports various document formats including:
- PDF (.pdf)
- Text files (.txt)
- Markdown (.md)
- And more (via LlamaIndex SimpleDirectoryReader)

## Adding Your Documents

1. Copy your English documents to the `english/` directory
2. Copy your French documents to the `french/` directory
3. Run the ingestion script: `python src/ingestion.py`

## Sample Documents

This directory includes sample documents about Artificial Intelligence to help you test the system:
- `english/sample_ai_document.txt` - English version
- `french/sample_ai_document.txt` - French version

Feel free to delete these and add your own documents!
