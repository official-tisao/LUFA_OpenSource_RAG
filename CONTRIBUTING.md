# Contributing to LUFA_OpenSource_RAG

Thank you for your interest in contributing to the Bilingual RAG System!

## Development Setup

1. Fork and clone the repository
2. Run the bootstrap script to set up your environment:
   ```bash
   ./bootstrap.sh
   ```
3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

## Code Structure

- `src/ingestion.py` - Document processing and indexing
- `src/rag_engine.py` - RAG query engine
- `src/app.py` - Streamlit web interface
- `data/` - Document storage
- `db/` - Vector database storage

## Testing

### Basic Structure Tests (no dependencies required)
```bash
python test_basic.py
```

### Full System Tests (after installation)
```bash
source venv/bin/activate
# Ensure Ollama is running and models are pulled
python test_integration.py
```

## Code Style

- Follow PEP 8 guidelines
- Add docstrings to all functions
- Keep functions focused and modular
- Add type hints where appropriate

## Pull Request Process

1. Create a new branch for your feature
2. Make your changes
3. Test your changes thoroughly
4. Update documentation if needed
5. Submit a pull request with a clear description

## Adding New Features

When adding new features, consider:
- Maintaining bilingual support
- Keeping the code modular
- Adding appropriate error handling
- Updating documentation and tests

## Reporting Issues

When reporting issues, please include:
- Python version
- Ollama version
- Steps to reproduce
- Error messages or logs
- Expected vs actual behavior

Thank you for contributing! 🎉
