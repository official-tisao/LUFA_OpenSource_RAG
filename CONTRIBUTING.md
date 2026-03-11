- `db/` - Vector database storage

## Testing

### Basic Structure Tests (no dependencies required)
```bash
python test_basic.py
```

### Full System Tests (after installation)
```bash
conda activate LUFA_OpenSource_RAG
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
