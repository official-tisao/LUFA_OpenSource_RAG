
### Enable verbose logging

**For ingestion:**
```bash
python src/ingestion.py 2>&1 | tee ingestion.log
```

**For app:**
```bash
streamlit run src/app.py --logger.level=debug
```

### Check Python environment
```bash
conda activate LUFA_OpenSource_RAG
pip list | grep llama
pip list | grep chroma
pip list | grep streamlit
```

## Getting Help

### Before asking for help, collect:
1. Python version: `python3 --version`
2. Ollama version: `ollama --version`
3. OS information: `uname -a` (Linux/Mac) or `ver` (Windows)
4. Error messages (full stack trace)
5. Steps to reproduce

### Where to get help:
1. Check this troubleshooting guide
2. Review [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md)
3. Search existing GitHub issues
4. Open a new issue with details above

## Quick Diagnostic Commands

```bash
# Check Python
python3 --version

# Check Ollama
curl -s http://localhost:11434/api/tags | head -20

# Check models
ollama list

# Check documents
find data/ -type f

# Check database
ls -lh db/

# Test imports
python3 -c "import llama_index; print('OK')"

# Run basic tests
python3 test_basic.py

# Run full tests (needs dependencies)
conda activate LUFA_OpenSource_RAG
python3 test_integration.py
```

## Reset Everything

If all else fails, start fresh:

```bash
# Backup your documents
cp -r data/ data_backup/

# Clean everything
rm -rf venv/ db/

# Re-run setup
./bootstrap.sh

# Restore documents
cp -r data_backup/* data/

# Re-ingest
conda activate LUFA_OpenSource_RAG
python src/ingestion.py
```

## Known Limitations

1. **Languages**: Currently optimized for English and French
2. **File formats**: Best with PDF and TXT
3. **Document size**: Very large files may need chunking
4. **Inference speed**: CPU-only can be slow
5. **Context window**: Limited by Llama 3.2 context size

## Tips for Best Results

1. **Quality documents**: Clean, well-formatted text
2. **Appropriate length**: Neither too short nor too long
3. **Clear queries**: Be specific in your questions
4. **Test incrementally**: Start with sample documents
5. **Monitor resources**: Check memory and CPU usage

---

Still having issues? Open an issue on GitHub with full details!
