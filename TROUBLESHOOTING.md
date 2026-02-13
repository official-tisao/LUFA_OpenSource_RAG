# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Issue: `python3` command not found
**Solution**: Install Python 3.8 or higher
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-pip

# macOS
brew install python3

# Windows
# Download from python.org
```

#### Issue: `./bootstrap.sh` permission denied
**Solution**: Make the script executable
```bash
chmod +x bootstrap.sh
./bootstrap.sh
```

#### Issue: Dependencies fail to install
**Solution**: Upgrade pip and try again
```bash
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Ollama Issues

#### Issue: Cannot connect to Ollama
**Check if Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

**Solution**: Start Ollama
```bash
# Usually just run:
ollama serve

# Or if using systemd:
systemctl start ollama
```

#### Issue: Model not found
**Check available models:**
```bash
ollama list
```

**Solution**: Pull required models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text-v2-moe
```

#### Issue: Models are slow
**Causes**:
- First run downloads models (large files)
- CPU inference is slower than GPU
- Large documents take time to process

**Solutions**:
- Be patient on first run
- Use smaller documents for testing
- Consider GPU if available

### Document Issues

#### Issue: No documents found
**Check directories:**
```bash
ls -la data/english/
ls -la data/french/
```

**Solution**: Add documents
```bash
cp your-doc.pdf data/english/
cp votre-doc.pdf data/french/
```

#### Issue: Ingestion fails
**Check error messages:**
```bash
python src/ingestion.py
```

**Common causes**:
- Corrupted PDF files
- Unsupported file formats
- Permission issues

**Solutions**:
- Try with sample documents first
- Check file permissions
- Use supported formats (PDF, TXT, MD)

#### Issue: Language detection is wrong
**Cause**: Short or ambiguous text

**Solution**: 
- Add more text to documents
- Manually verify sample_ai_document.txt works
- Check document encoding (should be UTF-8)

### Application Issues

#### Issue: Streamlit won't start
**Check if port is in use:**
```bash
lsof -i :8501
```

**Solution**: Use different port
```bash
streamlit run src/app.py --server.port 8502
```

#### Issue: "RAG engine not loaded"
**Causes**:
- Ollama not running
- Models not available
- Database not created

**Solutions**:
1. Start Ollama: `ollama serve`
2. Pull models: `ollama pull llama3.2 && ollama pull nomic-embed-text-v2-moe`
3. Run ingestion: `python src/ingestion.py`

#### Issue: Query returns empty results
**Causes**:
- Database is empty
- Documents not ingested
- Query too specific

**Solutions**:
1. Check if ingestion ran successfully
2. Try broader queries
3. Verify documents are in data directories

#### Issue: Responses are in wrong language
**Cause**: Language detection or prompt issue

**Solutions**:
- Try more explicit queries
- Check detected language in UI
- Verify document language tags

### Performance Issues

#### Issue: Slow responses
**Causes**:
- CPU-only inference
- Large document collection
- High top-k value

**Solutions**:
- Reduce top-k in settings (default: 3)
- Use fewer documents for testing
- Consider GPU acceleration

#### Issue: High memory usage
**Causes**:
- Large embedding model
- Many documents in memory

**Solutions**:
- Close other applications
- Process documents in batches
- Increase system RAM

### Testing Issues

#### Issue: Basic tests fail
**Expected**: Import tests fail without dependencies

**Solution**: Install dependencies first
```bash
./bootstrap.sh
source venv/bin/activate
python test_integration.py
```

#### Issue: Integration tests fail
**Check**:
1. Dependencies installed?
2. Ollama running?
3. Models available?

**Solution**: Follow setup instructions completely

### Database Issues

#### Issue: ChromaDB errors
**Solution**: Delete and recreate database
```bash
rm -rf db/chroma_db
python src/ingestion.py
```

#### Issue: "Collection not found"
**Solution**: Run ingestion to create collection
```bash
python src/ingestion.py
```

#### Issue: Database size grows large
**Cause**: Many documents ingested

**Solutions**:
- This is normal for large collections
- Clean up by deleting db/ and re-ingesting
- ChromaDB is efficient with storage

## Debug Mode

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
source venv/bin/activate
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
source venv/bin/activate
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
source venv/bin/activate
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
