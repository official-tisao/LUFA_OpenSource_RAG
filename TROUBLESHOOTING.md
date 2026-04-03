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
conda activate LUFA_OpenSource_RAG
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
