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
conda activate LUFA_OpenSource_RAG
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
