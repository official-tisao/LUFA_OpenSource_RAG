- **Affected versions**: llama-index < 0.12.28
- **Fixed in**: 0.12.28
- **Severity**: Critical
- **Description**: The library was vulnerable to SQL injection attacks

### Updates Applied

**Previous versions:**
```
llama-index==0.10.17
llama-index-llms-ollama==0.1.2
llama-index-embeddings-ollama==0.1.2
llama-index-vector-stores-chroma==0.1.4
```

**Updated to:**
```
llama-index==0.13.0
llama-index-llms-ollama==0.3.8
llama-index-embeddings-ollama==0.4.1
llama-index-vector-stores-chroma==0.4.1
```

### Impact

- All three vulnerabilities are now patched
- The code is fully compatible with the updated versions
- No breaking changes to the application API
- Users should update immediately by running: `pip install -r requirements.txt --upgrade`
