# Security Advisory - Vulnerability Fixes

## Date: 2026-02-11

### Critical Updates Applied

This repository has been updated to address security vulnerabilities in the llama-index dependency.

### Vulnerabilities Addressed

#### 1. Insecure Temporary File (CVE TBD)
- **Affected versions**: llama-index < 0.13.0
- **Fixed in**: 0.13.0
- **Severity**: High
- **Description**: The library created temporary files insecurely

#### 2. Creation of Temporary File in Directory with Insecure Permissions
- **Affected versions**: llama-index < 0.12.3
- **Fixed in**: 0.12.3
- **Severity**: High
- **Description**: Temporary files were created in directories with insecure permissions

#### 3. SQL Injection Vulnerability
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

### Recommendation

**For existing installations:**
1. Pull the latest changes
2. Run: `source venv/bin/activate`
3. Run: `pip install -r requirements.txt --upgrade`
4. Verify: `pip list | grep llama-index`

**For new installations:**
- The bootstrap script will automatically install the patched versions

### References

- LlamaIndex Security Advisories
- GitHub Advisory Database
- Python Package Index (PyPI) vulnerability reports

### Contact

If you have questions about these security updates, please open an issue on GitHub.

---

**Status**: ✅ All vulnerabilities patched
**Action Required**: Update dependencies immediately
