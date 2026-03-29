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
