#!/bin/bash

# Bootstrap script for Bilingual RAG System (Conda version)
# This script sets up the conda environment and provides commands to run the system

set -e  # Exit on error

echo "=========================================="
echo "Bilingual RAG System Bootstrap (Conda)"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# -------- CONFIGURABLE ENV NAME --------
CONDA_ENV_NAME="${CONDA_ENV_NAME:-LUFA_OpenSource_RAG}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"   # change to 3.12 or 3.10 if you prefer

# -------- CHECK PREREQUISITES --------

# Check if conda is available
# Check if conda works at all (not just command -v)
if ! conda --version >/dev/null 2>&1; then
    print_error "conda is not working in this shell. On Windows, run this script from 'Anaconda Prompt' or a terminal where conda is initialized."
    exit 1
fi

print_info "Conda found: $(conda --version)"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    print_warning "Ollama doesn't seem to be running at http://localhost:11434"
    print_info "Please start Ollama first: https://ollama.ai/"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_info "Ollama is running"
fi

# -------- CREATE / UPDATE CONDA ENV --------

# Check if env exists
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
