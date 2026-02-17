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
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lufa_rag}"
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
    print_info "Conda environment '$CONDA_ENV_NAME' already exists"
else
    print_info "Creating conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION"
fi

# Show which python is in the env
print_info "Python in '$CONDA_ENV_NAME':"
conda run -n "$CONDA_ENV_NAME" python --version

# -------- INSTALL DEPENDENCIES --------

print_info "Installing / updating dependencies in '$CONDA_ENV_NAME'..."
conda run -n "$CONDA_ENV_NAME" python -m pip install --upgrade pip
conda run -n "$CONDA_ENV_NAME" python -m pip install -r requirements.txt

# -------- CREATE REQUIRED DIRECTORIES --------

print_info "Creating necessary directories..."
mkdir -p data/english data/french db config tests

# -------- CHECK OLLAMA MODELS --------

print_info "Checking for required Ollama models..."

check_model() {
    local model=$1
    local tags
    tags=$(curl -s http://localhost:11434/api/tags || echo "")

    if echo "$tags" | grep -q "\"name\":\"$model\"" || \
       echo "$tags" | grep -q "\"name\":\"${model}:latest\""; then
        print_info "Model $model is available"
        return 0
    else
        print_warning "Model $model is not available"
        return 1
    fi
}

MODELS_OK=true

if ! check_model "llama3.2:3b-instruct-q4_K_M"; then
    print_warning "Please pull the llama3.2:3b-instruct-q4_K_M model:"
    echo "    ollama pull llama3.2:3b-instruct-q4_K_M"
    MODELS_OK=false
fi

if ! check_model "nomic-embed-text-v2-moe"; then
    print_warning "Please pull the nomic-embed-text-v2-moe model:"
    echo "    ollama pull nomic-embed-text-v2-moe"
    MODELS_OK=false
fi

if [ "$MODELS_OK" = false ]; then
    print_warning "Some models are missing. The system may not work correctly."
    print_info "Run:"
    echo "    ollama pull llama3.2:3b-instruct-q4_K_M && ollama pull nomic-embed-text-v2-moe"
fi

# -------- SUMMARY / NEXT STEPS --------

echo ""
echo "=========================================="
echo "Setup Complete (Conda)!"
echo "=========================================="
echo ""
print_info "Conda environment: $CONDA_ENV_NAME"
echo ""
print_info "Next steps (manual use):"
echo "  1. Activate the environment:"
echo "     conda activate $CONDA_ENV_NAME"
echo ""
echo "  2. Add your documents:"
echo "     - Place English documents in: data/english/"
echo "     - Place French documents in:  data/french/"
echo ""
echo "  3. Run document ingestion:"
echo "     conda activate $CONDA_ENV_NAME"
echo "     python src/ingestion.py"
echo ""
echo "  4. Start the Streamlit app:"
echo "     conda activate $CONDA_ENV_NAME"
echo "     streamlit run src/app.py"
echo ""
print_info "Or use helper modes directly with this script:"
echo "  ./bootstrap.sh ingest   # Run document ingestion (via conda run)"
echo "  ./bootstrap.sh run      # Start the Streamlit app (via conda run)"
echo ""

# -------- HANDLE SUBCOMMANDS --------

if [ "$1" = "ingest" ]; then
    print_info "Running document ingestion inside '$CONDA_ENV_NAME'..."
    conda run -n "$CONDA_ENV_NAME" python src/ingestion.py
elif [ "$1" = "run" ]; then
    print_info "Starting Streamlit app inside '$CONDA_ENV_NAME'..."
    conda run -n "$CONDA_ENV_NAME" streamlit run src/app.py
fi