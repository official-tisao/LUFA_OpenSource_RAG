#!/bin/bash

# Bootstrap script for Bilingual RAG System
# This script sets up the environment and provides commands to run the system

set -e  # Exit on error

echo "=========================================="
echo "Bilingual RAG System Bootstrap"
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

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

print_info "Python 3 found: $(python3 --version)"

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

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
print_info "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p data/english data/french db config tests

# Check for required Ollama models
print_info "Checking for required Ollama models..."

check_model() {
    local model=$1
    # Check for model with or without :latest tag (avoid duplicate API calls)
    local tags=$(curl -s http://localhost:11434/api/tags)
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

if ! check_model "mistral:7b"; then
    print_warning "Please pull the mistral:7b model: ollama pull mistral:7b"
    MODELS_OK=false
fi

if ! check_model "nomic-embed-text-v2-moe"; then
    print_warning "Please pull the nomic-embed-text-v2-moe model: ollama pull nomic-embed-text-v2-moe"
    MODELS_OK=false
fi

if [ "$MODELS_OK" = false ]; then
    print_warning "Some models are missing. The system may not work correctly."
    print_info "Run: ollama pull mistral:7b && ollama pull nomic-embed-text-v2-moe"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
print_info "Next steps:"
echo "  1. Add your documents:"
echo "     - Place English documents in: data/english/"
echo "     - Place French documents in: data/french/"
echo "     - Supported formats: PDF, TXT, and more"
echo ""
echo "  2. Run document ingestion:"
echo "     source venv/bin/activate"
echo "     python src/ingestion.py"
echo ""
echo "  3. Start the Streamlit app:"
echo "     source venv/bin/activate"
echo "     streamlit run src/app.py"
echo ""
print_info "Or use the provided helper commands:"
echo "  ./bootstrap.sh ingest  - Run document ingestion"
echo "  ./bootstrap.sh run     - Start the Streamlit app"
echo ""

# Handle commands
if [ "$1" = "ingest" ]; then
    print_info "Running document ingestion..."
    python src/ingestion.py
elif [ "$1" = "run" ]; then
    print_info "Starting Streamlit app..."
    streamlit run src/app.py
fi
