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
