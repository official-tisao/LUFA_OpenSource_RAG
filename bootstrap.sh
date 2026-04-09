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

#conda init
#conda activate "$CONDA_ENV_NAME"

if [ "$1" = "ingest" ]; then
    print_info "Running document ingestion inside '$CONDA_ENV_NAME'..."
    conda run -n "$CONDA_ENV_NAME" python src/pdf_ocr_converter.py
    conda run -n "$CONDA_ENV_NAME" python src/bilingual_pdf_splitter.py
    conda run -n "$CONDA_ENV_NAME" python src/ingestion.py

elif [ "$1" = "run" ]; then
    print_info "Starting Streamlit app inside '$CONDA_ENV_NAME'..."
    conda run -n "$CONDA_ENV_NAME" streamlit run src/app.py
fi