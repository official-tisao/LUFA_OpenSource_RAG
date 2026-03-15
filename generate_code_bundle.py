#!/usr/bin/env python3
import os
from pathlib import Path


def bundle_code(src_dir="src", output_file="all_python_code.txt"):
    src_path = Path(src_dir)

    if not src_path.exists():
        print(f"Error: Directory '{src_dir}' not found.")
        return

    with open(output_file, "w", encoding="utf-8") as outfile:
        # Walk through the src directory
