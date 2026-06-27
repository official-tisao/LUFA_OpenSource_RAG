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
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file

                    # Write header
                    outfile.write(f"Filename= {file_path}\n")
                    outfile.write("CODE:\n")
                    outfile.write("```\n")

                    # Write content
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())

                    # Write footer and spacing
                    outfile.write("\n```\n\n")

    print(f"Successfully bundled all code into {output_file}")


if __name__ == "__main__":
    bundle_code()