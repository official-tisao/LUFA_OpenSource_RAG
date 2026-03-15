        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file

                    # Write header
                    outfile.write(f"Filename= {file_path}\n")
                    outfile.write("CODE:\n")
                    outfile.write("```\n")

                    # Write content
