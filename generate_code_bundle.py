                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())

                    # Write footer and spacing
                    outfile.write("\n```\n\n")

    print(f"Successfully bundled all code into {output_file}")


if __name__ == "__main__":
    bundle_code()