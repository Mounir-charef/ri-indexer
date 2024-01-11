from pathlib import Path
import re


def main():
    current_dir = Path.cwd()
    lisa_dir = current_dir / "lisa" / "collection"
    output_dir = current_dir / "lisa_collection"
    # create output dir if it doesn't exist or overwrite it
    if output_dir.exists():
        for file in output_dir.iterdir():
            file.unlink()
    else:
        output_dir.mkdir()
    document_id_regex = re.compile(r"Document\s+(\d+)")

    for current_file in lisa_dir.iterdir():
        with open(current_file, "r") as file:
            content = file.readlines()
        current_document = []
        current_doc_id = None
        for line in content:
            line = line.strip()
            if document_id_regex.match(line):
                current_doc_id = document_id_regex.match(line).group(1)
            elif line.startswith("*") and line.endswith("*"):
                with open(output_dir / f"D{current_doc_id}.txt", "w") as file:
                    file.writelines(" ".join(current_document))
                current_document = []
            else:
                current_document.append(line)


if __name__ == "__main__":
    main()
