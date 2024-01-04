from pathlib import Path


def main():
    current_dir = Path.cwd()
    judgements_file = current_dir / "lisa" / "LISA.REL"
    output_dir = current_dir / "lisa_eval"
    # if create output dir if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "Judgements.txt"
    if output_file.exists():
        output_file.unlink()

    with open(judgements_file, "r") as file:
        lines = file.readlines()

    current_query = None
    for line in lines:
        line = line.strip()

        if line.startswith("Query"):
            current_query = line.split()[1]
        elif "Relevant Refs" in line:
            continue
        elif line.endswith("-1"):
            *doc_ids, _ = line.split()
            for doc_id in doc_ids:
                with open(output_file, "a") as file:
                    file.write(f"{current_query} {doc_id}\n")


if __name__ == "__main__":
    main()
