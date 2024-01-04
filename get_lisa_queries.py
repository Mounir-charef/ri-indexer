from pathlib import Path
import re


def main():
    current_dir = Path.cwd()
    query_file = current_dir / "lisa" / "LISA.QUE"
    output_dir = current_dir / "lisa_eval"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "Queries.txt"
    if output_file.exists():
        output_file.unlink()

    start_query_regex = re.compile(r"^\d+$")
    end_query_regex = re.compile(r".*#$")
    with open(query_file, "r") as file:
        content = file.readlines()

    current_query = []
    for line in content:
        line = line.strip()
        if start_query_regex.match(line):
            continue
        elif end_query_regex.match(line):
            current_query.append(line[:-1])
            # Write query to file
            with open(output_file, "a") as file:
                file.writelines(" ".join(current_query))
                file.write("\n")
            current_query = []
        else:
            current_query.append(line)


if __name__ == "__main__":
    main()
