import sys
import argparse
from PyQt5.QtWidgets import QApplication
from pathlib import Path
from Indexer import MyWindow
from Indexer.indexer import Indexer

current_file = Path(__file__).resolve()
DIR_PATH = current_file.parent

collection_dir = DIR_PATH / "lisa_collection"
evaluation_dir = DIR_PATH / "lisa_eval"

judgement = evaluation_dir / "Judgements.txt"
queries = evaluation_dir / "Queries.txt"
results_dir = DIR_PATH / "results"


def init_indexer():
    indexer = Indexer(
        documents_dir=collection_dir,
        results_dir=results_dir,
        judgements_path=judgement,
        queries_path=queries,
        doc_prefix="D",
    )

    return indexer


def main():
    parser = argparse.ArgumentParser(description="Indexer application")
    parser.add_argument(
        "-r",
        "--reprocess",
        action="store_true",
        help="Regenerate index files if this flag is provided",
    )

    indexer = init_indexer()

    args = parser.parse_args()
    if args.reprocess:
        indexer.processor.process_docs()
    app = QApplication(sys.argv)
    window = MyWindow(indexer)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
