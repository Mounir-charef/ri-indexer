import sys
from PyQt5.QtWidgets import QApplication
from Indexer.gui import MyWindow
from pathlib import Path
from Indexer import TextProcessor

current_file = Path(__file__).resolve()
DIR_PATH = current_file.parent


def init_indexer():
    collection_dir = DIR_PATH / "Collection"
    evaluation_dir = DIR_PATH / "eval"

    docs = [file.resolve() for file in collection_dir.iterdir() if file.suffix == ".txt"]
    judgement = evaluation_dir / "Judgements.txt"
    queries = evaluation_dir / "Queries.txt"

    results_dir = DIR_PATH / "results"

    return TextProcessor(docs, results_dir, judgements_path=judgement, queries_path=queries)


def main():
    processor = init_indexer()
    app = QApplication(sys.argv)
    window = MyWindow(processor)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
